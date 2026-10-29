"""
FastAPI Backend for Evidence-Grounded Scientific Chart Verifier.
Serves the interactive web demo and provides REST APIs for zero-GT verification.
"""

import base64
import json
import os
from pathlib import Path
import sys
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

# Ensure project root and src are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
for p in (str(BASE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import dotenv
    dotenv.load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

from src.extraction.calibration import calibrate_chart
from src.verification.visual_verifier import verify_chart_visually
from src.verification.visual_debugger import (
    render_verification_overlay,
    extract_evidence_crops,
    generate_html_certificate,
    generate_verification_package,
)
from src.vlm.benchmark import run_vlm_benchmark
from src.vlm.models import SimulatedVLM, GeminiVLM, AnthropicVLM, OpenAIVLM, GrokVLM

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHARTS_DIR = BASE_DIR / "data" / "charts"
PRED_DIR = BASE_DIR / "data" / "predictions"
OVERLAYS_DIR = BASE_DIR / "data" / "verification_overlays"
GT_DIR = BASE_DIR / "data" / "ground_truth"

OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Evidence-Grounded Scientific Chart Verifier",
    description="Interactive Web Demo & Verification API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def decode_image_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file: Could not decode pixels.")
    return img


def encode_image_base64(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image to PNG.")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -----------------------------------------------------------------------------
# Frontend Root Route
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Web demo frontend not found.</h1>", status_code=404)
    return FileResponse(str(index_file))


# -----------------------------------------------------------------------------
# Benchmark Scenarios
# -----------------------------------------------------------------------------
@app.get("/api/scenarios")
async def get_scenarios():
    """Returns list of curated benchmark charts with injected VLM hallucinations."""
    scenarios_meta = [
        {
            "id": "chart_001",
            "title": "Model Accuracy Across Epochs",
            "chart_type": "Standard Line Plot",
            "category": "Numerical Value Drift",
            "description": "True mark at x=2 is y=76.0, but VLM claimed y=78.0 (+2.04 numerical shift).",
            "image_url": "/api/scenarios/chart_001/image",
        },
        {
            "id": "chart_002",
            "title": "Training Loss Over Iterations",
            "chart_type": "Exponential Decay Curve",
            "category": "Fabricated Intermediate Point",
            "description": "VLM invented an ungrounded point at x=25.0, y=0.80 where no visual mark exists.",
            "image_url": "/api/scenarios/chart_002/image",
        },
        {
            "id": "chart_003",
            "title": "Global Temperature Anomaly",
            "chart_type": "Negative & Positive Coordinates",
            "category": "Sign Inversion & Title Hallucination",
            "description": "Negative point x=2016 claimed as +0.4 (true -0.4); title hallucinated as 'Trends'.",
            "image_url": "/api/scenarios/chart_003/image",
        },
        {
            "id": "chart_004",
            "title": "Enzyme Reaction Velocity",
            "chart_type": "Biochemical Kinetics (Diamonds)",
            "category": "Numerical Shift on Kinetic Curve",
            "description": "Point at x=6 claimed as y=46.0 instead of true y=41.0 (+12.2% error).",
            "image_url": "/api/scenarios/chart_004/image",
        },
        {
            "id": "chart_005",
            "title": "Bandpass Filter Response",
            "chart_type": "Decibel Gain (Pentagons & Negative Axis)",
            "category": "Omitted Point & Fabricated Peak",
            "description": "VLM omitted marker at x=50, and fabricated an extra point at x=35.",
            "image_url": "/api/scenarios/chart_005/image",
        },
    ]

    for s in scenarios_meta:
        cid = s["id"]
        pred_p = PRED_DIR / f"{cid}_prediction.json"
        if pred_p.exists():
            with open(pred_p, "r", encoding="utf-8") as f:
                s["prediction"] = json.load(f)
        else:
            s["prediction"] = {}

    return JSONResponse(scenarios_meta)


@app.get("/api/scenarios/{chart_id}/image")
async def get_scenario_image(chart_id: str):
    img_p = CHARTS_DIR / f"{chart_id}.png"
    if not img_p.exists():
        raise HTTPException(status_code=404, detail=f"Chart image '{chart_id}.png' not found.")
    return FileResponse(str(img_p), media_type="image/png")


# -----------------------------------------------------------------------------
# Core Verification API
# -----------------------------------------------------------------------------
@app.post("/api/verify")
async def verify_chart_endpoint(
    chart_id: Optional[str] = Form(None),
    tolerance: float = Form(0.03),
    prediction_json: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
):
    """
    Executes Zero Ground-Truth Verification for either an uploaded chart image
    or a selected benchmark scenario.
    """
    # 1. Determine Image
    img_cv2: Optional[np.ndarray] = None
    target_cid = chart_id or "custom_chart"

    if image_file and image_file.filename:
        data = await image_file.read()
        try:
            img_cv2 = decode_image_bytes(data)
            target_cid = Path(image_file.filename).stem or "custom_chart"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")
    elif chart_id:
        img_p = CHARTS_DIR / f"{chart_id}.png"
        if not img_p.exists():
            raise HTTPException(status_code=404, detail=f"Benchmark chart '{chart_id}' not found.")
        img_cv2 = cv2.imread(str(img_p))
        if img_cv2 is None:
            raise HTTPException(status_code=500, detail="Failed to load benchmark chart image.")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image or a valid chart_id.")

    # 2. Parse Prediction JSON
    if not prediction_json:
        raise HTTPException(status_code=400, detail="Prediction JSON payload is required.")

    try:
        pred_dict = json.loads(prediction_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in prediction: {e}")

    # 3. Perform Calibration & Visual Verification
    try:
        calib = calibrate_chart(img_cv2)
        v_res = verify_chart_visually(img_cv2, pred_dict, calibration=calib, rel_tolerance=tolerance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification engine failure: {e}")

    # 4. Generate Visual Debugger Artifacts
    try:
        out_overlay_path = OVERLAYS_DIR / f"{target_cid}_verified.png"
        out_html_path = OVERLAYS_DIR / f"{target_cid}_certificate.html"

        overlay_img = render_verification_overlay(img_cv2, v_res, calib, output_path=out_overlay_path)
        crop_records = extract_evidence_crops(img_cv2, v_res, output_dir=None, crop_size=70)
        generate_html_certificate(v_res, out_overlay_path, crop_records, out_html_path)

        overlay_b64 = encode_image_base64(overlay_img)
        orig_b64 = encode_image_base64(img_cv2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifact generation failure: {e}")

    # 5. Build Comprehensive Response
    return JSONResponse({
        "chart_id": target_cid,
        "is_calibrated": v_res.is_calibrated,
        "calibration": {
            "x_axis": {
                "r_squared": calib.x_axis.r_squared,
                "slope": calib.x_axis.slope,
                "inlier_count": calib.x_axis.inlier_count,
            },
            "y_axis": {
                "r_squared": calib.y_axis.r_squared,
                "slope": calib.y_axis.slope,
                "inlier_count": calib.y_axis.inlier_count,
            },
            "plot_box": calib.plot_box.to_dict(),
        },
        "counts": v_res.counts,
        "metrics": v_res.metrics,
        "points": [p.to_dict() for p in v_res.point_results],
        "proof_crops": crop_records,
        "text_verification": {k: v.to_dict() for k, v in v_res.text_results.items()},
        "omitted_points": v_res.omitted_points,
        "original_image_base64": orig_b64,
        "overlay_image_base64": overlay_b64,
        "certificate_url": f"/api/certificate/{target_cid}",
    })


@app.get("/api/certificate/{chart_id}")
async def get_certificate(chart_id: str):
    html_p = OVERLAYS_DIR / f"{chart_id}_certificate.html"
    if not html_p.exists():
        raise HTTPException(status_code=404, detail=f"Certificate for '{chart_id}' not found. Run verification first.")
    return FileResponse(str(html_p), media_type="text/html")


# -----------------------------------------------------------------------------
# Live VLM Benchmark & Single-Chart Extraction APIs
# -----------------------------------------------------------------------------
@app.get("/api/benchmark")
async def run_benchmark_endpoint():
    """
    Runs comparative evaluation across the multimodal VLM fleet
    (Google Gemini, Anthropic Claude, OpenAI GPT, xAI Grok, and Simulated Baseline).
    Tracks genuine vs. simulated execution modes transparently.
    """
    test_models = [
        GeminiVLM(model_name="Google Gemini 2.5 Flash"),
        AnthropicVLM(model_name="Anthropic Claude 3.5 Sonnet"),
        OpenAIVLM(model_name="OpenAI GPT-4o-mini"),
        GrokVLM(model_name="xAI Grok-2 Vision"),
        SimulatedVLM(model_name="Baseline Simulation (Ungrounded)", profile="fabricator", random_seed=404),
    ]
    all_charts = ["chart_001", "chart_002", "chart_003", "chart_004", "chart_005"]
    bench_results = run_vlm_benchmark(test_models, all_charts, CHARTS_DIR)

    output = []
    for m_name, res in bench_results.items():
        output.append(res.to_dict())

    return JSONResponse(output)


@app.post("/api/extract_vlm")
async def extract_vlm_endpoint(
    chart_id: Optional[str] = Form(None),
    model_choice: str = Form("gemini"),
    image_file: Optional[UploadFile] = File(None),
):
    """
    Extracts structured chart claims from an image using the selected live LLM/VLM.
    If the API call fails or is unconfigured, seamlessly falls back to simulation and
    clearly flags whether the output is GENUINE or SIMULATED.
    """
    img_input = None
    target_cid = chart_id or "chart"

    if image_file and image_file.filename:
        data = await image_file.read()
        img_input = data
        target_cid = Path(image_file.filename).stem or "chart"
    elif chart_id:
        img_p = CHARTS_DIR / f"{chart_id}.png"
        if not img_p.exists():
            raise HTTPException(status_code=404, detail=f"Chart '{chart_id}' not found.")
        img_input = str(img_p)
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image or chart_id.")

    model_key = model_choice.lower()
    if "gemini" in model_key:
        vlm = GeminiVLM(model_name="Google Gemini 2.5 Flash")
    elif "anthropic" in model_key or "claude" in model_key:
        vlm = AnthropicVLM(model_name="Anthropic Claude 3.5 Sonnet")
    elif "openai" in model_key or "gpt" in model_key:
        vlm = OpenAIVLM(model_name="OpenAI GPT-4o-mini")
    elif "grok" in model_key or "xai" in model_key:
        vlm = GrokVLM(model_name="xAI Grok-2 Vision")
    else:
        vlm = SimulatedVLM(model_name="Simulated-Baseline", profile="precise")

    extraction = vlm.extract_chart_data(img_input)

    return JSONResponse({
        "chart_id": target_cid,
        "model_name": vlm.model_name,
        "provider": vlm.provider,
        "is_genuine": vlm.is_genuine,
        "execution_mode": vlm.execution_mode,
        "status_note": vlm.status_note,
        "prediction": extraction,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=True)
