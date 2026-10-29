"""
Visual Evidence Debugger & Overlay Generator (Phase 3).

Generates human-interpretable and machine-actionable visual artifacts:
1. Annotated Verification Overlay Images (chart_xxx_verified.png):
   - 🟢 Green concentric rings for SUPPORTED points.
   - 🔴 Red crosses and error vectors for VALUE_CONTRADICTED hallucinations.
   - 🟣 Magenta dashed markers for FABRICATED_POINT hallucinations.
   - 🔵 Cyan diamonds for OMITTED_POINT unextracted markers.
   - 🟠 Warning boxes on UNGROUNDED or CONTRADICTED text labels.
   - Sleek translucent diagnostic HUD banner with score badges.
2. High-resolution ROI Evidence Crops for each claimed coordinate.
3. Standalone Interactive HTML Verification Certificate with embedded
   visual proof cards.
"""

import base64
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ..extraction.calibration import calibrate_chart, ChartCalibration, PlotBox
    from .visual_verifier import VerificationResult, VisualProbeResult, TextGroundingResult
except ImportError:
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    from extraction.calibration import calibrate_chart, ChartCalibration, PlotBox
    from verification.visual_verifier import VerificationResult, VisualProbeResult, TextGroundingResult


# Color Palette (BGR format for OpenCV)
COLOR_GREEN = (46, 204, 113)      # 🟢 SUPPORTED
COLOR_RED = (52, 73, 235)         # 🔴 CONTRADICTED
COLOR_MAGENTA = (200, 50, 220)    # 🟣 FABRICATED
COLOR_CYAN = (235, 180, 50)       # 🔵 OMITTED
COLOR_ORANGE = (39, 134, 245)     # 🟠 TEXT WARNING
COLOR_DARK_BG = (25, 25, 30)      # Translucent HUD
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)


def draw_text_badge(
    image: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = COLOR_WHITE,
    font_scale: float = 0.42,
    thickness: int = 1,
    padding: int = 4
) -> None:
    """Draws a crisp text badge with a solid background pill."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = min(image.shape[1] - 1, x + tw + padding)
    y2 = min(image.shape[0] - 1, y + baseline + padding)

    # Draw rounded-like box
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def render_verification_overlay(
    image_input: Union[str, Path, np.ndarray],
    verification_result: VerificationResult,
    calibration: ChartCalibration,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Renders an annotated verification overlay on the chart image.
    Visualizes supported claims, error vectors for hallucinations,
    fabricated points, and diagnostic score badges.
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    overlay = image.copy()
    h, w = overlay.shape[:2]
    pb = calibration.plot_box

    # 1. Subtle outline around the detected plot canvas
    cv2.rectangle(
        overlay,
        (pb.x_min, pb.y_min),
        (pb.x_max, pb.y_max),
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )

    # 2. Draw Diagnostic Header HUD Banner
    banner_h = 50
    header_crop = overlay[0:banner_h, :]
    dark_mask = np.full(header_crop.shape, 20, dtype=np.uint8)
    blended_banner = cv2.addWeighted(header_crop, 0.25, dark_mask, 0.75, 0)
    overlay[0:banner_h, :] = blended_banner
    cv2.line(overlay, (0, banner_h), (w, banner_h), (80, 80, 80), 1)

    # Title in HUD
    hud_title = f"VERIFIED: {verification_result.chart_id}"
    cv2.putText(overlay, hud_title, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

    # Metric pills on the right of HUD
    supp_cnt = verification_result.counts.get("supported", 0)
    contra_cnt = verification_result.counts.get("contradicted", 0)
    fab_cnt = verification_result.counts.get("fabricated", 0)
    ground_score = verification_result.metrics.get("overall_grounding_score", 0.0) * 100

    pill_y = 32
    draw_text_badge(overlay, f"GROUNDING: {ground_score:.1f}%", (w - 180, pill_y), (80, 80, 80), COLOR_WHITE, 0.45)
    draw_text_badge(overlay, f"PASS: {supp_cnt}", (w - 280, pill_y), COLOR_GREEN, COLOR_WHITE, 0.45)
    if contra_cnt > 0:
        draw_text_badge(overlay, f"CONTRADICTED: {contra_cnt}", (w - 435, pill_y), COLOR_RED, COLOR_WHITE, 0.45)
    if fab_cnt > 0:
        draw_text_badge(overlay, f"FABRICATED: {fab_cnt}", (w - 565, pill_y), COLOR_MAGENTA, COLOR_WHITE, 0.45)

    # 3. Annotate Data Points
    for pr in verification_result.point_results:
        c_px, c_py = int(round(pr.claimed_pixel[0])), int(round(pr.claimed_pixel[1]))

        if pr.status == "SUPPORTED":
            # Green concentric target rings
            cv2.circle(overlay, (c_px, c_py), 13, COLOR_GREEN, 2, cv2.LINE_AA)
            cv2.circle(overlay, (c_px, c_py), 3, COLOR_GREEN, -1, cv2.LINE_AA)
            badge_pos = (min(w - 110, c_px + 14), max(banner_h + 15, c_py - 10))
            draw_text_badge(overlay, f"✓ ({pr.x:g}, {pr.claimed_y:g})", badge_pos, COLOR_GREEN, COLOR_WHITE)

        elif pr.status == "VALUE_CONTRADICTED":
            # Red circle + cross at claimed location
            cv2.circle(overlay, (c_px, c_py), 14, COLOR_RED, 2, cv2.LINE_AA)
            cv2.line(overlay, (c_px - 8, c_py - 8), (c_px + 8, c_py + 8), COLOR_RED, 2, cv2.LINE_AA)
            cv2.line(overlay, (c_px - 8, c_py + 8), (c_px + 8, c_py - 8), COLOR_RED, 2, cv2.LINE_AA)

            # If observed marker was found, draw green target and arrow vector
            if pr.observed_pixel is not None:
                o_px, o_py = int(round(pr.observed_pixel[0])), int(round(pr.observed_pixel[1]))
                cv2.circle(overlay, (o_px, o_py), 13, COLOR_GREEN, 2, cv2.LINE_AA)
                cv2.circle(overlay, (o_px, o_py), 3, COLOR_GREEN, -1, cv2.LINE_AA)

                # Vector line connecting claimed to observed location
                cv2.arrowedLine(overlay, (c_px, c_py), (o_px, o_py), COLOR_RED, 2, cv2.LINE_AA, tipLength=0.2)

            badge_text = f"FAIL: Claim {pr.claimed_y:g} != Obs {pr.observed_y:.2f} (Δ={pr.discrepancy:.2f})"
            badge_pos = (min(w - 240, max(10, c_px + 15)), max(banner_h + 15, c_py - 12))
            draw_text_badge(overlay, badge_text, badge_pos, COLOR_RED, COLOR_WHITE)

        elif pr.status == "FABRICATED_POINT":
            # Magenta dashed square
            d = 12
            cv2.rectangle(overlay, (c_px - d, c_py - d), (c_px + d, c_py + d), COLOR_MAGENTA, 2, cv2.LINE_AA)
            cv2.line(overlay, (c_px - d, c_py), (c_px + d, c_py), COLOR_MAGENTA, 1, cv2.LINE_AA)
            cv2.line(overlay, (c_px, c_py - d), (c_px, c_py + d), COLOR_MAGENTA, 1, cv2.LINE_AA)

            badge_text = f"FABRICATED POINT: x={pr.x:g}"
            badge_pos = (min(w - 200, max(10, c_px + 15)), max(banner_h + 15, c_py - 12))
            draw_text_badge(overlay, badge_text, badge_pos, COLOR_MAGENTA, COLOR_WHITE)

        elif pr.status == "OUT_OF_BOUNDS":
            draw_text_badge(overlay, f"OUT OF BOUNDS: ({pr.x:g}, {pr.claimed_y:g})", (max(10, c_px), max(banner_h + 15, c_py)), (0, 0, 180), COLOR_WHITE)

    # 4. Highlight Omitted Visual Marks (if any)
    for op in verification_result.omitted_points:
        opx, opy = int(round(op["px"])), int(round(op["py"]))
        # Cyan diamond
        pts = np.array([[opx, opy - 9], [opx + 9, opy], [opx, opy + 9], [opx - 9, opy]], np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=COLOR_CYAN, thickness=2, lineType=cv2.LINE_AA)
        draw_text_badge(overlay, f"OMITTED: ({op['x']:.2f}, {op['y']:.2f})", (min(w - 130, opx + 12), opy), COLOR_CYAN, COLOR_WHITE)

    # 5. Legend Pill Box in bottom-right corner
    legend_y = h - 25
    draw_text_badge(overlay, "LEGEND: ", (w - 470, legend_y), (60, 60, 60), COLOR_WHITE, 0.38)
    draw_text_badge(overlay, "SUPPORTED", (w - 400, legend_y), COLOR_GREEN, COLOR_WHITE, 0.38)
    draw_text_badge(overlay, "CONTRADICTED", (w - 305, legend_y), COLOR_RED, COLOR_WHITE, 0.38)
    draw_text_badge(overlay, "FABRICATED", (w - 185, legend_y), COLOR_MAGENTA, COLOR_WHITE, 0.38)
    draw_text_badge(overlay, "OMITTED", (w - 85, legend_y), COLOR_CYAN, COLOR_WHITE, 0.38)

    # Save to disk if requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)

    return overlay


# =========================================================================
# High-Resolution ROI Crop Generator
# =========================================================================

def extract_evidence_crops(
    image: np.ndarray,
    verification_result: VerificationResult,
    output_dir: Optional[Union[str, Path]] = None,
    crop_size: int = 70
) -> List[dict]:
    """
    Extracts high-resolution visual evidence crops centered at each claimed coordinate.
    Embeds crosshairs and verdict tags. Returns crop metadata list.
    """
    h, w = image.shape[:2]
    half = crop_size // 2
    crop_records = []

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    for idx, pr in enumerate(verification_result.point_results):
        c_px = int(round(pr.claimed_pixel[0]))
        c_py = int(round(pr.claimed_pixel[1]))

        # Calculate bounding window
        x1 = max(0, c_px - half)
        y1 = max(0, c_py - half)
        x2 = min(w, c_px + half)
        y2 = min(h, c_py + half)

        patch = image[y1:y2, x1:x2].copy()
        if patch.size == 0:
            continue

        annotated_patch = cv2.resize(patch, (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        ph, pw = annotated_patch.shape[:2]
        center_x = int((c_px - x1) / (x2 - x1) * pw)
        center_y = int((c_py - y1) / (y2 - y1) * ph)

        # Crosshairs at claimed position
        color = COLOR_GREEN if pr.status == "SUPPORTED" else (COLOR_RED if pr.status == "VALUE_CONTRADICTED" else COLOR_MAGENTA)
        cv2.drawMarker(annotated_patch, (center_x, center_y), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        # Border according to status
        cv2.rectangle(annotated_patch, (0, 0), (pw - 1, ph - 1), color, 3)

        # Save patch if out_dir given
        file_path = None
        if out_dir is not None:
            clean_status = pr.status.lower()
            fname = f"{verification_result.chart_id}_pt{idx+1}_x{pr.x}_y{pr.claimed_y}_{clean_status}.png"
            file_path = str(out_dir / fname)
            cv2.imwrite(file_path, annotated_patch)

        # Convert to base64 for direct HTML embedding
        _, buf = cv2.imencode(".png", annotated_patch)
        b64_str = base64.b64encode(buf).decode("utf-8")

        crop_records.append({
            "point_index": idx + 1,
            "x": pr.x,
            "claimed_y": pr.claimed_y,
            "observed_y": pr.observed_y,
            "status": pr.status,
            "explanation": pr.explanation,
            "file_path": file_path,
            "base64": b64_str,
        })

    return crop_records


# =========================================================================
# Standalone Interactive HTML Verification Certificate
# =========================================================================

def generate_html_certificate(
    verification_result: VerificationResult,
    overlay_image_path: Union[str, Path],
    crop_records: List[dict],
    output_html_path: Union[str, Path]
) -> None:
    """
    Compiles an Evidence-Grounded HTML Verification Certificate with dark aesthetics,
    embedded overlay graphics, and visual proof cards.
    """
    out_html = Path(output_html_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Encode overlay image to base64 for self-contained portability
    with open(overlay_image_path, "rb") as f:
        overlay_b64 = base64.b64encode(f.read()).decode("utf-8")

    m = verification_result.metrics
    c = verification_result.counts
    grounding_pct = f"{m.get('overall_grounding_score', 0.0) * 100:.1f}%"
    support_pct = f"{m.get('support_rate', 0.0) * 100:.1f}%"
    halluc_pct = f"{m.get('hallucination_rate', 0.0) * 100:.1f}%"

    cards_html = ""
    for cr in crop_records:
        st = cr["status"]
        if st == "SUPPORTED":
            badge_class = "badge-supported"
            status_text = "SUPPORTED"
        elif st == "VALUE_CONTRADICTED":
            badge_class = "badge-contradicted"
            status_text = "VALUE CONTRADICTED"
        elif st == "FABRICATED_POINT":
            badge_class = "badge-fabricated"
            status_text = "FABRICATED POINT"
        else:
            badge_class = "badge-other"
            status_text = st

        obs_val = f"{cr['observed_y']:.2f}" if cr['observed_y'] is not None else "None (Blank)"
        diff_val = f"{abs(cr['claimed_y'] - cr['observed_y']):.2f}" if cr['observed_y'] is not None else "N/A"

        cards_html += f"""
        <div class="evidence-card">
            <div class="crop-container">
                <img src="data:image/png;base64,{cr['base64']}" alt="Point Proof Crop" />
            </div>
            <div class="card-details">
                <div class="card-header">
                    <span class="point-tag">Point #{cr['point_index']} (x = {cr['x']})</span>
                    <span class="badge {badge_class}">{status_text}</span>
                </div>
                <div class="reading-row">
                    <div class="reading-col">
                        <span class="label">Claimed Y</span>
                        <span class="value">{cr['claimed_y']}</span>
                    </div>
                    <div class="reading-col">
                        <span class="label">Observed Y</span>
                        <span class="value">{obs_val}</span>
                    </div>
                    <div class="reading-col">
                        <span class="label">Error (Δ)</span>
                        <span class="value">{diff_val}</span>
                    </div>
                </div>
                <p class="explanation">{cr['explanation']}</p>
            </div>
        </div>
        """

    text_rows_html = ""
    for field_name, tr in verification_result.text_results.items():
        st_class = "badge-supported" if tr.status == "SUPPORTED" else ("badge-contradicted" if tr.status == "CONTRADICTED" else "badge-fabricated")
        text_rows_html += f"""
        <tr>
            <td style="font-weight:600; text-transform:capitalize;">{field_name}</td>
            <td><code>{tr.claimed_text}</code></td>
            <td><code>{tr.observed_text or 'None detected'}</code></td>
            <td>{(tr.similarity * 100):.1f}%</td>
            <td><span class="badge {st_class}">{tr.status}</span></td>
            <td style="color:#a0aec0; font-size:0.85rem;">{tr.explanation}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Verification Certificate - {verification_result.chart_id}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
            --magenta: #d946ef;
            --cyan: #06b6d4;
            --accent: #38bdf8;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--surface-border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 6px 0;
            font-size: 1.75rem;
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 10px;
            padding: 16px 20px;
        }}
        .stat-card .title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .stat-card .val {{
            font-size: 1.8rem;
            font-weight: 700;
        }}
        .overlay-section {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .overlay-img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }}
        .section-title {{
            font-size: 1.3rem;
            margin: 30px 0 16px 0;
            font-weight: 600;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            display: inline-block;
        }}
        .badge-supported {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
        .badge-contradicted {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-fabricated {{ background: rgba(217, 70, 239, 0.2); color: #e879f9; border: 1px solid rgba(217, 70, 239, 0.4); }}
        .evidence-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 16px;
        }}
        .evidence-card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 10px;
            display: flex;
            padding: 14px;
            gap: 14px;
            align-items: center;
        }}
        .crop-container img {{
            width: 90px;
            height: 90px;
            border-radius: 8px;
            display: block;
        }}
        .card-details {{
            flex: 1;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .point-tag {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .reading-row {{
            display: flex;
            gap: 12px;
            background: rgba(15, 23, 42, 0.5);
            padding: 6px 10px;
            border-radius: 6px;
            margin-bottom: 8px;
        }}
        .reading-col .label {{
            font-size: 0.68rem;
            color: var(--text-secondary);
            display: block;
            text-transform: uppercase;
        }}
        .reading-col .value {{
            font-size: 0.9rem;
            font-weight: 700;
        }}
        .explanation {{
            margin: 0;
            font-size: 0.82rem;
            color: #cbd5e1;
            line-height: 1.3;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--surface-border);
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--surface-border);
            font-size: 0.9rem;
        }}
        th {{
            background: rgba(15, 23, 42, 0.6);
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Evidence Grounding Certificate</h1>
                <div class="subtitle">Deterministic Zero Ground-Truth Verification | Chart ID: <code>{verification_result.chart_id}</code></div>
            </div>
            <div>
                <span class="badge badge-supported" style="font-size: 0.9rem; padding: 6px 14px;">
                    Calibration: {'VERIFIED' if verification_result.is_calibrated else 'FAILED'}
                </span>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="title">Overall Grounding Score</div>
                <div class="val" style="color: var(--accent);">{grounding_pct}</div>
            </div>
            <div class="stat-card">
                <div class="title">Supported Points</div>
                <div class="val" style="color: var(--green);">{c.get('supported', 0)} / {c.get('total_claimed', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="title">Hallucination Rate</div>
                <div class="val" style="color: var(--red);">{halluc_pct}</div>
            </div>
            <div class="stat-card">
                <div class="title">Fabricated Points</div>
                <div class="val" style="color: var(--magenta);">{c.get('fabricated', 0)}</div>
            </div>
        </div>

        <div class="overlay-section">
            <h2 style="margin: 0 0 14px 0; font-size: 1.15rem;">Visual Evidence Overlay</h2>
            <img class="overlay-img" src="data:image/png;base64,{overlay_b64}" alt="Verification Overlay Image" />
        </div>

        <h2 class="section-title">Point-by-Point Visual Proof Cards</h2>
        <div class="evidence-grid">
            {cards_html}
        </div>

        <h2 class="section-title">Semantic Text Evidence Grounding</h2>
        <table>
            <thead>
                <tr>
                    <th>Element</th>
                    <th>Claimed Extracted Text</th>
                    <th>Observed OCR Evidence</th>
                    <th>Similarity</th>
                    <th>Status</th>
                    <th>Explanation</th>
                </tr>
            </thead>
            <tbody>
                {text_rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)


# =========================================================================
# High-Level Debugger Runner
# =========================================================================

def generate_verification_package(
    image_input: Union[str, Path, np.ndarray],
    prediction_input: Union[str, Path, dict],
    output_dir: Union[str, Path],
    chart_id: Optional[str] = None,
    rel_tolerance: float = 0.03
) -> Tuple[VerificationResult, Path, Path]:
    """
    Runs the complete Phase 3 Visual Verification pipeline:
    1. Calibrates coordinate system and verifies points visually.
    2. Renders annotated verification overlay image.
    3. Extracts high-resolution visual proof crops.
    4. Generates standalone interactive HTML verification certificate.

    Returns: (verification_result, overlay_image_path, html_certificate_path)
    """
    from .visual_verifier import verify_chart_visually

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run verification
    res = verify_chart_visually(image_input, prediction_input, rel_tolerance=rel_tolerance)
    cid = chart_id or res.chart_id

    # Load image for overlay drawing
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input.copy()

    # Get calibration
    calib = calibrate_chart(img)

    # 1. Overlay image
    overlay_path = out_dir / f"{cid}_verified.png"
    render_verification_overlay(img, res, calib, output_path=overlay_path)

    # 2. Evidence crops
    crops_dir = out_dir / "crops"
    crop_records = extract_evidence_crops(img, res, output_dir=crops_dir, crop_size=70)

    # 3. HTML certificate
    html_path = out_dir / f"{cid}_certificate.html"
    generate_html_certificate(res, overlay_path, crop_records, html_path)

    return res, overlay_path, html_path
