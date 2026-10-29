"""
Main Runner for Evidence-Grounded Scientific Chart Verification.

Integrates all 4 Research Phases:
1. Phase 1: Automatic Axis Calibration & Coordinate Mapping
2. Phase 2: Evidence-Grounded Visual Verification (Zero Ground-Truth Lie Detector)
3. Phase 3: Visual Evidence Debugger (Overlays, Proof Crops, HTML Certificates)
4. Phase 4: VLM Benchmark & Statistical Lie Detector Validation Experiment
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure both src and project root are in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SRC_DIR.parent
for _p in (str(_SRC_DIR), str(_BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.extraction.calibration import calibrate_chart
    from src.extraction.evidence import build_evidence_package
    from src.verification.visual_verifier import verify_chart_visually
    from src.verification.visual_debugger import generate_verification_package
    from src.vlm.benchmark import run_vlm_benchmark, generate_markdown_table
    from src.vlm.models import SimulatedVLM
    from src.vlm.evaluation_experiment import (
        run_lie_detector_validation,
        generate_experiment_markdown_table,
    )
except ImportError:
    from extraction.calibration import calibrate_chart
    from extraction.evidence import build_evidence_package
    from verification.visual_verifier import verify_chart_visually
    from verification.visual_debugger import generate_verification_package
    from vlm.benchmark import run_vlm_benchmark, generate_markdown_table
    from vlm.models import SimulatedVLM
    from vlm.evaluation_experiment import (
        run_lie_detector_validation,
        generate_experiment_markdown_table,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evidence-Grounded Scientific Chart Verifier (Zero Ground-Truth Visual Lie Detector)"
    )
    parser.add_argument("--image", type=str, help="Path to chart image file (.png, .jpg)")
    parser.add_argument("--prediction", type=str, help="Path to VLM prediction JSON file")
    parser.add_argument("--output", type=str, default=None, help="Directory to save overlay and HTML certificate")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Relative numerical tolerance (default: 0.03 = 3%%)")
    args = parser.parse_args()

    base_dir = _BASE_DIR
    charts_dir = base_dir / "data" / "charts"
    preds_dir = base_dir / "data" / "predictions"
    overlays_dir = base_dir / "data" / "verification_overlays"
    evidence_dir = base_dir / "data" / "evidence"
    gt_dir = base_dir / "data" / "ground_truth"

    # Custom single chart verification from CLI
    if args.image and args.prediction:
        img_p = Path(args.image)
        pred_p = Path(args.prediction)
        out_dir = Path(args.output) if args.output else overlays_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not img_p.exists():
            print(f"Error: Image file not found: {img_p}")
            sys.exit(1)
        if not pred_p.exists():
            print(f"Error: Prediction file not found: {pred_p}")
            sys.exit(1)

        print("=" * 78)
        print(f"  VERIFYING CHART: {img_p.name}")
        print("=" * 78)
        res, overlay_path, html_path = generate_verification_package(
            image_input=img_p,
            prediction_input=pred_p,
            output_dir=out_dir,
            chart_id=img_p.stem,
            rel_tolerance=args.tolerance,
        )
        print(res.summary())
        print(f"-> Annotated Overlay Saved: {overlay_path}")
        print(f"-> HTML Certificate Saved:  {html_path}")
        return

    overlays_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EVIDENCE-GROUNDED SCIENTIFIC CHART VERIFIER (PHASES 1, 2, 3, 4)")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # PART 1: End-to-End Verification on Core Benchmark Scenarios
    # -------------------------------------------------------------------------
    benchmark_cases = [
        ("chart_001", "Numerical Discrepancy Hallucination (x=2, y=78 vs true 76)"),
        ("chart_002", "Fabricated Point Hallucination (invented point x=25, y=0.8)"),
        ("chart_003", "Sign & Title Hallucinations (y=+0.4 vs true -0.4)"),
    ]

    print("\n>>> PART 1: ZERO GROUND-TRUTH VISUAL EVIDENCE VERIFICATION")
    for cid, description in benchmark_cases:
        img_p = charts_dir / f"{cid}.png"
        pred_p = preds_dir / f"{cid}_prediction.json"

        if not (img_p.exists() and pred_p.exists()):
            continue

        res, overlay_path, html_path = generate_verification_package(
            image_input=img_p,
            prediction_input=pred_p,
            output_dir=overlays_dir,
            chart_id=cid,
        )

        print(res.summary())
        print(f"-> Annotated Overlay Saved: {overlay_path}")
        print(f"-> HTML Certificate Saved:  {html_path}")

    # -------------------------------------------------------------------------
    # PART 2: Comparative VLM Benchmark Across 5 Scientific Charts
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(">>> PART 2: VLM COMPARATIVE BENCHMARK ACROSS 5 SCIENTIFIC CHARTS")
    print("=" * 78)

    test_models = [
        SimulatedVLM(model_name="GPT-4o (Simulated)", profile="precise", random_seed=101),
        SimulatedVLM(model_name="Claude-3.5 (Simulated)", profile="precise", random_seed=202),
        SimulatedVLM(model_name="LLaVA-Dense (Simulated)", profile="numerical_drift", random_seed=303),
        SimulatedVLM(model_name="Ungrounded-VLM (Simulated)", profile="fabricator", random_seed=404),
    ]

    all_charts = ["chart_001", "chart_002", "chart_003", "chart_004", "chart_005"]
    bench_results = run_vlm_benchmark(test_models, all_charts, charts_dir, output_dir=evidence_dir)
    print("\n" + generate_markdown_table(bench_results))

    # -------------------------------------------------------------------------
    # PART 3: Statistical Lie Detector Validation Experiment (N=100)
    # -------------------------------------------------------------------------
    print("=" * 78)
    print(">>> PART 3: STATISTICAL LIE DETECTOR VALIDATION EXPERIMENT")
    print("=" * 78)

    metrics, cat_rates = run_lie_detector_validation(charts_dir, gt_dir, output_dir=evidence_dir)
    print("\n" + generate_experiment_markdown_table(metrics, cat_rates))

    # Build and save multi-modal evidence package for chart_001
    sample_img = charts_dir / "chart_001.png"
    sample_evidence = evidence_dir / "chart_001_evidence.json"
    if sample_img.exists():
        build_evidence_package(sample_img, sample_evidence)

    print("=" * 78)
    print("VERIFICATION SUITE COMPLETE: ALL ARTIFACTS AND BENCHMARKS READY.")
    print("=" * 78)


if __name__ == "__main__":
    main()