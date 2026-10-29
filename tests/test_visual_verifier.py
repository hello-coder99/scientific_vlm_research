"""
Unit and integration tests for Visual Evidence Point Verifier (Phase 2).
Tests zero ground-truth hallucination detection directly against image pixels.
"""

import json
from pathlib import Path
import sys
import pytest
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.extraction.calibration import calibrate_chart
from src.verification.visual_verifier import (
    verify_chart_visually,
    detect_visual_markers,
    VerificationResult,
)

CHARTS_DIR = BASE_DIR / "data" / "charts"
PRED_DIR = BASE_DIR / "data" / "predictions"
GT_DIR = BASE_DIR / "data" / "ground_truth"


def test_marker_detection_accuracy():
    """Verify that detect_visual_markers detects exactly 5 data points on each chart."""
    for cid in ["chart_001", "chart_002", "chart_003"]:
        img_p = CHARTS_DIR / f"{cid}.png"
        calib = calibrate_chart(img_p)
        img = cv2_imread = __import__("cv2").imread(str(img_p))

        markers = detect_visual_markers(img, calib.plot_box, calib)
        assert len(markers) == 5, f"{cid}: Expected 5 visual markers, found {len(markers)}"


def test_detect_numerical_hallucination():
    """
    On chart_001, the VLM predicted (x=2, y=78), but the true mark is at y=76.
    Verify that visual verifier detects VALUE_CONTRADICTED.
    """
    img_p = CHARTS_DIR / "chart_001.png"
    pred_p = PRED_DIR / "chart_001_prediction.json"

    res = verify_chart_visually(img_p, pred_p)

    assert res.is_calibrated is True
    assert res.counts["supported"] == 4
    assert res.counts["contradicted"] == 1
    assert res.counts["fabricated"] == 0

    p2 = [p for p in res.point_results if p.x == 2.0][0]
    assert p2.status == "VALUE_CONTRADICTED"
    assert p2.claimed_y == 78.0
    assert abs(p2.observed_y - 76.0) <= 0.25
    assert abs(p2.discrepancy - 2.0) <= 0.25


def test_detect_fabricated_point():
    """
    On chart_002, the VLM claimed an extra point at (x=25, y=0.8), where no data mark exists.
    Verify that visual verifier detects FABRICATED_POINT.
    """
    img_p = CHARTS_DIR / "chart_002.png"
    pred_p = PRED_DIR / "chart_002_prediction.json"

    res = verify_chart_visually(img_p, pred_p)

    assert res.counts["supported"] == 5
    assert res.counts["fabricated"] == 1
    assert res.counts["contradicted"] == 0

    p_fab = [p for p in res.point_results if p.x == 25.0][0]
    assert p_fab.status == "FABRICATED_POINT"
    assert p_fab.observed_y is None
    assert "no visual marker exists" in p_fab.explanation.lower()


def test_detect_sign_and_title_hallucinations():
    """
    On chart_003:
    1. Point x=2016 was predicted as +0.4 (real value is -0.4). Verify VALUE_CONTRADICTED.
    2. Title was claimed as 'Global Surface Temperature Trends' (real is 'Global Temperature Anomaly').
       Verify CONTRADICTED title.
    """
    img_p = CHARTS_DIR / "chart_003.png"
    pred_p = PRED_DIR / "chart_003_prediction.json"

    res = verify_chart_visually(img_p, pred_p)

    # Point check
    p_sign = [p for p in res.point_results if p.x == 2016.0][0]
    assert p_sign.status == "VALUE_CONTRADICTED"
    assert abs(p_sign.observed_y - (-0.4)) <= 0.15
    assert abs(p_sign.discrepancy - 0.8) <= 0.15

    # Title check
    title_res = res.text_results["title"]
    assert title_res.status in ["CONTRADICTED", "UNGROUNDED"]
    assert title_res.similarity < 0.75


def test_perfect_clean_prediction():
    """
    When given a 100% veridical prediction (from ground truth),
    verify that 100% of points are marked SUPPORTED and hallucination rate is 0%.
    """
    img_p = CHARTS_DIR / "chart_001.png"
    gt_p = GT_DIR / "chart_001.json"

    with open(gt_p, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # Convert GT format to prediction format
    clean_pred = {
        "chart_id": "chart_001",
        "title": gt["title"],
        "x_axis": gt["x_axis"]["label"],
        "y_axis": gt["y_axis"]["label"],
        "points": gt["series"][0]["points"],
    }

    res = verify_chart_visually(img_p, clean_pred)

    assert res.counts["supported"] == 5
    assert res.counts["contradicted"] == 0
    assert res.counts["fabricated"] == 0
    assert res.metrics["support_rate"] == 1.0
    assert res.metrics["hallucination_rate"] == 0.0


def test_out_of_bounds_detection():
    """Verify that points placed outside the canvas boundaries are flagged as OUT_OF_BOUNDS."""
    img_p = CHARTS_DIR / "chart_001.png"
    bad_pred = {
        "chart_id": "chart_001",
        "title": "Model Accuracy Across Epochs",
        "x_axis": "Epoch",
        "y_axis": "Accuracy (%)",
        "points": [
            {"x": 99.0, "y": 999.0},  # Wildly out of bounds
            {"x": 1.0, "y": 72.0},
        ],
    }

    res = verify_chart_visually(img_p, bad_pred)
    bad_pt = [p for p in res.point_results if p.x == 99.0][0]
    assert bad_pt.status == "OUT_OF_BOUNDS"
