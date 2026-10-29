"""
Unit and integration tests for Visual Evidence Debugger (Phase 3).
Tests overlay rendering, ROI crop extraction, and HTML certificate generation.
"""

from pathlib import Path
import sys
import pytest
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.extraction.calibration import calibrate_chart
from src.verification.visual_verifier import verify_chart_visually
from src.verification.visual_debugger import (
    render_verification_overlay,
    extract_evidence_crops,
    generate_html_certificate,
    generate_verification_package,
)

CHARTS_DIR = BASE_DIR / "data" / "charts"
PRED_DIR = BASE_DIR / "data" / "predictions"
OUT_DIR = BASE_DIR / "data" / "verification_overlays"


def test_render_verification_overlay():
    """Verify overlay image dimensions, drawing, and file output."""
    img_p = CHARTS_DIR / "chart_001.png"
    pred_p = PRED_DIR / "chart_001_prediction.json"
    img = cv2.imread(str(img_p))

    calib = calibrate_chart(img)
    res = verify_chart_visually(img, pred_p, calibration=calib)

    out_p = OUT_DIR / "test_overlay_001.png"
    overlay = render_verification_overlay(img, res, calib, output_path=out_p)

    assert overlay.shape == img.shape, "Overlay dimensions must match original image"
    assert out_p.exists(), "Overlay file was not saved to disk"
    assert out_p.stat().st_size > 5000, "Overlay file size unexpectedly small"

    # Clean up test file
    if out_p.exists():
        out_p.unlink()


def test_extract_evidence_crops():
    """Verify that high-resolution ROI proof crops are extracted for every claimed point."""
    img_p = CHARTS_DIR / "chart_001.png"
    pred_p = PRED_DIR / "chart_001_prediction.json"
    img = cv2.imread(str(img_p))

    calib = calibrate_chart(img)
    res = verify_chart_visually(img, pred_p, calibration=calib)

    crops_dir = OUT_DIR / "test_crops"
    crop_records = extract_evidence_crops(img, res, output_dir=crops_dir, crop_size=70)

    assert len(crop_records) == len(res.point_results) == 5
    for cr in crop_records:
        assert cr["base64"] != ""
        assert cr["status"] in ["SUPPORTED", "VALUE_CONTRADICTED", "FABRICATED_POINT", "OUT_OF_BOUNDS"]
        if cr["file_path"]:
            p = Path(cr["file_path"])
            assert p.exists()
            p.unlink()  # Clean up

    if crops_dir.exists():
        crops_dir.rmdir()


def test_generate_html_certificate():
    """Verify that the interactive HTML certificate is generated with all sections."""
    img_p = CHARTS_DIR / "chart_001.png"
    pred_p = PRED_DIR / "chart_001_prediction.json"
    img = cv2.imread(str(img_p))

    calib = calibrate_chart(img)
    res = verify_chart_visually(img, pred_p, calibration=calib)

    temp_overlay = OUT_DIR / "temp_overlay.png"
    render_verification_overlay(img, res, calib, output_path=temp_overlay)

    crop_records = extract_evidence_crops(img, res, crop_size=70)

    temp_html = OUT_DIR / "test_certificate.html"
    generate_html_certificate(res, temp_overlay, crop_records, temp_html)

    assert temp_html.exists()
    assert temp_html.stat().st_size > 20000, "HTML certificate is unexpectedly small"

    content = temp_html.read_text(encoding="utf-8")
    assert "Evidence Grounding Certificate" in content
    assert "chart_001" in content
    assert "Visual Evidence Overlay" in content
    assert "Point-by-Point Visual Proof Cards" in content

    # Clean up
    if temp_overlay.exists():
        temp_overlay.unlink()
    if temp_html.exists():
        temp_html.unlink()


def test_generate_verification_package():
    """Verify end-to-end package generation across multiple chart cases."""
    img_p = CHARTS_DIR / "chart_002.png"
    pred_p = PRED_DIR / "chart_002_prediction.json"

    res, overlay_p, html_p = generate_verification_package(img_p, pred_p, OUT_DIR, chart_id="chart_002")

    assert res.counts["fabricated"] == 1
    assert overlay_p.exists()
    assert html_p.exists()
