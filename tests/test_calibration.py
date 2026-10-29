"""
Unit and integration tests for Axis Calibration and Coordinate Mapping (Phase 1).
"""

from pathlib import Path
import sys
import json
import pytest
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.extraction.calibration import (
    calibrate_chart,
    detect_plot_box,
    robust_axis_linear_fit,
    ChartCalibration,
    PlotBox,
    TickMark,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CHARTS_DIR = BASE_DIR / "data" / "charts"
GT_DIR = BASE_DIR / "data" / "ground_truth"


@pytest.fixture(scope="module")
def benchmark_charts():
    """Ensure benchmark charts are available."""
    charts = ["chart_001", "chart_002", "chart_003"]
    paths = {}
    for cid in charts:
        p = CHARTS_DIR / f"{cid}.png"
        assert p.exists(), f"Benchmark chart {p} does not exist. Run notebooks/create_dataset.py first."
        paths[cid] = p
    return paths


def test_plot_box_detection(benchmark_charts):
    """Verify that the plot area is accurately bounded on all benchmark charts."""
    for cid, path in benchmark_charts.items():
        calib = calibrate_chart(path)
        box = calib.plot_box

        # Sanity checks on plot box dimensions
        assert box.x_min >= 0, f"{cid}: x_min must be non-negative"
        assert box.y_min >= 0, f"{cid}: y_min must be non-negative"
        assert box.x_max <= calib.image_width, f"{cid}: x_max must be within image width"
        assert box.y_max <= calib.image_height, f"{cid}: y_max must be within image height"
        assert box.width >= 500, f"{cid}: Plot box width ({box.width}) is unexpectedly small"
        assert box.height >= 300, f"{cid}: Plot box height ({box.height}) is unexpectedly small"


def test_axis_goodness_of_fit(benchmark_charts):
    """Verify that linear regression on tick marks achieves R^2 >= 0.99 for all charts."""
    for cid, path in benchmark_charts.items():
        calib = calibrate_chart(path)

        assert calib.is_valid, f"{cid} calibration should be valid"
        assert calib.x_axis.r_squared >= 0.99, (
            f"{cid}: X-axis R^2 ({calib.x_axis.r_squared:.5f}) below threshold 0.99"
        )
        assert calib.y_axis.r_squared >= 0.99, (
            f"{cid}: Y-axis R^2 ({calib.y_axis.r_squared:.5f}) below threshold 0.99"
        )
        assert calib.x_axis.inlier_count >= 3, f"{cid}: Insufficient X inlier ticks"
        assert calib.y_axis.inlier_count >= 3, f"{cid}: Insufficient Y inlier ticks"

        # Verify correct orientation:
        # X: pixel increases -> data increases (slope > 0)
        # Y: pixel increases (downwards) -> data decreases (slope < 0)
        assert calib.x_axis.slope > 0, f"{cid}: X-axis slope should be positive"
        assert calib.y_axis.slope < 0, f"{cid}: Y-axis slope should be negative"


def test_bidirectional_coordinate_mapping(benchmark_charts):
    """Verify mathematical invertibility: pixel_to_data(data_to_pixel(x, y)) == (x, y)."""
    for cid, path in benchmark_charts.items():
        calib = calibrate_chart(path)

        # Test on 10 synthetic points across data range
        x_min, x_max = calib.x_axis.data_range
        y_min, y_max = calib.y_axis.data_range

        test_xs = np.linspace(x_min, x_max, 10)
        test_ys = np.linspace(y_min, y_max, 10)

        for x_val, y_val in zip(test_xs, test_ys):
            px, py = calib.data_to_pixel(x_val, y_val)
            rx, ry = calib.pixel_to_data(px, py)

            assert abs(rx - x_val) < 1e-4, f"{cid}: Inversion error on X: {rx} vs {x_val}"
            assert abs(ry - y_val) < 1e-4, f"{cid}: Inversion error on Y: {ry} vs {y_val}"


def test_ground_truth_points_project_inside_canvas(benchmark_charts):
    """Verify that all true data points project to pixel coordinates inside the plot canvas."""
    for cid, path in benchmark_charts.items():
        gt_file = GT_DIR / f"{cid}.json"
        with open(gt_file, "r", encoding="utf-8") as f:
            gt = json.load(f)

        calib = calibrate_chart(path)
        box = calib.plot_box

        for series in gt["series"]:
            for pt in series["points"]:
                px, py = calib.data_to_pixel(pt["x"], pt["y"])

                # Allow 5px tolerance near edges
                margin = 5
                assert box.x_min - margin <= px <= box.x_max + margin, (
                    f"{cid}: Point x={pt['x']} mapped to px={px:.1f}, outside box [{box.x_min}, {box.x_max}]"
                )
                assert box.y_min - margin <= py <= box.y_max + margin, (
                    f"{cid}: Point y={pt['y']} mapped to py={py:.1f}, outside box [{box.y_min}, {box.y_max}]"
                )


def test_robust_outlier_rejection_and_decimal_repair():
    """Verify that robust_axis_linear_fit rejects wild outliers and repairs missing decimal points."""
    # Synthetic Y-ticks with OCR drop of decimal point (775 instead of 77.5) and an outlier (999.0)
    ticks = [
        TickMark(pixel=116.0, value=87.5, raw_text="87.5", confidence=95.0),
        TickMark(pixel=193.0, value=85.0, raw_text="85.0", confidence=95.0),
        TickMark(pixel=271.0, value=82.5, raw_text="82.5", confidence=95.0),
        TickMark(pixel=300.0, value=999.0, raw_text="999.0", confidence=30.0),  # Wild noise outlier
        TickMark(pixel=425.0, value=775.0, raw_text="775", confidence=88.0),   # Missing decimal point (77.5)
        TickMark(pixel=502.0, value=75.0, raw_text="75.0", confidence=95.0),
    ]

    fit = robust_axis_linear_fit(ticks, "y")

    assert fit.r_squared >= 0.999, f"R^2 was {fit.r_squared:.5f}, expected >= 0.999"
    assert fit.slope < 0, "Slope should be negative for Y axis"

    # Verify outlier was flagged as outlier
    outlier_tick = [t for t in fit.ticks if t.raw_text == "999.0"][0]
    assert outlier_tick.status == "outlier"

    # Verify 775 was repaired to 77.5
    repaired_tick = [t for t in fit.ticks if t.raw_text == "775"][0]
    assert repaired_tick.status == "repaired"
    assert abs(repaired_tick.value - 77.5) < 0.1


def test_calibration_serialization_roundtrip(benchmark_charts):
    """Verify that ChartCalibration can be saved to JSON and loaded without data loss."""
    path = benchmark_charts["chart_001"]
    calib = calibrate_chart(path)

    temp_path = BASE_DIR / "data" / "evidence" / "temp_test_calibration.json"
    calib.save(temp_path)

    with open(temp_path, "r", encoding="utf-8") as f:
        loaded_dict = json.load(f)

    reloaded = ChartCalibration.from_dict(loaded_dict)

    assert reloaded.is_valid == calib.is_valid
    assert abs(reloaded.x_axis.slope - calib.x_axis.slope) < 1e-9
    assert abs(reloaded.y_axis.slope - calib.y_axis.slope) < 1e-9
    assert abs(reloaded.x_axis.intercept - calib.x_axis.intercept) < 1e-9
    assert abs(reloaded.y_axis.intercept - calib.y_axis.intercept) < 1e-9
    assert reloaded.plot_box.x_min == calib.plot_box.x_min
    assert reloaded.plot_box.y_max == calib.plot_box.y_max

    # Cleanup temp test file
    if temp_path.exists():
        temp_path.unlink()
