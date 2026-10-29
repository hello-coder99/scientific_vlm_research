"""
Unit and integration tests for VLM Benchmarking and Statistical Lie Detector Validation (Phase 4).
"""

from pathlib import Path
import sys
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.vlm.models import SimulatedVLM
from src.vlm.benchmark import run_vlm_benchmark
from src.vlm.evaluation_experiment import run_lie_detector_validation

CHARTS_DIR = BASE_DIR / "data" / "charts"
GT_DIR = BASE_DIR / "data" / "ground_truth"


def test_simulated_vlm_profiles():
    """Verify that different VLM simulation profiles behave according to specification."""
    img_p = CHARTS_DIR / "chart_001.png"

    # 1. Precise Model
    m_precise = SimulatedVLM(profile="precise", random_seed=42)
    ext_precise = m_precise.extract_chart_data(img_p)
    assert len(ext_precise["points"]) == 5
    assert abs(ext_precise["points"][0]["y"] - 72.0) <= 0.1

    # 2. Fabricator Model
    m_fab = SimulatedVLM(profile="fabricator", random_seed=42)
    ext_fab = m_fab.extract_chart_data(img_p)
    assert len(ext_fab["points"]) == 6, "Fabricator model should have injected an extra point"

    # 3. Sign Confused Model on chart_003 (which has negative values)
    img_p3 = CHARTS_DIR / "chart_003.png"
    m_sign = SimulatedVLM(profile="sign_confused", random_seed=42)
    ext_sign = m_sign.extract_chart_data(img_p3)
    p0 = ext_sign["points"][0]
    assert p0["y"] > 0, "Sign confused model should have flipped negative value to positive"


def test_run_vlm_benchmark():
    """Verify multi-model comparative benchmarking execution and metric calculations."""
    models = [
        SimulatedVLM(model_name="VLM-HighFidelity", profile="precise", random_seed=1),
        SimulatedVLM(model_name="VLM-Fabricator", profile="fabricator", random_seed=2),
    ]
    chart_ids = ["chart_001", "chart_002"]

    bench_results = run_vlm_benchmark(models, chart_ids, CHARTS_DIR)

    assert "VLM-HighFidelity" in bench_results
    assert "VLM-Fabricator" in bench_results

    m_hi = bench_results["VLM-HighFidelity"]
    m_fab = bench_results["VLM-Fabricator"]

    assert m_hi.support_rate >= 0.95
    assert m_hi.model_reliability_index >= 90.0
    assert m_fab.fabricated_points >= 1
    assert m_fab.model_reliability_index < m_hi.model_reliability_index


def test_lie_detector_statistical_validation():
    """
    Verify that the zero ground-truth verifier acts as an accurate lie detector,
    achieving Precision >= 95%, Specificity >= 95%, and F1 >= 90%.
    """
    metrics, cat_rates = run_lie_detector_validation(CHARTS_DIR, GT_DIR, random_seed=999)

    assert metrics.total_samples > 50
    assert metrics.precision >= 0.95, f"Precision was {metrics.precision:.3f}, expected >= 0.95"
    assert metrics.specificity >= 0.95, f"Specificity was {metrics.specificity:.3f}, expected >= 0.95"
    assert metrics.f1_score >= 0.90, f"F1-score was {metrics.f1_score:.3f}, expected >= 0.90"
    assert metrics.accuracy >= 0.90, f"Accuracy was {metrics.accuracy:.3f}, expected >= 0.90"

    # Extreme hallucinations and sign inversions should have near 100% detection rate
    assert cat_rates.get("Extreme", 0.0) >= 0.95
    assert cat_rates.get("Sign", 0.0) >= 0.95
