"""
Tests for FastAPI interactive web demo application and API endpoints.
"""

import json
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.web.app import app

client = TestClient(app)


def test_serve_index():
    """Verify that root URL serves the interactive HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Scientific Chart Verifier" in response.text
    assert "Zero Ground-Truth VLM Lie Detector" in response.text


def test_get_scenarios():
    """Verify that /api/scenarios returns all 5 benchmark scenarios."""
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) == 5
    ids = [s["id"] for s in scenarios]
    assert "chart_001" in ids
    assert "chart_002" in ids
    assert "chart_003" in ids
    assert "chart_004" in ids
    assert "chart_005" in ids


def test_get_scenario_image():
    """Verify that scenario image endpoint returns binary PNG data."""
    response = client.get("/api/scenarios/chart_001/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 1000


def test_verify_benchmark_scenario_endpoint():
    """Verify that /api/verify performs zero-GT visual verification."""
    pred_path = BASE_DIR / "data" / "predictions" / "chart_001_prediction.json"
    pred_json = pred_path.read_text(encoding="utf-8")

    response = client.post(
        "/api/verify",
        data={
            "chart_id": "chart_001",
            "tolerance": 0.03,
            "prediction_json": pred_json,
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert data["chart_id"] == "chart_001"
    assert data["is_calibrated"] is True
    assert data["calibration"]["x_axis"]["r_squared"] >= 0.99
    assert data["counts"]["supported"] == 4
    assert data["counts"]["contradicted"] == 1
    assert "overlay_image_base64" in data
    assert data["overlay_image_base64"].startswith("data:image/png;base64,")
    assert len(data["proof_crops"]) == 5
    assert "certificate_url" in data


def test_benchmark_endpoint():
    """Verify that /api/benchmark executes comparative VLM evaluation."""
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 5
    names = [m["model_name"] for m in models]
    assert any("GPT-4o" in n for n in names)
    assert any("Ungrounded" in n for n in names)
    assert any("Grok" in n for n in names)
    # Check that execution_mode and is_genuine are present
    for m in models:
        assert "is_genuine" in m
        assert "execution_mode" in m
        assert "status_note" in m


def test_extract_vlm_endpoint():
    """Verify that /api/extract_vlm extracts structured claims and reports genuine vs fallback mode."""
    # Test simulated
    response = client.post(
        "/api/extract_vlm",
        data={
            "chart_id": "chart_001",
            "model_choice": "simulated",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chart_id"] == "chart_001"
    assert "prediction" in data
    assert "points" in data["prediction"]
    assert "is_genuine" in data
    assert "execution_mode" in data

    # Test Grok extraction (with automatic fallback on invalid key)
    response_grok = client.post(
        "/api/extract_vlm",
        data={
            "chart_id": "chart_001",
            "model_choice": "grok",
        }
    )
    assert response_grok.status_code == 200
    grok_data = response_grok.json()
    assert "Grok" in grok_data["model_name"]
    assert "prediction" in grok_data
    assert "execution_mode" in grok_data
