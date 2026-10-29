"""
Synthetic Dataset Generator for Scientific Chart Verification.

Generates:
1. Diverse chart types (Line plots, multi-series, negative numbers, varied scales)
2. Ground-truth structured JSONs
3. Corresponding simulated VLM predictions with intentional hallucinations:
   - Veridical extractions (clean)
   - Fabricated points (hallucinated data points where no visual mark exists)
   - Numerical value distortions (shifted y-value)
   - Label hallucinations (fabricated axis or title names)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_benchmark_datasets():
    base_dir = Path(__file__).resolve().parent.parent
    charts_dir = base_dir / "data" / "charts"
    gt_dir = base_dir / "data" / "ground_truth"
    pred_dir = base_dir / "data" / "predictions"

    charts_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Chart 001: Model Accuracy Across Epochs (Standard Line Chart)
    # =========================================================================
    c1_data = {
        "chart_id": "chart_001",
        "chart_type": "line",
        "title": "Model Accuracy Across Epochs",
        "x_axis": {
            "label": "Epoch",
            "values": [1, 2, 3, 4, 5]
        },
        "y_axis": {
            "label": "Accuracy (%)",
            "values": [72, 76, 81, 85, 89]
        },
        "series": [
            {
                "name": "Model A",
                "points": [
                    {"x": 1, "y": 72},
                    {"x": 2, "y": 76},
                    {"x": 3, "y": 81},
                    {"x": 4, "y": 85},
                    {"x": 5, "y": 89}
                ]
            }
        ]
    }

    x1 = [1, 2, 3, 4, 5]
    y1 = [72, 76, 81, 85, 89]
    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    plt.title(c1_data["title"], fontsize=14, pad=12)
    plt.xlabel(c1_data["x_axis"]["label"], fontsize=12)
    plt.ylabel(c1_data["y_axis"]["label"], fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(charts_dir / "chart_001.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(gt_dir / "chart_001.json", "w", encoding="utf-8") as f:
        json.dump(c1_data, f, indent=4)

    # Injected hallucination prediction for chart_001 (x=2 has y=78 instead of 76)
    c1_pred = {
        "chart_id": "chart_001",
        "title": "Model Accuracy Across Epochs",
        "x_axis": "Epoch",
        "y_axis": "Accuracy (%)",
        "points": [
            {"x": 1, "y": 72},
            {"x": 2, "y": 78},  # Hallucinated value
            {"x": 3, "y": 81},
            {"x": 4, "y": 85},
            {"x": 5, "y": 89}
        ]
    }
    with open(pred_dir / "chart_001_prediction.json", "w", encoding="utf-8") as f:
        json.dump(c1_pred, f, indent=4)

    # =========================================================================
    # Chart 002: Training Loss Over Iterations (Decay Curve, Different Scale)
    # =========================================================================
    c2_data = {
        "chart_id": "chart_002",
        "chart_type": "line",
        "title": "Training Loss Over Iterations",
        "x_axis": {
            "label": "Iteration",
            "values": [10, 20, 30, 40, 50]
        },
        "y_axis": {
            "label": "Loss",
            "values": [1.45, 0.95, 0.62, 0.38, 0.22]
        },
        "series": [
            {
                "name": "Loss",
                "points": [
                    {"x": 10, "y": 1.45},
                    {"x": 20, "y": 0.95},
                    {"x": 30, "y": 0.62},
                    {"x": 40, "y": 0.38},
                    {"x": 50, "y": 0.22}
                ]
            }
        ]
    }

    x2 = [10, 20, 30, 40, 50]
    y2 = [1.45, 0.95, 0.62, 0.38, 0.22]
    plt.figure(figsize=(8, 5))
    plt.plot(x2, y2, marker="s", color="#d62728", linewidth=2.5, markersize=8)
    plt.title(c2_data["title"], fontsize=14, pad=12)
    plt.xlabel(c2_data["x_axis"]["label"], fontsize=12)
    plt.ylabel(c2_data["y_axis"]["label"], fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(charts_dir / "chart_002.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(gt_dir / "chart_002.json", "w", encoding="utf-8") as f:
        json.dump(c2_data, f, indent=4)

    # Simulated prediction for chart_002 with fabricated point (x=25, y=0.80)
    c2_pred = {
        "chart_id": "chart_002",
        "title": "Training Loss Over Iterations",
        "x_axis": "Iteration",
        "y_axis": "Loss",
        "points": [
            {"x": 10, "y": 1.45},
            {"x": 20, "y": 0.95},
            {"x": 25, "y": 0.80},  # Hallucinated extra point!
            {"x": 30, "y": 0.62},
            {"x": 40, "y": 0.38},
            {"x": 50, "y": 0.22}
        ]
    }
    with open(pred_dir / "chart_002_prediction.json", "w", encoding="utf-8") as f:
        json.dump(c2_pred, f, indent=4)

    # =========================================================================
    # Chart 003: Annual Temperature Anomaly (Negative & Positive Values)
    # =========================================================================
    c3_data = {
        "chart_id": "chart_003",
        "chart_type": "line",
        "title": "Global Temperature Anomaly",
        "x_axis": {
            "label": "Year",
            "values": [2016, 2017, 2018, 2019, 2020]
        },
        "y_axis": {
            "label": "Anomaly (C)",
            "values": [-0.4, 0.2, 0.6, 0.9, 1.3]
        },
        "series": [
            {
                "name": "Temperature Anomaly",
                "points": [
                    {"x": 2016, "y": -0.4},
                    {"x": 2017, "y": 0.2},
                    {"x": 2018, "y": 0.6},
                    {"x": 2019, "y": 0.9},
                    {"x": 2020, "y": 1.3}
                ]
            }
        ]
    }

    x3 = [2016, 2017, 2018, 2019, 2020]
    y3 = [-0.4, 0.2, 0.6, 0.9, 1.3]
    plt.figure(figsize=(8, 5))
    plt.plot(x3, y3, marker="^", color="#2ca02c", linewidth=2.5, markersize=8)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    plt.title(c3_data["title"], fontsize=14, pad=12)
    plt.xlabel(c3_data["x_axis"]["label"], fontsize=12)
    plt.ylabel(c3_data["y_axis"]["label"], fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(charts_dir / "chart_003.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(gt_dir / "chart_003.json", "w", encoding="utf-8") as f:
        json.dump(c3_data, f, indent=4)

    # Simulated prediction for chart_003 with fabricated label and sign error
    c3_pred = {
        "chart_id": "chart_003",
        "title": "Global Surface Temperature Trends",  # Hallucinated title
        "x_axis": "Year",
        "y_axis": "Anomaly (C)",
        "points": [
            {"x": 2016, "y": 0.4},   # Sign error hallucination (+0.4 instead of -0.4)
            {"x": 2017, "y": 0.2},
            {"x": 2018, "y": 0.6},
            {"x": 2019, "y": 0.9},
            {"x": 2020, "y": 1.3}
        ]
    }
    with open(pred_dir / "chart_003_prediction.json", "w", encoding="utf-8") as f:
        json.dump(c3_pred, f, indent=4)

    # =========================================================================
    # Chart 004: Enzyme Reaction Velocity (Biochemical Kinetics)
    # =========================================================================
    c4_data = {
        "chart_id": "chart_004",
        "chart_type": "line",
        "title": "Enzyme Reaction Velocity",
        "x_axis": {
            "label": "Substrate Concentration (mM)",
            "values": [2, 4, 6, 8, 10]
        },
        "y_axis": {
            "label": "Velocity (umol/min)",
            "values": [18.2, 31.5, 41.0, 48.2, 53.0]
        },
        "series": [
            {
                "name": "Reaction Velocity",
                "points": [
                    {"x": 2, "y": 18.2},
                    {"x": 4, "y": 31.5},
                    {"x": 6, "y": 41.0},
                    {"x": 8, "y": 48.2},
                    {"x": 10, "y": 53.0}
                ]
            }
        ]
    }

    x4 = [2, 4, 6, 8, 10]
    y4 = [18.2, 31.5, 41.0, 48.2, 53.0]
    plt.figure(figsize=(8, 5))
    plt.plot(x4, y4, marker="D", color="#9467bd", linewidth=2.5, markersize=8)
    plt.title(c4_data["title"], fontsize=14, pad=12)
    plt.xlabel(c4_data["x_axis"]["label"], fontsize=12)
    plt.ylabel(c4_data["y_axis"]["label"], fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(charts_dir / "chart_004.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(gt_dir / "chart_004.json", "w", encoding="utf-8") as f:
        json.dump(c4_data, f, indent=4)

    # Simulated prediction for chart_004 with numerical drift at x=6
    c4_pred = {
        "chart_id": "chart_004",
        "title": "Enzyme Reaction Velocity",
        "x_axis": "Substrate Concentration (mM)",
        "y_axis": "Velocity (umol/min)",
        "points": [
            {"x": 2, "y": 18.2},
            {"x": 4, "y": 31.5},
            {"x": 6, "y": 46.0},  # Drift hallucination (46.0 vs true 41.0)
            {"x": 8, "y": 48.2},
            {"x": 10, "y": 53.0}
        ]
    }
    with open(pred_dir / "chart_004_prediction.json", "w", encoding="utf-8") as f:
        json.dump(c4_pred, f, indent=4)

    # =========================================================================
    # Chart 005: Bandpass Filter Response (Engineering / Negative values)
    # =========================================================================
    c5_data = {
        "chart_id": "chart_005",
        "chart_type": "line",
        "title": "Bandpass Filter Response",
        "x_axis": {
            "label": "Frequency (kHz)",
            "values": [10, 20, 30, 40, 50]
        },
        "y_axis": {
            "label": "Gain (dB)",
            "values": [-24.0, -12.0, 0.0, -12.0, -24.0]
        },
        "series": [
            {
                "name": "Frequency Response",
                "points": [
                    {"x": 10, "y": -24.0},
                    {"x": 20, "y": -12.0},
                    {"x": 30, "y": 0.0},
                    {"x": 40, "y": -12.0},
                    {"x": 50, "y": -24.0}
                ]
            }
        ]
    }

    x5 = [10, 20, 30, 40, 50]
    y5 = [-24.0, -12.0, 0.0, -12.0, -24.0]
    plt.figure(figsize=(8, 5))
    plt.plot(x5, y5, marker="p", color="#ff7f0e", linewidth=2.5, markersize=8)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    plt.title(c5_data["title"], fontsize=14, pad=12)
    plt.xlabel(c5_data["x_axis"]["label"], fontsize=12)
    plt.ylabel(c5_data["y_axis"]["label"], fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(charts_dir / "chart_005.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(gt_dir / "chart_005.json", "w", encoding="utf-8") as f:
        json.dump(c5_data, f, indent=4)

    # Simulated prediction for chart_005 with omitted point at x=50 and fabricated point at x=35
    c5_pred = {
        "chart_id": "chart_005",
        "title": "Bandpass Filter Response",
        "x_axis": "Frequency (kHz)",
        "y_axis": "Gain (dB)",
        "points": [
            {"x": 10, "y": -24.0},
            {"x": 20, "y": -12.0},
            {"x": 30, "y": 0.0},
            {"x": 35, "y": -6.0},  # Fabricated point
            {"x": 40, "y": -12.0}
            # x=50 omitted
        ]
    }
    with open(pred_dir / "chart_005_prediction.json", "w", encoding="utf-8") as f:
        json.dump(c5_pred, f, indent=4)

    print("Generated 5 diverse benchmark charts, ground truths, and hallucination predictions.")


if __name__ == "__main__":
    generate_benchmark_datasets()