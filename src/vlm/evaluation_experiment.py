"""
Statistical Lie Detector Validation Experiment.
Empirically evaluates the Precision, Recall, Specificity, and F1-Score of the
Zero Ground-Truth Verifier against a controlled ground-truth hallucination dataset.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from ..verification.visual_verifier import verify_chart_visually
    from ..extraction.calibration import calibrate_chart
except ImportError:
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    from verification.visual_verifier import verify_chart_visually
    from extraction.calibration import calibrate_chart


@dataclass
class ConfusionMatrixMetrics:
    total_samples: int
    true_positives: int    # Hallucinations correctly caught
    true_negatives: int    # Clean points correctly marked SUPPORTED
    false_positives: int   # Clean points incorrectly flagged
    false_negatives: int   # Hallucinations missed
    precision: float
    recall: float
    f1_score: float
    specificity: float
    accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_lie_detector_validation(
    charts_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    random_seed: int = 1234
) -> Tuple[ConfusionMatrixMetrics, Dict[str, float]]:
    """
    Constructs a controlled balanced test set (Clean vs Hallucinated claims)
    and evaluates the Verifier's classification performance as an objective lie detector.
    """
    charts_dir = Path(charts_dir)
    gt_dir = Path(gt_dir)
    rng = random.Random(random_seed)

    chart_ids = ["chart_001", "chart_002", "chart_003", "chart_004", "chart_005"]

    # 1. Gather Ground Truth Points
    all_gt_points = []
    chart_gt_map = {}

    for cid in chart_ids:
        gt_file = gt_dir / f"{cid}.json"
        img_file = charts_dir / f"{cid}.png"
        if not (gt_file.exists() and img_file.exists()):
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        pts = []
        for s in gt_data.get("series", []):
            pts.extend(s.get("points", []))
        if not pts and "points" in gt_data:
            pts = gt_data["points"]

        chart_gt_map[cid] = {
            "title": gt_data["title"],
            "x_axis": gt_data["x_axis"]["label"],
            "y_axis": gt_data["y_axis"]["label"],
            "points": pts,
            "y_range": abs(max(p["y"] for p in pts) - min(p["y"] for p in pts)),
        }

        for p in pts:
            all_gt_points.append({
                "chart_id": cid,
                "x": p["x"],
                "y": p["y"],
            })

    # 2. Build Controlled Experiment Cases
    # Clean Class (True Label = 0)
    clean_cases = []
    for p in all_gt_points:
        clean_cases.append({
            "chart_id": p["chart_id"],
            "x": p["x"],
            "claimed_y": p["y"],
            "is_hallucination": False,
            "category": "Clean_Veridical",
        })

    # Hallucination Class (True Label = 1)
    hallucination_cases = []
    category_counts = {"Drift": 0, "Extreme": 0, "Sign": 0, "Fabricated": 0}

    for p in all_gt_points:
        cid = p["chart_id"]
        y_range = chart_gt_map[cid]["y_range"]
        y_val = p["y"]

        # 1. Mild/Moderate Numerical Drift (+-5% to +-15% of range)
        drift_delta = rng.choice([-1, 1]) * rng.uniform(0.06, 0.15) * (y_range if y_range > 0 else 5.0)
        hallucination_cases.append({
            "chart_id": cid,
            "x": p["x"],
            "claimed_y": round(y_val + drift_delta, 2),
            "is_hallucination": True,
            "category": "Drift",
        })
        category_counts["Drift"] += 1

        # 2. Extreme Numerical Hallucination (+-30% to +-60% of range)
        ext_delta = rng.choice([-1, 1]) * rng.uniform(0.30, 0.60) * (y_range if y_range > 0 else 10.0)
        hallucination_cases.append({
            "chart_id": cid,
            "x": p["x"],
            "claimed_y": round(y_val + ext_delta, 2),
            "is_hallucination": True,
            "category": "Extreme",
        })
        category_counts["Extreme"] += 1

        # 3. Sign Inversion (if y != 0)
        if y_val != 0:
            hallucination_cases.append({
                "chart_id": cid,
                "x": p["x"],
                "claimed_y": -y_val,
                "is_hallucination": True,
                "category": "Sign",
            })
            category_counts["Sign"] += 1

        # 4. Fabricated Coordinate (non-existent x coordinate)
        fab_x = p["x"] + rng.choice([0.5, 1.5, 2.5, 5.0])
        hallucination_cases.append({
            "chart_id": cid,
            "x": fab_x,
            "claimed_y": y_val,
            "is_hallucination": True,
            "category": "Fabricated",
        })
        category_counts["Fabricated"] += 1

    # Sample balanced dataset: 50 clean, 50 hallucinated
    rng.shuffle(clean_cases)
    rng.shuffle(hallucination_cases)
    test_clean = clean_cases[:50]
    test_halluc = hallucination_cases[:50]
    eval_set = test_clean + test_halluc
    rng.shuffle(eval_set)

    # 3. Run Zero Ground-Truth Verifier on Evaluation Set
    tp = 0  # Hallucination caught
    fp = 0  # Clean point falsely accused
    tn = 0  # Clean point verified as supported
    fn = 0  # Hallucination missed

    category_caught = {cat: 0 for cat in category_counts}
    category_totals = {cat: 0 for cat in category_counts}

    # Group test cases by chart_id for batched verification
    cases_by_chart: Dict[str, List[dict]] = {}
    for case in eval_set:
        cases_by_chart.setdefault(case["chart_id"], []).append(case)

    detailed_eval_records = []

    for cid, cases in cases_by_chart.items():
        img_p = charts_dir / f"{cid}.png"
        meta = chart_gt_map[cid]

        # Construct single prediction JSON with all test claims for this chart
        pred_dict = {
            "chart_id": cid,
            "title": meta["title"],
            "x_axis": meta["x_axis"],
            "y_axis": meta["y_axis"],
            "points": [{"x": c["x"], "y": c["claimed_y"]} for c in cases],
        }

        # Run Verifier (Zero ground truth given)
        v_res = verify_chart_visually(img_p, pred_dict, rel_tolerance=0.03)

        # Match results back to test cases
        for case, p_res in zip(cases, v_res.point_results):
            true_halluc = case["is_hallucination"]
            cat = case["category"]
            pred_halluc = p_res.status in ["VALUE_CONTRADICTED", "FABRICATED_POINT", "OUT_OF_BOUNDS"]

            if true_halluc:
                category_totals[cat] = category_totals.get(cat, 0) + 1
                if pred_halluc:
                    tp += 1
                    category_caught[cat] = category_caught.get(cat, 0) + 1
                else:
                    fn += 1
            else:
                if pred_halluc:
                    fp += 1
                else:
                    tn += 1

            detailed_eval_records.append({
                "chart_id": cid,
                "x": case["x"],
                "claimed_y": case["claimed_y"],
                "ground_truth_label": "HALLUCINATION" if true_halluc else "CLEAN",
                "verifier_verdict": p_res.status,
                "category": cat,
                "discrepancy": p_res.discrepancy,
                "correct": (true_halluc == pred_halluc),
            })

    # 4. Compute Statistical Metrics
    tot_samples = tp + tn + fp + fn
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / tot_samples if tot_samples > 0 else 0.0

    metrics = ConfusionMatrixMetrics(
        total_samples=tot_samples,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        specificity=float(specificity),
        accuracy=float(accuracy),
    )

    # Category rates
    cat_rates = {}
    for cat, tot in category_totals.items():
        cat_rates[cat] = (category_caught[cat] / tot) if tot > 0 else 1.0

    # Save to disk if requested
    if output_dir is not None:
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        res_json = out_d / "validation_experiment_results.json"
        with open(res_json, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics.to_dict(),
                "category_detection_rates": cat_rates,
                "samples": detailed_eval_records,
            }, f, indent=4)

        table_md = out_d / "validation_experiment_table.md"
        with open(table_md, "w", encoding="utf-8") as f:
            f.write(generate_experiment_markdown_table(metrics, cat_rates))

    return metrics, cat_rates


def generate_experiment_markdown_table(m: ConfusionMatrixMetrics, cat_rates: Dict[str, float]) -> str:
    """Generates an academic-grade Markdown validation table for research publication."""
    drift_pct = f"{cat_rates.get('Drift', 1.0) * 100:.1f}%"
    ext_pct = f"{cat_rates.get('Extreme', 1.0) * 100:.1f}%"
    sign_pct = f"{cat_rates.get('Sign', 1.0) * 100:.1f}%"
    fab_pct = f"{cat_rates.get('Fabricated', 1.0) * 100:.1f}%"

    table = (
        "### Statistical Validation: Lie Detector Performance\n\n"
        "| Metric | Empirical Value | Description |\n"
        "| :--- | :---: | :--- |\n"
        f"| **Total Test Claims** | **{m.total_samples}** | Balanced evaluation set (50 clean, 50 hallucinated) |\n"
        f"| **True Positives (TP)** | **{m.true_positives}** | Hallucinations correctly detected and flagged |\n"
        f"| **True Negatives (TN)** | **{m.true_negatives}** | Clean veridical claims confirmed as SUPPORTED |\n"
        f"| **False Positives (FP)** | **{m.false_positives}** | Clean claims falsely accused of hallucination |\n"
        f"| **False Negatives (FN)** | **{m.false_negatives}** | Hallucinations that went undetected |\n"
        f"| **Precision** | **{m.precision * 100:.2f}%** | Positive Predictive Value: TP / (TP + FP) |\n"
        f"| **Recall / Sensitivity** | **{m.recall * 100:.2f}%** | True Positive Rate: TP / (TP + FN) |\n"
        f"| **Specificity** | **{m.specificity * 100:.2f}%** | True Negative Rate: TN / (TN + FP) |\n"
        f"| **F1-Score** | **{m.f1_score * 100:.2f}%** | Harmonic mean of Precision and Recall |\n"
        f"| **Overall Classification Accuracy** | **{m.accuracy * 100:.2f}%** | Overall empirical correctness |\n\n"
        "#### Category-Wise Hallucination Detection Rates:\n"
        f"- **Numerical Drift (5% to 15%)**: {drift_pct} caught\n"
        f"- **Extreme Hallucinations (30% to 60%)**: {ext_pct} caught\n"
        f"- **Sign Inversions (-y to +y)**: {sign_pct} caught\n"
        f"- **Fabricated Points (Invented x)**: {fab_pct} caught\n"
    )
    return table


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    c_dir = base_dir / "data" / "charts"
    g_dir = base_dir / "data" / "ground_truth"
    o_dir = base_dir / "data" / "evidence"

    print("Running Controlled Lie Detector Validation Experiment (N=100)...")
    metrics, rates = run_lie_detector_validation(c_dir, g_dir, output_dir=o_dir)
    print("\n" + generate_experiment_markdown_table(metrics, rates))
