"""
VLM Comparative Benchmarking Suite.
Evaluates multimodal models on scientific chart extraction and measures their
empirical hallucination rates and reliability scores using the Zero Ground-Truth Verifier.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from .base import BaseVLM
    from .models import SimulatedVLM, OpenAIVLM, GeminiVLM, AnthropicVLM, GrokVLM
    from ..verification.visual_verifier import verify_chart_visually, VerificationResult
except ImportError:
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    from vlm.base import BaseVLM
    from vlm.models import SimulatedVLM, OpenAIVLM, GeminiVLM, AnthropicVLM, GrokVLM
    from verification.visual_verifier import verify_chart_visually, VerificationResult


@dataclass
class ModelBenchmarkSummary:
    """Aggregated evaluation metrics for a single VLM across the test suite."""
    model_name: str
    total_charts: int
    total_points: int
    supported_points: int
    contradicted_points: int
    fabricated_points: int
    omitted_points: int
    support_rate: float
    hallucination_rate: float
    fabrication_rate: float
    text_grounding_score: float
    model_reliability_index: float  # MRI in range [0, 100]
    is_genuine: bool = False
    execution_mode: str = "SIMULATED / RANDOM"
    status_note: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_vlm_benchmark(
    models: List[BaseVLM],
    chart_ids: List[str],
    charts_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, ModelBenchmarkSummary]:
    """
    Runs an automated comparative benchmark across multiple VLMs and chart types.
    """
    charts_dir = Path(charts_dir)
    results: Dict[str, ModelBenchmarkSummary] = {}
    detailed_logs: Dict[str, List[dict]] = {}

    for model in models:
        m_name = model.model_name
        detailed_logs[m_name] = []

        tot_points = 0
        supp_pts = 0
        contra_pts = 0
        fab_pts = 0
        omit_pts = 0
        text_scores = []

        for cid in chart_ids:
            img_p = charts_dir / f"{cid}.png"
            if not img_p.exists():
                continue

            # 1. Model extraction
            try:
                extraction = model.extract_chart_data(img_p)
            except Exception as e:
                print(f"Warning: {m_name} failed extraction on {cid}: {e}")
                continue

            # 2. Run Zero-GT Evidence Verifier
            v_res = verify_chart_visually(img_p, extraction)

            tot_points += v_res.counts.get("total_claimed", 0)
            supp_pts += v_res.counts.get("supported", 0)
            contra_pts += v_res.counts.get("contradicted", 0)
            fab_pts += v_res.counts.get("fabricated", 0)
            omit_pts += v_res.counts.get("omitted", 0)
            text_scores.append(v_res.metrics.get("text_grounding_score", 1.0))

            detailed_logs[m_name].append({
                "chart_id": cid,
                "verification_result": v_res.to_dict(),
            })

        # Calculate model aggregate rates
        supp_rate = (supp_pts / tot_points) if tot_points > 0 else 0.0
        halluc_rate = ((contra_pts + fab_pts) / tot_points) if tot_points > 0 else 0.0
        fab_rate = (fab_pts / tot_points) if tot_points > 0 else 0.0
        mean_text_score = float(np.mean(text_scores)) if text_scores else 0.0

        # Model Reliability Index (MRI)
        # 70% point support + 30% text grounding - 40% penalty for severe hallucinations
        mri = max(0.0, min(100.0, (0.70 * supp_rate + 0.30 * mean_text_score - 0.40 * halluc_rate) * 100.0))

        results[m_name] = ModelBenchmarkSummary(
            model_name=m_name,
            total_charts=len(chart_ids),
            total_points=tot_points,
            supported_points=supp_pts,
            contradicted_points=contra_pts,
            fabricated_points=fab_pts,
            omitted_points=omit_pts,
            support_rate=float(supp_rate),
            hallucination_rate=float(halluc_rate),
            fabrication_rate=float(fab_rate),
            text_grounding_score=float(mean_text_score),
            model_reliability_index=float(mri),
            is_genuine=getattr(model, "is_genuine", False),
            execution_mode=getattr(model, "execution_mode", "SIMULATED / RANDOM"),
            status_note=getattr(model, "status_note", ""),
            provider=getattr(model, "provider", ""),
        )

    # Save outputs if output_dir provided
    if output_dir is not None:
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        # JSON Summary
        summary_file = out_d / "benchmark_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=4)

        # Markdown Table
        table_file = out_d / "benchmark_table.md"
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(generate_markdown_table(results))

    return results


def generate_markdown_table(results: Dict[str, ModelBenchmarkSummary]) -> str:
    """Formats benchmark results as a clean publication-grade Markdown table."""
    sorted_models = sorted(results.values(), key=lambda r: r.model_reliability_index, reverse=True)

    header = (
        "| Model Name | Origin / Mode | Evaluated Points | Support Rate (%) | Hallucination Rate (%) | "
        "Fabrications | Omissions | Reliability Index (MRI) | Status Note |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
    )

    rows = []
    for m in sorted_models:
        origin_badge = "**[GENUINE - Live API]**" if m.is_genuine else "**[SIMULATED / RANDOM]**"
        rows.append(
            f"| **{m.model_name}** | {origin_badge} | {m.total_points} | {m.support_rate*100:.1f}% | "
            f"{m.hallucination_rate*100:.1f}% | {m.fabricated_points} | {m.omitted_points} | "
            f"**{m.model_reliability_index:.1f} / 100** | {m.status_note} |"
        )

    return header + "\n".join(rows) + "\n"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    base_dir = Path(__file__).resolve().parent.parent.parent
    charts_folder = base_dir / "data" / "charts"
    out_folder = base_dir / "data" / "evidence"

    # Define fleet of models to benchmark
    test_models: List[BaseVLM] = [
        SimulatedVLM(model_name="GPT-4o (Simulated)", profile="precise", random_seed=101),
        SimulatedVLM(model_name="Claude-3.5 (Simulated)", profile="precise", random_seed=202),
        SimulatedVLM(model_name="LLaVA-Dense (Simulated)", profile="numerical_drift", random_seed=303),
        SimulatedVLM(model_name="Ungrounded-VLM (Simulated)", profile="fabricator", random_seed=404),
    ]

    all_charts = ["chart_001", "chart_002", "chart_003", "chart_004", "chart_005"]
    print("Running VLM Benchmark across 5 scientific charts...")
    bench_results = run_vlm_benchmark(test_models, all_charts, charts_folder, output_dir=out_folder)
    print("\n" + generate_markdown_table(bench_results))
