# VLM benchmarking and evaluation package
from .base import BaseVLM, EXTRACTION_SYSTEM_PROMPT
from .models import SimulatedVLM, OpenAIVLM, GeminiVLM
from .benchmark import run_vlm_benchmark, ModelBenchmarkSummary, generate_markdown_table
from .evaluation_experiment import (
    run_lie_detector_validation,
    ConfusionMatrixMetrics,
    generate_experiment_markdown_table,
)
