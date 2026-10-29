"""
Base interfaces and standard prompting templates for Vision-Language Models (VLMs).
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Dict, Optional, Union


EXTRACTION_SYSTEM_PROMPT = """You are an expert scientific data extractor.
Your task is to examine a scientific chart and extract all objective data into a strictly valid JSON object.

Output JSON format:
{
    "chart_id": "<string identifier or filename>",
    "title": "<exact chart title>",
    "x_axis": "<exact label of the x-axis>",
    "y_axis": "<exact label of the y-axis>",
    "points": [
        {"x": <numerical or category value>, "y": <numerical value>},
        ...
    ]
}

Guidelines:
1. Do not invent or estimate points that are not marked on the plot.
2. Read axis numbers and tick marks carefully. Pay attention to negative signs and decimals.
3. Return ONLY valid JSON, with no surrounding markdown or explanation.
"""


class BaseVLM(ABC):
    """Abstract base class for all VLM extractors."""

    def __init__(
        self,
        model_name: str,
        provider: str = "Simulation",
        is_genuine: bool = False,
        execution_mode: str = "SIMULATED / RANDOM",
        status_note: str = "",
    ):
        self.model_name = model_name
        self.provider = provider
        self.is_genuine = is_genuine
        self.execution_mode = execution_mode
        self.status_note = status_note

    @abstractmethod
    def extract_chart_data(self, image_input: Union[str, Path]) -> dict:
        """
        Extracts structured data from a chart image.
        Returns a dictionary conforming to the standard chart JSON schema.
        """
        pass

    def __repr__(self) -> str:
        mode_tag = "GENUINE" if self.is_genuine else "SIMULATED"
        return f"{self.__class__.__name__}(model='{self.model_name}', mode='{mode_tag}')"
