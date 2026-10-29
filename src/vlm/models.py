"""
VLM Model Implementations: Real API Integrations and Realistic Simulation Profiles.
Supports Google Gemini, Anthropic Claude, and OpenAI GPT with automatic live execution
and graceful, transparent fallback to realistic simulation when APIs are unavailable.
"""

import base64
import json
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Union

import requests

try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from .base import BaseVLM, EXTRACTION_SYSTEM_PROMPT


def _load_image_base64_and_mime(image_input: Union[str, Path, bytes]) -> tuple[str, str]:
    """Reads image from path or bytes and returns (base64_str, mime_type)."""
    if isinstance(image_input, bytes):
        raw_bytes = image_input
        mime_type = "image/png"
    else:
        img_path = Path(image_input)
        suffix = img_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif suffix == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/png"
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return b64_str, mime_type


def _normalize_chart_json(raw_data: Union[dict, str], default_chart_id: str = "chart") -> dict:
    """Normalizes various LLM extraction shapes into the standard verifier dictionary format."""
    if isinstance(raw_data, str):
        # Extract JSON object from string
        m = re.search(r"\{.*\}", raw_data, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            data = json.loads(raw_data)
    else:
        data = raw_data

    chart_id = data.get("chart_id") or default_chart_id
    title = data.get("title") or "Chart"
    x_axis = data.get("x_axis") or "X"
    y_axis = data.get("y_axis") or "Y"

    # Normalize axes if returned as dicts
    if isinstance(x_axis, dict):
        x_axis = x_axis.get("label", "X")
    if isinstance(y_axis, dict):
        y_axis = y_axis.get("label", "Y")

    # Extract points
    points_raw = []
    if "points" in data and isinstance(data["points"], list):
        points_raw = data["points"]
    elif "series" in data and isinstance(data["series"], list):
        for s in data["series"]:
            if isinstance(s, dict) and "points" in s:
                points_raw.extend(s["points"])

    clean_points = []
    for p in points_raw:
        if not isinstance(p, dict) or "x" not in p or "y" not in p:
            continue
        try:
            x_val = float(p["x"]) if not isinstance(p["x"], str) or p["x"].replace(".", "", 1).replace("-", "", 1).isdigit() else p["x"]
            y_val = float(p["y"])
            clean_points.append({"x": x_val, "y": y_val})
        except (ValueError, TypeError):
            continue

    return {
        "chart_id": str(chart_id),
        "title": str(title),
        "x_axis": str(x_axis),
        "y_axis": str(y_axis),
        "points": clean_points,
    }


class SimulatedVLM(BaseVLM):
    """
    Simulated VLM with configurable behavioral profiles for reproducible offline benchmarking.

    Profiles:
    - 'precise': State-of-the-art model. Clean extractions with negligible sensor rounding (<= 0.2%).
    - 'numerical_drift': Plausible model that drifts on dense numeric intervals (+-5% to +-15% error).
    - 'fabricator': Prone to inventing non-existent intermediate data points.
    - 'sign_confused': Prone to sign flips on negative axes.
    - 'omitter': Skips hard-to-read marks.
    """

    def __init__(
        self,
        model_name: str = "VLM-Simulated",
        profile: str = "precise",
        ground_truth_dir: Optional[Union[str, Path]] = None,
        random_seed: int = 42,
    ):
        super().__init__(
            model_name=model_name,
            provider="Simulation",
            is_genuine=False,
            execution_mode="SIMULATED / RANDOM",
            status_note=f"Synthetic simulation baseline (Profile: {profile})",
        )
        self.profile = profile
        self.gt_dir = (
            Path(ground_truth_dir)
            if ground_truth_dir
            else Path(__file__).resolve().parent.parent.parent / "data" / "ground_truth"
        )
        self.rng = random.Random(random_seed)

    def extract_chart_data(self, image_input: Union[str, Path, bytes]) -> dict:
        chart_id = "chart"
        if not isinstance(image_input, bytes):
            chart_id = Path(image_input).stem

        # Look for corresponding ground truth template
        gt_file = self.gt_dir / f"{chart_id}.json"
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as f:
                template = json.load(f)

            points = []
            for s in template.get("series", []):
                points.extend(s.get("points", []))
            if not points and "points" in template:
                points = template["points"]

            extracted_points = []
            for pt in points:
                extracted_points.append({"x": pt["x"], "y": float(pt["y"])})

            extracted_title = template.get("title", f"Chart {chart_id}")
            x_axis_meta = template.get("x_axis", {})
            extracted_x_axis = x_axis_meta.get("label", "X-Axis") if isinstance(x_axis_meta, dict) else str(x_axis_meta)
            y_axis_meta = template.get("y_axis", {})
            extracted_y_axis = y_axis_meta.get("label", "Y-Axis") if isinstance(y_axis_meta, dict) else str(y_axis_meta)
        else:
            # Graceful synthetic fallback for custom charts without ground truth
            extracted_title = f"Chart {chart_id}"
            extracted_x_axis = "X"
            extracted_y_axis = "Y"
            extracted_points = [
                {"x": 1.0, "y": round(self.rng.uniform(10.0, 30.0), 2)},
                {"x": 2.0, "y": round(self.rng.uniform(30.0, 50.0), 2)},
                {"x": 3.0, "y": round(self.rng.uniform(50.0, 70.0), 2)},
                {"x": 4.0, "y": round(self.rng.uniform(65.0, 85.0), 2)},
                {"x": 5.0, "y": round(self.rng.uniform(75.0, 95.0), 2)},
            ]

        # Apply profile-specific behavioral characteristics
        if self.profile == "precise":
            for p in extracted_points:
                noise = self.rng.uniform(-0.001, 0.001) * abs(p["y"] if p["y"] != 0 else 1.0)
                p["y"] = round(p["y"] + noise, 3)

        elif self.profile == "numerical_drift":
            if extracted_points:
                drift_idx = self.rng.randint(0, len(extracted_points) - 1)
                y_val = extracted_points[drift_idx]["y"]
                shift = self.rng.choice([1.08, 0.92, 1.15, 0.85])
                extracted_points[drift_idx]["y"] = round(y_val * shift if y_val != 0 else y_val + 2.5, 2)

        elif self.profile == "fabricator":
            if len(extracted_points) >= 2:
                p1 = extracted_points[len(extracted_points) // 2 - 1]
                p2 = extracted_points[len(extracted_points) // 2]
                mid_x = (p1["x"] + p2["x"]) / 2.0 if isinstance(p1["x"], (int, float)) and isinstance(p2["x"], (int, float)) else p1["x"]
                mid_y = round((p1["y"] + p2["y"]) / 2.0, 2)
                extracted_points.insert(len(extracted_points) // 2, {"x": mid_x, "y": mid_y})

        elif self.profile == "sign_confused":
            for p in extracted_points:
                if p["y"] < 0:
                    p["y"] = abs(p["y"])
                    break
            extracted_title += " Estimated"

        elif self.profile == "omitter":
            if len(extracted_points) > 2:
                extracted_points.pop()

        result = {
            "chart_id": chart_id,
            "title": extracted_title,
            "x_axis": extracted_x_axis,
            "y_axis": extracted_y_axis,
            "points": extracted_points,
            "_meta": {
                "is_genuine": False,
                "execution_mode": self.execution_mode,
                "provider": self.provider,
                "model_name": self.model_name,
                "note": self.status_note,
            },
        }
        return result


class GeminiVLM(BaseVLM):
    """
    Client for Google Gemini Multimodal Models (e.g. gemini-2.5-flash).
    Executes live API calls when GEMINI_API_KEY is available; falls back to realistic
    simulation if network, key, or quotas are unavailable.
    """

    def __init__(
        self,
        model_name: str = "Google Gemini 2.5 Flash",
        api_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            provider="Google Gemini",
            is_genuine=False,
            execution_mode="GENUINE (Live API)",
            status_note="Initializing Gemini client...",
        )
        self.api_model = api_model or ("gemini-2.5-flash" if "gemini-" not in model_name else model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def extract_chart_data(self, image_input: Union[str, Path, bytes]) -> dict:
        chart_id = "chart" if isinstance(image_input, bytes) else Path(image_input).stem

        if not self.api_key:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = "GEMINI_API_KEY is missing from environment. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

        try:
            b64_img, mime_type = _load_image_base64_and_mime(image_input)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.api_model}:generateContent?key={self.api_key}"
            prompt = EXTRACTION_SYSTEM_PROMPT + "\nReturn ONLY valid JSON with keys: chart_id, title, x_axis, y_axis, points (list of {x, y})."

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": b64_img,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.0,
                },
            }

            resp = requests.post(url, json=payload, timeout=18)
            if resp.status_code == 200:
                res_data = resp.json()
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                clean_dict = _normalize_chart_json(text, default_chart_id=chart_id)
                self.is_genuine = True
                self.execution_mode = "GENUINE (Live API)"
                self.status_note = f"Live Google Gemini API extraction succeeded ({self.model_name})"
                clean_dict["_meta"] = {
                    "is_genuine": True,
                    "execution_mode": self.execution_mode,
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "note": self.status_note,
                }
                return clean_dict
            else:
                err_text = resp.text[:120].replace("\n", " ")
                raise RuntimeError(f"HTTP {resp.status_code}: {err_text}")

        except Exception as e:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = f"Gemini API failure ({str(e)}). Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

    def _fallback_extract(self, image_input: Union[str, Path, bytes], chart_id: str) -> dict:
        fallback_sim = SimulatedVLM(model_name=self.model_name, profile="precise")
        res = fallback_sim.extract_chart_data(image_input)
        res["_meta"] = {
            "is_genuine": False,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model_name": self.model_name,
            "note": self.status_note,
        }
        return res


class AnthropicVLM(BaseVLM):
    """
    Client for Anthropic Claude Multimodal Models (e.g. claude-3-5-sonnet-20241022).
    Executes live API calls when ANTHROPIC_API_KEY is available; falls back to realistic
    simulation if quota, key, or credit balance is exhausted.
    """

    def __init__(
        self,
        model_name: str = "Anthropic Claude 3.5 Sonnet",
        api_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            provider="Anthropic Claude",
            is_genuine=False,
            execution_mode="GENUINE (Live API)",
            status_note="Initializing Anthropic client...",
        )
        self.api_model = api_model or ("claude-3-5-sonnet-20241022" if "claude-" not in model_name else model_name)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def extract_chart_data(self, image_input: Union[str, Path, bytes]) -> dict:
        chart_id = "chart" if isinstance(image_input, bytes) else Path(image_input).stem

        if not self.api_key:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = "ANTHROPIC_API_KEY is missing from environment. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

        try:
            b64_img, mime_type = _load_image_base64_and_mime(image_input)
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.api_model,
                "max_tokens": 1024,
                "system": EXTRACTION_SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": b64_img,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract all structured chart data into valid JSON with chart_id, title, x_axis, y_axis, and points.",
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                res_data = resp.json()
                content = res_data["content"][0]["text"]
                clean_dict = _normalize_chart_json(content, default_chart_id=chart_id)
                self.is_genuine = True
                self.execution_mode = "GENUINE (Live API)"
                self.status_note = f"Live Anthropic Claude extraction succeeded ({self.model_name})"
                clean_dict["_meta"] = {
                    "is_genuine": True,
                    "execution_mode": self.execution_mode,
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "note": self.status_note,
                }
                return clean_dict
            else:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text[:100]
                raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

        except Exception as e:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = f"Anthropic API: {str(e)[:90]}. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

    def _fallback_extract(self, image_input: Union[str, Path, bytes], chart_id: str) -> dict:
        fallback_sim = SimulatedVLM(model_name=self.model_name, profile="precise", random_seed=202)
        res = fallback_sim.extract_chart_data(image_input)
        res["_meta"] = {
            "is_genuine": False,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model_name": self.model_name,
            "note": self.status_note,
        }
        return res


class OpenAIVLM(BaseVLM):
    """
    Client for OpenAI Vision Models (e.g. gpt-4o, gpt-4o-mini).
    Executes live API calls when OPENAI_API_KEY is available; falls back to realistic
    simulation if key is invalid or quota is exhausted.
    """

    def __init__(
        self,
        model_name: str = "OpenAI GPT-4o-mini",
        api_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            provider="OpenAI GPT",
            is_genuine=False,
            execution_mode="GENUINE (Live API)",
            status_note="Initializing OpenAI client...",
        )
        self.api_model = api_model or ("gpt-4o-mini" if "gpt-" not in model_name else model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def extract_chart_data(self, image_input: Union[str, Path, bytes]) -> dict:
        chart_id = "chart" if isinstance(image_input, bytes) else Path(image_input).stem

        if not self.api_key:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = "OPENAI_API_KEY is missing from environment. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

        try:
            b64_img, mime_type = _load_image_base64_and_mime(image_input)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.api_model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all structured data from this scientific chart into valid JSON."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }

            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                res_data = resp.json()
                content = res_data["choices"][0]["message"]["content"]
                clean_dict = _normalize_chart_json(content, default_chart_id=chart_id)
                self.is_genuine = True
                self.execution_mode = "GENUINE (Live API)"
                self.status_note = f"Live OpenAI extraction succeeded ({self.model_name})"
                clean_dict["_meta"] = {
                    "is_genuine": True,
                    "execution_mode": self.execution_mode,
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "note": self.status_note,
                }
                return clean_dict
            else:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text[:100]
                raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

        except Exception as e:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = f"OpenAI API: {str(e)[:90]}. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

    def _fallback_extract(self, image_input: Union[str, Path, bytes], chart_id: str) -> dict:
        fallback_sim = SimulatedVLM(model_name=self.model_name, profile="precise", random_seed=101)
        res = fallback_sim.extract_chart_data(image_input)
        res["_meta"] = {
            "is_genuine": False,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model_name": self.model_name,
            "note": self.status_note,
        }
        return res


class GrokVLM(BaseVLM):
    """
    Client for xAI Grok Vision Models (e.g. grok-2-vision-1212).
    Executes live API calls when XAI_API_KEY is available; falls back to realistic
    simulation if key is invalid, quota is exhausted, or endpoint returns error.
    """

    def __init__(
        self,
        model_name: str = "xAI Grok-2 Vision",
        api_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            provider="xAI Grok",
            is_genuine=False,
            execution_mode="GENUINE (Live API)",
            status_note="Initializing xAI Grok client...",
        )
        self.api_model = api_model or "grok-2-vision-1212"
        self.api_key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")

    def extract_chart_data(self, image_input: Union[str, Path, bytes]) -> dict:
        chart_id = "chart" if isinstance(image_input, bytes) else Path(image_input).stem

        if not self.api_key:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = "XAI_API_KEY is missing from environment. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

        try:
            b64_img, mime_type = _load_image_base64_and_mime(image_input)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.api_model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all structured data from this scientific chart into valid JSON."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                        ],
                    },
                ],
                "temperature": 0.0,
            }

            resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                res_data = resp.json()
                content = res_data["choices"][0]["message"]["content"]
                clean_dict = _normalize_chart_json(content, default_chart_id=chart_id)
                self.is_genuine = True
                self.execution_mode = "GENUINE (Live API)"
                self.status_note = f"Live xAI Grok extraction succeeded ({self.model_name})"
                clean_dict["_meta"] = {
                    "is_genuine": True,
                    "execution_mode": self.execution_mode,
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "note": self.status_note,
                }
                return clean_dict
            else:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", err_json.get("message", resp.text))
                except Exception:
                    err_msg = resp.text[:100]
                raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

        except Exception as e:
            self.is_genuine = False
            self.execution_mode = "SIMULATED / RANDOM (Fallback)"
            self.status_note = f"xAI Grok API: {str(e)[:90]}. Reverted to simulation."
            return self._fallback_extract(image_input, chart_id)

    def _fallback_extract(self, image_input: Union[str, Path, bytes], chart_id: str) -> dict:
        fallback_sim = SimulatedVLM(model_name=self.model_name, profile="precise", random_seed=303)
        res = fallback_sim.extract_chart_data(image_input)
        res["_meta"] = {
            "is_genuine": False,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model_name": self.model_name,
            "note": self.status_note,
        }
        return res


XAIVLM = GrokVLM
