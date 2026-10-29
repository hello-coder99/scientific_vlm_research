"""
Automatic Axis Calibration and Coordinate Mapping Module.

Extracts objective visual-geometric evidence from a scientific chart image:
1. Detects the plot area canvas (spines / bounding box).
2. Extracts numerical tick marks from OCR with robust filtering and decimal repair.
3. Snaps ticks to detected visual gridlines or tick marks.
4. Performs robust consensus linear regression (RANSAC-style) to establish
   mathematical pixel-to-data affine coordinate transformations.
5. Produces a ChartCalibration object for bidirectional coordinate mapping
   (data space <-> pixel space).
"""

from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract


@dataclass
class PlotBox:
    """Bounding box of the chart plotting area in pixel coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_min + self.width / 2.0, self.y_min + self.height / 2.0)

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TickMark:
    """Individual tick mark extracted from the chart image."""
    pixel: float
    value: float
    raw_text: str
    confidence: float
    status: str = "inlier"  # "inlier", "repaired", "outlier"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AxisCalibration:
    """
    Mathematical calibration for a single axis.
    Model: data_value = slope * pixel_coord + intercept
    Inverse: pixel_coord = (data_value - intercept) / slope
    """
    axis_name: str  # 'x' or 'y'
    slope: float
    intercept: float
    r_squared: float
    ticks: List[TickMark] = field(default_factory=list)
    inlier_count: int = 0
    data_range: Tuple[float, float] = (0.0, 0.0)
    pixel_range: Tuple[float, float] = (0.0, 0.0)
    scale_type: str = "linear"

    def pixel_to_data(self, pixel_coord: float) -> float:
        """Transform pixel coordinate to chart data value."""
        return float(self.slope * pixel_coord + self.intercept)

    def data_to_pixel(self, data_val: float) -> float:
        """Transform chart data value to pixel coordinate."""
        if abs(self.slope) < 1e-12:
            raise ZeroDivisionError(f"Axis '{self.axis_name}' slope is near zero ({self.slope}).")
        return float((data_val - self.intercept) / self.slope)

    def to_dict(self) -> dict:
        return {
            "axis_name": self.axis_name,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "scale_type": self.scale_type,
            "inlier_count": self.inlier_count,
            "data_range": list(self.data_range),
            "pixel_range": list(self.pixel_range),
            "ticks": [t.to_dict() for t in self.ticks],
        }


@dataclass
class ChartCalibration:
    """
    Complete bidirectional coordinate calibration for a scientific chart.
    Maps between pixel space (x_pix, y_pix) and data space (x_data, y_data).
    """
    image_path: str
    image_width: int
    image_height: int
    plot_box: PlotBox
    x_axis: AxisCalibration
    y_axis: AxisCalibration
    title_text: Optional[str] = None
    x_label_text: Optional[str] = None
    y_label_text: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Calibration is considered statistically valid if R^2 >= 0.95 on both axes."""
        return (
            self.x_axis is not None
            and self.y_axis is not None
            and self.x_axis.r_squared >= 0.95
            and self.y_axis.r_squared >= 0.95
            and self.x_axis.inlier_count >= 2
            and self.y_axis.inlier_count >= 2
        )

    def pixel_to_data(self, px: float, py: float) -> Tuple[float, float]:
        """Convert image pixel coordinate (px, py) to chart data coordinate (x, y)."""
        x_data = self.x_axis.pixel_to_data(px)
        y_data = self.y_axis.pixel_to_data(py)
        return x_data, y_data

    def data_to_pixel(self, dx: float, dy: float) -> Tuple[float, float]:
        """Convert chart data coordinate (dx, dy) to image pixel coordinate (px, py)."""
        px = self.x_axis.data_to_pixel(dx)
        py = self.y_axis.data_to_pixel(dy)
        return px, py

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "plot_box": self.plot_box.to_dict(),
            "x_axis": self.x_axis.to_dict(),
            "y_axis": self.y_axis.to_dict(),
            "title_text": self.title_text,
            "x_label_text": self.x_label_text,
            "y_label_text": self.y_label_text,
            "is_valid": self.is_valid,
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """Save calibration metadata to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, d: dict) -> "ChartCalibration":
        plot_box = PlotBox(
            x_min=d["plot_box"]["x_min"],
            y_min=d["plot_box"]["y_min"],
            x_max=d["plot_box"]["x_max"],
            y_max=d["plot_box"]["y_max"],
        )

        def make_axis(ad: dict) -> AxisCalibration:
            ticks = [
                TickMark(
                    pixel=t["pixel"],
                    value=t["value"],
                    raw_text=t["raw_text"],
                    confidence=t.get("confidence", 100.0),
                    status=t.get("status", "inlier"),
                    bbox=tuple(t.get("bbox", (0, 0, 0, 0))),
                )
                for t in ad.get("ticks", [])
            ]
            return AxisCalibration(
                axis_name=ad["axis_name"],
                slope=ad["slope"],
                intercept=ad["intercept"],
                r_squared=ad["r_squared"],
                ticks=ticks,
                inlier_count=ad.get("inlier_count", len(ticks)),
                data_range=tuple(ad.get("data_range", (0.0, 0.0))),
                pixel_range=tuple(ad.get("pixel_range", (0.0, 0.0))),
                scale_type=ad.get("scale_type", "linear"),
            )

        return cls(
            image_path=d.get("image_path", ""),
            image_width=d.get("image_width", 0),
            image_height=d.get("image_height", 0),
            plot_box=plot_box,
            x_axis=make_axis(d["x_axis"]),
            y_axis=make_axis(d["y_axis"]),
            title_text=d.get("title_text"),
            x_label_text=d.get("x_label_text"),
            y_label_text=d.get("y_label_text"),
        )

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID (Low R^2 or insufficient ticks)"
        return (
            f"--- Chart Calibration Summary [{status}] ---\n"
            f"Image: {self.image_path} ({self.image_width}x{self.image_height})\n"
            f"Plot Area: x=[{self.plot_box.x_min}, {self.plot_box.x_max}], "
            f"y=[{self.plot_box.y_min}, {self.plot_box.y_max}] "
            f"({self.plot_box.width}x{self.plot_box.height} px)\n"
            f"X-Axis: dx = {self.x_axis.slope:.6f} * px + {self.x_axis.intercept:.6f} "
            f"(R^2 = {self.x_axis.r_squared:.5f}, {self.x_axis.inlier_count} ticks, "
            f"data range: {self.x_axis.data_range})\n"
            f"Y-Axis: dy = {self.y_axis.slope:.6f} * py + {self.y_axis.intercept:.6f} "
            f"(R^2 = {self.y_axis.r_squared:.5f}, {self.y_axis.inlier_count} ticks, "
            f"data range: {self.y_axis.data_range})\n"
            f"Labels: Title='{self.title_text}', X='{self.x_label_text}', Y='{self.y_label_text}'"
        )


# =========================================================================
# Computer Vision & Extraction Primitives
# =========================================================================

def detect_plot_box(image: np.ndarray) -> PlotBox:
    """
    Detect the rectangular chart plotting area (canvas) inside the image.
    Uses morphological line extraction combined with Hough spine detection.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Adaptive thresholding to extract line structures
    thresh = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    # Morphological kernels for horizontal and vertical lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 40), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    grid = cv2.add(h_lines, v_lines)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes = []
    min_w = int(w * 0.25)
    min_h = int(h * 0.25)

    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw >= min_w and bh >= min_h:
            candidate_boxes.append((bx, by, bx + bw, by + bh, bw * bh))

    # Refine using Canny + Hough lines to snap box edges to true axis spines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=min_w // 2, maxLineGap=10)

    v_spines = []
    h_spines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x1 - x2) <= 3 and abs(y2 - y1) >= min_h // 2:
                v_spines.append(int((x1 + x2) / 2))
            elif abs(y1 - y2) <= 3 and abs(x2 - x1) >= min_w // 2:
                h_spines.append(int((y1 + y2) / 2))

    if candidate_boxes:
        candidate_boxes.sort(key=lambda b: b[4], reverse=True)
        best_box = candidate_boxes[0]
        x_min, y_min, x_max, y_max = best_box[:4]
    else:
        # Fallback to margins if no closed contour found
        x_min, y_min, x_max, y_max = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.85)

    # Snap to closest detected long spines if within 15 pixels
    if v_spines:
        left_spines = [vx for vx in v_spines if abs(vx - x_min) <= 25]
        if left_spines:
            x_min = int(np.median(left_spines))
        right_spines = [vx for vx in v_spines if abs(vx - x_max) <= 25]
        if right_spines:
            x_max = int(np.median(right_spines))

    if h_spines:
        top_spines = [hy for hy in h_spines if abs(hy - y_min) <= 25]
        if top_spines:
            y_min = int(np.median(top_spines))
        bottom_spines = [hy for hy in h_spines if abs(hy - y_max) <= 25]
        if bottom_spines:
            y_max = int(np.median(bottom_spines))

    return PlotBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def detect_gridlines(image: np.ndarray, plot_box: PlotBox) -> Tuple[List[int], List[int]]:
    """
    Detect visual gridlines / tick marks within or along the plot box.
    Returns: (x_gridlines, y_gridlines) sorted lists of pixel coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=5)

    x_coords = []
    y_coords = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Vertical line
            if abs(x1 - x2) <= 2:
                # Must be within plot x range and reasonably long
                mx = int((x1 + x2) / 2)
                if (plot_box.x_min - 5) <= mx <= (plot_box.x_max + 5) and abs(y2 - y1) >= 40:
                    x_coords.append(mx)
            # Horizontal line
            elif abs(y1 - y2) <= 2:
                my = int((y1 + y2) / 2)
                if (plot_box.y_min - 5) <= my <= (plot_box.y_max + 5) and abs(x2 - x1) >= 40:
                    y_coords.append(my)

    # Cluster close lines within 4 pixels
    def cluster_coords(coords: List[int]) -> List[int]:
        if not coords:
            return []
        coords = sorted(coords)
        clusters = [[coords[0]]]
        for c in coords[1:]:
            if c - clusters[-1][-1] <= 4:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [int(np.mean(cl)) for cl in clusters]

    return cluster_coords(x_coords), cluster_coords(y_coords)


def extract_clean_ocr_words(image: np.ndarray, plot_box: PlotBox) -> List[dict]:
    """
    Extract OCR word tokens, confidences, and bounding boxes.
    Applies axis-spine masking to prevent tick dashes/spines from fusing with digits.
    """
    h, w = image.shape[:2]
    ocr_img = image.copy()

    # Mask thin vertical band around left axis spine (x_min - 4 to x_min + 4)
    # and bottom spine (y_max - 4 to y_max + 4) with white to separate ticks from digits
    pad = 5
    x_spine_start = max(0, plot_box.x_min - pad)
    x_spine_end = min(w, plot_box.x_min + pad)
    ocr_img[max(0, plot_box.y_min - 10):min(h, plot_box.y_max + 10), x_spine_start:x_spine_end] = (255, 255, 255)

    y_spine_start = max(0, plot_box.y_max - pad)
    y_spine_end = min(h, plot_box.y_max + pad)
    ocr_img[y_spine_start:y_spine_end, max(0, plot_box.x_min - 10):min(w, plot_box.x_max + 10)] = (255, 255, 255)

    data = pytesseract.image_to_data(ocr_img, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0.0

        words.append({
            "text": txt,
            "confidence": conf,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "cx": float(data["left"][i] + data["width"][i] / 2.0),
            "cy": float(data["top"][i] + data["height"][i] / 2.0),
        })
    return words


def parse_numerical_value(text: str) -> Optional[float]:
    """
    Parse a clean float/int from an OCR string token.
    Handles percentages, currency, and strips trailing noise.
    """
    cleaned = text.strip().rstrip("%").lstrip("$").rstrip(",").strip()
    # Check standard number format: -12.34 or 1234
    m = re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", cleaned)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None

    # Check if a decimal is embedded, e.g. "87.5." or "87.5-"
    m2 = re.search(r"-?\d+\.\d+", cleaned)
    if m2:
        try:
            return float(m2.group(0))
        except ValueError:
            pass

    # Pure integer embedded
    m3 = re.search(r"-?\d+", cleaned)
    if m3:
        try:
            return float(m3.group(0))
        except ValueError:
            pass

    return None


def extract_axis_ticks_and_labels(
    words: List[dict],
    plot_box: PlotBox,
    img_width: int,
    img_height: int,
    gridlines_x: List[int],
    gridlines_y: List[int],
) -> Tuple[List[TickMark], List[TickMark], Optional[str], Optional[str], Optional[str]]:
    """
    Categorizes OCR tokens into:
    1. Y-axis numerical ticks
    2. X-axis numerical ticks
    3. Title text
    4. X-axis label
    5. Y-axis label
    """
    y_tick_candidates = []
    x_tick_candidates = []
    title_tokens = []
    x_label_tokens = []
    y_label_tokens = []

    # Regions:
    # Title: y < plot_box.y_min - 5, centered horizontally
    # Y-ticks: x < plot_box.x_min, and y within [y_min - 15, y_max + 15]
    # X-ticks: y > plot_box.y_max, and x within [x_min - 15, x_max + 15]

    for w in words:
        cx, cy = w["cx"], w["cy"]
        txt = w["text"]
        num_val = parse_numerical_value(txt)

        # Title detection
        if cy < plot_box.y_min - 10:
            title_tokens.append(w)
            continue

        # Y-axis margin detection (left of plot area)
        if w["x"] < plot_box.x_min and (plot_box.y_min - 20 <= cy <= plot_box.y_max + 20):
            if num_val is not None:
                y_tick_candidates.append((w, num_val))
            else:
                y_label_tokens.append(w)
            continue

        # X-axis margin detection (below plot area)
        if cy > plot_box.y_max - 5 and (plot_box.x_min - 20 <= cx <= plot_box.x_max + 20):
            # Check if this token is further down (likely x-axis label, e.g. "Epoch")
            # If cy is more than 35px below y_max and is non-numeric, it's label
            if num_val is not None:
                x_tick_candidates.append((w, num_val))
            else:
                x_label_tokens.append(w)
            continue

    # -------------------------------------------------------------
    # Snap ticks to nearest gridlines if available, otherwise text center
    # -------------------------------------------------------------
    def make_ticks(
        candidates: List[Tuple[dict, float]],
        axis: str,
        gridlines: List[int]
    ) -> List[TickMark]:
        ticks = []
        for w, val in candidates:
            coord = w["cx"] if axis == "x" else w["cy"]
            # Try to snap to nearest gridline within 15px
            snapped = coord
            if gridlines:
                nearest = min(gridlines, key=lambda g: abs(g - coord))
                if abs(nearest - coord) <= 15:
                    snapped = float(nearest)

            ticks.append(
                TickMark(
                    pixel=snapped,
                    value=val,
                    raw_text=w["text"],
                    confidence=w["confidence"],
                    status="inlier",
                    bbox=(w["x"], w["y"], w["width"], w["height"]),
                )
            )
        # Sort ticks by pixel coordinate
        ticks.sort(key=lambda t: t.pixel)
        return ticks

    y_ticks = make_ticks(y_tick_candidates, "y", gridlines_y)
    x_ticks = make_ticks(x_tick_candidates, "x", gridlines_x)

    # Reconstruct text labels (filter low-confidence noise and unprintable characters)
    def clean_label_tokens(tokens: List[dict]) -> Optional[str]:
        valid = [
            t["text"] for t in tokens
            if t.get("confidence", 0) >= 50.0 and re.search(r"[a-zA-Z0-9]", t["text"])
        ]
        return " ".join(valid) if valid else None

    title_tokens.sort(key=lambda t: t["x"])
    title_str = clean_label_tokens(title_tokens)

    x_label_tokens.sort(key=lambda t: t["x"])
    x_label_str = clean_label_tokens(x_label_tokens)

    # For Y label, reading vertically
    y_label_tokens.sort(key=lambda t: t["y"])
    y_label_str = clean_label_tokens(y_label_tokens)

    return x_ticks, y_ticks, title_str, x_label_str, y_label_str


def robust_axis_linear_fit(ticks: List[TickMark], axis_name: str) -> AxisCalibration:
    """
    Fits: data_value = slope * pixel_coord + intercept
    Uses consensus / RANSAC line fitting to discard OCR noise and repair missing decimals.
    """
    if len(ticks) < 2:
        return AxisCalibration(
            axis_name=axis_name,
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            ticks=ticks,
            inlier_count=len(ticks),
        )

    pixels = np.array([t.pixel for t in ticks], dtype=float)
    values = np.array([t.value for t in ticks], dtype=float)
    n = len(ticks)

    best_score = -float("inf")
    best_m = 0.0
    best_c = 0.0
    best_inliers: List[int] = []

    # Robust pairwise consensus (exhaustive if n <= 20, which is always true for chart ticks)
    for i in range(n):
        for j in range(i + 1, n):
            dp = pixels[j] - pixels[i]
            if abs(dp) < 1e-3:
                continue
            m = (values[j] - values[i]) / dp
            c = values[i] - m * pixels[i]

            # In standard image coords:
            # X axis: pixel increases left-to-right, value should increase -> slope > 0
            # Y axis: pixel increases top-to-bottom, value should decrease -> slope < 0
            if axis_name == "x" and m <= 0:
                continue
            if axis_name == "y" and m >= 0:
                continue

            # Evaluate residuals
            preds = m * pixels + c
            residuals = np.abs(values - preds)
            val_span = np.max(values) - np.min(values)
            tol = max(0.05 * val_span, 0.1)

            inliers = np.where(residuals <= tol)[0]
            if len(inliers) >= 2:
                # Refit on inliers
                m_refit, c_refit = np.polyfit(pixels[inliers], values[inliers], 1)
                preds_inf = m_refit * pixels[inliers] + c_refit
                ss_tot = np.sum((values[inliers] - np.mean(values[inliers])) ** 2)
                ss_res = np.sum((values[inliers] - preds_inf) ** 2)
                r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

                # Score combines inlier count and R^2
                score = len(inliers) * 100.0 + r2
                if score > best_score:
                    best_score = score
                    best_m = float(m_refit)
                    best_c = float(c_refit)
                    best_inliers = list(inliers)

    # If no valid pair met orientation constraints, fall back to simple polyfit
    if len(best_inliers) < 2:
        best_m, best_c = np.polyfit(pixels, values, 1)
        best_inliers = list(range(n))

    # Check non-inliers for OCR decimal drop (e.g. 775 instead of 77.5, 15 instead of 1.5)
    for idx in range(n):
        if idx not in best_inliers:
            raw_v = values[idx]
            p = pixels[idx]
            expected_v = best_m * p + best_c
            # Test scale factors 10^-1, 10^-2
            for scale in [0.1, 0.01]:
                repaired_v = raw_v * scale
                if abs(repaired_v - expected_v) <= max(0.05 * abs(expected_v), 0.1):
                    ticks[idx].value = repaired_v
                    ticks[idx].status = "repaired"
                    values[idx] = repaired_v
                    best_inliers.append(idx)
                    break
            else:
                ticks[idx].status = "outlier"

    # Final refit including any repaired ticks
    best_inliers.sort()
    final_pixels = pixels[best_inliers]
    final_values = values[best_inliers]

    if len(best_inliers) >= 2:
        best_m, best_c = np.polyfit(final_pixels, final_values, 1)
        preds = best_m * final_pixels + best_c
        ss_tot = np.sum((final_values - np.mean(final_values)) ** 2)
        ss_res = np.sum((final_values - preds) ** 2)
        final_r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    else:
        final_r2 = 0.0

    # Label inlier status on ticks
    for idx in best_inliers:
        if ticks[idx].status != "repaired":
            ticks[idx].status = "inlier"

    data_min = float(np.min(final_values)) if len(final_values) > 0 else 0.0
    data_max = float(np.max(final_values)) if len(final_values) > 0 else 0.0
    pixel_min = float(np.min(final_pixels)) if len(final_pixels) > 0 else 0.0
    pixel_max = float(np.max(final_pixels)) if len(final_pixels) > 0 else 0.0

    return AxisCalibration(
        axis_name=axis_name,
        slope=float(best_m),
        intercept=float(best_c),
        r_squared=float(final_r2),
        ticks=ticks,
        inlier_count=len(best_inliers),
        data_range=(data_min, data_max),
        pixel_range=(pixel_min, pixel_max),
        scale_type="linear",
    )


# =========================================================================
# Main Calibration Orchestrator
# =========================================================================

def calibrate_chart(
    image_input: Union[str, Path, np.ndarray],
    ocr_words: Optional[List[dict]] = None
) -> ChartCalibration:
    """
    Main entry point: Automatically calibrate a chart image to establish
    its pixel <-> data space coordinate transformation.

    Args:
        image_input: File path (str/Path) or pre-loaded BGR image numpy array.
        ocr_words: Optional pre-extracted OCR word list. If None, runs OCR automatically.

    Returns:
        ChartCalibration with verified mathematical mappings for X and Y axes.
    """
    if isinstance(image_input, (str, Path)):
        img_path_str = str(image_input)
        image = cv2.imread(img_path_str)
        if image is None:
            raise FileNotFoundError(f"Could not read chart image at '{img_path_str}'")
    elif isinstance(image_input, np.ndarray):
        img_path_str = "in_memory_image"
        image = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    img_h, img_w = image.shape[:2]

    # Step 1: Detect Plot Box Area
    plot_box = detect_plot_box(image)

    # Step 2: Detect Gridlines
    grid_x, grid_y = detect_gridlines(image, plot_box)

    # Step 3: Extract Clean OCR (with spine masking) if not supplied
    if ocr_words is None:
        ocr_words = extract_clean_ocr_words(image, plot_box)

    # Step 4: Parse Ticks and Labels
    x_ticks, y_ticks, title_text, x_label, y_label = extract_axis_ticks_and_labels(
        ocr_words, plot_box, img_w, img_h, grid_x, grid_y
    )

    # Step 5: Fit Axes with Robust Consensus
    x_calib = robust_axis_linear_fit(x_ticks, "x")
    y_calib = robust_axis_linear_fit(y_ticks, "y")

    # Step 6: Assemble Complete Calibration
    calibration = ChartCalibration(
        image_path=img_path_str,
        image_width=img_w,
        image_height=img_h,
        plot_box=plot_box,
        x_axis=x_calib,
        y_axis=y_calib,
        title_text=title_text,
        x_label_text=x_label,
        y_label_text=y_label,
    )

    return calibration


if __name__ == "__main__":
    # Test on chart_001.png
    base_dir = Path(__file__).resolve().parent.parent.parent
    sample_chart = base_dir / "data" / "charts" / "chart_001.png"

    if sample_chart.exists():
        print(f"Calibrating {sample_chart}...")
        calib = calibrate_chart(sample_chart)
        print(calib.summary())
        print(f"Calibration Valid: {calib.is_valid}")

        # Test point mapping
        print("\nTest Ground Truth Mapping:")
        for x_val, y_val in [(1, 72), (2, 76), (3, 81), (4, 85), (5, 89)]:
            px, py = calib.data_to_pixel(x_val, y_val)
            rx, ry = calib.pixel_to_data(px, py)
            print(f"Data ({x_val}, {y_val}) -> Pixel ({px:.1f}, {py:.1f}) -> Reconstructed ({rx:.2f}, {ry:.2f})")
    else:
        print(f"Sample chart not found at {sample_chart}")
