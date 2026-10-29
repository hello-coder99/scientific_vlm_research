"""
Evidence-Grounded Visual Verifier (Zero Ground-Truth Verification Engine).

Directly cross-checks structured VLM extractions (prediction.json) against
the objective visual, geometric, and textual evidence of the chart image:
1. Detects physical data markers (circles, squares, triangles) from raw image pixels.
2. Projects predicted points (x, y) into calibrated pixel space.
3. Probes the visual canvas:
   - Identifies SUPPORTED claims (visual mark confirmed within tolerance).
   - Detects VALUE_CONTRADICTED hallucinations (mark exists at different Y).
   - Detects FABRICATED_POINT hallucinations (invented points with no mark).
   - Detects OUT_OF_BOUNDS coordinates (outside physical canvas).
   - Detects OMITTED_POINT omissions (visual marks the VLM failed to extract).
4. Verifies Title and Axis Labels against OCR text evidence.
"""

from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ..extraction.calibration import calibrate_chart, ChartCalibration, PlotBox
except ImportError:
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    from extraction.calibration import calibrate_chart, ChartCalibration, PlotBox


@dataclass
class VisualProbeResult:
    """Individual verification verdict for a claimed data point."""
    x: float
    claimed_y: float
    claimed_pixel: Tuple[float, float]
    status: str  # "SUPPORTED", "VALUE_CONTRADICTED", "FABRICATED_POINT", "OUT_OF_BOUNDS"
    observed_y: Optional[float] = None
    observed_pixel: Optional[Tuple[float, float]] = None
    discrepancy: Optional[float] = None
    relative_error: Optional[float] = None
    confidence: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextGroundingResult:
    """Verification verdict for a semantic text claim (title, axis label)."""
    field_name: str
    claimed_text: str
    observed_text: Optional[str]
    similarity: float
    status: str  # "SUPPORTED", "CONTRADICTED", "UNGROUNDED"
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    """Complete Evidence-Grounded Verification Report."""
    chart_id: str
    is_calibrated: bool
    point_results: List[VisualProbeResult] = field(default_factory=list)
    text_results: Dict[str, TextGroundingResult] = field(default_factory=dict)
    omitted_points: List[dict] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chart_id": self.chart_id,
            "is_calibrated": self.is_calibrated,
            "counts": self.counts,
            "metrics": self.metrics,
            "text_verification": {k: v.to_dict() for k, v in self.text_results.items()},
            "point_verification": [p.to_dict() for p in self.point_results],
            "omitted_points": self.omitted_points,
        }

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 60,
            "EVIDENCE-GROUNDED VERIFICATION REPORT (ZERO GROUND-TRUTH)",
            "=" * 60,
            f"Chart ID: {self.chart_id} (Calibration: {'VALID' if self.is_calibrated else 'FAILED'})",
            "\n--- Semantic Text Grounding ---"
        ]
        for field_name, tr in self.text_results.items():
            lines.append(
                f"{field_name:<12}: [{tr.status:<12}] "
                f"Claimed='{tr.claimed_text}' | Observed='{tr.observed_text or 'None'}' "
                f"(similarity={tr.similarity:.2f})"
            )

        lines.extend([
            "\n--- Visual Point Verification ---",
            f"Supported Points     : {self.counts.get('supported', 0)}",
            f"Contradicted Values  : {self.counts.get('contradicted', 0)} (Numerical Hallucinations)",
            f"Fabricated Points    : {self.counts.get('fabricated', 0)} (Invented Points)",
            f"Omitted Points       : {self.counts.get('omitted', 0)} (Missed in extraction)",
            "\n--- Verification Metrics ---",
            f"Support Rate         : {self.metrics.get('support_rate', 0.0) * 100:.2f}%",
            f"Hallucination Rate   : {self.metrics.get('hallucination_rate', 0.0) * 100:.2f}%",
            f"Overall Grounding    : {self.metrics.get('overall_grounding_score', 0.0) * 100:.2f}%",
            "\n--- Point-by-Point Evidence Breakdown ---"
        ])

        for pr in self.point_results:
            obs_str = f"observed={pr.observed_y:.2f}" if pr.observed_y is not None else "no marker found"
            lines.append(
                f"x={pr.x} | claimed={pr.claimed_y} | {obs_str:<22} | [{pr.status}] {pr.explanation}"
            )

        if self.omitted_points:
            lines.append("\n--- Unextracted Visual Marks Found On Chart ---")
            for op in self.omitted_points:
                lines.append(f"Visual Mark at x={op['x']:.2f}, y={op['y']:.2f} (omitted by VLM)")

        lines.append("=" * 60)
        return "\n".join(lines)


# =========================================================================
# Computer Vision Marker Detection Engine
# =========================================================================

def detect_visual_markers(
    image: np.ndarray,
    plot_box: PlotBox,
    calibration: ChartCalibration
) -> List[dict]:
    """
    Detects discrete visual data marks (circles, squares, triangles) from
    the chart canvas pixels. Returns list of detected marker dicts:
    [{'x': data_x, 'y': data_y, 'px': pixel_x, 'py': pixel_y, 'area': area}]
    """
    canvas = image[plot_box.y_min:plot_box.y_max, plot_box.x_min:plot_box.x_max]
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Foreground mask: high color saturation OR distinct dark contrast from white canvas
    sat_mask = hsv[:, :, 1] > 60
    contrast_mask = gray < 180
    fg = (sat_mask | contrast_mask).astype(np.uint8) * 255

    # Morphological opening with (8, 8) kernel cleanly severs the thin connecting
    # line segments (width ~ 2-3px) while preserving 2D marker blobs (diameter ~ 12-16px)
    k_marker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    markers_binary = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_marker)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(markers_binary)
    detected_markers: List[dict] = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Filter out tiny residual specks and oversized bounding boxes
        aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
        if 40 <= area <= 1200 and cw <= 55 and ch <= 55 and aspect <= 2.2:
            local_cx, local_cy = centroids[i]
            global_px = float(plot_box.x_min + local_cx)
            global_py = float(plot_box.y_min + local_cy)

            data_x, data_y = calibration.pixel_to_data(global_px, global_py)
            detected_markers.append({
                "x": float(data_x),
                "y": float(data_y),
                "px": global_px,
                "py": global_py,
                "area": area,
                "width": cw,
                "height": ch,
            })

    # Sort markers along the x-axis
    detected_markers.sort(key=lambda m: m["x"])
    return detected_markers


# =========================================================================
# Text & Semantic Grounding
# =========================================================================

def string_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Calculates normalized token-aware similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    s1_clean = s1.lower().strip()
    s2_clean = s2.lower().strip()
    if s1_clean == s2_clean:
        return 1.0

    # Token overlap (Jaccard)
    tok1 = set(s1_clean.split())
    tok2 = set(s2_clean.split())
    if tok1 and tok2:
        jaccard = len(tok1 & tok2) / len(tok1 | tok2)
    else:
        jaccard = 0.0

    # Sequence matcher ratio
    seq_ratio = SequenceMatcher(None, s1_clean, s2_clean).ratio()

    # Harmonic combination
    return 0.6 * seq_ratio + 0.4 * jaccard


def verify_text_grounding(
    prediction: dict,
    calibration: ChartCalibration
) -> Dict[str, TextGroundingResult]:
    """
    Verifies claimed Title, X-axis label, and Y-axis label against OCR evidence.
    """
    results: Dict[str, TextGroundingResult] = {}

    checks = [
        ("title", prediction.get("title", ""), calibration.title_text),
        ("x_axis", prediction.get("x_axis", ""), calibration.x_label_text),
        ("y_axis", prediction.get("y_axis", ""), calibration.y_label_text),
    ]

    for field_name, claimed, observed in checks:
        sim = string_similarity(claimed, observed)
        if sim >= 0.75:
            status = "SUPPORTED"
            expl = f"Grounds closely with visual text evidence '{observed}'."
        elif sim >= 0.40:
            status = "CONTRADICTED"
            expl = f"Partial mismatch: claimed '{claimed}', but image text is '{observed}'."
        else:
            status = "UNGROUNDED"
            expl = f"Fabricated or ungrounded label: no matching text observed in chart."

        results[field_name] = TextGroundingResult(
            field_name=field_name,
            claimed_text=claimed,
            observed_text=observed,
            similarity=float(sim),
            status=status,
            explanation=expl,
        )

    return results


# =========================================================================
# Point Verification Engine
# =========================================================================

def verify_points_against_visual_evidence(
    claimed_points: List[dict],
    observed_markers: List[dict],
    calibration: ChartCalibration,
    rel_tolerance: float = 0.03
) -> Tuple[List[VisualProbeResult], List[dict]]:
    """
    Cross-checks claimed data points directly against detected visual markers.

    rel_tolerance: relative error tolerance proportional to Y data range
                   (default: 0.03 = 3% of full Y axis range).
    """
    pb = calibration.plot_box
    y_range = abs(calibration.y_axis.data_range[1] - calibration.y_axis.data_range[0])
    abs_tolerance = max(rel_tolerance * y_range, 0.1)

    # Pixel margin tolerance for matching x coordinate (typically ~15-20 pixels)
    x_span = abs(calibration.x_axis.data_range[1] - calibration.x_axis.data_range[0])
    px_span = pb.width
    x_val_tol = max(18.0 * (x_span / (px_span + 1e-5)), 0.2)

    point_results: List[VisualProbeResult] = []
    matched_marker_indices = set()

    for pt in claimed_points:
        qx = float(pt["x"])
        qy = float(pt["y"])

        # 1. Check canvas bounds
        try:
            px, py = calibration.data_to_pixel(qx, qy)
        except Exception as e:
            point_results.append(VisualProbeResult(
                x=qx, claimed_y=qy, claimed_pixel=(0.0, 0.0),
                status="OUT_OF_BOUNDS", explanation=f"Mapping failed: {e}"
            ))
            continue

        margin = 15
        if not (pb.x_min - margin <= px <= pb.x_max + margin and pb.y_min - margin <= py <= pb.y_max + margin):
            point_results.append(VisualProbeResult(
                x=qx, claimed_y=qy, claimed_pixel=(px, py),
                status="OUT_OF_BOUNDS",
                explanation=f"Point ({qx}, {qy}) maps to pixel ({px:.1f}, {py:.1f}), outside canvas boundaries."
            ))
            continue

        # 2. Find closest observed marker along X axis
        candidate_markers = [
            (idx, m) for idx, m in enumerate(observed_markers)
            if abs(m["x"] - qx) <= x_val_tol
        ]

        if candidate_markers:
            # Pick marker closest along X
            candidate_markers.sort(key=lambda cm: abs(cm[1]["x"] - qx))
            best_idx, best_m = candidate_markers[0]
            matched_marker_indices.add(best_idx)

            obs_y = best_m["y"]
            discrepancy = abs(qy - obs_y)
            rel_err = discrepancy / (y_range + 1e-12)

            if discrepancy <= abs_tolerance:
                status = "SUPPORTED"
                expl = f"Confirmed by visual mark at ({best_m['x']:.2f}, {obs_y:.2f})."
            else:
                status = "VALUE_CONTRADICTED"
                expl = (
                    f"Numerical hallucination: claimed y={qy}, but visual mark is at "
                    f"y={obs_y:.2f} (error = {discrepancy:.2f})."
                )

            point_results.append(VisualProbeResult(
                x=qx,
                claimed_y=qy,
                claimed_pixel=(px, py),
                status=status,
                observed_y=obs_y,
                observed_pixel=(best_m["px"], best_m["py"]),
                discrepancy=float(discrepancy),
                relative_error=float(rel_err),
                confidence=float(max(0.0, 1.0 - rel_err)),
                explanation=expl,
            ))
        else:
            # No marker at this X coordinate: Fabricated Point!
            point_results.append(VisualProbeResult(
                x=qx,
                claimed_y=qy,
                claimed_pixel=(px, py),
                status="FABRICATED_POINT",
                observed_y=None,
                observed_pixel=None,
                discrepancy=None,
                relative_error=None,
                confidence=0.0,
                explanation=f"Hallucinated point: no visual marker exists in the chart at x={qx}.",
            ))

    # Identify any observed visual markers that were omitted from prediction
    omitted = []
    for idx, m in enumerate(observed_markers):
        if idx not in matched_marker_indices:
            omitted.append({
                "x": m["x"],
                "y": m["y"],
                "px": m["px"],
                "py": m["py"],
            })

    return point_results, omitted


# =========================================================================
# High-Level Deterministic Verifier Entry Point
# =========================================================================

def verify_chart_visually(
    image_input: Union[str, Path, np.ndarray],
    prediction_input: Union[str, Path, dict],
    rel_tolerance: float = 0.03,
    calibration: Optional[ChartCalibration] = None
) -> VerificationResult:
    """
    Deterministically verifies a VLM extraction against a chart image.
    ZERO ground truth required.

    Args:
        image_input: Path to chart image or preloaded BGR numpy array.
        prediction_input: Path to prediction JSON or parsed prediction dictionary.
        rel_tolerance: Relative numerical error tolerance (default 0.03 = 3%).
        calibration: Optional precomputed ChartCalibration.

    Returns:
        VerificationResult with point-by-point verdicts, text grounding, and scores.
    """
    # 1. Load Image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        chart_id = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
    elif isinstance(image_input, np.ndarray):
        chart_id = "in_memory_chart"
        image = image_input
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # 2. Load Prediction
    if isinstance(prediction_input, (str, Path)):
        pred_path = Path(prediction_input)
        with open(pred_path, "r", encoding="utf-8") as f:
            prediction = json.load(f)
    elif isinstance(prediction_input, dict):
        prediction = prediction_input
    else:
        raise TypeError(f"Unsupported prediction type: {type(prediction_input)}")

    chart_id = prediction.get("chart_id", chart_id)

    # 3. Calibrate Chart Coordinate System
    if calibration is None:
        calibration = calibrate_chart(image)

    # 4. Detect Visual Markers directly from pixels
    observed_markers = detect_visual_markers(image, calibration.plot_box, calibration)

    # 5. Extract claimed points from prediction
    claimed_points = prediction.get("points", [])
    if not claimed_points and "series" in prediction:
        # Handle nested series format
        claimed_points = [
            pt for series in prediction["series"]
            for pt in series.get("points", [])
        ]

    # 6. Verify Points
    point_results, omitted_points = verify_points_against_visual_evidence(
        claimed_points, observed_markers, calibration, rel_tolerance
    )

    # 7. Verify Text Elements
    text_results = verify_text_grounding(prediction, calibration)

    # 8. Compute Summary Counts and Metrics
    supported_cnt = sum(1 for p in point_results if p.status == "SUPPORTED")
    contradicted_cnt = sum(1 for p in point_results if p.status == "VALUE_CONTRADICTED")
    fabricated_cnt = sum(1 for p in point_results if p.status == "FABRICATED_POINT")
    out_of_bounds_cnt = sum(1 for p in point_results if p.status == "OUT_OF_BOUNDS")
    omitted_cnt = len(omitted_points)
    total_claimed = len(point_results)

    if total_claimed > 0:
        support_rate = supported_cnt / total_claimed
        hallucination_rate = (contradicted_cnt + fabricated_cnt + out_of_bounds_cnt) / total_claimed
    else:
        support_rate = 0.0
        hallucination_rate = 0.0

    # Text score
    text_scores = [tr.similarity for tr in text_results.values()]
    text_score = float(np.mean(text_scores)) if text_scores else 1.0

    # Overall Grounding Score: 70% numerical accuracy, 30% text grounding
    overall_score = float(0.7 * support_rate + 0.3 * text_score)

    result = VerificationResult(
        chart_id=chart_id,
        is_calibrated=calibration.is_valid,
        point_results=point_results,
        text_results=text_results,
        omitted_points=omitted_points,
        counts={
            "supported": supported_cnt,
            "contradicted": contradicted_cnt,
            "fabricated": fabricated_cnt,
            "out_of_bounds": out_of_bounds_cnt,
            "omitted": omitted_cnt,
            "total_claimed": total_claimed,
        },
        metrics={
            "support_rate": float(support_rate),
            "hallucination_rate": float(hallucination_rate),
            "text_grounding_score": float(text_score),
            "overall_grounding_score": float(overall_score),
        },
    )

    return result


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Test visual verification across all 3 benchmark charts
    test_cases = [
        ("chart_001.png", "chart_001_prediction.json"),
        ("chart_002.png", "chart_002_prediction.json"),
        ("chart_003.png", "chart_003_prediction.json"),
    ]

    for img_name, pred_name in test_cases:
        img_p = base_dir / "data" / "charts" / img_name
        pred_p = base_dir / "data" / "predictions" / pred_name

        if img_p.exists() and pred_p.exists():
            v_report = verify_chart_visually(img_p, pred_p)
            print(v_report.summary())
            print()
