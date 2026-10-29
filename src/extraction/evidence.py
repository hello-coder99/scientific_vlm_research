"""
Evidence Extraction Module.
Consolidates OCR, geometric line detections, and coordinate calibration
into an Evidence Package JSON for deterministic verification.
"""

import json
from pathlib import Path
from typing import Optional, Union

import cv2

try:
    from .calibration import calibrate_chart, ChartCalibration
    from .ocr import extract_text
    from .geometry import detect_lines
except ImportError:
    # Support direct execution
    from calibration import calibrate_chart, ChartCalibration
    from ocr import extract_text
    from geometry import detect_lines


def build_evidence_package(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Builds a complete evidence package for a chart image:
    1. Coordinate Calibration (Plot box, X/Y axis linear maps, R^2, tick marks)
    2. OCR extracted tokens and confidences
    3. Geometric line segments (axes, gridlines, data lines)

    Returns the evidence dictionary, and optionally writes it to output_path.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 1. Coordinate Calibration
    calibration: ChartCalibration = calibrate_chart(image_path)

    # 2. Raw OCR data
    ocr_data = extract_text(image_path)

    # 3. Geometric line segments
    lines = detect_lines(str(image_path))

    evidence = {
        "chart_id": image_path.stem,
        "image_path": str(image_path),
        "image_width": calibration.image_width,
        "image_height": calibration.image_height,
        "calibration": calibration.to_dict(),
        "ocr": ocr_data,
        "geometry": {
            "num_lines": len(lines),
            "lines": lines,
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=4)
        print(f"Evidence package successfully saved to: {output_path}")

    return evidence


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    sample_img = base_dir / "data" / "charts" / "chart_001.png"
    out_json = base_dir / "data" / "evidence" / "chart_001_evidence.json"

    if sample_img.exists():
        pkg = build_evidence_package(sample_img, out_json)
        print(f"Calibration status: Valid={pkg['calibration']['is_valid']}")
        print(f"Gridlines / Lines detected: {pkg['geometry']['num_lines']}")
        print(f"Words extracted: {len(pkg['ocr']['words'])}")
    else:
        print(f"Sample image not found: {sample_img}")