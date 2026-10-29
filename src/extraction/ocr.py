"""
OCR extraction module for scientific charts.
Extracts text tokens, confidence scores, and bounding boxes using PyTesseract.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pytesseract


def preprocess_image(image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess chart image for OCR.
    Avoids aggressive equalization to preserve thin fonts and decimal points.
    """
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Slight bilateral filter reduces compression artifacts while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 5, 20, 20)

    return image, denoised


def extract_text(image_input: Union[str, Path, np.ndarray]) -> Dict:
    """
    Extract words, bounding boxes, and confidences from a chart image.
    """
    original, processed = preprocess_image(image_input)

    text_data = pytesseract.image_to_data(
        processed,
        output_type=pytesseract.Output.DICT
    )

    results: List[Dict] = []
    num_entries = len(text_data["text"])

    for i in range(num_entries):
        word = text_data["text"][i].strip()
        if not word:
            continue

        try:
            confidence = float(text_data["conf"][i])
        except (ValueError, TypeError):
            confidence = 0.0

        results.append({
            "text": word,
            "confidence": confidence,
            "x": int(text_data["left"][i]),
            "y": int(text_data["top"][i]),
            "width": int(text_data["width"][i]),
            "height": int(text_data["height"][i]),
        })

    return {
        "image_width": int(original.shape[1]),
        "image_height": int(original.shape[0]),
        "words": results,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    sample_img = base_dir / "data" / "charts" / "chart_001.png"

    if sample_img.exists():
        result = extract_text(sample_img)
        print(f"Extracted {len(result['words'])} words from {sample_img.name}:")
        for word in result["words"][:10]:
            print(f"  {word['text']:<15} conf={word['confidence']:<5.1f} bbox=({word['x']}, {word['y']})")
    else:
        print(f"Sample image not found: {sample_img}")