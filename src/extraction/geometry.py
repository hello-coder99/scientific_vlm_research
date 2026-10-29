import cv2
import numpy as np


def detect_lines(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Detect straight lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )

    detected = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            detected.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

    return detected


if __name__ == "__main__":

    lines = detect_lines(
        "../../data/charts/chart_001.png"
    )

    print(f"Detected {len(lines)} lines")

    for line in lines:
        print(line)