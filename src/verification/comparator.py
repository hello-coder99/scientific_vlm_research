import json



def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_text(expected, predicted):
    return {
        "expected": expected,
        "predicted": predicted,
        "match": expected == predicted
    }


def compare_points(expected_points, predicted_points, tolerance=0):
    expected = {
        point["x"]: point["y"]
        for point in expected_points
    }

    predicted = {
        point["x"]: point["y"]
        for point in predicted_points
    }

    results = []

    supported = 0
    contradicted = 0
    missing = 0
    extra = 0

    # --------------------------------
    # 1. Check expected points
    # --------------------------------

    for x, expected_y in expected.items():

        # AI did not report this point
        if x not in predicted:

            results.append({
                "x": x,
                "expected": expected_y,
                "predicted": None,
                "status": "MISSING"
            })

            missing += 1
            continue

        predicted_y = predicted[x]

        # Check numerical difference
        difference = abs(expected_y - predicted_y)

        if difference <= tolerance:

            results.append({
                "x": x,
                "expected": expected_y,
                "predicted": predicted_y,
                "difference": difference,
                "status": "SUPPORTED"
            })

            supported += 1

        else:

            results.append({
                "x": x,
                "expected": expected_y,
                "predicted": predicted_y,
                "difference": difference,
                "status": "CONTRADICTED"
            })

            contradicted += 1

    # --------------------------------
    # 2. Check for extra points
    # --------------------------------

    for x, predicted_y in predicted.items():

        if x not in expected:

            results.append({
                "x": x,
                "expected": None,
                "predicted": predicted_y,
                "status": "EXTRA"
            })

            extra += 1

    total_expected = len(expected)

    if total_expected > 0:

        support_rate = supported / total_expected
        contradiction_rate = contradicted / total_expected
        missing_rate = missing / total_expected

    else:

        support_rate = 0
        contradiction_rate = 0
        missing_rate = 0

    return {
        "results": results,

        "counts": {
            "supported": supported,
            "contradicted": contradicted,
            "missing": missing,
            "extra": extra
        },

        "metrics": {
            "support_rate": support_rate,
            "contradiction_rate": contradiction_rate,
            "missing_rate": missing_rate
        }
    }


def verify_chart(ground_truth_path, prediction_path):

    ground_truth = load_json(ground_truth_path)
    prediction = load_json(prediction_path)

    # --------------------------------
    # Text verification
    # --------------------------------

    title_result = compare_text(
        ground_truth["title"],
        prediction["title"]
    )

    x_axis_result = compare_text(
        ground_truth["x_axis"]["label"],
        prediction["x_axis"]
    )

    y_axis_result = compare_text(
        ground_truth["y_axis"]["label"],
        prediction["y_axis"]
    )

    # --------------------------------
    # Numerical verification
    # --------------------------------

    point_result = compare_points(
        ground_truth["series"][0]["points"],
        prediction["points"],
        tolerance=0
    )

    # --------------------------------
    # Overall score
    # --------------------------------

    text_checks = [
        title_result["match"],
        x_axis_result["match"],
        y_axis_result["match"]
    ]

    text_score = sum(text_checks) / len(text_checks)

    numerical_score = point_result["metrics"]["support_rate"]

    overall_score = (
        0.3 * text_score +
        0.7 * numerical_score
    )

    return {

        "chart_id": ground_truth["chart_id"],

        "text_verification": {
            "title": title_result,
            "x_axis": x_axis_result,
            "y_axis": y_axis_result,
            "score": text_score
        },

        "numerical_verification": point_result,

        "overall": {
            "score": overall_score
        }
    }


if __name__ == "__main__":
    from pathlib import Path
    try:
        from .report import generate_report
    except ImportError:
        from report import generate_report

    base_dir = Path(__file__).resolve().parent.parent.parent
    gt_file = base_dir / "data" / "ground_truth" / "chart_001.json"
    pred_file = base_dir / "data" / "predictions" / "chart_001_prediction.json"

    result = verify_chart(gt_file, pred_file)
    generate_report(result)
    