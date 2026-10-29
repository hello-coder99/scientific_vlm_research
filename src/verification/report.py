import json


def generate_report(result):

    numerical = result["numerical_verification"]
    counts = numerical["counts"]
    metrics = numerical["metrics"]

    print("\n")
    print("=" * 50)
    print("SCIENTIFIC CHART VERIFICATION REPORT")
    print("=" * 50)

    print(f"\nChart ID: {result['chart_id']}")

    print("\n--- Text Verification ---")

    text = result["text_verification"]

    print(
        f"Title       : "
        f"{'PASS' if text['title']['match'] else 'FAIL'}"
    )

    print(
        f"X-axis      : "
        f"{'PASS' if text['x_axis']['match'] else 'FAIL'}"
    )

    print(
        f"Y-axis      : "
        f"{'PASS' if text['y_axis']['match'] else 'FAIL'}"
    )

    print("\n--- Numerical Verification ---")

    print(f"Supported    : {counts['supported']}")
    print(f"Contradicted : {counts['contradicted']}")
    print(f"Missing      : {counts['missing']}")
    print(f"Extra        : {counts['extra']}")

    print("\n--- Metrics ---")

    print(
        f"Support Rate       : "
        f"{metrics['support_rate'] * 100:.2f}%"
    )

    print(
        f"Contradiction Rate : "
        f"{metrics['contradiction_rate'] * 100:.2f}%"
    )

    print(
        f"Missing Rate       : "
        f"{metrics['missing_rate'] * 100:.2f}%"
    )

    print(
        f"\nOverall Score      : "
        f"{result['overall']['score'] * 100:.2f}%"
    )

    print("\n--- Point-Level Results ---")

    for item in numerical["results"]:

        print(
            f"x={item['x']} | "
            f"expected={item['expected']} | "
            f"predicted={item['predicted']} | "
            f"{item['status']}"
        )

    print("=" * 50)