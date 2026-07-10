"""
Analyze predicted probabilities produced by WalkForwardCV.

This script summarizes the probability distribution and
helps determine a sensible threshold search range.
"""

from pathlib import Path

import pandas as pd

PREDICTIONS_FILE = Path("reports/evaluation/logistic_regression_predictions.csv")


def main() -> None:
    df = pd.read_csv(PREDICTIONS_FILE)

    print("=" * 80)
    print("Prediction Dataset")
    print("=" * 80)

    print(f"Rows : {len(df):,}")
    print()

    print("=" * 80)
    print("Probability Summary")
    print("=" * 80)

    print(df["predicted_probability"].describe())

    print()

    print("=" * 80)
    print("Probability Quantiles")
    print("=" * 80)

    print(
        df["predicted_probability"].quantile(
            [
                0.50,
                0.75,
                0.90,
                0.95,
                0.97,
                0.98,
                0.99,
                0.995,
                0.999,
            ]
        )
    )

    print()

    print("=" * 80)
    print("Largest Probabilities")
    print("=" * 80)

    print(
        df.nlargest(
            50,
            "predicted_probability",
        )[
            [
                "predicted_probability",
                "true_label",
                "predicted_label",
                "test_year",
            ]
        ]
    )

    print()

    print("=" * 80)
    print("Positive Predictions")
    print("=" * 80)

    thresholds = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
    ]

    for threshold in thresholds:
        positives = (df["predicted_probability"] >= threshold).sum()

        print(f"Threshold={threshold:>5.2f} Predicted Positives={positives}")

    print()

    print("=" * 80)
    print("Probability Distribution by True Label")
    print("=" * 80)

    print()

    print("Fraud cases")

    print(
        df.loc[
            df["true_label"] == 1,
            "predicted_probability",
        ].describe()
    )

    print()

    print("Non-fraud cases")

    print(
        df.loc[
            df["true_label"] == 0,
            "predicted_probability",
        ].describe()
    )


if __name__ == "__main__":
    main()
