"""
Statistical validation of linguistic fraud hypotheses.

Responsibilities
----------------
- Load the feature-engineered development dataset.
- Produce descriptive statistics.
- Compute correlation matrix.
- Compute variance inflation factors (VIF).
- Perform hypothesis testing.
- Apply Holm-Bonferroni correction.
- Export statistical reports.

This module DOES NOT

- train models
- tune hyperparameters
- evaluate predictive performance
- perform cross-validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import configs.settings as settings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalTester:
    """
    Statistical validation of the preregistered
    linguistic hypotheses.
    """

    INPUT_FILE = (
        settings.FEATURES_DIR
        / "trainval_features.parquet"
    )

    OUTPUT_DIR = (
        settings.REPORTS_DIR
        / "statistical_tests"
    )

    SUMMARY_FILE = (
        OUTPUT_DIR
        / "hypothesis_test_results.csv"
    )

    CORRELATION_FILE = (
        OUTPUT_DIR
        / "correlation_matrix.csv"
    )

    VIF_FILE = (
        OUTPUT_DIR
        / "vif.csv"
    )

    REPORT_FILE = (
        OUTPUT_DIR
        / "statistical_report.json"
    )

    FEATURES = [

        "negative_density",

        "positive_density",

        "uncertainty_density",

        "litigious_density",

        "weak_modal_density",

        "strong_modal_density",

        "constraining_density",

    ]

    def __init__(self) -> None:

        self.df: pd.DataFrame | None = None

        self.fraud: pd.DataFrame | None = None

        self.clean: pd.DataFrame | None = None

        self.results: pd.DataFrame | None = None

        self.correlation: pd.DataFrame | None = None

        self.vif: pd.DataFrame | None = None

    # ============================================================
    # Loading
    # ============================================================

    def load_dataset(self) -> None:

        logger.info("=" * 70)
        logger.info(
            "Loading feature-engineered dataset..."
        )

        self.df = pd.read_parquet(
            self.INPUT_FILE
        )

        self.fraud = self.df[
            self.df["fraudulent"] == 1
        ]

        self.clean = self.df[
            self.df["fraudulent"] == 0
        ]

        self._extracted_from_log_feature_summary_20()

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def cohens_d(
        x: pd.Series,
        y: pd.Series,
    ) -> float:
        """
        Compute Cohen's d using pooled variance.
        """

        nx = len(x)
        ny = len(y)

        if nx < 2 or ny < 2:
            return np.nan

        pooled_sd = np.sqrt(

            (

                (nx - 1)
                * x.var(ddof=1)

                +

                (ny - 1)
                * y.var(ddof=1)

            )

            /

            (nx + ny - 2)

        )

        return np.nan if pooled_sd == 0 else (x.mean() - y.mean()) / pooled_sd

    @staticmethod
    def percent_difference(
        fraud_mean: float,
        clean_mean: float,
    ) -> float:

        if clean_mean == 0:
            return np.nan

        return (
            (fraud_mean - clean_mean)
            / clean_mean
            * 100
        )

    @staticmethod
    def round_dataframe(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        return df.round(
            {
                "mean_fraud": 6,
                "mean_clean": 6,
                "diff_pct": 1,
                "cohens_d": 3,
                "t_stat": 4,
                "p_value": 6,
                "p_holm": 6,
                "vif": 3,
            }
        )
        # ============================================================
    # Descriptive Statistics
    # ============================================================

    def descriptive_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for each linguistic
        feature by fraud class.
        """

        logger.info(
            "Computing descriptive statistics..."
        )

        rows = []

        for feature in self.FEATURES:

            fraud_values = self.fraud[feature].dropna()
            clean_values = self.clean[feature].dropna()

            rows.append(
                {
                    "feature": feature,

                    "fraud_n": len(fraud_values),
                    "clean_n": len(clean_values),

                    "fraud_mean": fraud_values.mean(),
                    "clean_mean": clean_values.mean(),

                    "fraud_std": fraud_values.std(),
                    "clean_std": clean_values.std(),

                    "fraud_median": fraud_values.median(),
                    "clean_median": clean_values.median(),

                    "fraud_min": fraud_values.min(),
                    "clean_min": clean_values.min(),

                    "fraud_max": fraud_values.max(),
                    "clean_max": clean_values.max(),
                }
            )

        descriptive = pd.DataFrame(rows)

        descriptive = descriptive.round(6)

        descriptive.to_csv(
            self.OUTPUT_DIR / "descriptive_statistics.csv",
            index=False,
        )

        logger.info(
            "Descriptive statistics written."
        )

        return descriptive

    # ============================================================
    # Correlation Matrix
    # ============================================================

    def compute_correlation_matrix(self) -> None:
        """
        Pearson correlation matrix for all linguistic
        density features.
        """

        logger.info(
            "Computing correlation matrix..."
        )

        self.correlation = (

            self.df[
                self.FEATURES
            ]

            .corr(
                method="pearson"
            )

            .round(6)

        )

        self.correlation.to_csv(
            self.CORRELATION_FILE
        )

        logger.info(
            "Correlation matrix saved."
        )

    # ============================================================
    # Variance Inflation Factors
    # ============================================================

    def compute_vif(self) -> None:
        """
        Compute Variance Inflation Factors on standardized features.

        Standardization doesn't change the theoretical VIF but
        improves numerical stability of the underlying regression.
        """



        logger.info("Computing variance inflation factors...")

        X = self.df[self.FEATURES].dropna()

        X = pd.DataFrame(
            StandardScaler().fit_transform(X),
            columns=self.FEATURES,
        )

        vif_rows = []

        vif_rows.extend(
            {
                "feature": column,
                "vif": variance_inflation_factor(X.values, i),
            }
            for i, column in enumerate(X.columns)
        )
        self.vif = (
            pd.DataFrame(vif_rows)
            .sort_values("vif", ascending=False)
            .reset_index(drop=True)
        )

        self.vif = self.round_dataframe(self.vif)
        self.vif.to_csv(self.VIF_FILE, index=False)

        logger.info("VIF analysis saved.")
    # ============================================================
    # Summary Logging
    # ============================================================

    def log_feature_summary(self) -> None:

        logger.info("=" * 70)
        logger.info(
            "Feature Summary"
        )
        logger.info("=" * 70)

        self._extracted_from_log_feature_summary_20()
        logger.info(
            "Features analysed : %d",
            len(self.FEATURES),
        )

        logger.info("=" * 70)

    # TODO Rename this here and in `load_dataset` and `log_feature_summary`
    def _extracted_from_log_feature_summary_20(self):
        logger.info("Total observations : %d", len(self.df))
        logger.info("Fraud observations : %d", len(self.fraud))
        logger.info("Clean observations : %d", len(self.clean))
        # ============================================================
    # Hypothesis Testing
    # ============================================================

    def perform_hypothesis_tests(self) -> None:
        """
        Perform Welch's t-tests for all preregistered
        linguistic hypotheses.

        Multiple comparisons are controlled using the
        Holm-Bonferroni correction.
        """

        logger.info(
            "Performing hypothesis tests..."
        )

        rows = []

        for feature in self.FEATURES:

            fraud_values = (
                self.fraud[feature]
                .dropna()
            )

            clean_values = (
                self.clean[feature]
                .dropna()
            )

            fraud_mean = fraud_values.mean()
            clean_mean = clean_values.mean()

            t_stat, p_value = stats.ttest_ind(
                fraud_values,
                clean_values,
                equal_var=False,
            )

            rows.append(
                {
                    "feature": feature,

                    "fraud_n": len(
                        fraud_values
                    ),

                    "clean_n": len(
                        clean_values
                    ),

                    "mean_fraud": fraud_mean,

                    "mean_clean": clean_mean,

                    "diff_pct": self.percent_difference(
                        fraud_mean,
                        clean_mean,
                    ),

                    "cohens_d": self.cohens_d(
                        fraud_values,
                        clean_values,
                    ),

                    "t_stat": t_stat,

                    "p_value": p_value,
                }
            )

        self.results = pd.DataFrame(
            rows
        )

        self.results["p_holm"] = (
            multipletests(
                self.results["p_value"],
                method="holm",
            )[1]
        )

        self.results[
            "significant_holm"
        ] = (
            self.results["p_holm"] < 0.05
        )

        self.results = self.round_dataframe(
            self.results
        )

        self.results.to_csv(
            self.SUMMARY_FILE,
            index=False,
        )

        logger.info(
            "Hypothesis testing complete."
        )

    # ============================================================
    # Report Generation
    # ============================================================

    def report(self) -> dict[str, Any]:

        significant = int(
            self.results[
                "significant_holm"
            ].sum()
        )

        return {

            "total_observations":
                len(self.df),

            "fraud_observations":
                len(self.fraud),

            "clean_observations":
                len(self.clean),

            "features_tested":
                len(self.FEATURES),

            "significant_features":
                significant,

            "holm_correction":
                True,

            "correlation_file":
                str(self.CORRELATION_FILE),

            "vif_file":
                str(self.VIF_FILE),

            "hypothesis_results":
                str(self.SUMMARY_FILE),
        }

    def write_report(self) -> None:

        logger.info(
            "Writing statistical report..."
        )

        with open(
            self.REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.report(),
                file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "Statistical report written."
        )

    # ============================================================
    # Console Output
    # ============================================================

    def print_results(self) -> None:

        self._extracted_from_print_results_3("Correlation Matrix")
        print(
            self.correlation.to_string()
        )

        self._extracted_from_print_results_3("Variance Inflation Factors")
        print(
            self.vif.to_string(
                index=False
            )
        )

        self._extracted_from_print_results_3("Hypothesis Test Results")
        print(
            self.results.to_string(
                index=False
            )
        )

        print("=" * 110)

    # TODO Rename this here and in `print_results`
    def _extracted_from_print_results_3(self, arg0):
        print()

        print("=" * 110)
        print(arg0)
        print("=" * 110)
    @staticmethod
    def interpret_cohens_d(d: float) -> str:

        d = abs(d)

        if d < 0.20:
            return "Negligible"

        if d < 0.50:
            return "Small"

        return "Medium" if d < 0.80 else "Large"
    # ============================================================
    # Pipeline
    # ============================================================

    def run(self) -> None:

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.load_dataset()
        self.log_feature_summary()
        self.descriptive_statistics()
        self.compute_correlation_matrix()
        self.compute_vif()
        self.perform_hypothesis_tests()
        self.write_report()
        self.print_results()


def run_statistical_tests() -> None:
    StatisticalTester().run()


def main() -> None:
    run_statistical_tests()


if __name__ == "__main__":
    main()