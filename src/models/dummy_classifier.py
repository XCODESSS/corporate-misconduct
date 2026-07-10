"""
Dummy classifier baseline.

Responsibilities
----------------
- Load the development feature dataset.
- Prepare feature matrix, target and filing years.
- Train a DummyClassifier baseline.
- Evaluate using WalkForwardCV.
- Save cross-validation reports.

This module DOES NOT

- engineer features
- tune hyperparameters
- evaluate the held-out test set
- perform SHAP analysis
"""

from __future__ import annotations

from typing import Any

import configs.settings as settings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from src.evaluation.cross_validation import WalkForwardCV
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DummyBaseline:
    """
    Train and evaluate a DummyClassifier baseline
    using walk-forward cross-validation.
    """

    INPUT_FILE = settings.FEATURES_DIR / "trainval_features.parquet"

    MODEL_NAME = "dummy_classifier"

    FEATURE_COLUMNS = list(settings.MODEL_FEATURE_COLUMNS)

    TARGET_COLUMN = "fraudulent"

    YEAR_COLUMN = "filing_year"

    RANDOM_STATE = getattr(
        settings,
        "RANDOM_STATE",
        42,
    )

    def __init__(
        self,
        min_fraud_per_fold: int = 30,
    ) -> None:
        self.min_fraud_per_fold = min_fraud_per_fold

        self.data: pd.DataFrame | None = None

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.years: np.ndarray | None = None

        self.cv_summary: dict[str, Any] = {}

    # ============================================================
    # Dataset Loading
    # ============================================================

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the train/validation feature dataset.
        """

        logger.info("=" * 70)
        logger.info("Loading development feature dataset...")

        table = pq.read_table(self.INPUT_FILE)

        df = table.to_pandas()

        logger.info(
            "Loaded %d rows and %d columns.",
            len(df),
            len(df.columns),
        )

        self.data = df

        return df

    # ============================================================
    # Validation
    # ============================================================

    def validate_dataset(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Ensure all required columns are present.
        """

        required_columns = self.FEATURE_COLUMNS + [
            self.TARGET_COLUMN,
            self.YEAR_COLUMN,
        ]

        if missing := [col for col in required_columns if col not in df.columns]:
            raise ValueError("Dataset missing required columns:\n" + "\n".join(missing))

        logger.info("Dataset validation successful.")

    # ============================================================
    # Feature Preparation
    # ============================================================

    def prepare_features(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Prepare X, y and filing years.
        """

        logger.info("Preparing feature matrix...")

        feature_frame = df[self.FEATURE_COLUMNS].copy()

        missing_before = feature_frame.isna().sum().sum()

        if missing_before > 0:
            logger.warning(
                "Replacing %d missing feature values with 0.0",
                missing_before,
            )

        feature_frame = feature_frame.fillna(0.0)

        self.X = feature_frame.to_numpy(dtype=np.float64)

        self.y = df[self.TARGET_COLUMN].astype(int).to_numpy()

        self.years = df[self.YEAR_COLUMN].astype(np.int32).to_numpy()

        logger.info(
            "Fraud prevalence: %.2f%%",
            self.y.mean() * 100,
        )

        logger.info(
            "Evaluation years: %d-%d",
            self.years.min(),
            self.years.max(),
        )
        logger.info(
            "Prepared X=%s y=%s",
            self.X.shape,
            self.y.shape,
        )

    # ============================================================
    # Model
    # ============================================================

    def build_model(
        self,
    ) -> DummyClassifier:
        """
        Create the DummyClassifier baseline.

        Strategy
        --------
        Uses the empirical class prior so predicted
        probabilities reflect the observed fraud rate.
        """

        logger.info("Building DummyClassifier...")

        model = DummyClassifier(
            strategy="prior",
            random_state=self.RANDOM_STATE,
        )

        logger.info(
            "Strategy: prior | random_state=%d",
            self.RANDOM_STATE,
        )

        return model

    # ============================================================
    # Cross-Validation
    # ============================================================

    def run_cross_validation(
        self,
        model: DummyClassifier,
    ) -> dict[str, Any]:
        """
        Evaluate the baseline model using the
        WalkForwardCV engine.
        """

        logger.info("=" * 70)
        logger.info("Starting walk-forward cross-validation...")

        if self.X is None or self.y is None or self.years is None:
            raise RuntimeError("Features have not been prepared.")

        cv = WalkForwardCV(
            min_fraud_per_fold=self.min_fraud_per_fold,
        )

        summary = cv.run(
            estimator=model,
            X=self.X,
            y=self.y,
            years=self.years,
            model_name=self.MODEL_NAME,
            decision_threshold=0.5,
        )

        self.cv_summary = summary

        logger.info("Cross-validation completed.")

        return summary

    # ============================================================
    # Reporting
    # ============================================================

    def log_summary(
        self,
    ) -> None:
        """
        Log the aggregated cross-validation metrics.
        """

        if not self.cv_summary:
            logger.warning("No CV summary available.")
            return

        self._extracted_from_run_15("Dummy Classifier Summary")
        logger.info(
            "Folds Evaluated : %d",
            self.cv_summary.get(
                "n_folds",
                0,
            ),
        )

        logger.info(
            "Years Evaluated : %s",
            self.cv_summary.get(
                "years_evaluated",
                [],
            ),
        )

        logger.info(
            "Overall Fraud Rate : %.2f%%",
            self.y.mean() * 100,
        )

        metric_names = [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "mcc",
            "balanced_acc",
            "brier_score",
        ]

        for metric in metric_names:
            if values := self.cv_summary.get(
                metric,
                {},
            ):
                logger.info(
                    "%-15s mean=%8.4f   std=%8.4f",
                    metric,
                    values.get("mean", 0.0),
                    values.get("std", 0.0),
                )

        logger.info("=" * 70)
        # ============================================================

    # Pipeline
    # ============================================================

    def run(self) -> dict[str, Any]:
        """
        Execute the complete DummyClassifier pipeline.

        Returns
        -------
        dict
            Cross-validation summary metrics.
        """

        self._extracted_from_run_15("Starting DummyClassifier baseline...")
        # --------------------------------------------------------
        # Load Dataset
        # --------------------------------------------------------

        df = self.load_dataset()

        # --------------------------------------------------------
        # Validate Dataset
        # --------------------------------------------------------

        self.validate_dataset(df)

        # --------------------------------------------------------
        # Prepare Features
        # --------------------------------------------------------

        self.prepare_features(df)

        # --------------------------------------------------------
        # Build Model
        # --------------------------------------------------------

        model = self.build_model()

        # --------------------------------------------------------
        # Evaluate
        # --------------------------------------------------------

        summary = self.run_cross_validation(model)

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        self.log_summary()

        logger.info("DummyClassifier baseline completed successfully.")
        logger.info("=" * 70)

        return summary

    # TODO Rename this here and in `log_summary` and `run`
    def _extracted_from_run_15(self, arg0):
        logger.info("=" * 70)
        logger.info(arg0)
        logger.info("=" * 70)


# ============================================================
# Public API
# ============================================================


def run_dummy_classifier() -> dict[str, Any]:
    """
    Train and evaluate the DummyClassifier baseline.

    Returns
    -------
    dict
        Cross-validation summary.
    """

    return DummyBaseline().run()


def main() -> None:
    """
    Script entry point.
    """

    run_dummy_classifier()


if __name__ == "__main__":
    main()
