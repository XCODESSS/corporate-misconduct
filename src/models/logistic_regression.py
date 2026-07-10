"""
Logistic Regression baseline.

Responsibilities
----------------
- Load the development feature dataset.
- Prepare feature matrix, target and filing years.
- Train a Logistic Regression classifier.
- Evaluate using WalkForwardCV.
- Save cross-validation reports.

This module DOES NOT

- engineer features
- tune hyperparameters
- evaluate the held-out test set
- perform SHAP analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import optuna

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import configs.settings as settings

from src.evaluation.cross_validation import WalkForwardCV
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogisticRegressionBaseline:
    """
    Train and evaluate a Logistic Regression model
    using walk-forward cross-validation.
    """

    INPUT_FILE = (
        settings.FEATURES_DIR
        / "trainval_features.parquet"
    )
    MODEL_NAME = "logistic_regression"

    OPTUNA_TRIALS = 200

    OPTUNA_STORAGE = (
        settings.REPORTS_DIR
        / "logistic_regression_optuna.db"
    )

    BEST_PARAMS_FILE = (
        settings.REPORTS_DIR
        / "logistic_regression_best_params.json"
    )

    TRIALS_FILE = (
        settings.REPORTS_DIR
        / "logistic_regression_trials.csv"
    )

    FEATURE_COLUMNS = [
        "negative_density",
        "positive_density",
        "uncertainty_density",
        "litigious_density",
        "weak_modal_density",
        "strong_modal_density",
        "constraining_density",
    ]

    TARGET_COLUMN = "fraudulent"

    YEAR_COLUMN = "filing_year"

    RANDOM_STATE = getattr(
        settings,
        "RANDOM_STATE",
        42,
    )

    DECISION_THRESHOLD = 0.05

    def __init__(
        self,
        min_fraud_per_fold: int = 30,
        calibrate: bool = True,
        calibration_method: str = "sigmoid",
    ) -> None:

        self.min_fraud_per_fold = min_fraud_per_fold
        self.calibrate = calibrate
        self.calibration_method = calibration_method

        self.data: pd.DataFrame | None = None

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.years: np.ndarray | None = None

        self.cv_summary: dict[str, Any] = {}
        self.study: optuna.study.Study | None = None
        self.best_params: dict[str, Any] = {}

    # ============================================================
    # Dataset Loading
    # ============================================================

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the train/validation feature dataset.
        """

        logger.info("=" * 70)
        logger.info(
            "Loading development feature dataset..."
        )

        table = pq.read_table(
            self.INPUT_FILE
        )

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
        Ensure all required columns exist.
        """

        required_columns = (
            self.FEATURE_COLUMNS
            + [
                self.TARGET_COLUMN,
                self.YEAR_COLUMN,
            ]
        )

        if missing := [
            column
            for column in required_columns
            if column not in df.columns
        ]:

            raise ValueError(
                "Dataset missing required columns:\n"
                + "\n".join(missing)
            )

        logger.info(
            "Dataset validation successful."
        )

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

        logger.info(
            "Preparing feature matrix..."
        )

        feature_frame = df[
            self.FEATURE_COLUMNS
        ].copy()

        missing_before = (
            feature_frame
            .isna()
            .sum()
            .sum()
        )

        if missing_before > 0:

            logger.warning(
                "Replacing %d missing feature values with 0.0",
                missing_before,
            )

        feature_frame = (
            feature_frame
            .fillna(0.0)
        )

        self.X = (
            feature_frame
            .to_numpy(dtype=np.float64)
        )

        self.y = (
            df[self.TARGET_COLUMN]
            .astype(np.int64)
            .to_numpy()
        )

        self.years = (
            df[self.YEAR_COLUMN]
            .astype(np.int32)
            .to_numpy()
        )

        logger.info(
            "Prepared X=%s y=%s",
            self.X.shape,
            self.y.shape,
        )

        logger.info(
            "Fraud prevalence: %.2f%%",
            self.y.mean() * 100,
        )

        logger.info(
            "Evaluation years: %d-%d",
            self.years.min(),
            self.years.max(),
        )
    
    # ============================================================
    # Model
    # ============================================================

    def build_model(
        self,
        trial: optuna.Trial | None = None,
    ) -> Pipeline:
        """
        Build the Logistic Regression pipeline.

        Standardization is performed inside the pipeline
        to prevent data leakage during walk-forward
        cross-validation.
        """

        logger.info(
            "Building Logistic Regression model..."
        )

        model = Pipeline(

            steps=[

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    LogisticRegression(

                        C=(
                            trial.suggest_float(
                                "C",
                                1e-4,
                                100.0,
                                log=True,
                            )
                            if trial
                            else 1.0
                        ),

                        solver=(
                            trial.suggest_categorical(
                                "solver",
                                [
                                    "lbfgs",
                                    "liblinear",
                                    "saga",
                                ],
                            )
                            if trial
                            else "lbfgs"
                        ),

                        class_weight=(
                            trial.suggest_categorical(
                                "class_weight",
                                [
                                    None,
                                    "balanced",
                                ],
                            )
                            if trial
                            else None
                        ),

                        max_iter=(
                            trial.suggest_int(
                                "max_iter",
                                500,
                                3000,
                                step=250,
                            )
                            if trial
                            else 1000
                        ),

                        random_state=self.RANDOM_STATE,
                    ),
                ),
            ]
        )

        logger.info(
            "Pipeline:"
        )

        logger.info(
            "  StandardScaler"
        )

        classifier = model.named_steps["classifier"]

        logger.info(
            "  LogisticRegression(%s)",
            classifier.get_params(),
        )

        return model

    # ============================================================
    # Cross Validation
    # ============================================================

    def run_cross_validation(
        self,
        model: Pipeline,
        calibrate: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate the Logistic Regression model
        using WalkForwardCV.
        """

        logger.info("=" * 70)
        logger.info(
            "Starting walk-forward cross-validation..."
        )

        if (
            self.X is None
            or self.y is None
            or self.years is None
        ):
            raise RuntimeError(
                "Features have not been prepared."
            )

        cv = WalkForwardCV(
            min_fraud_per_fold=self.min_fraud_per_fold,
        )

        summary = cv.run(
            estimator=model,
            X=self.X,
            y=self.y,
            years=self.years,
            model_name=self.MODEL_NAME,
            decision_threshold=self.DECISION_THRESHOLD,
            calibrate=calibrate,
            calibration_method=self.calibration_method,
        )
        

        self.cv_summary = summary

        logger.info(
            "Cross-validation completed."
        )

        return summary
    # ============================================================
    # Optuna
    # ============================================================
    def objective(
        self,
        trial: optuna.Trial,
    ) -> float:
        """
        Objective function for Optuna.
        """

        model = self.build_model(
            trial=trial,
        )

        summary = self.run_cross_validation(
            model
        )

        score = summary["pr_auc"]["mean"]

        trial.set_user_attr(
            "roc_auc",
            summary["roc_auc"]["mean"],
        )

        trial.set_user_attr(
            "f1",
            summary["f1"]["mean"],
        )

        trial.set_user_attr(
            "mcc",
            summary["mcc"]["mean"],
        )

        trial.set_user_attr(
            "balanced_acc",
            summary["balanced_acc"]["mean"],
        )

        return score
    
    def optimize(
        self,
    ) -> None:
        """
        Optimize Logistic Regression hyperparameters
        using Optuna.
        """

        logger.info("=" * 70)
        logger.info(
            "Starting Optuna optimization..."
        )
        logger.info("=" * 70)

        self.study = optuna.create_study(
            study_name="logistic_regression",
            direction="maximize",
            storage=f"sqlite:///{self.OPTUNA_STORAGE}",
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.OPTUNA_TRIALS,
            show_progress_bar=True,
        )

        self.best_params = dict(
            self.study.best_trial.params
        )

        logger.info(
            "Best PR-AUC : %.6f",
            self.study.best_value,
        )

        logger.info(
            "Best Parameters:"
        )

        for key, value in self.best_params.items():

            logger.info(
                "%s = %s",
                key,
                value,
            )

        with open(
            self.BEST_PARAMS_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.best_params,
                f,
                indent=4,
            )

        trials = self.study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "params",
                "user_attrs",
                "state",
            )
        )

        trials = trials.sort_values(
            "value",
            ascending=False,
        )

        trials.to_csv(
            self.TRIALS_FILE,
            index=False,
        )

        logger.info(
            "Best parameters saved to %s",
            self.BEST_PARAMS_FILE,
        )

        logger.info(
            "Trials saved to %s",
            self.TRIALS_FILE,
        )

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

            logger.warning(
                "No CV summary available."
            )
            return

        self._extracted_from_run_15("Logistic Regression Summary")
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

        logger.info(
            "Total Fraud Cases : %d",
            self.cv_summary.get(
                "total_test_fraud",
                0,
            ),
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

    def run(
        self,
        optimize: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the complete Logistic Regression pipeline.

        Returns
        -------
        dict
            Cross-validation summary metrics.
        """

        self._extracted_from_run_15("Starting Logistic Regression...")
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
        # Hyperparameter Optimization
        # --------------------------------------------------------

        if optimize:

            self.optimize()

            model = self.build_model()

            model.named_steps[
                "classifier"
            ].set_params(
                **self.best_params,
            )

        else:

            model = self.build_model()
        # --------------------------------------------------------
        # Evaluate Best Model
        # --------------------------------------------------------

        summary = self.run_cross_validation(
            model,
            calibrate=self.calibrate,
        )

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        self.log_summary()

        logger.info(
            "Logistic Regression completed successfully."
        )

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


def run_logistic_regression(
    optimize: bool = False,
) -> dict[str, Any]:
        
    """
    Train and evaluate the Logistic Regression model.

    Returns
    -------
    dict
        Cross-validation summary metrics.
    """

    return LogisticRegressionBaseline().run(
        optimize=optimize,
    )


def main() -> None:
    """
    Script entry point.
    """

    run_logistic_regression(
        optimize=False,
    )


if __name__ == "__main__":

    main()