"""
Module: xgboost_model.

Responsibilities
----------------
- Train and evaluate an XGBoost classifier for corporate misconduct prediction.
- Perform walk-forward temporal cross-validation.
- Support Optuna hyperparameter optimization.
- Support probability calibration.
- Persist experiment artifacts.
- Produce production-grade experiment logs.

This module DOES NOT

- perform feature engineering
- perform walk-forward splitting
- compute evaluation metrics
- compute calibration metrics
"""

from __future__ import annotations

import json
from typing import Any

import configs.settings as settings
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import shap
from sklearn.pipeline import Pipeline
from src.evaluation.cross_validation import WalkForwardCV
from src.utils.logger import get_logger
from xgboost import XGBClassifier

logger = get_logger(__name__)


class XGBoostBaseline:
    """
    Production implementation of XGBoost for
    corporate misconduct prediction.

    The class is responsible only for model
    construction, optimization and orchestration.

    Evaluation logic is delegated to WalkForwardCV.
    """

    MODEL_NAME = "xgboost"

    RANDOM_STATE = 52

    TARGET_COLUMN = "fraudulent"

    YEAR_COLUMN = "filing_year"

    DEFAULT_DECISION_THRESHOLD = 0.50

    DEFAULT_MIN_FRAUD_PER_FOLD = 30

    DEFAULT_OPTUNA_TRIALS = 200

    DEFAULT_CALIBRATION_METHOD = "sigmoid"

    DEFAULT_CALIBRATION_FOLDS = 5

    INPUT_FILE = settings.FEATURES_DIR / "trainval_features.parquet"

    OUTPUT_DIR = settings.REPORTS_DIR / "models" / MODEL_NAME

    OPTUNA_DIRECTORY = settings.REPORTS_DIR / "optuna"

    OPTUNA_STORAGE = OPTUNA_DIRECTORY / "xgboost.db"

    BEST_PARAMS_FILE = OPTUNA_DIRECTORY / "xgboost_best_params.json"

    TRIALS_FILE = OPTUNA_DIRECTORY / "xgboost_trials.csv"

    FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"

    MODEL_METADATA_FILE = OUTPUT_DIR / "model_metadata.json"

    FEATURE_COLUMNS = [
        "negative_density",
        "positive_density",
        "uncertainty_density",
        "litigious_density",
        "weak_modal_density",
        "strong_modal_density",
        "constraining_density",
    ]

    def __init__(
        self,
        decision_threshold: float | None = None,
        min_fraud_per_fold: int | None = None,
        optuna_trials: int | None = None,
    ) -> None:
        """
        Initialize the XGBoost pipeline.
        """

        self.X: np.ndarray | None = None

        self.y: np.ndarray | None = None

        self.years: np.ndarray | None = None

        self.cv_summary: dict[str, Any] | None = None

        self.study: optuna.Study | None = None

        self.best_params: dict[str, Any] = {}

        self.best_model: Pipeline | None = None

        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else self.DEFAULT_DECISION_THRESHOLD
        )

        self.min_fraud_per_fold = (
            min_fraud_per_fold
            if min_fraud_per_fold is not None
            else self.DEFAULT_MIN_FRAUD_PER_FOLD
        )

        self.optuna_trials = (
            optuna_trials if optuna_trials is not None else self.DEFAULT_OPTUNA_TRIALS
        )

        self._create_directories()

    def _create_directories(
        self,
    ) -> None:
        """
        Create all required output directories.
        """

        self.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.OPTUNA_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Load the processed training dataset.

        Returns
        -------
        pd.DataFrame
            Complete development dataset.
        """

        logger.info("=" * 70)
        logger.info("Loading processed training dataset...")

        if not self.INPUT_FILE.exists():
            raise FileNotFoundError(f"Dataset not found: {self.INPUT_FILE}")

        try:
            table = pq.read_table(
                self.INPUT_FILE,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Unable to read parquet dataset: {self.INPUT_FILE}"
            ) from exc

        dataset = table.to_pandas()

        if dataset.empty:
            raise ValueError("Dataset contains zero rows.")

        logger.info(
            "Rows Loaded    : %d",
            len(dataset),
        )

        logger.info(
            "Columns Loaded : %d",
            len(dataset.columns),
        )

        return dataset

    def validate_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """
        Validate dataset integrity before training.
        """

        logger.info("Validating dataset...")

        required_columns = self.FEATURE_COLUMNS + [
            self.TARGET_COLUMN,
            self.YEAR_COLUMN,
        ]

        if missing_columns := [
            column for column in required_columns if column not in dataset.columns
        ]:
            raise ValueError(f"Missing required columns:\n{missing_columns}")

        labels = dataset[self.TARGET_COLUMN].dropna().unique()

        labels = sorted(labels.tolist())

        if labels != [0, 1]:
            raise ValueError("Target column must contain only {0,1}.")

        duplicate_rows = dataset.duplicated().sum()

        if duplicate_rows > 0:
            logger.warning(
                "%d duplicate rows detected.",
                duplicate_rows,
            )

        feature_frame = dataset[self.FEATURE_COLUMNS]

        if empty_columns := feature_frame.columns[feature_frame.isna().all()].tolist():
            raise ValueError(f"Completely empty feature columns:\n{empty_columns}")

        logger.info("Dataset validation passed.")

    def prepare_features(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """
        Prepare feature matrix and target arrays.
        """

        logger.info("Preparing training arrays...")

        feature_frame = dataset[self.FEATURE_COLUMNS].copy()

        missing_values = int(feature_frame.isna().sum().sum())

        if missing_values > 0:
            logger.warning(
                "Replacing %d missing values with 0.0",
                missing_values,
            )

            feature_frame = feature_frame.fillna(
                0.0,
            )

        self.X = feature_frame.to_numpy(
            dtype=np.float32,
            copy=True,
        )

        self.y = dataset[self.TARGET_COLUMN].astype(np.int8).to_numpy()

        self.years = dataset[self.YEAR_COLUMN].astype(np.int32).to_numpy()

        if len(self.X) != len(self.y) or len(self.X) != len(self.years):
            raise RuntimeError(
                "Feature matrix, labels and years have different lengths."
            )

        fraud_cases = int(self.y.sum())

        fraud_rate = fraud_cases / len(self.y)

        logger.info(
            "Feature Matrix : %s",
            self.X.shape,
        )

        logger.info(
            "Target Shape   : %s",
            self.y.shape,
        )

        logger.info(
            "Years Shape    : %s",
            self.years.shape,
        )

        logger.info(
            "Fraud Cases    : %d",
            fraud_cases,
        )

        logger.info(
            "Fraud Rate     : %.2f%%",
            fraud_rate * 100,
        )

        logger.info(
            "Evaluation Period : %d - %d",
            int(self.years.min()),
            int(self.years.max()),
        )

    def _default_parameters(
        self,
    ) -> dict[str, Any]:
        """
        Return the default XGBoost configuration.

        These parameters are intended to provide a strong,
        reproducible baseline before Optuna optimization.
        """

        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": self.RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
        }

    def _validate_parameters(
        self,
        parameters: dict[str, Any],
    ) -> None:
        """
        Validate model hyperparameters before model construction.
        """

        if parameters["learning_rate"] <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if parameters["n_estimators"] <= 0:
            raise ValueError("n_estimators must be greater than zero.")

        if parameters["max_depth"] <= 0:
            raise ValueError("max_depth must be greater than zero.")

        if parameters["min_child_weight"] <= 0:
            raise ValueError("min_child_weight must be greater than zero.")

        if not 0 < parameters["subsample"] <= 1:
            raise ValueError("subsample must lie in (0,1].")

        if not 0 < parameters["colsample_bytree"] <= 1:
            raise ValueError("colsample_bytree must lie in (0,1].")

        if parameters["scale_pos_weight"] <= 0:
            raise ValueError("scale_pos_weight must be greater than zero.")

    def build_model(
        self,
    ) -> Pipeline:
        """
        Build the baseline XGBoost model.
        """

        parameters = self._default_parameters()

        self._validate_parameters(
            parameters,
        )

        classifier = XGBClassifier(
            **parameters,
        )

        model = Pipeline(
            steps=[
                (
                    "classifier",
                    classifier,
                ),
            ],
        )

        logger.info(
            "=" * 70,
        )

        logger.info(
            "Building baseline XGBoost model...",
        )

        logger.info(
            "=" * 70,
        )

        logger.info(
            "Model Parameters",
        )

        for key, value in parameters.items():
            logger.info(
                "%-20s : %s",
                key,
                value,
            )

        return model

    def build_tuned_model(
        self,
    ) -> Pipeline:
        """
        Build the best XGBoost model discovered
        during Optuna optimization.
        """

        if not self.best_params:
            raise RuntimeError("No tuned parameters available.")

        parameters = self._default_parameters()

        parameters.update(
            self.best_params,
        )

        self._validate_parameters(
            parameters,
        )

        classifier = XGBClassifier(
            **parameters,
        )

        model = Pipeline(
            steps=[
                (
                    "classifier",
                    classifier,
                ),
            ],
        )

        logger.info(
            "=" * 70,
        )

        logger.info(
            "Building tuned XGBoost model...",
        )

        logger.info(
            "=" * 70,
        )

        logger.info(
            "Model Parameters",
        )

        for key, value in parameters.items():
            logger.info(
                "%-20s : %s",
                key,
                value,
            )

        return model

    def _sample_parameters(
        self,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        """
        Sample XGBoost hyperparameters.
        """

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                1200,
                step=50,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.20,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                8,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1.0,
                20.0,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.60,
                1.00,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.60,
                1.00,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                10.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                5.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                20.0,
                log=True,
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                5.0,
                50.0,
            ),
        }

    def objective(
        self,
        trial: optuna.Trial,
    ) -> float:
        """
        Optuna objective.

        Optimize mean PR-AUC from WalkForwardCV.
        """

        self.best_params = self._sample_parameters(
            trial,
        )

        model = self.build_tuned_model()

        summary = self.run_cross_validation(
            model,
        )

        score = summary.get("pr_auc", {}).get("mean")

        if score is None:
            raise optuna.TrialPruned("PR-AUC not available.")

        if not np.isfinite(score):
            raise optuna.TrialPruned("Non-finite PR-AUC.")

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

        return float(score)

    def _save_best_parameters(
        self,
    ) -> None:
        """
        Persist the best Optuna parameters.
        """

        with open(
            self.BEST_PARAMS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.best_params,
                file,
                indent=4,
            )

    def _save_trials(
        self,
    ) -> None:
        """
        Save the Optuna trial history.
        """

        if self.study is None:
            return

        trials = self.study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "params",
                "user_attrs",
                "state",
            ),
        ).sort_values(
            "value",
            ascending=False,
        )

        trials.to_csv(
            self.TRIALS_FILE,
            index=False,
        )

    def optimize(
        self,
    ) -> None:
        """
        Run Optuna hyperparameter optimization.
        """

        logger.info("=" * 70)
        logger.info("Starting Optuna optimization...")
        logger.info("=" * 70)

        self.study = optuna.create_study(
            study_name=self.MODEL_NAME,
            direction="maximize",
            storage=f"sqlite:///{self.OPTUNA_STORAGE}",
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.optuna_trials,
            show_progress_bar=True,
        )

        self.best_params = dict(
            self.study.best_trial.params,
        )

        self._save_best_parameters()

        self._save_trials()

        logger.info(
            "Best PR-AUC : %.6f",
            self.study.best_value,
        )

        logger.info("Best Parameters")

        for parameter, value in self.best_params.items():
            logger.info(
                "%-20s : %s",
                parameter,
                value,
            )

    def run_cross_validation(
        self,
        model: Pipeline,
        calibrate: bool = False,
        calibration_method: str | None = None,
        calibration_cv: int | None = None,
        optimize_threshold: bool = True,
        fit_raw_reference: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate the XGBoost model using
        expanding-window walk-forward validation.

        Parameters
        ----------
        model
            XGBoost pipeline.

        calibrate
            Enable probability calibration.

        calibration_method
            "sigmoid" or "isotonic".

        calibration_cv
            Internal folds used by
            CalibratedClassifierCV.

        optimize_threshold
            Recompute the optimal
            decision threshold on the
            training fold.

        fit_raw_reference
            Fit an additional raw model
            for calibration comparison.
        """

        if self.X is None:
            raise RuntimeError("Feature matrix has not been prepared.")

        if self.y is None:
            raise RuntimeError("Target vector has not been prepared.")

        if self.years is None:
            raise RuntimeError("Year vector has not been prepared.")

        logger.info("=" * 70)
        logger.info("Starting WalkForwardCV...")
        logger.info("=" * 70)

        cv = WalkForwardCV(
            min_fraud_per_fold=self.min_fraud_per_fold,
        )

        summary = cv.run(
            estimator=model,
            X=self.X,
            y=self.y,
            years=self.years,
            model_name=self.MODEL_NAME,
            decision_threshold=self.decision_threshold,
            calibrate=calibrate,
            calibration_method=(calibration_method or self.DEFAULT_CALIBRATION_METHOD),
            calibration_cv=(calibration_cv or self.DEFAULT_CALIBRATION_FOLDS),
            optimize_threshold=optimize_threshold,
            fit_raw_reference=fit_raw_reference,
        )

        self.cv_summary = summary

        logger.info("WalkForwardCV completed.")

        return summary

    def evaluate_baseline(
        self,
    ) -> dict[str, Any]:
        """
        Evaluate the default
        XGBoost configuration.
        """

        logger.info("=" * 70)
        logger.info("Running baseline model...")
        logger.info("=" * 70)

        model = self.build_model()

        return self.run_cross_validation(
            model=model,
            calibrate=False,
            optimize_threshold=True,
        )

    def evaluate_best_model(
        self,
        calibrate: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate the Optuna tuned
        XGBoost model.
        """

        if not self.best_params:
            raise RuntimeError("No tuned parameters available.")

        logger.info("=" * 70)
        logger.info("Running tuned model...")
        logger.info("=" * 70)

        model = self.build_tuned_model()

        return self.run_cross_validation(
            model=model,
            calibrate=calibrate,
            optimize_threshold=True,
            fit_raw_reference=False,
        )

    def _extract_feature_importance(
        self,
        model: Pipeline,
        importance_type: str,
    ) -> pd.DataFrame:
        """
        Extract feature importance from the trained
        XGBoost model.

        Parameters
        ----------
        importance_type
            "gain", "weight", or "cover"
        """

        classifier: XGBClassifier = model.named_steps["classifier"]

        booster = classifier.get_booster()

        raw_scores = booster.get_score(
            importance_type=importance_type,
        )

        feature_scores: list[dict[str, Any]] = []

        for index, feature_name in enumerate(
            self.FEATURE_COLUMNS,
        ):
            score = raw_scores.get(
                f"f{index}",
                0.0,
            )

            feature_scores.append(
                {
                    "feature": feature_name,
                    "importance": float(score),
                }
            )

        return (
            pd.DataFrame(feature_scores)
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def save_feature_importance(
        self,
        model: Pipeline,
    ) -> None:
        """
        Save feature importance using multiple
        XGBoost importance definitions.
        """

        logger.info("Saving feature importance...")

        output_directory = self.OUTPUT_DIR / "feature_importance"

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for importance_type in (
            "gain",
            "weight",
            "cover",
        ):
            importance = self._extract_feature_importance(
                model=model,
                importance_type=importance_type,
            )

            output_file = output_directory / f"{importance_type}.csv"

            importance.to_csv(
                output_file,
                index=False,
            )

            logger.info(
                "%s importance saved to %s",
                importance_type,
                output_file,
            )

    def save_model_metadata(
        self,
    ) -> None:
        """
        Save experiment metadata.
        """

        metadata = {
            "model": self.MODEL_NAME,
            "random_state": self.RANDOM_STATE,
            "decision_threshold": self.decision_threshold,
            "min_fraud_per_fold": self.min_fraud_per_fold,
            "optuna_trials": self.optuna_trials,
            "best_parameters": self.best_params,
        }

        with open(
            self.MODEL_METADATA_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

        logger.info("Metadata saved.")

    def save_model(
        self,
        model: Pipeline,
    ) -> None:
        """
        Persist the trained model.
        """

        import joblib

        model_path = self.OUTPUT_DIR / "xgboost_model.joblib"

        joblib.dump(
            model,
            model_path,
        )

        logger.info(
            "Model saved to %s",
            model_path,
        )

    def save_artifacts(
        self,
        model: Pipeline,
    ) -> None:
        """
        Save all model artifacts.
        """

        self.save_feature_importance(
            model,
        )

        self.save_model_metadata()

        self.save_model(
            model,
        )

    def _get_trained_classifier(
        self,
        model: Pipeline,
    ) -> XGBClassifier:
        """
        Return the trained XGBoost classifier.
        """

        classifier = model.named_steps.get(
            "classifier",
        )

        if classifier is None:
            raise RuntimeError("Pipeline does not contain an XGBClassifier.")

        return classifier

    def _sample_shap_dataset(
        self,
        sample_size: int = 1000,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Sample observations for SHAP analysis.

        Large datasets are subsampled to keep
        explanation time reasonable.
        """

        if self.X is None:
            raise RuntimeError("Feature matrix has not been prepared.")

        total_rows = len(self.X)

        if total_rows <= sample_size:
            return self.X, self.FEATURE_COLUMNS

        rng = np.random.default_rng(
            self.RANDOM_STATE,
        )

        sample_indices = rng.choice(
            total_rows,
            size=sample_size,
            replace=False,
        )

        return (
            self.X[sample_indices],
            self.FEATURE_COLUMNS,
        )

    def compute_shap_values(
        self,
        model: Pipeline,
        sample_size: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values.
        """

        classifier = self._get_trained_classifier(
            model,
        )

        shap_features, _ = self._sample_shap_dataset(
            sample_size,
        )

        explainer = shap.TreeExplainer(
            classifier,
        )

        shap_values = explainer.shap_values(
            shap_features,
        )

        return (
            shap_features,
            np.asarray(shap_values),
        )

    def save_shap_importance(
        self,
        model: Pipeline,
        sample_size: int = 1000,
    ) -> None:
        """
        Save global SHAP feature importance.
        """

        logger.info("Computing SHAP importance...")

        features, shap_values = self.compute_shap_values(
            model=model,
            sample_size=sample_size,
        )

        importance = np.abs(
            shap_values,
        ).mean(axis=0)

        shap_importance = (
            pd.DataFrame(
                {
                    "feature": self.FEATURE_COLUMNS,
                    "mean_abs_shap": importance,
                }
            )
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True,
            )
        )

        output_directory = self.OUTPUT_DIR / "shap"

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        shap_importance.to_csv(
            output_directory / "shap_importance.csv",
            index=False,
        )

        logger.info("SHAP importance saved.")

    def save_shap_summary_plot(
        self,
        model: Pipeline,
        sample_size: int = 1000,
    ) -> None:
        """
        Save SHAP summary plot.
        """

        features, shap_values = self.compute_shap_values(
            model=model,
            sample_size=sample_size,
        )

        output_directory = self.OUTPUT_DIR / "shap"

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        shap.summary_plot(
            shap_values,
            features,
            feature_names=self.FEATURE_COLUMNS,
            show=False,
        )

        import matplotlib.pyplot as plt

        plt.tight_layout()

        plt.savefig(
            output_directory / "summary_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info("SHAP summary plot saved.")

    def run_shap_analysis(
        self,
        model: Pipeline,
    ) -> None:
        """
        Execute complete SHAP analysis.
        """

        logger.info("=" * 70)
        logger.info("Running SHAP analysis...")
        logger.info("=" * 70)

        self.save_shap_importance(
            model,
        )

        self.save_shap_summary_plot(
            model,
        )

        logger.info("SHAP analysis completed.")

    def log_summary(
        self,
    ) -> None:
        """
        Log the aggregate cross-validation summary.
        """

        if self.cv_summary is None:
            raise RuntimeError("Cross-validation summary is unavailable.")

        logger.info("=" * 70)
        logger.info("XGBoost Summary")
        logger.info("=" * 70)

        logger.info(
            "Folds Evaluated : %d",
            self.cv_summary["n_folds"],
        )

        logger.info(
            "Years Evaluated : %s",
            self.cv_summary["years_evaluated"],
        )

        logger.info(
            "Overall Fraud Rate : %.2f%%",
            self.cv_summary["overall_fraud_rate"] * 100,
        )

        logger.info(
            "Total Fraud Cases : %d",
            self.cv_summary["total_test_fraud"],
        )

        metric_names = (
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "mcc",
            "balanced_acc",
            "brier_score",
        )

        for metric_name in metric_names:
            statistics = self.cv_summary[metric_name]

            logger.info(
                "%-15s mean=%8.4f std=%8.4f",
                metric_name,
                statistics["mean"],
                statistics["std"],
            )

        logger.info("=" * 70)

    def train_model(
        self,
        optimize: bool,
    ) -> Pipeline:
        """
        Train either the baseline or
        Optuna-tuned model.
        """

        if not optimize:
            return self.build_model()

        self.optimize()

        return self.build_tuned_model()

    def evaluate_model(
        self,
        model: Pipeline,
        calibrate: bool,
    ) -> dict[str, Any]:
        """
        Evaluate a trained model.
        """

        return self.run_cross_validation(
            model=model,
            calibrate=calibrate,
            optimize_threshold=True,
            fit_raw_reference=False,
        )

    def run(
        self,
        optimize: bool = False,
        calibrate: bool = False,
        run_shap: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the complete XGBoost pipeline.
        """

        logger.info("=" * 70)
        logger.info("Starting XGBoost pipeline...")
        logger.info("=" * 70)

        dataset = self.load_dataset()

        self.validate_dataset(
            dataset,
        )

        self.prepare_features(
            dataset,
        )

        model = self.train_model(
            optimize=optimize,
        )

        summary = self.evaluate_model(
            model=model,
            calibrate=calibrate,
        )

        self.best_model = model

        self.save_artifacts(
            model,
        )

        if run_shap:
            self.run_shap_analysis(
                model,
            )

        self.log_summary()

        logger.info("XGBoost completed successfully.")

        logger.info("=" * 70)

        return summary


def run_xgboost(
    optimize: bool = False,
    calibrate: bool = False,
    run_shap: bool = True,
) -> dict[str, Any]:
    """
    Public API for running XGBoost.
    """

    pipeline = XGBoostBaseline()

    return pipeline.run(
        optimize=optimize,
        calibrate=calibrate,
        run_shap=run_shap,
    )


def main() -> None:
    """
    Script entry point.
    """

    run_xgboost(
        optimize=False,
        calibrate=False,
        run_shap=True,
    )


if __name__ == "__main__":
    main()
