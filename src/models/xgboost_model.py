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
- Run the one-time sealed held-out test evaluation for the selected winner.

This module DOES NOT

- perform feature engineering
- perform walk-forward splitting
- compute evaluation metrics
- compute calibration metrics
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import configs.settings as settings
import joblib
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import shap
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from src.evaluation.calibration import ProbabilityCalibrator, evaluate_calibration
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

    EXPERIMENT_NAME = "xgboost_lm_text_surface_probability_v2"

    MODEL_NAME = EXPERIMENT_NAME

    RANDOM_STATE = 52

    TARGET_COLUMN = "fraudulent"

    YEAR_COLUMN = "filing_year"

    FILING_DATE_COLUMN = "filing_date"

    FINAL_TEST_START_YEAR = 2019

    FINAL_TEST_END_YEAR = 2022

    DEFAULT_DECISION_THRESHOLD = 0.50

    FINAL_TEST_DECISION_THRESHOLD = 0.50

    DEFAULT_MIN_FRAUD_PER_FOLD = 30

    DEFAULT_OPTUNA_TRIALS = 200

    DEFAULT_CALIBRATION_METHOD = "sigmoid"

    DEFAULT_CALIBRATION_FOLDS = 5

    DEFAULT_CALIBRATION_STRATEGY = "chronological_holdout"

    DEFAULT_CALIBRATION_HOLDOUT_FRACTION = 0.20

    INPUT_FILE = settings.FEATURES_DIR / "trainval_features.parquet"

    TEST_INPUT_FILE = settings.FEATURES_DIR / "test_features.parquet"

    OUTPUT_DIR = settings.REPORTS_DIR / "models" / EXPERIMENT_NAME

    CV_OUTPUT_DIR = OUTPUT_DIR / "cross_validation"

    FINAL_TEST_OUTPUT_DIR = OUTPUT_DIR / "final_test"

    OPTUNA_DIRECTORY = settings.REPORTS_DIR / "optuna"

    OPTUNA_STORAGE = OPTUNA_DIRECTORY / f"{EXPERIMENT_NAME}.db"

    BEST_PARAMS_FILE = OPTUNA_DIRECTORY / f"{EXPERIMENT_NAME}_best_params.json"

    TRIALS_FILE = OPTUNA_DIRECTORY / f"{EXPERIMENT_NAME}_trials.csv"

    FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"

    MODEL_METADATA_FILE = OUTPUT_DIR / "model_metadata.json"

    EXPERIMENT_MANIFEST_FILE = OUTPUT_DIR / "experiment_manifest.json"

    FEATURE_COLUMNS = list(settings.MODEL_FEATURE_COLUMNS)

    OPTIMIZATION_METRIC = "pr_auc"

    CANDIDATE_REVIEW_SIZE = 10

    CANDIDATE_REVIEW_FILE = OUTPUT_DIR / "candidate_review_full_refit.json"

    XGBOOST_DEVICE = "cuda"

    PROBABILITY_SCALE_POS_WEIGHT = 1.0

    SUMMARY_METRICS = (
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "mcc",
        "balanced_acc",
        "brier_score",
        "naive_brier_score",
        "brier_skill_score",
        "recall_at_5_percent",
        "precision_at_5_percent",
    )

    TRIAL_LOGGER_NAMES = (
        "src.models.xgboost_model",
        "src.evaluation.cross_validation",
        "src.evaluation.calibration",
    )

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

        self.best_trial_number: int | None = None

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

    def _create_directories(
        self,
    ) -> None:
        """
        Create all required output directories.
        """

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OPTUNA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    def _feature_schema_hash(self) -> str:
        feature_schema = json.dumps(self.FEATURE_COLUMNS, separators=(",", ":"))
        return hashlib.sha256(feature_schema.encode("utf-8")).hexdigest()

    def save_experiment_manifest(self) -> None:
        """Persist the fixed inputs and settings for this experiment."""
        manifest = {
            "experiment_name": self.EXPERIMENT_NAME,
            "model_name": self.MODEL_NAME,
            "input_file": str(self.INPUT_FILE),
            "target_column": self.TARGET_COLUMN,
            "year_column": self.YEAR_COLUMN,
            "feature_columns": self.FEATURE_COLUMNS,
            "feature_schema_sha256": self._feature_schema_hash(),
            "optimization_metric": self.OPTIMIZATION_METRIC,
            "min_fraud_per_fold": self.min_fraud_per_fold,
            "calibration_method": self.DEFAULT_CALIBRATION_METHOD,
            "calibration_folds": self.DEFAULT_CALIBRATION_FOLDS,
            "calibration_strategy": self.DEFAULT_CALIBRATION_STRATEGY,
            "calibration_holdout_fraction": self.DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
            "chronological_calibration_refits_full_training_fold": True,
            "scale_pos_weight": self.PROBABILITY_SCALE_POS_WEIGHT,
            "xgboost_device": self.XGBOOST_DEVICE,
        }

        with open(self.EXPERIMENT_MANIFEST_FILE, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4)

        logger.info("Experiment manifest saved to %s", self.EXPERIMENT_MANIFEST_FILE)

    def _load_parquet(self, path) -> pd.DataFrame:
        """Shared parquet loader used for both dev and sealed test data."""

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise RuntimeError(f"Unable to read parquet dataset: {path}") from exc

        dataset = table.to_pandas()

        if dataset.empty:
            raise ValueError(f"Dataset contains zero rows: {path}")

        return dataset

    def load_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Load the processed development (train/validation) dataset.

        Returns
        -------
        pd.DataFrame
            Complete development dataset.
        """

        logger.info("=" * 70)
        logger.info("Loading processed training dataset...")

        dataset = self._load_parquet(self.INPUT_FILE)

        logger.info("Rows Loaded    : %d", len(dataset))
        logger.info("Columns Loaded : %d", len(dataset.columns))

        return dataset

    def load_test_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Load the sealed 2019-2022 held-out test dataset.

        This dataset must only be touched by run_final_test_evaluation().
        """

        logger.info("=" * 70)
        logger.info("Loading sealed test dataset...")

        dataset = self._load_parquet(self.TEST_INPUT_FILE)

        logger.info("Rows Loaded    : %d", len(dataset))
        logger.info("Columns Loaded : %d", len(dataset.columns))

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

        target_values = dataset[self.TARGET_COLUMN]
        if target_values.isna().any():
            raise ValueError("Target column cannot contain missing values.")

        labels = sorted(target_values.unique().tolist())

        if labels != [0, 1]:
            raise ValueError("Target column must contain only {0,1}.")

        duplicate_rows = dataset.duplicated().sum()

        if duplicate_rows > 0:
            logger.warning("%d duplicate rows detected.", duplicate_rows)

        feature_frame = dataset[self.FEATURE_COLUMNS]

        if empty_columns := feature_frame.columns[feature_frame.isna().all()].tolist():
            raise ValueError(f"Completely empty feature columns:\n{empty_columns}")

        logger.info("Dataset validation passed.")

    def _extract_arrays(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared feature/target/year extraction used for dev and test data."""

        feature_frame = dataset[self.FEATURE_COLUMNS].copy()

        missing_values = int(feature_frame.isna().sum().sum())

        if missing_values > 0:
            logger.warning("Replacing %d missing values with 0.0", missing_values)
            feature_frame = feature_frame.fillna(0.0)

        X = feature_frame.to_numpy(dtype=np.float32, copy=True)
        y = dataset[self.TARGET_COLUMN].astype(np.int8).to_numpy()
        years = dataset[self.YEAR_COLUMN].astype(np.int32).to_numpy()

        if len(X) != len(y) or len(X) != len(years):
            raise RuntimeError(
                "Feature matrix, labels and years have different lengths."
            )

        return X, y, years

    def prepare_features(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """
        Prepare feature matrix and target arrays for the development set.
        """

        logger.info("Preparing training arrays...")

        self.X, self.y, self.years = self._extract_arrays(dataset)

        fraud_cases = int(self.y.sum())
        fraud_rate = fraud_cases / len(self.y)

        logger.info("Feature Matrix : %s", self.X.shape)
        logger.info("Target Shape   : %s", self.y.shape)
        logger.info("Years Shape    : %s", self.years.shape)
        logger.info("Fraud Cases    : %d", fraud_cases)
        logger.info("Fraud Rate     : %.2f%%", fraud_rate * 100)
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

        The probability-calibrated v2 study deliberately uses the fixed
        PROBABILITY_SCALE_POS_WEIGHT value. Candidate selection and the
        sealed test must therefore use the same value.
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
            "scale_pos_weight": self.PROBABILITY_SCALE_POS_WEIGHT,
            "device": self.XGBOOST_DEVICE,
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

        return self._build_pipeline(parameters, "baseline")

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
        parameters.update(self.best_params)

        return self._build_pipeline(parameters, "tuned")

    def _load_study(self) -> optuna.Study:
        """Open the completed PR-AUC study without adding trials."""
        self.study = optuna.load_study(
            study_name=self.MODEL_NAME,
            storage=f"sqlite:///{self.OPTUNA_STORAGE}",
        )
        return self.study

    def _top_pr_auc_trials(self) -> list[optuna.trial.FrozenTrial]:
        """Return completed, finite PR-AUC trials in preliminary rank order."""
        study = self.study or self._load_study()
        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        ]
        return sorted(
            completed_trials,
            key=lambda trial: (-float(trial.value), trial.number),
        )[: self.CANDIDATE_REVIEW_SIZE]

    def review_top_candidates(self) -> dict[str, Any]:
        """Select a production candidate without evaluating the held-out test set."""
        if self.X is None or self.y is None or self.years is None:
            raise RuntimeError("Features and target must be prepared before review.")

        self._create_directories()
        candidates: list[dict[str, Any]] = []
        for trial in self._top_pr_auc_trials():
            self.best_params = dict(trial.params)
            self.best_trial_number = trial.number
            summary = self.run_cross_validation(
                model=self.build_tuned_model(),
                calibrate=True,
                optimize_threshold=True,
                fit_raw_reference=False,
                persist_results=True,
                model_name=f"{self.MODEL_NAME}_trial_{trial.number}_full_refit",
            )
            brier_skill = summary["brier_skill_score"]["mean"]
            candidates.append(
                {
                    "trial_number": trial.number,
                    "preliminary_pr_auc": float(trial.value),
                    "parameters": dict(trial.params),
                    "metrics": summary,
                    "eligible": bool(np.isfinite(brier_skill) and brier_skill > 0),
                }
            )

        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if not eligible:
            return self._save_rejected_review(candidates)
        selected = sorted(
            eligible,
            key=lambda candidate: (
                -candidate["metrics"]["recall_at_5_percent"]["mean"],
                -candidate["metrics"]["pr_auc"]["mean"],
                candidate["trial_number"],
            ),
        )[0]
        self.best_trial_number = selected["trial_number"]
        self.best_params = dict(selected["parameters"])
        review = {
            "status": "selected",
            "selection_rule": (
                "positive mean Brier skill score; highest mean Recall@5%; "
                "mean PR-AUC tie-breaker"
            ),
            "candidate_count": len(candidates),
            "selected_trial_number": self.best_trial_number,
            "candidates": candidates,
        }
        with open(self.CANDIDATE_REVIEW_FILE, "w", encoding="utf-8") as file:
            json.dump(review, file, indent=4)

        logger.info(
            "Candidate review selected trial=%d | Recall@5%%=%.6f | PR-AUC=%.6f",
            self.best_trial_number,
            selected["metrics"]["recall_at_5_percent"]["mean"],
            selected["metrics"]["pr_auc"]["mean"],
        )
        return review

    def _save_rejected_review(self, candidates):
        self.best_trial_number = None
        self.best_params = {}
        review = {
            "status": "no_eligible_candidate",
            "selection_rule": (
                "positive mean Brier skill score; highest mean Recall@5%; "
                "mean PR-AUC tie-breaker"
            ),
            "candidate_count": len(candidates),
            "selected_trial_number": None,
            "candidates": candidates,
        }
        with open(self.CANDIDATE_REVIEW_FILE, "w", encoding="utf-8") as file:
            json.dump(review, file, indent=4)
        logger.warning(
            "Candidate review rejected all %d trials: no positive mean Brier "
            "skill score. No model or SHAP artifacts will be produced.",
            len(candidates),
        )
        return review

    def _load_selected_candidate_review(self) -> dict[str, Any]:
        """
        Load candidate_review_full_refit.json and refuse to proceed unless a
        winner was actually selected. Used only by the sealed test evaluation.
        """

        if not self.CANDIDATE_REVIEW_FILE.exists():
            raise FileNotFoundError(
                f"Candidate review not found: {self.CANDIDATE_REVIEW_FILE}. "
                "Run the development candidate review before the sealed test "
                "evaluation."
            )

        with open(self.CANDIDATE_REVIEW_FILE, encoding="utf-8") as file:
            review = json.load(file)

        if review.get("status") != "selected":
            raise RuntimeError(
                "Candidate review has no eligible winner "
                f"(status={review.get('status')!r}); the sealed test set must "
                "not be evaluated without a selected candidate."
            )

        return review

    def _build_pipeline(
        self,
        parameters: dict[str, Any],
        model_kind: str,
    ) -> Pipeline:
        self._validate_parameters(parameters)
        model = Pipeline(steps=[("classifier", XGBClassifier(**parameters))])

        logger.info("=" * 70)
        logger.info("Building %s XGBoost model...", model_kind)
        logger.info("=" * 70)
        logger.info("Model Parameters")

        for parameter_name, parameter_value in parameters.items():
            logger.info("%-20s : %s", parameter_name, parameter_value)

        return model

    def _sample_parameters(
        self,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        """
        Sample XGBoost hyperparameters.
        """

        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.20, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        }

    def objective(
        self,
        trial: optuna.Trial,
    ) -> float:
        """
        Optuna objective.

        Optimize mean PR-AUC from WalkForwardCV.
        """

        self.best_params = self._sample_parameters(trial)

        model = self.build_tuned_model()

        summary = self.run_cross_validation(model, persist_results=False)

        score = summary.get("pr_auc", {}).get("mean")

        if score is None:
            raise optuna.TrialPruned("PR-AUC not available.")

        if not np.isfinite(score):
            raise optuna.TrialPruned("Non-finite PR-AUC.")

        trial.set_user_attr("roc_auc", summary["roc_auc"]["mean"])
        trial.set_user_attr("f1", summary["f1"]["mean"])
        trial.set_user_attr("mcc", summary["mcc"]["mean"])
        trial.set_user_attr("balanced_acc", summary["balanced_acc"]["mean"])

        return float(score)

    def _save_best_parameters(
        self,
    ) -> None:
        """
        Persist the best Optuna parameters.
        """

        with open(self.BEST_PARAMS_FILE, "w", encoding="utf-8") as file:
            json.dump(self.best_params, file, indent=4)

    def _save_trials(
        self,
    ) -> None:
        """
        Save the Optuna trial history.
        """

        if self.study is None:
            return

        trials = self.study.trials_dataframe(
            attrs=("number", "value", "params", "user_attrs", "state"),
        ).sort_values("value", ascending=False)

        trials.to_csv(self.TRIALS_FILE, index=False)

    @contextmanager
    def _quiet_trial_logging(self) -> Iterator[None]:
        """Keep the Optuna progress bar readable during repeated CV runs."""

        logger_levels: list[tuple[logging.Logger, int]] = []
        for logger_name in self.TRIAL_LOGGER_NAMES:
            trial_logger = logging.getLogger(logger_name)
            logger_levels.append((trial_logger, trial_logger.level))
            trial_logger.setLevel(logging.WARNING)

        optuna_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        try:
            yield
        finally:
            for trial_logger, previous_level in logger_levels:
                trial_logger.setLevel(previous_level)
            optuna.logging.set_verbosity(optuna_verbosity)

    def optimize(
        self,
    ) -> None:
        """
        Run Optuna hyperparameter optimization.
        """

        self._log_section_header("Starting Optuna optimization...")
        self._create_directories()

        self.study = optuna.create_study(
            study_name=self.MODEL_NAME,
            direction="maximize",
            storage=f"sqlite:///{self.OPTUNA_STORAGE}",
            load_if_exists=True,
        )

        logger.info(
            "Optimizing %d trial(s); the progress bar shows completion and ETA.",
            self.optuna_trials,
        )
        with self._quiet_trial_logging():
            self.study.optimize(
                self.objective,
                n_trials=self.optuna_trials,
                show_progress_bar=True,
            )

        self.best_params = dict(self.study.best_trial.params)
        self.best_trial_number = self.study.best_trial.number

        self._save_best_parameters()
        self._save_trials()

        logger.info(
            "Optimization complete | best trial=%d | PR-AUC=%.6f",
            self.best_trial_number,
            self.study.best_value,
        )

    def run_cross_validation(
        self,
        model: Pipeline,
        calibrate: bool = False,
        calibration_method: str | None = None,
        calibration_cv: int | None = None,
        calibration_strategy: str | None = None,
        calibration_holdout_fraction: float | None = None,
        optimize_threshold: bool = True,
        fit_raw_reference: bool = False,
        persist_results: bool = True,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate the XGBoost model using
        expanding-window walk-forward validation.
        """

        if self.X is None:
            raise RuntimeError("Feature matrix has not been prepared.")

        if self.y is None:
            raise RuntimeError("Target vector has not been prepared.")

        if self.years is None:
            raise RuntimeError("Year vector has not been prepared.")

        self._log_section_header("Starting WalkForwardCV...")
        cv = WalkForwardCV(
            min_fraud_per_fold=self.min_fraud_per_fold,
            output_dir=self.CV_OUTPUT_DIR,
        )

        summary = cv.run(
            estimator=model,
            X=self.X,
            y=self.y,
            years=self.years,
            model_name=model_name or self.MODEL_NAME,
            decision_threshold=self.decision_threshold,
            calibrate=calibrate,
            calibration_method=(
                self.DEFAULT_CALIBRATION_METHOD
                if calibration_method is None
                else calibration_method
            ),
            calibration_cv=(
                self.DEFAULT_CALIBRATION_FOLDS
                if calibration_cv is None
                else calibration_cv
            ),
            calibration_strategy=(
                self.DEFAULT_CALIBRATION_STRATEGY
                if calibration_strategy is None
                else calibration_strategy
            ),
            calibration_holdout_fraction=(
                self.DEFAULT_CALIBRATION_HOLDOUT_FRACTION
                if calibration_holdout_fraction is None
                else calibration_holdout_fraction
            ),
            optimize_threshold=optimize_threshold,
            fit_raw_reference=fit_raw_reference,
            persist_results=persist_results,
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

        self._log_section_header("Running baseline model...")
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

        self._log_section_header("Running tuned model...")
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
        raw_scores = booster.get_score(importance_type=importance_type)

        feature_scores: list[dict[str, Any]] = []

        for index, feature_name in enumerate(self.FEATURE_COLUMNS):
            score = raw_scores.get(f"f{index}", 0.0)
            feature_scores.append({"feature": feature_name, "importance": float(score)})

        return (
            pd.DataFrame(feature_scores)
            .sort_values("importance", ascending=False)
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
        output_directory.mkdir(parents=True, exist_ok=True)

        for importance_type in ("gain", "weight", "cover"):
            importance = self._extract_feature_importance(
                model=model,
                importance_type=importance_type,
            )
            output_file = output_directory / f"{importance_type}.csv"
            importance.to_csv(output_file, index=False)
            logger.info("%s importance saved to %s", importance_type, output_file)

    def save_model_metadata(
        self,
    ) -> None:
        """
        Save experiment metadata.
        """

        metadata = {
            "experiment_name": self.EXPERIMENT_NAME,
            "model": self.MODEL_NAME,
            "random_state": self.RANDOM_STATE,
            "decision_threshold": self.decision_threshold,
            "min_fraud_per_fold": self.min_fraud_per_fold,
            "optuna_trials": self.optuna_trials,
            "best_trial_number": self.best_trial_number,
            "best_parameters": self.best_params,
            "feature_schema_sha256": self._feature_schema_hash(),
        }

        with open(self.MODEL_METADATA_FILE, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        logger.info("Metadata saved.")

    def save_model(
        self,
        model: Pipeline,
    ) -> None:
        """
        Persist the trained model.

        NOTE: this saves the raw full-data model only. It does NOT
        include a fitted calibration mapper. Do not load this file and
        call predict_proba() expecting calibrated output — see
        run_final_test_evaluation(), which reconstructs the calibrated
        model from scratch instead of loading this artifact.
        """

        model_path = self.OUTPUT_DIR / "xgboost_model.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)

    def save_artifacts(
        self,
        model: Pipeline,
    ) -> None:
        """
        Save all model artifacts.
        """

        self._create_directories()
        self.save_feature_importance(model)
        self.save_model_metadata()
        self.save_model(model)

    def _get_trained_classifier(
        self,
        model: Pipeline,
    ) -> XGBClassifier:
        """
        Return the trained XGBoost classifier.
        """

        classifier = model.named_steps.get("classifier")

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

        rng = np.random.default_rng(self.RANDOM_STATE)
        sample_indices = rng.choice(total_rows, size=sample_size, replace=False)

        return self.X[sample_indices], self.FEATURE_COLUMNS

    def compute_shap_values(
        self,
        model: Pipeline,
        sample_size: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values.
        """

        classifier = self._get_trained_classifier(model)
        shap_features, _ = self._sample_shap_dataset(sample_size)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(shap_features)

        return shap_features, np.asarray(shap_values)

    def save_shap_importance(
        self,
        features: np.ndarray,
        shap_values: np.ndarray,
    ) -> None:
        """
        Save global SHAP feature importance.
        """

        if features.shape[1] != len(self.FEATURE_COLUMNS):
            raise ValueError("SHAP feature count does not match the configured schema.")

        importance = np.abs(shap_values).mean(axis=0)

        shap_importance = (
            pd.DataFrame({"feature": self.FEATURE_COLUMNS, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        output_directory = self.OUTPUT_DIR / "shap"
        output_directory.mkdir(parents=True, exist_ok=True)

        shap_importance.to_csv(output_directory / "shap_importance.csv", index=False)

        logger.info("SHAP importance saved.")

    def save_shap_summary_plot(
        self,
        features: np.ndarray,
        shap_values: np.ndarray,
    ) -> None:
        """
        Save SHAP summary plot.
        """

        output_directory = self.OUTPUT_DIR / "shap"
        output_directory.mkdir(parents=True, exist_ok=True)

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

        self._log_section_header("Running SHAP analysis...")
        features, shap_values = self.compute_shap_values(model)
        self.save_shap_importance(features, shap_values)
        self.save_shap_summary_plot(features, shap_values)

        logger.info("SHAP analysis completed.")

    def log_summary(
        self,
    ) -> None:
        """
        Log the aggregate cross-validation summary.
        """

        if self.cv_summary is None:
            raise RuntimeError("Cross-validation summary is unavailable.")

        self._log_section_header("XGBoost Summary")
        if self.best_trial_number is not None:
            logger.info("Selected Optuna Trial : %d", self.best_trial_number)

        logger.info("Folds Evaluated : %d", self.cv_summary["n_folds"])
        logger.info("Years Evaluated : %s", self.cv_summary["years_evaluated"])

        if self.y is None:
            raise RuntimeError("Target vector has not been prepared.")

        logger.info("Overall Fraud Rate : %.2f%%", self.y.mean() * 100)
        logger.info("Total Fraud Cases : %d", self.cv_summary["total_test_fraud"])

        for metric_name in self.SUMMARY_METRICS:
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

        if self.best_trial_number is not None:
            logger.info(
                "Evaluating selected best Optuna model | trial=%d",
                self.best_trial_number,
            )

        return self.run_cross_validation(
            model=model,
            calibrate=calibrate,
            optimize_threshold=True,
            fit_raw_reference=False,
        )

    def fit_full_dataset_model(self, model: Pipeline) -> Pipeline:
        """Fit a fresh model on all development data for artifact generation."""

        if self.X is None or self.y is None:
            raise RuntimeError("Features and target must be prepared before fitting.")

        fitted_model = clone(model)
        fitted_model.fit(self.X, self.y)
        return fitted_model

    def run(
        self,
        optimize: bool = False,
        calibrate: bool = False,
        run_shap: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the complete XGBoost pipeline.
        """

        self._prepare_development_pipeline("Starting XGBoost pipeline...")
        selected_model = self.train_model(optimize=optimize)

        if optimize:
            review = self.review_top_candidates()
            if review["selected_trial_number"] is None:
                raise RuntimeError(
                    "Candidate review found no eligible model; "
                    "see candidate_review.json."
                )
            selected_model = self.build_tuned_model()

        summary = self.evaluate_model(model=selected_model, calibrate=calibrate)

        fitted_model = self.fit_full_dataset_model(selected_model)
        self._finalize_artifacts(fitted_model, run_shap)
        logger.info("XGBoost completed successfully.")
        logger.info("=" * 70)

        return summary

    def run_candidate_review(
        self,
        run_shap: bool = True,
    ) -> dict[str, Any]:
        """Review an existing completed study and create selected-model artifacts."""
        self._prepare_development_pipeline("Starting XGBoost candidate review...")
        review = self.review_top_candidates()
        if review["selected_trial_number"] is None:
            logger.warning(
                "Candidate review ended without a winner; no model or SHAP artifacts "
                "were produced."
            )
            return review
        selected = next(
            candidate
            for candidate in review["candidates"]
            if candidate["trial_number"] == self.best_trial_number
        )
        self.cv_summary = selected["metrics"]

        fitted_model = self.fit_full_dataset_model(self.build_tuned_model())
        self._finalize_artifacts(fitted_model, run_shap)
        return self.cv_summary

    def _finalize_artifacts(self, fitted_model: Pipeline, run_shap: bool) -> None:
        self.best_model = fitted_model
        self.save_artifacts(fitted_model)
        if run_shap:
            self.run_shap_analysis(fitted_model)
        self.log_summary()

    def _prepare_development_pipeline(self, header: str) -> pd.DataFrame:
        self._log_section_header(header)
        self._create_directories()
        dataset = self.load_dataset()
        self.validate_dataset(dataset)
        self.prepare_features(dataset)
        self.save_experiment_manifest()
        return dataset

    def run_selected_candidate_shap(self) -> None:
        """Fit and explain the persisted candidate-review winner only."""
        if not self.CANDIDATE_REVIEW_FILE.exists():
            raise FileNotFoundError(
                "Candidate review is missing; run the development review first."
            )

        with open(self.CANDIDATE_REVIEW_FILE, encoding="utf-8") as file:
            review = json.load(file)
        if review.get("status") != "selected":
            raise RuntimeError(
                "Candidate review has no eligible winner; SHAP must not be run."
            )

        selected_trial = review["selected_trial_number"]
        selected = next(
            candidate
            for candidate in review["candidates"]
            if candidate["trial_number"] == selected_trial
        )
        self._create_directories()
        dataset = self.load_dataset()
        self.validate_dataset(dataset)
        self.prepare_features(dataset)
        self.best_trial_number = selected_trial
        self.best_params = dict(selected["parameters"])
        fitted_model = self.fit_full_dataset_model(self.build_tuned_model())
        self.best_model = fitted_model
        self.save_artifacts(fitted_model)
        self.run_shap_analysis(fitted_model)

    # ============================================================
    # Sealed Test Evaluation (one-time, 2019-2022)
    # ============================================================

    def _parse_filing_dates(
        self,
        dataset: pd.DataFrame,
        dataset_name: str,
    ) -> pd.Series:
        """Parse the date used by the temporal split and validate it fully."""

        if self.FILING_DATE_COLUMN not in dataset.columns:
            raise ValueError(
                f"{dataset_name} data is missing {self.FILING_DATE_COLUMN!r}."
            )

        filing_dates = pd.to_datetime(
            dataset[self.FILING_DATE_COLUMN],
            dayfirst=True,
            errors="coerce",
        )
        if filing_dates.isna().any():
            missing_count = int(filing_dates.isna().sum())
            raise ValueError(
                f"{dataset_name} data contains {missing_count} invalid filing dates."
            )
        return filing_dates

    def _select_final_test_period(
        self,
        test_dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Restrict the test feature file to the pre-registered 2019-2022 window."""

        filing_dates = self._parse_filing_dates(test_dataset, "Test")
        period_mask = filing_dates.dt.year.between(
            self.FINAL_TEST_START_YEAR,
            self.FINAL_TEST_END_YEAR,
        )
        if not period_mask.any():
            raise ValueError(
                "Test data has no rows in the required "
                f"{self.FINAL_TEST_START_YEAR}-{self.FINAL_TEST_END_YEAR} period."
            )

        final_test_dataset = test_dataset.loc[period_mask].copy()
        final_test_dates = filing_dates.loc[period_mask]
        excluded_count = int((~period_mask).sum())
        if excluded_count:
            logger.info(
                "Excluded %d test rows outside the fixed %d-%d filing-date window.",
                excluded_count,
                self.FINAL_TEST_START_YEAR,
                self.FINAL_TEST_END_YEAR,
            )
        return final_test_dataset, final_test_dates

    def _validate_final_test_dates(
        self,
        test_filing_dates: pd.Series,
        dev_filing_dates: pd.Series,
    ) -> None:
        """Ensure the selected final-test window is strictly after development."""

        min_test_year = int(test_filing_dates.dt.year.min())
        max_test_year = int(test_filing_dates.dt.year.max())
        if (
            min_test_year != self.FINAL_TEST_START_YEAR
            or max_test_year != self.FINAL_TEST_END_YEAR
        ):
            raise ValueError(
                "Selected final-test period does not match the required "
                f"{self.FINAL_TEST_START_YEAR}-{self.FINAL_TEST_END_YEAR} window."
            )

        if dev_filing_dates.max() >= test_filing_dates.min():
            raise ValueError(
                "Development and final-test filing dates overlap or are out of order."
            )

    def _final_test_artifact_paths(self) -> tuple[Path, Path, Path]:
        """Return the three immutable artifacts of a sealed test evaluation."""

        return (
            self.FINAL_TEST_OUTPUT_DIR / "final_test_summary.json",
            self.FINAL_TEST_OUTPUT_DIR / "final_test_predictions.csv",
            self.FINAL_TEST_OUTPUT_DIR / "final_test_calibration.json",
        )

    def _ensure_final_test_has_not_run(self) -> None:
        """Refuse to overwrite artifacts from an already-opened test set."""

        existing_paths = [
            path for path in self._final_test_artifact_paths() if path.exists()
        ]
        if existing_paths:
            existing_names = ", ".join(path.name for path in existing_paths)
            raise RuntimeError(
                "Final-test artifacts already exist; sealed evaluation cannot be "
                f"rerun. Existing artifacts: {existing_names}"
            )

    def run_final_test_evaluation(self) -> dict[str, Any]:
        """
        One-time sealed evaluation on the 2019-2022 held-out test set.

        This is a ONE-SHOT evaluation. Do not call it more than once for
        a given candidate selection — rerunning after inspecting the
        result defeats the purpose of holding out a test set at all.

        Procedure
        ---------
        1. Load candidate_review_full_refit.json; refuse to run unless
           status == "selected".
        2. Load the selected trial's hyperparameters from that file.
        3. Load development data (trainval_features.parquet) and the
           sealed test data (test_features.parquet).
        4. Restrict test rows to the pre-registered 2019-2022 filing-date
           window and verify that it follows the development period.
        5. Reconstruct the calibrated model exactly as scored during
           development: fit the raw estimator on the early chronological
           portion of ALL development data, fit the probability mapper
           on the late chronological holdout, refit the estimator on
           all development data, then score test rows through that
           frozen mapper. Does NOT load xgboost_model.joblib — that
           artifact has no fitted calibration mapper attached.
        6. Score at a fixed 0.50 threshold. No threshold tuning against
           test data.
        7. Compute the full metric set (ROC-AUC, PR-AUC, precision,
           recall, F1, F1-macro, MCC, balanced accuracy, Brier, naive
           Brier, Brier skill score, Recall@5%, Precision@5%, ECE, MCE,
           reliability curve) by reusing WalkForwardCV's own metric
           calculations rather than reimplementing them.
        8. Save to reports/models/<experiment>/final_test/ only:
           final_test_summary.json, final_test_predictions.csv,
           final_test_calibration.json.

        Returns
        -------
        dict
            The full final_test_summary.json payload.
        """

        review = self._load_selected_candidate_review()
        selected = next(
            candidate
            for candidate in review["candidates"]
            if candidate["trial_number"] == review["selected_trial_number"]
        )
        self.best_trial_number = review["selected_trial_number"]
        self.best_params = dict(selected["parameters"])

        self._create_directories()
        self.FINAL_TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_final_test_has_not_run()

        dev_dataset = self.load_dataset()
        self.validate_dataset(dev_dataset)
        self.prepare_features(dev_dataset)
        dev_filing_dates = self._parse_filing_dates(dev_dataset, "Development")

        test_dataset = self.load_test_dataset()
        self.validate_dataset(test_dataset)
        test_dataset, test_filing_dates = self._select_final_test_period(test_dataset)
        X_test, y_test, years_test = self._extract_arrays(test_dataset)
        self._validate_final_test_dates(test_filing_dates, dev_filing_dates)

        logger.info("=" * 70)
        logger.info(
            "SEALED TEST EVALUATION | trial=%d | dev_dates=%s-%s | test_dates=%s-%s",
            self.best_trial_number,
            dev_filing_dates.min().date(),
            dev_filing_dates.max().date(),
            test_filing_dates.min().date(),
            test_filing_dates.max().date(),
        )
        logger.info("=" * 70)

        unfitted_model = self.build_tuned_model()

        calibrator = ProbabilityCalibrator(
            method=self.DEFAULT_CALIBRATION_METHOD,
            cv=self.DEFAULT_CALIBRATION_FOLDS,
            strategy=self.DEFAULT_CALIBRATION_STRATEGY,
            holdout_fraction=self.DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
        ).fit(unfitted_model, self.X, self.y, years=self.years)

        test_scores = calibrator.predict_proba(X_test)
        y_pred = (test_scores >= self.FINAL_TEST_DECISION_THRESHOLD).astype(int)

        # Reuse WalkForwardCV's own metric math instead of duplicating it.
        metrics_helper = WalkForwardCV(min_fraud_per_fold=self.min_fraud_per_fold)
        fold_metrics = metrics_helper._calculate_fold_metrics(
            y_test=y_test,
            y_score=test_scores,
            y_pred=y_pred,
            test_year=0,
            train_size=len(self.X),
            decision_threshold=self.FINAL_TEST_DECISION_THRESHOLD,
            calibrate=True,
            raw_brier=None,
        )
        fold_metrics.pop("test_year")
        fold_metrics.pop("ece", None)
        fold_metrics.pop("mce", None)
        fold_metrics["test_period"] = (
            f"{self.FINAL_TEST_START_YEAR}-{self.FINAL_TEST_END_YEAR}"
        )
        fold_metrics["selected_trial_number"] = self.best_trial_number
        fold_metrics["threshold_tuned_on_test"] = False

        calibration_metrics = evaluate_calibration(y_test, test_scores)

        predictions_df = pd.DataFrame(
            {
                "filing_year": years_test,
                "filing_date": test_filing_dates.dt.strftime("%Y-%m-%d"),
                "true_label": y_test,
                "predicted_probability": test_scores,
                "predicted_label": y_pred,
                "decision_threshold": self.FINAL_TEST_DECISION_THRESHOLD,
            }
        )

        summary_path, predictions_path, calibration_path = (
            self._final_test_artifact_paths()
        )

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(fold_metrics, file, indent=4)
        predictions_df.to_csv(predictions_path, index=False)
        with open(calibration_path, "w", encoding="utf-8") as file:
            json.dump(calibration_metrics, file, indent=4)

        logger.info("Final test summary saved to %s", summary_path)
        logger.info("Final test predictions saved to %s", predictions_path)
        logger.info("Final test calibration saved to %s", calibration_path)

        logger.info("=" * 70)
        logger.info("SEALED TEST RESULT (2019-2022, one-time)")
        for metric_name in self.SUMMARY_METRICS:
            if metric_name in fold_metrics:
                logger.info("%-15s : %.6f", metric_name, fold_metrics[metric_name])
        logger.info("=" * 70)

        return fold_metrics

    def _log_section_header(self, title: str) -> None:
        logger.info("=" * 70)
        logger.info(title)
        logger.info("=" * 70)


def run_xgboost(
    optimize: bool = True,
    calibrate: bool = True,
    run_shap: bool = False,
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


def review_xgboost_candidates(
    run_shap: bool = True,
) -> dict[str, Any]:
    """Review the completed Optuna study without adding trials or using test data."""
    return XGBoostBaseline().run_candidate_review(run_shap=run_shap)


def run_selected_xgboost_shap() -> None:
    """Run SHAP for a persisted eligible v2 winner without retraining candidates."""
    XGBoostBaseline().run_selected_candidate_shap()


def evaluate_selected_xgboost_on_test() -> dict[str, Any]:
    """Run the one-time sealed 2019-2022 evaluation for the selected winner."""
    return XGBoostBaseline().run_final_test_evaluation()


def main() -> None:
    """
    Script entry point.
    """

    run_xgboost(
        optimize=True,
        calibrate=True,
        run_shap=False,
    )


if __name__ == "__main__":
    main()
