"""
Walk-forward cross-validation engine.

Responsibilities
----------------
- Generate expanding-window temporal folds by year.
- Skip folds where test-year fraud count is below threshold.
- Fit any sklearn-compatible estimator on each fold.
- Compute per-fold evaluation metrics.
- Aggregate metrics across folds.
- Save fold results and summary report.

This module DOES NOT

- load data from disk
- engineer features
- select or tune models
- know anything about Parquet files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import configs.settings as settings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.evaluation.calibration import ProbabilityCalibrator, evaluate_calibration
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_CALIBRATION_METHOD = "sigmoid"
DEFAULT_CALIBRATION_CV = 5
THRESHOLD_CANDIDATES = np.arange(0.01, 1.00, 0.01)
METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "mcc",
    "balanced_acc",
    "brier_score",
)


class WalkForwardCV:
    """
    Expanding-window walk-forward cross-validation.

    Each fold trains on all data before the test year
    and evaluates on the test year only.

    Folds where the test year contains fewer than
    `min_fraud_per_fold` fraud cases are skipped to
    prevent unstable metrics on near-empty test sets.

    Parameters
    ----------
    min_fraud_per_fold : int
        Minimum number of fraud cases required in a
        test fold for it to be included. Default: 30.

    output_dir : Path
        Directory where fold_results.csv and
        cv_summary.json are written.
    """

    OUTPUT_DIR = settings.REPORTS_DIR / "evaluation"

    def __init__(
        self,
        min_fraud_per_fold: int = 30,
        output_dir: Path | None = None,
    ) -> None:
        self.min_fraud_per_fold = min_fraud_per_fold
        self.output_dir = output_dir or self.OUTPUT_DIR
        self.fold_results: list[dict[str, Any]] = []
        self.fold_predictions: list[pd.DataFrame] = []
        self.calibration_curves: list[dict[str, Any]] = []

    # ============================================================
    # Fold Generation
    # ============================================================

    def generate_folds(
        self,
        years: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Yield (train_idx, test_idx, test_year) tuples.

        Train set  : all indices where year < test_year
        Test set   : all indices where year == test_year
        Skip       : test folds with fraud_count < min_fraud_per_fold
        """

        if len(years) != len(y):
            raise ValueError("years and y must have the same length.")

        unique_years = sorted(np.unique(years))

        for test_year in unique_years:
            train_idx = np.where(years < test_year)[0]
            test_idx = np.where(years == test_year)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            fraud_count = int(y[test_idx].sum())

            if fraud_count < self.min_fraud_per_fold:
                logger.info(
                    "Skipping fold year=%d | fraud_count=%d < min=%d",
                    test_year,
                    fraud_count,
                    self.min_fraud_per_fold,
                )
                continue

            logger.info(
                "Fold year=%d | train_n=%d | test_n=%d | test_fraud=%d",
                test_year,
                len(train_idx),
                len(test_idx),
                fraud_count,
            )

            yield train_idx, test_idx, test_year

    def find_best_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        default_threshold: float = DEFAULT_DECISION_THRESHOLD,
    ) -> float:
        """
        Find the threshold that maximizes F1 score.
        """

        if len(np.unique(y_true)) < 2:
            return default_threshold

        best_threshold = default_threshold
        best_f1 = -1.0

        for threshold in THRESHOLD_CANDIDATES:
            y_pred = (y_score >= threshold).astype(int)

            score = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        return best_threshold

    # ============================================================
    # Fold Evaluation
    # ============================================================

    def evaluate_fold(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        test_year: int,
        decision_threshold: float,
        calibrate: bool = False,
        calibration_method: str = DEFAULT_CALIBRATION_METHOD,
        calibration_cv: int = DEFAULT_CALIBRATION_CV,
        fit_raw_reference: bool = True,
        optimize_threshold: bool = True,
    ) -> dict[str, Any]:
        """
        Fit estimator on train split, evaluate on test split.

        If calibrate=True, probabilities are calibrated via
        ProbabilityCalibrator (Platt/isotonic), fit ONLY on the
        training fold.

        fit_raw_reference : bool
            If True (default), also fits an uncalibrated reference
            model to report Brier-score improvement from calibration.
            Costs one extra full fit on top of the `calibration_cv`
            internal fits CalibratedClassifierCV already performs —
            for expensive estimators (XGBoost/CatBoost/LightGBM) set
            this to False once you've confirmed calibration helps and
            you no longer need the raw comparison every run.

        optimize_threshold : bool
            If True (default), searches for the F1-optimal threshold
            on the training fold, using whichever score (raw or
            calibrated) is being evaluated. If False, uses
            `decision_threshold` directly without search — use this
            to hold the threshold fixed across a calibrated vs.
            uncalibrated comparison, to isolate the effect of
            calibration on F1/precision/recall independent of
            re-thresholding. Brier score and ECE are threshold-
            independent either way and are the more direct measure
            of calibration quality.

        Returns a dict of metrics for this fold.
        """

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if not self._has_both_classes(y_train):
            logger.warning(
                "Skipping year=%d because training fold has only one class.",
                test_year,
            )
            return {}

        train_score, test_score, raw_brier = self._fit_and_score_fold(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            calibrate=calibrate,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            fit_raw_reference=fit_raw_reference,
        )

        best_threshold = self._select_threshold(
            y_train=y_train,
            train_score=train_score,
            default_threshold=decision_threshold,
            should_optimize=optimize_threshold,
        )

        logger.info(
            "Year %d | Optimal Threshold = %.2f",
            test_year,
            best_threshold,
        )

        y_pred = (test_score >= best_threshold).astype(int)
        fold_metrics = self._calculate_fold_metrics(
            y_test=y_test,
            y_score=test_score,
            y_pred=y_pred,
            test_year=test_year,
            train_size=len(train_idx),
            decision_threshold=best_threshold,
            calibrate=calibrate,
            raw_brier=raw_brier,
        )
        self._record_predictions(
            test_year=test_year,
            y_test=y_test,
            y_score=test_score,
            y_pred=y_pred,
            decision_threshold=best_threshold,
        )

        if not calibrate:
            return fold_metrics

        return self._record_calibration_metrics(fold_metrics, y_test, test_score)

    @staticmethod
    def _has_both_classes(labels: np.ndarray) -> bool:
        return len(np.unique(labels)) == 2

    def _fit_and_score_fold(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        calibrate: bool,
        calibration_method: str,
        calibration_cv: int,
        fit_raw_reference: bool,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        if not calibrate:
            fitted_model = clone(estimator).fit(X_train, y_train)
            return (
                self._predict_probabilities(fitted_model, X_train),
                self._predict_probabilities(fitted_model, X_test),
                None,
            )

        self._validate_calibration_estimator(estimator)
        raw_brier = self._calculate_raw_brier(
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            should_fit=fit_raw_reference,
        )
        calibrator = ProbabilityCalibrator(
            method=calibration_method,
            cv=calibration_cv,
        ).fit(clone(estimator), X_train, y_train)
        return (
            calibrator.predict_proba(X_train),
            calibrator.predict_proba(X_test),
            raw_brier,
        )

    @staticmethod
    def _predict_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            raise AttributeError("Estimator must implement predict_proba().")
        return model.predict_proba(features)[:, 1]

    @staticmethod
    def _validate_calibration_estimator(estimator: Any) -> None:
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "Calibration requires an estimator that implements predict_proba()."
            )

    def _calculate_raw_brier(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        should_fit: bool,
    ) -> float | None:
        if not should_fit:
            return None

        raw_model = clone(estimator).fit(X_train, y_train)
        raw_scores = self._predict_probabilities(raw_model, X_test)
        return float(brier_score_loss(y_test, raw_scores))

    def _select_threshold(
        self,
        y_train: np.ndarray,
        train_score: np.ndarray,
        default_threshold: float,
        should_optimize: bool,
    ) -> float:
        if not should_optimize:
            return default_threshold
        return self.find_best_threshold(y_train, train_score, default_threshold)

    def _calculate_fold_metrics(
        self,
        y_test: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
        test_year: int,
        train_size: int,
        decision_threshold: float,
        calibrate: bool,
        raw_brier: float | None,
    ) -> dict[str, Any]:
        has_both_test_classes = self._has_both_classes(y_test)
        roc_auc = float("nan")
        pr_auc = float("nan")
        if has_both_test_classes:
            roc_auc = float(roc_auc_score(y_test, y_score))
            pr_auc = float(average_precision_score(y_test, y_score))

        return {
            "test_year": test_year,
            "train_n": train_size,
            "test_n": len(y_test),
            "test_fraud_n": int(y_test.sum()),
            "decision_threshold": decision_threshold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "brier_score": brier_score_loss(y_test, y_score),
            "calibrated": calibrate,
            "raw_brier_score": raw_brier,
            "ece": None,
            "mce": None,
        }

    def _record_predictions(
        self,
        test_year: int,
        y_test: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
        decision_threshold: float,
    ) -> None:
        prediction_frame = pd.DataFrame(
            {
                "test_year": test_year,
                "true_label": y_test,
                "predicted_probability": y_score,
                "predicted_label": y_pred,
                "decision_threshold": decision_threshold,
            }
        )
        self.fold_predictions.append(prediction_frame)

    def _record_calibration_metrics(
        self,
        fold_metrics: dict[str, Any],
        y_test: np.ndarray,
        y_score: np.ndarray,
    ) -> dict[str, Any]:
        calibration_metrics = evaluate_calibration(y_test, y_score)
        self.calibration_curves.append(
            {**calibration_metrics, "test_year": int(fold_metrics["test_year"])}
        )
        calibrated_metrics = {
            **fold_metrics,
            "ece": calibration_metrics["ece"],
            "mce": calibration_metrics["mce"],
        }
        self._log_calibration_metrics(calibrated_metrics)
        return calibrated_metrics

    @staticmethod
    def _log_calibration_metrics(fold_metrics: dict[str, Any]) -> None:
        raw_brier = fold_metrics["raw_brier_score"]
        calibrated_brier = fold_metrics["brier_score"]
        if raw_brier is None:
            logger.info(
                "Year %d | Brier calibrated=%.4f | ECE=%.4f MCE=%.4f",
                fold_metrics["test_year"],
                calibrated_brier,
                fold_metrics["ece"],
                fold_metrics["mce"],
            )
            return

        brier_improvement = raw_brier - calibrated_brier
        logger.info(
            "Year %d | Brier raw=%.4f -> calibrated=%.4f (%s%.4f) | ECE=%.4f MCE=%.4f",
            fold_metrics["test_year"],
            raw_brier,
            calibrated_brier,
            "+" if brier_improvement >= 0 else "",
            brier_improvement,
            fold_metrics["ece"],
            fold_metrics["mce"],
        )

    # ============================================================
    # Aggregation
    # ============================================================

    def aggregate_metrics(self) -> dict[str, Any]:
        """
        Compute mean and std across all folds for each metric.
        """

        if not self.fold_results:
            logger.warning("No fold results available.")
            return {}

        df = pd.DataFrame(self.fold_results)

        summary: dict[str, Any] = {
            "n_folds": len(df),
            "years_evaluated": df["test_year"].tolist(),
            "total_test_fraud": int(df["test_fraud_n"].sum()),
            "decision_threshold_mean": round(float(df["decision_threshold"].mean()), 6),
            "decision_threshold_std": round(
                float(df["decision_threshold"].std(ddof=0)), 6
            ),
        }

        for metric_name in METRIC_NAMES:
            summary[metric_name] = {
                "mean": round(float(df[metric_name].mean()), 6),
                "std": round(float(df[metric_name].std(ddof=0)), 6),
            }

        return summary

    # ============================================================
    # Saving
    # ============================================================

    def save_results(
        self,
        model_name: str,
    ) -> None:
        """
        Save per-fold results and aggregate summary.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fold_path = self.output_dir / f"{model_name}_fold_results.csv"
        fold_df = pd.DataFrame(self.fold_results)
        fold_df.to_csv(
            fold_path,
            index=False,
        )

        logger.info(
            "Fold results saved to %s",
            fold_path,
        )
        prediction_path = self.output_dir / f"{model_name}_predictions.csv"

        if self.fold_predictions:
            prediction_df = pd.concat(
                self.fold_predictions,
                ignore_index=True,
            )
            prediction_df.to_csv(
                prediction_path,
                index=False,
            )

            logger.info(
                "Fold predictions saved to %s",
                prediction_path,
            )
        else:
            logger.warning(
                "No fold predictions to save for %s",
                model_name,
            )

        summary = self.aggregate_metrics()
        self._write_json_report(
            filename=f"{model_name}_cv_summary.json",
            payload=summary,
            log_message="CV summary saved to %s",
        )
        if self.calibration_curves:
            curves_path = self.output_dir / f"{model_name}_calibration_curves.json"
            with open(curves_path, "w", encoding="utf-8") as f:
                json.dump(self.calibration_curves, f, indent=4)
            logger.info("Calibration curves saved to %s", curves_path)

            if calibrated_folds := [
                r for r in self.fold_results if r.get("calibrated")
            ]:
                cal_df = pd.DataFrame(calibrated_folds)

                improvements = (
                    cal_df["raw_brier_score"] - cal_df["brier_score"]
                ).dropna()

                calibration_summary: dict[str, Any] = {
                    "n_calibrated_folds": len(cal_df),
                    "brier_score_calibrated": {
                        "mean": round(float(cal_df["brier_score"].mean()), 6),
                        "std": round(float(cal_df["brier_score"].std(ddof=0)), 6),
                    },
                    "ece": {
                        "mean": round(float(cal_df["ece"].mean()), 6),
                        "std": round(float(cal_df["ece"].std(ddof=0)), 6),
                    },
                    "mce": {
                        "mean": round(float(cal_df["mce"].mean()), 6),
                        "std": round(float(cal_df["mce"].std(ddof=0)), 6),
                    },
                }

                if cal_df["raw_brier_score"].notna().any():
                    calibration_summary["brier_score_raw"] = {
                        "mean": round(float(cal_df["raw_brier_score"].mean()), 6),
                        "std": round(float(cal_df["raw_brier_score"].std(ddof=0)), 6),
                    }
                    calibration_summary["brier_improvement"] = {
                        "mean": (
                            round(float(improvements.mean()), 6)
                            if len(improvements)
                            else None
                        ),
                        "std": (
                            round(float(improvements.std(ddof=0)), 6)
                            if len(improvements)
                            else None
                        ),
                        "n_folds_improved": int((improvements > 0).sum()),
                        "n_folds_worsened": int((improvements < 0).sum()),
                    }

                self._write_json_report(
                    filename=f"{model_name}_calibration_summary.json",
                    payload=calibration_summary,
                    log_message="Calibration summary saved to %s",
                )

    def _write_json_report(
        self,
        filename: str,
        payload: dict[str, Any],
        log_message: str,
    ) -> None:
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=4)
        logger.info(log_message, report_path)

    # ============================================================
    # Main Entry Point
    # ============================================================

    def run(
        self,
        estimator,
        X,
        y,
        years,
        model_name="model",
        decision_threshold: float = 0.5,
        calibrate: bool = False,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 5,
        fit_raw_reference: bool = True,
        optimize_threshold: bool = True,
        persist_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run walk-forward cross-validation.

        Parameters
        ----------
        estimator : sklearn-compatible
            Must implement fit(), predict(), predict_proba().
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Binary target (1=fraud, 0=clean).
        years : np.ndarray
            Filing year for each row. Same length as X and y.
        model_name : str
            Used for output file naming.
        calibrate : bool
            If True, calibrate probabilities per fold via
            ProbabilityCalibrator before thresholding/scoring.
        calibration_method : "sigmoid" | "isotonic"
        calibration_cv : int
            Internal CV folds used to fit the calibration map.
        fit_raw_reference : bool
            If True (default), fits an extra uncalibrated model per
            fold purely to report Brier-score improvement. Set False
            for expensive estimators once you don't need the
            comparison every run — see evaluate_fold docstring.
        optimize_threshold : bool
            If False, skips per-fold threshold search and uses
            `decision_threshold` directly. Use to hold the threshold
            fixed when isolating calibration's effect from
            re-thresholding — see evaluate_fold docstring.

        Returns
        -------
        dict
            Aggregated metrics summary.
        """

        logger.info("=" * 70)
        logger.info(
            "Walk-forward CV | model=%s | min_fraud_per_fold=%d",
            model_name,
            self.min_fraud_per_fold,
        )
        logger.info("=" * 70)

        self.fold_results = []
        self.fold_predictions = []
        self.calibration_curves = []

        for train_idx, test_idx, test_year in self.generate_folds(years, y):
            fold_metrics = self.evaluate_fold(
                estimator=estimator,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                test_year=test_year,
                decision_threshold=decision_threshold,
                calibrate=calibrate,
                calibration_method=calibration_method,
                calibration_cv=calibration_cv,
                fit_raw_reference=fit_raw_reference,
                optimize_threshold=optimize_threshold,
            )

            if not fold_metrics:
                continue

            self.fold_results.append(fold_metrics)

            logger.info(
                "year=%d | roc_auc=%.4f | pr_auc=%.4f | f1=%.4f | mcc=%.4f",
                test_year,
                fold_metrics["roc_auc"],
                fold_metrics["pr_auc"],
                fold_metrics["f1"],
                fold_metrics["mcc"],
            )

        summary = self.aggregate_metrics()

        if persist_results:
            self.save_results(model_name)

        logger.info("=" * 70)
        logger.info("CV complete | n_folds=%d", summary.get("n_folds", 0))

        for metric in ["roc_auc", "pr_auc", "f1", "mcc", "balanced_acc"]:
            stats = summary.get(metric, {})
            logger.info(
                "%s: mean=%.4f std=%.4f",
                metric,
                stats.get("mean", 0),
                stats.get("std", 0),
            )

        logger.info("=" * 70)

        return summary
