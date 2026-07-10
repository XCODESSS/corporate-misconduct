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

import numpy as np
import pandas as pd
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

import configs.settings as settings
from src.evaluation.calibration import ProbabilityCalibrator, evaluate_calibration
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            raise ValueError(
                "years and y must have the same length."
            )

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
        default_threshold: float = 0.5,
    ) -> float:
        """
        Find the threshold that maximizes F1 score.
        """

        if len(np.unique(y_true)) < 2:
            return default_threshold

        thresholds = np.arange(0.01, 1.00, 0.01)

        best_threshold = default_threshold
        best_f1 = -1.0

        for threshold in thresholds:
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
        estimator,
        X,
        y,
        train_idx,
        test_idx,
        test_year,
        decision_threshold: float,
        calibrate: bool = False,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 5,
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

        import sklearn.base as skbase

        model = skbase.clone(estimator)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            logger.warning(
                "Skipping year=%d because training fold has only one class.",
                test_year,
            )
            return {}

        raw_brier = None

        if calibrate:
            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    "Calibration requires an estimator that implements "
                    "predict_proba()."
                )

            if fit_raw_reference:
                # Uncalibrated reference fit, for Brier comparison only.
                raw_model = skbase.clone(estimator)
                raw_model.fit(X_train, y_train)
                raw_test_score = raw_model.predict_proba(X_test)[:, 1]
                raw_brier = brier_score_loss(y_test, raw_test_score)

            calibrator = ProbabilityCalibrator(
                method=calibration_method,
                cv=calibration_cv,
            )
            calibrator.fit(model, X_train, y_train)

            train_score = calibrator.predict_proba(X_train)
            test_score = calibrator.predict_proba(X_test)
        else:
            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                train_score = model.predict_proba(X_train)[:, 1]
                test_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                train_score = model.decision_function(X_train)
                test_score = model.decision_function(X_test)
            else:
                raise AttributeError(
                    "Estimator must implement predict_proba() or decision_function()."
                )

        if optimize_threshold:
            best_threshold = self.find_best_threshold(
                y_train,
                train_score,
                default_threshold=decision_threshold,
            )
        else:
            best_threshold = decision_threshold

        logger.info(
            "Year %d | Optimal Threshold = %.2f",
            test_year,
            best_threshold,
        )

        y_pred = (test_score >= best_threshold).astype(int)
        y_score = test_score

        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except ValueError:
            roc_auc = float("nan")

        try:
            pr_auc = average_precision_score(y_test, y_score)
        except ValueError:
            pr_auc = float("nan")

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_score)

        prediction_df = pd.DataFrame(
            {
                "test_year": test_year,
                "true_label": y_test,
                "predicted_probability": y_score,
                "predicted_label": y_pred,
                "decision_threshold": best_threshold,
            }
        )

        self.fold_predictions.append(prediction_df)

        fold_metrics = {
            "test_year": test_year,
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "test_fraud_n": int(y_test.sum()),
            "decision_threshold": best_threshold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "mcc": mcc,
            "balanced_acc": bal_acc,
            "brier_score": brier,
            "calibrated": calibrate,
            "raw_brier_score": raw_brier,
            "ece": None,
            "mce": None,
        }

        if calibrate:
            curve = evaluate_calibration(y_test, y_score)
            curve["test_year"] = int(test_year)
            self.calibration_curves.append(curve)

            fold_metrics["ece"] = curve["ece"]
            fold_metrics["mce"] = curve["mce"]

            if raw_brier is not None:
                logger.info(
                    "Year %d | Brier raw=%.4f -> calibrated=%.4f (%s%.4f) | ECE=%.4f MCE=%.4f",
                    test_year,
                    raw_brier,
                    brier,
                    "+" if raw_brier - brier >= 0 else "",
                    raw_brier - brier,
                    curve["ece"],
                    curve["mce"],
                )
            else:
                logger.info(
                    "Year %d | Brier calibrated=%.4f | ECE=%.4f MCE=%.4f",
                    test_year,
                    brier,
                    curve["ece"],
                    curve["mce"],
                )

        return fold_metrics

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

        metric_cols = [
            "roc_auc", "pr_auc", "f1",
            "precision", "recall", "mcc",
            "balanced_acc", "brier_score",
        ]

        summary: dict[str, Any] = {
            "n_folds": len(df),
            "years_evaluated": df["test_year"].tolist(),
            "total_test_fraud": int(df["test_fraud_n"].sum()),
            "decision_threshold_mean": round(float(df["decision_threshold"].mean()), 6),
            "decision_threshold_std": round(float(df["decision_threshold"].std(ddof=0)), 6),
        }

        for col in metric_cols:
            summary[col] = {
                "mean": round(float(df[col].mean()), 6),
                "std": round(float(df[col].std(ddof=0)), 6),
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
        prediction_path = (
            self.output_dir
            / f"{model_name}_predictions.csv"
        )

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
        self._extracted_from_save_results_98(
            model_name, '_cv_summary.json', summary, "CV summary saved to %s"
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
                    (cal_df["raw_brier_score"] - cal_df["brier_score"]).dropna()
                )

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
                        "mean": round(float(improvements.mean()), 6) if len(improvements) else None,
                        "std": round(float(improvements.std(ddof=0)), 6) if len(improvements) else None,
                        "n_folds_improved": int((improvements > 0).sum()),
                        "n_folds_worsened": int((improvements < 0).sum()),
                    }

                self._extracted_from_save_results_98(
                    model_name,
                    '_calibration_summary.json',
                    calibration_summary,
                    "Calibration summary saved to %s",
                )

    # TODO Rename this here and in `save_results`
    def _extracted_from_save_results_98(self, model_name, arg1, arg2, arg3):
        cal_summary_path = self.output_dir / f"{model_name}{arg1}"
        with open(cal_summary_path, "w", encoding="utf-8") as f:
            json.dump(arg2, f, indent=4)
        logger.info(arg3, cal_summary_path)

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