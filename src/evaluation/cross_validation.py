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
from typing import Any

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

    # ============================================================
    # Fold Generation
    # ============================================================

    def generate_folds(
        self,
        years: np.ndarray,
        y: np.ndarray,
    ):
        """
        Yield (train_idx, test_idx, test_year) tuples.

        Train set  : all indices where year < test_year
        Test set   : all indices where year == test_year
        Skip       : test folds with fraud_count < min_fraud_per_fold
        """

        unique_years = sorted(np.unique(years))

        for test_year in unique_years:

            train_idx = np.where(years < test_year)[0]
            test_idx  = np.where(years == test_year)[0]

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
    ) -> float:
        """
        Find the threshold that maximizes F1 score.
        Returns
        -------
        float
            Optimal decision threshold.
        """

        thresholds = np.arange(
            0.01,
            1.00,
            0.01,
        )

        best_threshold = 0.050
        best_f1 = -1.0

        for threshold in thresholds:

            y_pred = (
                y_score >= threshold
            ).astype(int)

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
    ) -> dict[str, Any]:
        """
        Fit estimator on train split, evaluate on test split.

        Returns a dict of metrics for this fold.
        """

        import sklearn.base as skbase

        # Clone to avoid state leakage between folds
        model = skbase.clone(estimator)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        model.fit(X_train, y_train)

        # ---------- TRAINING PROBABILITIES ----------

        if hasattr(model, "predict_proba"):

            train_score = model.predict_proba(
                X_train
            )[:, 1]

            test_score = model.predict_proba(
                X_test
            )[:, 1]

        elif hasattr(model, "decision_function"):

            train_score = model.decision_function(
                X_train
            )

            test_score = model.decision_function(
                X_test
            )

        else:

            raise AttributeError(
                "Estimator must implement predict_proba() "
                "or decision_function()."
            )
        
        # ---------- AUTOMATIC THRESHOLD ----------
        
        best_threshold = self.find_best_threshold(
            y_train,
            train_score,
        )

        logger.info(
            "Year %d | Optimal Threshold = %.2f",
            test_year,
            best_threshold,
        )

        y_pred = (
            test_score >= best_threshold
        ).astype(int)

        y_score = test_score

        roc_auc  = roc_auc_score(y_test, y_score)
        pr_auc   = average_precision_score(y_test, y_score)
        f1       = f1_score(y_test, y_pred, zero_division=0)
        prec     = precision_score(y_test, y_pred, zero_division=0)
        rec      = recall_score(y_test, y_pred, zero_division=0)
        mcc      = matthews_corrcoef(y_test, y_pred)
        bal_acc  = balanced_accuracy_score(y_test, y_pred)
        brier    = brier_score_loss(y_test, y_score)

        prediction_df = pd.DataFrame(
            {
                "test_year": test_year,
                "true_label": y_test,
                "predicted_probability": y_score,
                "predicted_label": y_pred,
            }
        )

        self.fold_predictions.append(prediction_df)

        return {
            "test_year": test_year,
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "test_fraud_n": int(y_test.sum()),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "mcc": mcc,
            "balanced_acc": bal_acc,
            "brier_score": brier,
        }

    # ============================================================
    # Aggregation
    # ============================================================

    def aggregate_metrics(self) -> dict[str, Any]:
        """
        Compute mean and std across all folds for each metric.
        """

        if not self.fold_results:
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
        }

        for col in metric_cols:
            summary[col] = {
                "mean": round(float(df[col].mean()), 6),
                "std":  round(float(df[col].std()),  6),
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

        summary = self.aggregate_metrics()
        summary_path = self.output_dir / f"{model_name}_cv_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info("CV summary saved to %s", summary_path)

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

        for train_idx, test_idx, test_year in self.generate_folds(years, y):

            fold_metrics = self.evaluate_fold(
                estimator=estimator,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                test_year=test_year,
                decision_threshold=decision_threshold,
            )

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