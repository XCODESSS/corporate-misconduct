"""
Module: calibration.

Responsibilities
----------------
- Calibrate predicted probabilities from an sklearn-compatible
  classifier using Platt scaling (sigmoid) or isotonic regression.
- Evaluate calibration quality via Brier score and reliability
  (calibration) curve data.

This module DOES NOT

- perform walk-forward splitting (see cross_validation.py)
- perform hyperparameter tuning (see tuning/*.py)
- perform threshold optimization (see cross_validation.find_best_threshold)
- compute ranking metrics such as ROC-AUC / PR-AUC (see metrics.py)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.utils.logger import get_logger

logger = get_logger(__name__)

CalibrationMethod = Literal["sigmoid", "isotonic"]


class ProbabilityCalibrator:
    """
    Wraps an UNFITTED sklearn-compatible estimator with
    CalibratedClassifierCV to produce calibrated probability estimates.

    Fit ONLY on training-fold data. CalibratedClassifierCV performs its
    own internal cross-validation on the training fold to generate
    out-of-fold predictions for fitting the calibration map, then
    refits the base estimator on the full training fold. Test-fold
    data is never seen during fit(), so this is safe to use inside
    WalkForwardCV without introducing leakage.

    Parameters
    ----------
    method : "sigmoid" | "isotonic"
        "sigmoid"   -> Platt scaling. Parametric, stable on small folds.
        "isotonic"  -> Isotonic regression. More flexible, needs more
                       positive samples per fold to avoid overfitting.
    cv : int
        Number of internal folds used to generate out-of-fold
        predictions for calibration. Reduced automatically if the
        training fold does not have enough positive samples.
    """

    def __init__(
        self,
        method: CalibrationMethod = "sigmoid",
        cv: int = 5,
    ) -> None:
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unsupported calibration method: {method}")

        self.method = method
        self.cv = cv
        self.calibrated_model: CalibratedClassifierCV | None = None
        self._fallback_model: Any = None

    def fit(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrated classifier on training-fold data only.

        `estimator` must be UNFITTED. Do not call estimator.fit()
        before passing it in — CalibratedClassifierCV owns fitting.
        """

        n_pos = int(y_train.sum())
        n_splits = min(self.cv, n_pos)

        if n_splits < 2:
            logger.warning(
                "Only %d positive samples in training fold; "
                "cannot run %d-fold calibration. "
                "Falling back to uncalibrated estimator.",
                n_pos,
                self.cv,
            )
            estimator.fit(X_train, y_train)
            self._fallback_model = estimator
            self.calibrated_model = None
            return self

        if n_splits < self.cv:
            logger.warning(
                "Reducing calibration folds from %d to %d "
                "due to limited positive samples (%d) in training fold.",
                self.cv,
                n_splits,
                n_pos,
            )

        self.calibrated_model = CalibratedClassifierCV(
            estimator,
            method=self.method,
            cv=n_splits,
        )
        self.calibrated_model.fit(X_train, y_train)
        self._fallback_model = None

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability of the positive class."""

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]

        if self._fallback_model is not None:
            return self._fallback_model.predict_proba(X)[:, 1]

        raise RuntimeError("ProbabilityCalibrator.fit() was not called.")


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration
    Error (MCE) using equal-width bins over [0, 1].

    ECE : weighted average of |accuracy - confidence| across bins,
          weighted by the fraction of samples in each bin.
    MCE : the single worst |accuracy - confidence| gap across bins.

    Both are 0.0 for a perfectly calibrated model. Bins with zero
    samples are skipped (they contribute nothing to ECE and are
    ignored for MCE).
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())

        if count == 0:
            continue

        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        gap = abs(bin_acc - bin_conf)

        ece += (count / n) * gap
        mce = max(mce, gap)

    return {
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
    }


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute Brier score, ECE/MCE, and reliability-curve data for a
    set of predicted probabilities.

    Returns
    -------
    dict with:
        brier_score       : float
        ece                : float, Expected Calibration Error
        mce                : float, Maximum Calibration Error
        calibration_curve  : {"prob_true": [...], "prob_pred": [...]}
            prob_true : observed fraud rate within each probability bin
            prob_pred : mean predicted probability within each bin
            A well-calibrated model has prob_true ~= prob_pred.
    """

    brier = brier_score_loss(y_true, y_prob)
    error_metrics = compute_ece(y_true, y_prob, n_bins=n_bins)

    try:
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy="uniform",
        )
    except ValueError:
        logger.warning(
            "Could not compute calibration curve (likely a single-class "
            "test fold); returning empty curve."
        )
        prob_true, prob_pred = np.array([]), np.array([])

    return {
        "brier_score": round(float(brier), 6),
        "ece": error_metrics["ece"],
        "mce": error_metrics["mce"],
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


def compare_calibration(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
) -> dict[str, Any]:
    """
    Compare Brier score before vs after calibration.

    Positive `brier_improvement` means calibration reduced Brier score
    (better). Negative means calibration made probabilities worse for
    this fold — this can legitimately happen on small/noisy folds and
    should be tracked, not hidden.
    """

    raw_brier = brier_score_loss(y_true, y_prob_raw)
    calibrated_brier = brier_score_loss(y_true, y_prob_calibrated)

    return {
        "raw_brier_score": round(float(raw_brier), 6),
        "calibrated_brier_score": round(float(calibrated_brier), 6),
        "brier_improvement": round(float(raw_brier - calibrated_brier), 6),
    }