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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from src.utils.logger import get_logger

logger = get_logger(__name__)

CalibrationMethod = Literal["sigmoid", "isotonic"]
CalibrationStrategy = Literal["cross_validation", "chronological_holdout"]
DEFAULT_CALIBRATION_BINS = 10
MINIMUM_CALIBRATION_SPLITS = 2
DEFAULT_CALIBRATION_HOLDOUT_FRACTION = 0.20
MINIMUM_PROBABILITY = 0.0
MAXIMUM_PROBABILITY = 1.0


class ProbabilityCalibrator:
    """
    Wraps an UNFITTED sklearn-compatible estimator with
    CalibratedClassifierCV to produce calibrated probability estimates.

    Fit ONLY on training-fold data. It supports either random internal
    cross-validation or a chronological holdout taken from the end of the
    training fold. Test-fold data is never seen during fit().

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
    strategy : "cross_validation" | "chronological_holdout"
        "chronological_holdout" learns a calibration map from a later
        training-fold period, then refits the base estimator on all training
        data before scoring the outer validation period.
    """

    def __init__(
        self,
        method: CalibrationMethod = "sigmoid",
        cv: int = 5,
        strategy: CalibrationStrategy = "cross_validation",
        holdout_fraction: float = DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
    ) -> None:
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unsupported calibration method: {method}")
        if cv < MINIMUM_CALIBRATION_SPLITS:
            raise ValueError(
                f"cv must be at least {MINIMUM_CALIBRATION_SPLITS}; received {cv}."
            )
        if strategy not in ("cross_validation", "chronological_holdout"):
            raise ValueError(f"Unsupported calibration strategy: {strategy}")
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must lie in (0, 1).")

        self.method = method
        self.cv = cv
        self.strategy = strategy
        self.holdout_fraction = holdout_fraction
        self.calibrated_model: CalibratedClassifierCV | None = None
        self._fallback_model: Any = None
        self._full_model: Any = None
        self._probability_mapper: Any = None

    def fit(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        years: np.ndarray | None = None,
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrated classifier on training-fold data only.

        `estimator` must be UNFITTED. Do not call estimator.fit()
        before passing it in — CalibratedClassifierCV owns fitting.
        """

        _validate_binary_labels(y_train)
        self._full_model = None
        self._probability_mapper = None
        if self.strategy == "chronological_holdout":
            return self._fit_chronological_holdout(
                estimator,
                X_train,
                y_train,
                years,
            )

        positive_count = int(y_train.sum())
        negative_count = len(y_train) - positive_count
        n_splits = min(self.cv, positive_count, negative_count)

        if n_splits < MINIMUM_CALIBRATION_SPLITS:
            logger.warning(
                "Training fold has %d positive and %d negative samples; "
                "cannot run %d-fold calibration. "
                "Falling back to uncalibrated estimator.",
                positive_count,
                negative_count,
                self.cv,
            )
            estimator.fit(X_train, y_train)
            self._fallback_model = estimator
            self.calibrated_model = None
            return self
        if n_splits < self.cv:
            logger.warning(
                "Reducing calibration folds from %d to %d "
                "due to the smallest class having %d samples in the training fold.",
                self.cv,
                n_splits,
                min(positive_count, negative_count),
            )

        self.calibrated_model = CalibratedClassifierCV(
            estimator,
            method=self.method,
            cv=n_splits,
        )
        self.calibrated_model.fit(X_train, y_train)
        self._fallback_model = None
        return self

    def _fit_chronological_holdout(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        years: np.ndarray | None,
    ) -> "ProbabilityCalibrator":
        if years is None:
            raise ValueError("Chronological calibration requires training-fold years.")

        fit_idx, calibration_idx = self._chronological_holdout_indices(years)
        if not self._has_both_classes(y_train[fit_idx]) or not self._has_both_classes(
            y_train[calibration_idx]
        ):
            logger.warning(
                "Chronological calibration split has a single-class partition; "
                "falling back to an uncalibrated estimator."
            )
            estimator.fit(X_train, y_train)
            self._fallback_model = estimator
            self.calibrated_model = None
            self._full_model = None
            return self

        calibration_model = clone(estimator).fit(
            X_train[fit_idx],
            y_train[fit_idx],
        )
        calibration_scores = self._predict_probability_scores(
            calibration_model,
            X_train[calibration_idx],
        )
        self._probability_mapper = self._fit_probability_mapper(
            calibration_scores,
            y_train[calibration_idx],
        )
        self._full_model = clone(estimator).fit(X_train, y_train)
        self.calibrated_model = None
        self._fallback_model = None
        return self

    def _chronological_holdout_indices(
        self,
        years: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        years = np.asarray(years)
        if years.ndim != 1 or len(years) < 2:
            raise ValueError("Training-fold years must be a one-dimensional array.")

        ordered_years = np.sort(years, kind="stable")
        split_position = max(1, int(np.ceil(len(years) * (1 - self.holdout_fraction))))
        split_position = min(split_position, len(years) - 1)
        calibration_start_year = ordered_years[split_position]
        fit_idx = np.flatnonzero(years < calibration_start_year)
        calibration_idx = np.flatnonzero(years >= calibration_start_year)
        if len(fit_idx) == 0 or len(calibration_idx) == 0:
            raise ValueError("Chronological calibration split cannot be empty.")
        return fit_idx, calibration_idx

    @staticmethod
    def _has_both_classes(labels: np.ndarray) -> bool:
        return len(np.unique(labels)) == 2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability of the positive class."""

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]

        if self._fallback_model is not None:
            return self._fallback_model.predict_proba(X)[:, 1]

        if self._full_model is not None and self._probability_mapper is not None:
            scores = self._predict_probability_scores(self._full_model, X)
            return self._map_probabilities(scores)

        raise RuntimeError("ProbabilityCalibrator.fit() was not called.")

    @staticmethod
    def _predict_probability_scores(model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def _fit_probability_mapper(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Any:
        if self.method == "sigmoid":
            mapper = LogisticRegression(C=1e6, solver="lbfgs")
            mapper.fit(scores.reshape(-1, 1), labels)
            return mapper

        mapper = IsotonicRegression(out_of_bounds="clip")
        mapper.fit(scores, labels)
        return mapper

    def _map_probabilities(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return self._probability_mapper.predict_proba(scores.reshape(-1, 1))[:, 1]
        return self._probability_mapper.predict(scores)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
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

    y_true, y_prob = _validate_calibration_inputs(y_true, y_prob, n_bins)

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
    n_bins: int = DEFAULT_CALIBRATION_BINS,
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

    y_true, y_prob = _validate_calibration_inputs(y_true, y_prob, n_bins)
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

    y_true, y_prob_raw = _validate_calibration_inputs(
        y_true,
        y_prob_raw,
        DEFAULT_CALIBRATION_BINS,
    )
    _, y_prob_calibrated = _validate_calibration_inputs(
        y_true,
        y_prob_calibrated,
        DEFAULT_CALIBRATION_BINS,
    )
    raw_brier = brier_score_loss(y_true, y_prob_raw)
    calibrated_brier = brier_score_loss(y_true, y_prob_calibrated)

    return {
        "raw_brier_score": round(float(raw_brier), 6),
        "calibrated_brier_score": round(float(calibrated_brier), 6),
        "brier_improvement": round(float(raw_brier - calibrated_brier), 6),
    }


def _validate_calibration_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_bin_count(n_bins)
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob)
    _validate_matching_non_empty_arrays(labels, probabilities)
    _validate_binary_labels(labels)
    _validate_probability_range(probabilities)
    return labels, probabilities


def _validate_bin_count(n_bins: int) -> None:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")


def _validate_matching_non_empty_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    if len(labels) == 0:
        raise ValueError("Calibration inputs cannot be empty.")
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length.")


def _validate_binary_labels(labels: np.ndarray) -> None:
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels (0 and 1).")


def _validate_probability_range(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob must contain only finite values.")
    if np.any(probabilities < MINIMUM_PROBABILITY) or np.any(
        probabilities > MAXIMUM_PROBABILITY
    ):
        raise ValueError("y_prob values must be between 0.0 and 1.0.")
