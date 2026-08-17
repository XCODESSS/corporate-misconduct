"""Regression tests for model-selection and sealed-test safeguards."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from src.models.xgboost_model import XGBoostBaseline


def test_probability_study_keeps_fixed_scale_pos_weight() -> None:
    """The final model must match the v2 candidate-review configuration."""

    pipeline = XGBoostBaseline()
    pipeline.y = np.array([0, 0, 0, 1], dtype=np.int8)

    assert (
        pipeline._default_parameters()["scale_pos_weight"]
        == pipeline.PROBABILITY_SCALE_POS_WEIGHT
        == 1.0
    )


def test_final_test_artifacts_cannot_be_overwritten() -> None:
    """A previously opened sealed test set must not be evaluated again."""

    pipeline = XGBoostBaseline()
    summary_path = Mock()
    predictions_path = Mock()
    calibration_path = Mock()
    summary_path.name = "final_test_summary.json"
    predictions_path.name = "final_test_predictions.csv"
    calibration_path.name = "final_test_calibration.json"
    for path in (summary_path, predictions_path, calibration_path):
        path.exists.return_value = False
    pipeline._final_test_artifact_paths = lambda: (
        summary_path,
        predictions_path,
        calibration_path,
    )

    pipeline._ensure_final_test_has_not_run()

    summary_path.exists.return_value = True

    with pytest.raises(RuntimeError, match="cannot be rerun"):
        pipeline._ensure_final_test_has_not_run()


def test_final_test_period_uses_filing_dates_not_fiscal_years() -> None:
    """Fiscal-year values may differ from the date used for the temporal split."""

    pipeline = XGBoostBaseline()
    test_dataset = pd.DataFrame(
        {
            "filing_date": ["31-12-2018", "01-01-2019", "31-12-2022", "01-01-2023"],
            "filing_year": [2018, 2000, 2022, 2023],
        }
    )

    selected, filing_dates = pipeline._select_final_test_period(test_dataset)

    assert selected["filing_year"].tolist() == [2000, 2022]
    assert filing_dates.dt.year.tolist() == [2019, 2022]
