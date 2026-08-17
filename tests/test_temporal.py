"""Strict temporal-order tests."""

import numpy as np
import pandas as pd
import pytest
from scripts.audit_temporal_folds import audit
from src.evaluation.temporal import (
    assert_strict_fold_order,
    filing_years,
    parse_filing_dates,
)


def test_filing_years_come_from_actual_filing_dates() -> None:
    assert filing_years(pd.Series(["15-03-2020"])).tolist() == [2020]


def test_invalid_filing_date_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid filing date"):
        parse_filing_dates(pd.Series(["31-12-2018", "not-a-date"]))


def test_strict_fold_order_rejects_overlap() -> None:
    dates = np.array(["2018-03-01", "2019-02-01", "2018-12-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="temporal overlap"):
        assert_strict_fold_order(dates, np.array([0, 1]), np.array([2]))


def test_audit_rejects_reporting_year_as_filing_year(tmp_path) -> None:
    path = tmp_path / "development.parquet"
    pd.DataFrame(
        {
            "filing_date": ["15-03-2020"],
            "reporting_date": ["31-12-2019"],
            "filing_year": [2019],
            "fraudulent": [1],
        }
    ).to_parquet(path)
    result = audit(path)
    assert result["filing_year_mismatch_count"] == 1
    assert result["status"] == "fail"
