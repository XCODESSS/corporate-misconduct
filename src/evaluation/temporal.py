"""Strict filing-time utilities used by development evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_filing_dates(values: pd.Series | np.ndarray) -> np.ndarray:
    """Parse filing dates and reject every missing or malformed value."""

    parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
    missing = int(pd.isna(parsed).sum())
    if missing:
        raise ValueError(f"invalid filing date count: {missing}")
    return np.asarray(parsed, dtype="datetime64[D]")


def filing_years(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return calendar years derived only from actual filing dates."""

    return pd.DatetimeIndex(parse_filing_dates(values)).year.to_numpy(dtype=np.int32)


def assert_strict_fold_order(
    filing_dates: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Reject a fold when any training filing is not earlier than the test fold."""

    dates = parse_filing_dates(filing_dates)
    if (
        len(train_idx)
        and len(test_idx)
        and dates[train_idx].max() >= dates[test_idx].min()
    ):
        raise ValueError("temporal overlap detected between training and test filings")
