"""Regression tests for temporal preprocessing semantics."""

import pandas as pd
from src.features.lm_features import LMFeatureEngineer


def test_prepare_dataset_separates_filing_and_reporting_year() -> None:
    frame = pd.DataFrame(
        {
            "cik": ["1750"],
            "filing_date": ["15-03-2020"],
            "reporting_date": ["31-12-2019"],
        }
    )

    result = LMFeatureEngineer().prepare_dataset(frame)

    assert result.loc[0, "filing_year"] == 2020
    assert result.loc[0, "reporting_year"] == 2019
