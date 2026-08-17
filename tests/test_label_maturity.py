from __future__ import annotations

import pandas as pd
from src.evaluation.label_maturity import LabelMaturityAudit


def test_mixed_aaer_timestamps_and_cik_normalization_are_parsed() -> None:
    timestamps = LabelMaturityAudit.parse_source_record_timestamps(
        pd.Series(["2024-12-19T21:08:41", "2017-01-18T08:24:41-05:00"])
    )

    assert timestamps.notna().all()
    assert timestamps.dt.year.tolist() == [2024, 2017]
    assert LabelMaturityAudit.normalize_cik("1750.0") == "0000001750"


def test_source_timestamp_attachment_uses_exact_fraud_period_key() -> None:
    filings = pd.DataFrame(
        {
            "cik_key": ["0000001750"],
            "matched_fraud_start": [pd.Timestamp("2015-11-01")],
            "matched_fraud_end": [pd.Timestamp("2020-02-29")],
            "filing_date": [pd.Timestamp("2019-07-18")],
        }
    )
    periods = pd.DataFrame(
        {
            "cik_key": ["0000001750"],
            "fraud_start_key": [pd.Timestamp("2015-11-01")],
            "fraud_end_key": [pd.Timestamp("2020-02-29")],
            "aaer_source_record_first": [pd.Timestamp("2024-12-19")],
            "aaer_source_record_last": [pd.Timestamp("2024-12-19")],
            "source_record_count": [1],
        }
    )

    attached = LabelMaturityAudit.attach_source_timestamps(filings, periods)

    assert attached.loc[0, "observed_label_lag_days"] == 1981
