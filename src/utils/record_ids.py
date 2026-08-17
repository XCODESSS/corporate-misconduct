"""Stable non-label identifiers for filing-level audit records."""

from __future__ import annotations

from typing import Any

import pandas as pd
from src.utils.fingerprints import sha256_json


def stable_filing_id(
    cik: Any, filing_date: Any, reporting_date: Any, filing_type: Any
) -> str:
    parsed_filing = pd.to_datetime(
        filing_date, format="mixed", dayfirst=True, errors="raise"
    )
    parsed_reporting = pd.to_datetime(
        reporting_date, format="mixed", dayfirst=True, errors="raise"
    )
    payload = {
        "cik": str(cik).removesuffix(".0").zfill(10),
        "filing_date": parsed_filing.date().isoformat(),
        "reporting_date": parsed_reporting.date().isoformat(),
        "filing_type": str(filing_type).strip().upper(),
    }
    return sha256_json(payload)
