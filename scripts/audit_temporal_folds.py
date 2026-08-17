"""Audit development features for year drift and temporal fold overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(path: Path) -> dict[str, int | float | str]:
    frame = pd.read_parquet(
        path, columns=["filing_date", "reporting_date", "filing_year", "fraudulent"]
    )
    dates = pd.to_datetime(frame["filing_date"], dayfirst=True, errors="coerce")
    if dates.isna().any():
        return {
            "rows": len(frame),
            "invalid_filing_dates": int(dates.isna().sum()),
            "status": "fail",
        }
    actual_years = dates.dt.year
    mismatch = int((actual_years != frame["filing_year"]).sum())
    evaluated = 0
    overlaps = 0
    max_overlap = 0
    for year in sorted(actual_years.unique()):
        train = frame.loc[actual_years < year]
        test = frame.loc[actual_years == year]
        if train.empty or int(test["fraudulent"].sum()) < 30:
            continue
        evaluated += 1
        test_start = dates.loc[test.index].min()
        overlap_count = int((dates.loc[train.index] >= test_start).sum())
        overlaps += int(overlap_count > 0)
        max_overlap = max(max_overlap, overlap_count)
    status = "pass" if mismatch == 0 and overlaps == 0 else "fail"
    return {
        "rows": len(frame),
        "filing_year_mismatch_count": mismatch,
        "filing_year_mismatch_rate": mismatch / len(frame),
        "evaluated_folds": evaluated,
        "folds_with_temporal_overlap": overlaps,
        "max_overlapping_train_rows": max_overlap,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    result = audit(parser.parse_args().input)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
