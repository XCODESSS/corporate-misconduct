# Temporal Validation Audit — 2026-08-17

## Conclusion

The historical Trial 196 development selection is invalidated. Feature engineering derived `filing_year` from `reporting_date`, so the purported walk-forward folds were not strictly ordered by the date each filing became available.

## Measured pre-fix evidence

- Development rows: 41,748.
- Rows whose stored year differed from actual filing year: 30,582 (73.2538%).
- Evaluated folds containing training filings on or after the first test filing: 21 of 21.
- Largest overlapping training count in one fold: 2,111.
- Historical development Recall@5% of 26.22% must not be used as evidence of out-of-time performance.
- The historical 2019-2022 result belongs to that invalidated selection pipeline. It is nonconfirmatory and permanently closed to reruns.

## Repair invariant

Every replacement development fold must satisfy:

```text
max(training filing_date) < min(test filing_date)
```

Run the development-only audit after rebuilding development features:

```powershell
python scripts/audit_temporal_folds.py --input data/processed/features/trainval_features.parquet
```

Modeling remains blocked unless the audit reports zero year mismatches and zero overlapping folds.
