# Model Card: Corporate Misconduct Warning

## Status

This is a research ranking project, not a production fraud score. Trial 196's reported development Recall@5% of 26.22% is invalidated because reporting-year folds admitted later filings into training.

The already-observed 2019-2022 result—Recall@5% 12.07% (7 of 58), Precision@5% 0.96%, and Brier skill -0.0201—is historical and nonconfirmatory. It belongs to the invalidated selection pipeline and is permanently closed to reruns.

## Intended use

The project tests whether filing language can help a human analyst prioritize a fixed 5% review queue. It must not drive enforcement, compliance, investment, or automated company-level decisions.

## Corrected evaluation contract

- `filing_date` alone determines temporal membership.
- `reporting_year` is separate metadata and cannot determine folds.
- Every fold must satisfy `max(training filing_date) < min(test filing_date)`.
- Recall@5% is primary; Precision@5% is reported alongside it.
- Positive mean Brier skill is required; mean PR-AUC breaks ties.
- Classification metrics use a fixed 0.50 threshold.
- New artifacts are bound to development-data and configuration hashes.
- `no_eligible_candidate` is a valid result and produces no model or SHAP artifact.

## Label limitations

Recorded misconduct labels arrive late. The historical audit observed a median delay of 1,236.5 days, with 39 of 58 final-period positives appearing after 2022. AAER `dateTime` is treated only as a source-record timestamp, not an enforcement or publication date.

## Portable demonstration

The demo reads tracked historical evidence and never trains, tunes, scores, or opens the historical final set:

```powershell
.\demo\open_demo.ps1
python demo\generate_xgboost_demo.py --check
```

Evidence:

- [historical evidence](../demo/artifacts/historical_evidence.json)
- [historical SHAP importance](../demo/artifacts/shap_importance.csv)
- [temporal validation audit](audits/2026-08-17-temporal-validation-audit.md)

## Next permitted step

Rebuild development features and pass the temporal audit. A new development-only Optuna/candidate-review run requires separate approval. SHAP requires approval after selection. The historical 2019-2022 period will not be evaluated again.
