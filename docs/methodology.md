# Methodology

## Research question

Can language features rank recorded misconduct cases into the highest-scoring 5% of corporate filings, representing a fixed analyst-review budget?

## Time and data availability

`filing_date` is the only field allowed to determine temporal folds. `reporting_year` describes the accounting period and may support source lookup, but it cannot control training or evaluation membership. Every development fold must satisfy:

```text
max(training filing_date) < min(test filing_date)
```

The automated audit rejects malformed filing dates, stored-year drift, and temporal overlap.

## Metrics and selection

- Recall@5% is primary and measures how many recorded positives appear in the fixed review queue.
- Precision@5% is reported alongside recall.
- Positive mean Brier skill is an eligibility gate.
- Mean PR-AUC is the tie-breaker.
- Classification metrics use the fixed 0.50 threshold; in-sample threshold optimization is prohibited.
- `no_eligible_candidate` is a valid result and creates no model or SHAP artifact.

## Calibration and provenance

Calibration partitions are chronological and use only development data. New candidate reviews embed hashes of the development feature file, ordered feature schema, calibration configuration, and selection policy. A mismatch blocks artifact reuse.

## Historical evaluation status

The Trial 196 development folds used reporting years and are invalidated. The 2019-2022 evaluation was already opened under that selection pipeline, so it is retained only as historical evidence and is permanently closed to reruns. A corrected model may report development results only until a genuinely new, preregistered, label-mature holdout exists.

## Label semantics

AAER `dateTime` is treated only as an AAER source-record timestamp. It is not asserted to be an enforcement or publication date, and absence of a recorded label does not establish absence of misconduct.
