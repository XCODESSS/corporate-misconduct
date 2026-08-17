# Model Card: Corporate Misconduct Warning

## Summary

This is a research-model demonstration of a leakage-aware XGBoost ranking
pipeline. It ranks corporate filings into a fixed 5% analyst-review queue; it
does not provide a production fraud score or an automated enforcement decision.

The selected development candidate was Trial 196 of
`xgboost_lm_text_surface_probability_v2`. It achieved mean development
Recall@5% of 26.22% in chronological walk-forward validation. In the frozen
2019-2022 evaluation, the same fixed 5% queue found 7 of 58 recorded cases
(12.07%) and had Brier skill of -0.0201. The later result does not support a
claim that its probabilities are calibrated or deployment-ready.

## Intended and non-intended use

**Intended use:** reproducible research, historical ranking analysis, and a
portfolio walkthrough of temporal model evaluation.

**Not intended for:** production fraud decisions, compliance triage, investment
decisions, automated enforcement, or estimating a real-world probability that a
specific company committed misconduct.

The model may surface misleading correlations, miss actual misconduct, and
reflect limitations in public enforcement labels. A human analyst and an
appropriate governance process would be required for any consequential use.

## Data and features

The experiment uses filing-level records with a binary `fraudulent` label. Its
23 features comprise:

- seven Loughran-McDonald language densities: negative, positive, uncertainty,
  litigious, weak-modal, strong-modal, and constraining;
- fifteen Management Discussion & Analysis text-surface features, including
  length, readability, digit/punctuation/uppercase ratios, word complexity,
  and lexical diversity; and
- a text-availability indicator.

The precise ordered feature schema and fingerprint are stored in the
[experiment manifest](../reports/models/xgboost_lm_text_surface_probability_v2/experiment_manifest.json).

## Temporal evaluation and selection

Development evaluation uses 21 chronological walk-forward folds, with test
years 1997-2017. Calibration uses a chronological holdout: it learns a
probability map on holdout scores while refitting the ranking model on the full
outer training fold. This prevents the calibrator from changing the model used
for ranking.

The selection rule was set before the sealed evaluation:

1. positive mean Brier skill score is required;
2. among eligible candidates, maximize mean Recall@5%; and
3. use mean PR-AUC as a tie-breaker.

Recall@5% ranks a period's filings by model score and sends the highest-scoring
5% to review. It then measures the share of recorded positive cases that occur
in that queue. It is useful for a fixed review-capacity exercise, but it is not
a probability-calibration claim.

## Results

| Evaluation | Recall@5% | Supporting evidence |
| --- | ---: | --- |
| Development (Trial 196, 21 folds) | 26.22% | Precision@5% 14.46%; PR-AUC 0.1495; Brier skill 0.0213 |
| Frozen final period (2019-2022) | 12.07% (7/58 cases) | 730 of 14,582 filings reviewed; precision@5% 0.96%; PR-AUC 0.0252; Brier skill -0.0201 |

The negative frozen-period Brier skill means predicted probabilities did not
outperform the naive fraud-rate baseline. Therefore, the project reports a
research result rather than an operational scoring claim.

## Label maturity limitation

The final-period audit observed a median label delay of 1,236.5 days, and 39 of
58 linked positive labels were recorded after 2022. The audit cannot establish
that the absence of a label means the absence of misconduct. It explicitly does
not certify any year as fully mature. Interpret the frozen-period result as an
evaluation on recorded labels with material maturity uncertainty.

## Explainability

The static demo displays mean absolute SHAP values from the selected development
model. They describe feature contribution patterns in that fitted model; they do
not establish causation, prove that a filing is fraudulent, or resolve the
calibration and label-delay limitations.

## Reproducibility

The demo is artifact-only and never invokes Optuna, XGBoost training, candidate
review, or the sealed final-test evaluation.

```powershell
# Generate and open the static page (no training)
.\demo\open_demo.ps1

# Validate its inputs without writing HTML
python demo\generate_xgboost_demo.py --check

# Run the fast demo checks
python -m pytest tests\test_demo.py -q
```

Source evidence:

- [candidate review](../reports/models/xgboost_lm_text_surface_probability_v2/candidate_review_full_refit.json)
- [frozen-test summary](../reports/models/xgboost_lm_text_surface_probability_v2/final_test/final_test_summary.json)
- [label-maturity audit](../reports/label_maturity/label_maturity_summary.json)
- [SHAP importance](../reports/models/xgboost_lm_text_surface_probability_v2/shap/shap_importance.csv)

## Next research direction

Before any new test evaluation, implement issuer-history language features with
strict no-lookahead construction: prior-year deltas, rolling issuer baselines,
and issuer- or industry-relative language abnormalities. Evaluate that new
feature family on development folds only. Beneish M-Score, hyperparameter
searches, and tuning against the frozen final period remain out of scope.
