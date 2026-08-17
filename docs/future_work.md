# Future Work Backlog

## Evaluation rules

- Rebuild development features and pass the strict temporal audit before modeling.
- Use actual filing-date folds for every feature-family comparison.
- Fit transformations only on information available before each test filing period.
- Track Recall@5%, Precision@5%, PR-AUC, Brier skill, and ECE.
- Do not select features from in-sample or threshold-tuned F1.
- Keep the historical 2019-2022 period permanently closed.
- Require separate approval for Optuna, candidate review, XGBoost/GPU jobs, and SHAP.

## Candidate research directions

- Issuer-history changes in tone, uncertainty, readability, and filing length.
- Rolling issuer baselines and peer-relative language abnormalities.
- Language interactions and filing-size-normalized tone measures.
- Point-in-time financial features only when their availability can be verified.

Beneish M-Score work remains out of scope.
