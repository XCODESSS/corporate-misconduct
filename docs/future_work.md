# Future Work Backlog

This document captures proposed improvements before they are scheduled. Items
should be evaluated with walk-forward validation and retained only when they
improve out-of-time performance or operational usefulness.

## User Ideas

### Model output and tuning experience

- Keep Optuna output minimal during repeated trials.
- Show progress as completed trials out of the configured total, with an ETA.
- Show the detailed model summary only after the final evaluation completes.

### XGBoost quality and maintainability

- Compute SHAP values once and reuse them for all SHAP artifacts.
- Use a class-imbalance-aware XGBoost baseline and tune its weight rather than
  assuming one value is always best.
- Keep model feature names in shared configuration.
- Reduce duplicated model-construction code.
- Treat SHAP as an optional dependency.
- Consider a shared base trainer as model pipelines converge.

## Feature Ideas

### Text structure and readability

- Filing length, sentence length, word complexity, numeric density, and section
  proportions.
- Readability scores and changes in writing style.

### Issuer history

- Prior-year and multi-year changes in tone, uncertainty, readability, and
  filing length for the same CIK.
- Peer-relative or issuer-relative abnormal language measures.

### Language interactions

- Negative-to-positive tone ratio.
- Uncertainty multiplied by negative or litigious language.
- Tone measures normalized by filing size and compared with an issuer baseline.

### Financial risk

- Profitability, leverage, liquidity, cash-flow, accrual, growth, and
  Beneish-style indicators when point-in-time financial data is available.

### Semantic features

- Document or section embeddings, topic features, and dimensionality-reduced
  semantic representations.

## Evaluation Rules

- Use the existing walk-forward folds for every feature-family comparison.
- Compare each new feature family with an ablation study.
- Track PR-AUC, calibration (Brier/ECE), and a pre-defined operating point.
- Fit transformations using only information available before each test year.
- Do not adopt a feature family based solely on in-sample or threshold-tuned F1.
