# Corporate Misconduct Warning

Leakage-aware XGBoost research pipeline for ranking corporate filings into a fixed misconduct-review queue. The project uses chronological walk-forward validation, probability calibration, Recall@5% as its operational metric, and a one-time sealed final evaluation.

## Research-demo result

The selected development candidate, Trial 196, achieved mean Recall@5% of 26.22% across 21 walk-forward folds. On the frozen 2019-2022 final period, it found 7 of 58 recorded cases in a 5% review queue (Recall@5% 12.07%), but its Brier skill was -0.0201. The model is therefore a research prototype, not a deployable probability model.

Generate the local static demo without training or reopening the final test:

```powershell
.\demo\open_demo.ps1
```

The demo is artifact-driven and opens in seconds. It shows the selection rule,
development versus final results, Recall@5% definition, SHAP feature ranking,
and the observed label-delay limitation. Read the full
[model card](docs/model_card.md) before interpreting the result.

## Why the timeline matters

Random validation can leak future context into a historical-looking score.
This project trains only on earlier filings and evaluates later periods through
walk-forward folds, then uses a one-time frozen 2019-2022 evaluation. The
result is deliberately mixed: the development ranking signal was promising,
but final-period probability calibration was not.

```text
Earlier filings -> chronological training -> walk-forward candidate selection
                                              |
                                              v
                               fixed 5% analyst-review queue
                                              |
                                              v
                                  one-time frozen 2019-2022 result
```

## Limitations and next step

This is not a production fraud-scoring system. It should not drive enforcement,
compliance, or investment decisions. Labels arrive with substantial observed
delay (median 1,237 days), so the frozen period has material label-maturity
uncertainty as well as failed calibration.

The next research-only improvement is issuer-history language features—prior
year deltas, rolling issuer baselines, and peer-relative language abnormalities.
They will be evaluated on development folds only; the final test remains sealed.

## Structure

- `configs/` for shared paths and settings
- `data/` for raw, interim, processed, and external data
- `src/` for reusable pipeline, feature, model, and analysis code
- `reports/` for figures, tables, and hypothesis notes
- `notebooks/` for exploratory analysis
- `tests/` for automated checks

## Data

Place the source CSV at `data/raw/lm_summaries/Loughran-McDonald_10X_Summaries_1993-2025.csv`.

## Code quality

Install development dependencies with `pip install -r requirements-dev.txt`, then run the following checks:

```powershell
ruff check --no-cache src tests configs
ruff format --check --no-cache src tests configs
```

Use `ruff check --fix --no-cache src tests configs` followed by `ruff format --no-cache src tests configs` to apply safe lint fixes and formatting.
