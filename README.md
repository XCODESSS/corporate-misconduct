# Corporate Misconduct Warning

Research pipeline for testing whether filing language can rank corporate filings into a fixed 5% misconduct-review queue. The project is repairing a discovered temporal-validation flaw and must not be treated as a production fraud detector.

## Historical result — validation invalidated

Trial 196 previously reported development Recall@5% of 26.22%, but that result is invalid because `filing_year` was derived from `reporting_date`; every evaluated fold contained later filings in training. Its already-observed 2019-2022 result—7 of 58 recorded cases in a 5% queue, Recall@5% 12.07%, Brier skill -0.0201—is retained only as a historical, nonconfirmatory artifact and will not be rerun.

Generate the local static demo without training or reopening the final test:

```powershell
.\demo\open_demo.ps1
```

The demo is artifact-driven and opens in seconds. It shows the selection rule,
development versus final results, Recall@5% definition, SHAP feature ranking,
and the observed label-delay limitation. Read the full
[model card](docs/model_card.md) before interpreting the result.

## Why the timeline matters

The repaired evaluation derives fold membership from the actual `filing_date` and requires `max(training filing_date) < min(test filing_date)`. `reporting_year` is retained separately and cannot control evaluation folds.

```text
Actual filing dates -> strict chronological training -> development selection
                                              |
                                              v
                               fixed 5% analyst-review queue
                                              |
                                              v
                                  development-only result
```

## Limitations and next step

This is not a production fraud-scoring system. It should not drive enforcement,
compliance, or investment decisions. Labels arrive with substantial observed
delay (median 1,237 days), so the frozen period has material label-maturity
uncertainty as well as failed calibration.

The next step is to rebuild development features, pass the temporal audit, and request separate approval for a new development-only experiment. The historical final period is permanently closed.

## Structure

- `configs/` for shared paths and settings
- `data/` for raw, interim, processed, and external data
- `src/` for reusable pipeline, feature, model, and analysis code
- `reports/` for figures, tables, and hypothesis notes
- `notebooks/` for exploratory analysis
- `tests/` for automated checks

## Data

Place the source CSV at `data/raw/lm/Loughran-McDonald_10X_Summaries_1993-2025.csv`.

Create a clean Python 3.12 environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

CUDA, training libraries, and model artifacts are not required to open the static demo.

## Code quality

Install development dependencies with `pip install -r requirements-dev.txt`, then run the following checks:

```powershell
ruff check --no-cache src tests configs
ruff format --check --no-cache src tests configs
```

Use `ruff check --fix --no-cache src tests configs` followed by `ruff format --no-cache src tests configs` to apply safe lint fixes and formatting.
