# Corporate Misconduct Warning

Project scaffold for analyzing corporate misconduct warning signals from the Loughran-McDonald summaries and derived features.

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
