# Repository Integrity Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair temporal leakage, reproducibility, pipeline, artifact-audit, testing, and presentation defects while permanently preserving the already-opened 2019-2022 final evaluation as a historical artifact rather than reusing it.

**Architecture:** Make actual SEC filing timestamps the single source of truth for temporal evaluation while retaining reporting year as a separate descriptive/lookup field. Put every development result behind strict temporal invariants and immutable artifact contracts, make the data pipeline import-safe and explicitly staged, and ship the static demo from small versioned evidence snapshots. Archive the current Trial 196 development and frozen metrics as historically observed but methodologically invalidated; any replacement research result must come from a new development-only experiment and must never rescore the old final period.

**Tech Stack:** Python 3.12, pandas, NumPy, PyArrow, scikit-learn, Optuna, XGBoost, SHAP, pytest, Ruff, PowerShell, JSON/CSV/Parquet artifacts.

**Spec:** No standalone specification exists. Requirements come from the user's 2026-08-17 repository-review request, the verified review findings, `README.md`, `docs/model_card.md`, and the current evaluation code.

## Global Constraints

- Work from `D:\coperate-misconduct-warning` on Windows/PowerShell.
- Do not read, rebuild, rewrite, or rescore `data/processed/features/test_features.parquet` during implementation or validation.
- Treat the existing 2019-2022 final evaluation as permanently closed. Never call `evaluate_selected_xgboost_on_test()` again.
- Do not start Optuna, XGBoost/GPU training, candidate review, SHAP, a new model, or any long walk-forward job without a separate explicit user approval.
- Fast unit tests must use synthetic fixtures or development-only inputs; no unit test may open the test-feature parquet.
- Preserve the selection policy for any approved replacement experiment: positive mean Brier skill eligibility; mean Recall@5% primary; mean Precision@5% alongside it; mean PR-AUC tie-breaker; SHAP only after selection.
- Keep `no_eligible_candidate` as a valid terminal development outcome that writes the review result and produces no model or SHAP artifact.
- Do not add Beneish M-Score work.
- Continue to frame the project as research ranking analysis, never as production fraud scoring, compliance triage, enforcement automation, or investment advice.
- Keep the demo artifact-only, launchable with `./demo/open_demo.ps1`, and independent of training libraries and untracked report directories.
- Preserve the historical frozen numbers only as archived evidence: Recall@5% `12.07%` (`7/58`) and Brier skill `-0.0201`, with label-maturity limitations visible beside them.
- Mark the historical Trial 196 development Recall@5% `26.22%` as invalidated by temporal overlap until a new strictly chronological development experiment is approved and completed.
- Use `python -m pytest` from the repository root; do not use bare `pytest`.
- Use test-first, small commits, and no unrelated refactors.

---

## File Responsibility Map

| File | Action | Single responsibility after this plan |
| --- | --- | --- |
| `src/evaluation/temporal.py` | Create | Parse filing timestamps and generate/validate strict expanding-window folds. |
| `src/features/lm_features.py` | Modify | Keep `filing_year` from `filing_date`, keep `reporting_year` separate, and support development-only feature rebuilding. |
| `src/evaluation/cross_validation.py` | Modify | Evaluate strict filing-time folds and prohibit in-sample threshold optimization. |
| `src/evaluation/calibration.py` | Modify | Split chronological calibration partitions using exact filing timestamps. |
| `src/models/dummy_classifier.py` | Modify | Pass filing timestamps to cross-validation. |
| `src/models/logistic_regression.py` | Modify | Pass filing timestamps to cross-validation and use fixed thresholds. |
| `src/models/xgboost_model.py` | Modify | Start a new development experiment, enforce provenance, use fixed thresholds, add auditable identifiers, and permanently block old-test reuse. |
| `src/utils/fingerprints.py` | Create | Produce canonical JSON and file SHA-256 fingerprints. |
| `src/utils/record_ids.py` | Create | Produce stable filing identifiers from non-label source fields. |
| `src/evaluation/one_shot.py` | Create | Atomically persist a preregistered holdout run with an exclusive started ledger and final completion manifest. |
| `src/ingestion/load_finnlp_dataset.py` | Create | Resolve raw FinNLP paths and stream JSON arrays without loading them fully. |
| `src/pipeline/run_pipeline.py` | Rewrite | Expose an import-safe, explicitly acknowledged data-stage runner with no model or final-test stage. |
| `src/pipeline/prepare_dataset.py` | Delete | Remove a machine-specific top-level debugging script. |
| `src/pipeline/build_features.py` | Delete | Remove a top-level data-printing script. |
| `src/pipeline/train_models.py` | Delete | Remove a misleading empty entrypoint; models remain explicit APIs. |
| `src/pipeline/evaluate_models.py` | Delete | Remove a misleading empty entrypoint. |
| `src/models/random_forest.py` | Delete | Remove an unimplemented model advertised as available. |
| `src/models/lightgbm_model.py` | Delete | Remove an unimplemented model advertised as available. |
| `src/models/catboost_model.py` | Delete | Remove an unimplemented model advertised as available. |
| `src/models/neural_network.py` | Delete | Remove an unimplemented model advertised as available. |
| `configs/models.yaml` | Modify | List only executable model implementations. |
| `requirements.txt` | Modify | Pin every runtime package imported by executable project code. |
| `requirements-dev.txt` | Modify | Pin test/lint tooling on top of runtime requirements. |
| `.gitignore` | Modify | Ignore generated data/reports/logs without ignoring Python source or versioned demo evidence. |
| `scripts/audit_temporal_folds.py` | Create | Audit a development feature parquet for reporting-year drift and fold overlap. |
| `scripts/export_demo_evidence.py` | Create | Validate historical report artifacts and export a small versioned demo snapshot. |
| `demo/artifacts/historical_evidence.json` | Create | Version the exact historical metrics, provenance hashes, and invalidation status needed by the demo. |
| `demo/artifacts/shap_importance.csv` | Create | Version the historical top-feature display data with historical-only status. |
| `demo/generate_xgboost_demo.py` | Modify | Render only versioned demo evidence and make invalidation status unavoidable. |
| `demo/open_demo.ps1` | Modify | Generate and open the portable static page with no report-directory dependency. |
| `tests/test_temporal.py` | Create | Test exact-date fold boundaries and overlap rejection. |
| `tests/test_cross_validation.py` | Create | Test fixed-threshold behavior and strict temporal inputs. |
| `tests/test_artifact_contracts.py` | Create | Test artifact fingerprints, provenance mismatches, and permanent final-test closure. |
| `tests/test_ingestion.py` | Rewrite | Test the restored raw-data loader and validator import. |
| `tests/test_preprocessing.py` | Rewrite | Test filing-year/reporting-year semantics and development-only feature preparation. |
| `tests/test_pipeline.py` | Rewrite | Test pipeline stage ordering, acknowledgement gates, and import safety. |
| `tests/test_demo.py` | Modify | Test clean-clone demo generation from versioned evidence and invalidation copy. |
| `tests/test_repository_integrity.py` | Create | Import public modules and enforce source/dependency/tracking invariants. |
| `docs/audits/2026-08-17-temporal-validation-audit.md` | Create | Preserve measured overlap evidence and its effect on claims. |
| `README.md` | Modify | Give reproducible setup/demo commands and correct historical-result language. |
| `docs/methodology.md` | Rewrite | Specify filing-time folds, calibration, fixed queue metrics, and artifact provenance. |
| `docs/model_card.md` | Modify | Mark historical development selection invalid and final metrics nonconfirmatory. |
| `docs/future_work.md` | Modify | Replace “existing folds” language with strict filing-time folds and approval gates. |

---

### Task 1: Make filing timestamps the temporal source of truth

**Files:**
- Create: `src/evaluation/temporal.py`
- Modify: `src/features/lm_features.py:139-200`
- Modify: `src/evaluation/cross_validation.py:111-155,192-349,705-820`
- Modify: `src/evaluation/calibration.py:94-211`
- Modify: `src/models/dummy_classifier.py`
- Modify: `src/models/logistic_regression.py`
- Modify: `src/models/xgboost_model.py:66-180,330-370,764-840`
- Create: `tests/test_temporal.py`
- Rewrite: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: raw `filing_date` strings parsed with `dayfirst=True`; raw `reporting_date` strings; binary labels.
- Produces: `parse_filing_dates(values: pd.Series | np.ndarray) -> np.ndarray`, `filing_years(values: pd.Series | np.ndarray) -> np.ndarray`, `WalkForwardCV.generate_folds(filing_dates: np.ndarray, y: np.ndarray)`, and module-level `engineer_development_features() -> Path`.

- [ ] **Step 1: Write failing year-semantics tests**

```python
def test_prepare_dataset_separates_filing_and_reporting_year() -> None:
    frame = pd.DataFrame(
        {
            "cik": ["1750"],
            "filing_date": ["15-03-2020"],
            "reporting_date": ["31-12-2019"],
        }
    )

    result = LMFeatureEngineer().prepare_dataset(frame)

    assert result.loc[0, "filing_year"] == 2020
    assert result.loc[0, "reporting_year"] == 2019
```

- [ ] **Step 2: Write failing strict-fold tests**

```python
def test_walk_forward_folds_are_strictly_ordered_by_filing_date() -> None:
    dates = np.array(
        ["2017-12-20", "2018-03-01", "2018-12-31", "2019-02-15"],
        dtype="datetime64[D]",
    )
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    cv = WalkForwardCV(min_fraud_per_fold=1)

    folds = list(cv.generate_folds(dates, labels))

    for train_idx, test_idx, _ in folds:
        assert dates[train_idx].max() < dates[test_idx].min()


def test_invalid_filing_date_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid filing date"):
        parse_filing_dates(pd.Series(["31-12-2018", "not-a-date"]))
```

- [ ] **Step 3: Run focused tests and confirm the current code fails**

Run: `python -m pytest tests/test_preprocessing.py tests/test_temporal.py -q`

Expected: FAIL because `reporting_year` does not exist and `generate_folds` currently accepts reporting-derived integer years.

- [ ] **Step 4: Add exact-date helpers**

```python
def parse_filing_dates(values: pd.Series | np.ndarray) -> np.ndarray:
    parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"invalid filing date count: {int(parsed.isna().sum())}")
    return parsed.to_numpy(dtype="datetime64[D]")


def filing_years(values: pd.Series | np.ndarray) -> np.ndarray:
    dates = pd.DatetimeIndex(parse_filing_dates(values))
    return dates.year.to_numpy(dtype=np.int32)
```

- [ ] **Step 5: Preserve both year meanings in feature engineering**

Replace `LMFeatureEngineer.prepare_dataset()` year assignment with:

```python
df["cik"] = self.normalize_cik(df["cik"])
df["filing_year"] = filing_years(df["filing_date"])
df["reporting_year"] = filing_years(df["reporting_date"])
return df
```

Use `reporting_year` only for the LM lookup key and use `filing_year` only for temporal folds and reporting. Rename local lookup variables from `filing_year` to `lm_document_year` where they refer to the LM CSV key.

- [ ] **Step 6: Generate folds from exact dates**

```python
def generate_folds(self, filing_dates: np.ndarray, y: np.ndarray):
    dates = parse_filing_dates(filing_dates)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    for test_year in sorted(np.unique(years)):
        test_start = np.datetime64(f"{int(test_year):04d}-01-01")
        test_end = np.datetime64(f"{int(test_year) + 1:04d}-01-01")
        train_idx = np.flatnonzero(dates < test_start)
        test_idx = np.flatnonzero((dates >= test_start) & (dates < test_end))
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if dates[train_idx].max() >= dates[test_idx].min():
            raise RuntimeError(f"temporal overlap detected for {int(test_year)}")
        fraud_count = int(y[test_idx].sum())
        if fraud_count < self.min_fraud_per_fold:
            continue
        yield train_idx, test_idx, int(test_year)
```

Change `WalkForwardCV.run()` and `evaluate_fold()` from `years` to `filing_dates`. Derive the calibration timestamps from `filing_dates[train_idx]`.

- [ ] **Step 7: Make calibration use exact timestamps**

Change the signature from `ProbabilityCalibrator.fit(estimator, X_train, y_train, years=None)` to `ProbabilityCalibrator.fit(estimator, X_train, y_train, filing_dates=None)`. In `_chronological_holdout_indices`, sort exact `datetime64[D]` values, choose a cutoff date, and enforce `max(fit_dates) < min(calibration_dates)`. Reject same-day boundary ambiguity by moving every row on the cutoff day into the calibration partition.

- [ ] **Step 8: Pass filing dates through all implemented models**

Store `self.filing_dates` beside `self.X` and `self.y` in Dummy, Logistic Regression, and XGBoost. Do not derive this array from `reporting_date` or the preexisting `filing_year` column. In final prediction serialization, derive display year from the parsed filing date.

- [ ] **Step 9: Add a development-only feature API**

```python
def engineer_development_features() -> Path:
    engineer = LMFeatureEngineer()
    development = engineer.prepare_dataset(
        engineer.load_dataset(engineer.TRAINVAL_FILE, "trainval")
    )
    required_keys = set(zip(development["cik"], development["reporting_year"]))
    lookup = engineer.build_lm_lookup(required_keys)
    engineer.process_dataset(
        engineer.TRAINVAL_FILE,
        engineer.TRAINVAL_OUTPUT,
        lookup,
        "trainval",
    )
    return engineer.TRAINVAL_OUTPUT
```

Change `attach_features()` to form its lookup key from `row.reporting_year`. This API must not read or write `TEST_FILE` or `TEST_OUTPUT`. Unit-test `load_dataset()` and `process_dataset()` calls with mocks and assert neither test path appears.

- [ ] **Step 10: Run focused and broader tests**

Run: `python -m pytest tests/test_preprocessing.py tests/test_temporal.py tests/test_models.py -q`

Expected: PASS, with no test opening `test_features.parquet`.

- [ ] **Step 11: Commit**

```powershell
git add src/evaluation/temporal.py src/evaluation/cross_validation.py src/evaluation/calibration.py src/features/lm_features.py src/models/dummy_classifier.py src/models/logistic_regression.py src/models/xgboost_model.py tests/test_temporal.py tests/test_preprocessing.py tests/test_models.py
git commit -m "fix: enforce filing-time validation folds"
```

---

### Task 2: Remove biased threshold optimization and isolate a new experiment

**Files:**
- Modify: `src/evaluation/cross_validation.py:157-186,266-289,705-820`
- Modify: `src/models/dummy_classifier.py`
- Modify: `src/models/logistic_regression.py`
- Modify: `src/models/xgboost_model.py:60-140,497-558,764-840`
- Create: `tests/test_cross_validation.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: fixed `decision_threshold: float`, strict-date folds from Task 1.
- Produces: threshold-independent development ranking/calibration metrics and fixed-threshold classification metrics; experiment name `xgboost_lm_text_surface_probability_v3_strict_time`.

- [ ] **Step 1: Write failing fixed-threshold tests**

```python
def test_cross_validation_rejects_in_sample_threshold_search() -> None:
    cv = WalkForwardCV(min_fraud_per_fold=1)
    with pytest.raises(ValueError, match="independent threshold-validation scores"):
        cv._select_threshold(
            y_train=np.array([0, 1]),
            train_score=np.array([0.1, 0.9]),
            default_threshold=0.5,
            should_optimize=True,
        )


def test_candidate_review_uses_fixed_threshold(monkeypatch, tmp_path: Path) -> None:
    pipeline = XGBoostBaseline()
    pipeline.X = np.ones((4, len(pipeline.FEATURE_COLUMNS)), dtype=np.float32)
    pipeline.y = np.array([0, 0, 1, 1], dtype=np.int8)
    pipeline.filing_dates = np.array(
        ["2017-01-01", "2017-02-01", "2018-01-01", "2018-02-01"],
        dtype="datetime64[D]",
    )
    trial = SimpleNamespace(number=7, value=0.2, params={})
    observed: list[bool] = []
    monkeypatch.setattr(pipeline, "_top_pr_auc_trials", lambda: [trial])
    monkeypatch.setattr(pipeline, "build_tuned_model", lambda: object())
    monkeypatch.setattr(pipeline, "CANDIDATE_REVIEW_FILE", tmp_path / "review.json")

    def fake_cross_validation(**kwargs):
        observed.append(kwargs["optimize_threshold"])
        return {
            "brier_skill_score": {"mean": 0.1},
            "recall_at_5_percent": {"mean": 0.2},
            "precision_at_5_percent": {"mean": 0.1},
            "pr_auc": {"mean": 0.15},
        }

    monkeypatch.setattr(pipeline, "run_cross_validation", fake_cross_validation)
    pipeline.review_top_candidates()
    assert observed == [False]
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/test_cross_validation.py tests/test_models.py -q`

Expected: FAIL because the current code optimizes F1 using fitted training predictions.

- [ ] **Step 3: Make fixed thresholds the only supported production path**

```python
def _select_threshold(
    self,
    y_train: np.ndarray,
    train_score: np.ndarray,
    default_threshold: float,
    should_optimize: bool,
) -> float:
    if should_optimize:
        raise ValueError(
            "threshold optimization requires independent threshold-validation scores"
        )
    return default_threshold
```

Set `optimize_threshold=False` in Dummy, Logistic Regression, XGBoost optimization, and candidate review. Keep Recall@5%, Precision@5%, PR-AUC, Brier, and Brier skill unchanged because they do not depend on the classification threshold.

- [ ] **Step 4: Start a new development namespace**

Change the active XGBoost experiment name to `xgboost_lm_text_surface_probability_v3_strict_time`. Never overwrite or reinterpret v2 artifacts. Add `evaluation_contract_version = "strict-filing-time-v1"` to all new summaries.

- [ ] **Step 5: Preserve the selection rule and no-winner behavior**

Add tests asserting:

```python
assert review["selection_rule"] == (
    "positive mean Brier skill score; highest mean Recall@5%; "
    "mean PR-AUC tie-breaker"
)
assert no_winner["status"] == "no_eligible_candidate"
assert no_winner["selected_trial_number"] is None
```

- [ ] **Step 6: Run checks**

Run: `python -m pytest tests/test_cross_validation.py tests/test_models.py -q`

Expected: PASS without fitting XGBoost.

- [ ] **Step 7: Commit**

```powershell
git add src/evaluation/cross_validation.py src/models/dummy_classifier.py src/models/logistic_regression.py src/models/xgboost_model.py tests/test_cross_validation.py tests/test_models.py
git commit -m "fix: remove resubstitution threshold tuning"
```

---

### Task 3: Bind candidate artifacts to immutable development provenance

**Files:**
- Create: `src/utils/fingerprints.py`
- Modify: `src/models/xgboost_model.py:196-230,497-605,929-982`
- Create: `tests/test_artifact_contracts.py`

**Interfaces:**
- Consumes: development feature parquet path, ordered feature names, calibration configuration, selection configuration, experiment name.
- Produces: `sha256_file(path: Path) -> str`, `sha256_json(payload: Any) -> str`, `XGBoostBaseline._current_artifact_contract() -> dict[str, Any]`, and strict contract validation before any final-test path is opened.

- [ ] **Step 1: Write failing fingerprint and mismatch tests**

```python
def test_sha256_json_is_key_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_selected_review_rejects_stale_development_hash(tmp_path: Path) -> None:
    pipeline = XGBoostBaseline()
    pipeline.CANDIDATE_REVIEW_FILE = tmp_path / "review.json"
    review = {
        "status": "selected",
        "artifact_contract": {"development_data_sha256": "stale"},
    }
    pipeline.CANDIDATE_REVIEW_FILE.write_text(json.dumps(review), encoding="utf-8")
    pipeline._current_artifact_contract = lambda: {
        "development_data_sha256": "current"
    }
    with pytest.raises(RuntimeError, match="development_data_sha256"):
        pipeline._load_selected_candidate_review()
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/test_artifact_contracts.py -q`

Expected: FAIL because canonical fingerprint helpers and contract validation do not exist.

- [ ] **Step 3: Add canonical fingerprint helpers**

```python
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Define and persist the contract**

```python
def _current_artifact_contract(self) -> dict[str, Any]:
    return {
        "contract_version": "strict-filing-time-v1",
        "experiment_name": self.EXPERIMENT_NAME,
        "development_data_sha256": sha256_file(self.INPUT_FILE),
        "feature_schema_sha256": self._feature_schema_hash(),
        "calibration": {
            "method": self.DEFAULT_CALIBRATION_METHOD,
            "strategy": self.DEFAULT_CALIBRATION_STRATEGY,
            "holdout_fraction": self.DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
        },
        "selection": {
            "review_size": self.CANDIDATE_REVIEW_SIZE,
            "primary": "recall_at_5_percent",
            "eligibility": "brier_skill_score > 0",
            "tie_breaker": "pr_auc",
        },
    }
```

Embed this payload and its canonical hash in candidate-review, experiment-manifest, model-metadata, and SHAP metadata outputs.

- [ ] **Step 5: Validate before any test access**

In `_load_selected_candidate_review()`, compare every persisted contract field with `_current_artifact_contract()` and raise a message naming the first mismatched field. Call this validation before `load_test_dataset()` or any test path existence check.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_artifact_contracts.py tests/test_models.py -q`

Expected: PASS using temporary files only.

- [ ] **Step 7: Commit**

```powershell
git add src/utils/fingerprints.py src/models/xgboost_model.py tests/test_artifact_contracts.py tests/test_models.py
git commit -m "fix: bind model artifacts to development provenance"
```

---

### Task 4: Permanently close the historical final test and make future output protocols auditable

**Files:**
- Create: `src/utils/record_ids.py`
- Create: `src/evaluation/one_shot.py`
- Modify: `src/models/xgboost_model.py:70-125,1297-1541,1579-1581`
- Modify: `tests/test_artifact_contracts.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `cik`, filing date, reporting date, filing type, existing historical final artifacts.
- Produces: `stable_filing_id(cik: Any, filing_date: Any, reporting_date: Any, filing_type: Any) -> str`, `OneShotEvaluationWriter.begin(metadata: dict[str, Any]) -> None`, `OneShotEvaluationWriter.commit_text_files(payloads: dict[str, str]) -> dict[str, Any]`, and an unconditional closure error for the old 2019-2022 evaluator.

- [ ] **Step 1: Write failing closure and identifier tests**

```python
def test_historical_final_test_is_permanently_closed() -> None:
    with pytest.raises(RuntimeError, match="permanently closed"):
        evaluate_selected_xgboost_on_test()


def test_stable_filing_id_is_deterministic() -> None:
    first = stable_filing_id("0000001750", "2019-07-18", "2018-12-31", "10-K")
    second = stable_filing_id("1750", "18-07-2019", "31-12-2018", "10-K")
    assert first == second
    assert len(first) == 64
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_artifact_contracts.py tests/test_models.py -q`

Expected: FAIL because the public function still calls the old evaluator and no stable ID helper exists.

- [ ] **Step 3: Add stable identifiers without labels**

```python
def stable_filing_id(cik, filing_date, reporting_date, filing_type) -> str:
    parsed_filing = pd.to_datetime(filing_date, dayfirst=True, errors="raise")
    parsed_reporting = pd.to_datetime(reporting_date, dayfirst=True, errors="raise")
    payload = {
        "cik": str(cik).removesuffix(".0").zfill(10),
        "filing_date": parsed_filing.date().isoformat(),
        "reporting_date": parsed_reporting.date().isoformat(),
        "filing_type": str(filing_type).strip().upper(),
    }
    return sha256_json(payload)
```

Do not include `fraudulent`, matched AAER periods, model scores, or row order in this identifier.

- [ ] **Step 4: Permanently close the existing evaluator**

```python
def evaluate_selected_xgboost_on_test() -> dict[str, Any]:
    raise RuntimeError(
        "The historical 2019-2022 final test is permanently closed and cannot be rerun."
    )
```

Keep the historical artifacts read-only for presentation. Remove any CLI path that can call `run_final_test_evaluation()` for the v2 or v3 experiment.

- [ ] **Step 5: Add an exact one-shot writer for a genuinely new holdout**

```python
class OneShotEvaluationWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.started_path = output_dir / "evaluation_started.json"
        self.complete_path = output_dir / "evaluation_complete.json"

    def begin(self, metadata: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.started_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, sort_keys=True, indent=2)

    def commit_text_files(self, payloads: dict[str, str]) -> dict[str, Any]:
        temporary = Path(
            tempfile.mkdtemp(prefix="evaluation-", dir=self.output_dir.parent)
        )
        hashes: dict[str, str] = {}
        for name, content in payloads.items():
            temporary_path = temporary / name
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(content, encoding="utf-8")
            hashes[name] = sha256_file(temporary_path)
        for name in payloads:
            destination = self.output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary / name, destination)
        manifest = {"status": "complete", "sha256": hashes}
        with self.complete_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
        return manifest
```

Call `begin()` before a separately preregistered holdout dataset is opened. Serialize summary, predictions, and calibration completely in memory, include `record_id` and `cik` in prediction CSV text, then pass all three strings to `commit_text_files()`. If serialization or commit fails, keep `evaluation_started.json`; a subsequent `begin()` must raise `FileExistsError`. Unit-test only with temporary synthetic strings and paths, never with 2019-2022 data.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_artifact_contracts.py tests/test_models.py -q`

Expected: PASS without opening any project Parquet file.

- [ ] **Step 7: Commit**

```powershell
git add src/utils/record_ids.py src/models/xgboost_model.py tests/test_artifact_contracts.py tests/test_models.py
git commit -m "fix: close historical holdout and harden audit outputs"
```

---

### Task 5: Restore raw ingestion and replace placeholder pipeline modules

**Files:**
- Create: `src/ingestion/load_finnlp_dataset.py`
- Rewrite: `src/pipeline/run_pipeline.py`
- Delete: `src/pipeline/prepare_dataset.py`
- Delete: `src/pipeline/build_features.py`
- Delete: `src/pipeline/train_models.py`
- Delete: `src/pipeline/evaluate_models.py`
- Delete: `src/models/random_forest.py`
- Delete: `src/models/lightgbm_model.py`
- Delete: `src/models/catboost_model.py`
- Delete: `src/models/neural_network.py`
- Modify: `configs/models.yaml`
- Rewrite: `tests/test_ingestion.py`
- Rewrite: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: paths from `configs.settings`, existing ingestion/preprocessing public functions.
- Produces: `FinNLPDatasetPaths`, `FinNLPDatasetLoader`, `run_data_pipeline(through: str, acknowledge_test_write: bool = False) -> list[str]`, and `python -m src.pipeline.run_pipeline --list`.

- [ ] **Step 1: Write failing loader tests**

```python
def test_loader_streams_firm_year_array(tmp_path: Path) -> None:
    firm_years = tmp_path / "firm_years.json"
    firm_years.write_text('[{"cik": "1750"}]', encoding="utf-8")
    labels = tmp_path / "firm_years_labels.json"
    labels.write_text("[]", encoding="utf-8")
    aaer = tmp_path / "aaer_mark5.csv"
    aaer.write_text("", encoding="utf-8")
    loader = FinNLPDatasetLoader(firm_years, labels, aaer)
    assert list(loader.stream_firm_years()) == [{"cik": "1750"}]


def test_raw_validator_imports() -> None:
    import src.ingestion.validate_raw_data  # noqa: F401
```

- [ ] **Step 2: Write failing pipeline safety tests**

```python
def test_pipeline_requires_acknowledgement_before_split(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="acknowledge-test-write"):
        run_data_pipeline("split", acknowledge_test_write=False)


def test_pipeline_contains_no_model_or_final_test_stage() -> None:
    assert not ({"optuna", "xgboost", "candidate-review", "shap", "final-test"} & STAGES)
```

- [ ] **Step 3: Run tests and confirm current failures**

Run: `python -m pytest tests/test_ingestion.py tests/test_pipeline.py -q`

Expected: FAIL because the loader is missing and pipeline modules are placeholders/debug scripts.

- [ ] **Step 4: Add the loader**

```python
@dataclass(frozen=True, slots=True)
class FinNLPDatasetPaths:
    firm_years: Path
    labels: Path
    aaer: Path


class FinNLPDatasetLoader:
    def __init__(self, firm_years=None, labels=None, aaer=None) -> None:
        self._paths = FinNLPDatasetPaths(
            Path(firm_years or settings.FIRM_YEARS_FILE),
            Path(labels or settings.LABELS_FILE),
            Path(aaer or settings.AAER_FILE),
        )

    def get_dataset_paths(self) -> FinNLPDatasetPaths:
        return self._paths

    def stream_firm_years(self):
        with self._paths.firm_years.open("rb") as stream:
            yield from ijson.items(stream, "item")
```

- [ ] **Step 5: Implement an explicit data-only stage registry**

```python
STAGES = {
    "validate-raw": validate_raw_data,
    "ingest-firm-years": ingest_firm_years,
    "ingest-labels": ingest_labels,
    "merge-labels": merge_labels,
    "clean-text": clean_firm_year_mda_text,
    "normalize": normalize_dataset,
    "quality-check": quality_check_dataset,
    "deduplicate": deduplicate_dataset,
    "split": split_dataset,
    "development-features": engineer_development_features,
}
```

`run_data_pipeline()` must require `acknowledge_test_write=True` before `split`, because the existing splitter writes both train/validation and test outputs. The `development-features` stage must call the development-only API from Task 1 and must not write test features.

- [ ] **Step 6: Add a safe CLI**

Support only:

```powershell
python -m src.pipeline.run_pipeline --list
python -m src.pipeline.run_pipeline --through quality-check
python -m src.pipeline.run_pipeline --through split --acknowledge-test-write
```

No default command may execute stages. Without `--list` or `--through`, argparse must exit with code 2 and a usage message.

- [ ] **Step 7: Remove false entrypoints and advertised models**

Use `git rm` for the listed placeholder/debug files. Update `configs/models.yaml` to:

```yaml
models:
  - dummy_classifier
  - logistic_regression
  - xgboost
```

- [ ] **Step 8: Run tests and import checks**

Run: `python -m pytest tests/test_ingestion.py tests/test_pipeline.py -q`

Expected: PASS without reading project raw data.

Run: `python -c "import src.ingestion.validate_raw_data; import src.pipeline.run_pipeline"`

Expected: exit code 0 and no dataset output printed.

- [ ] **Step 9: Commit**

```powershell
git add src/ingestion/load_finnlp_dataset.py src/pipeline/run_pipeline.py configs/models.yaml tests/test_ingestion.py tests/test_pipeline.py
git rm src/pipeline/prepare_dataset.py src/pipeline/build_features.py src/pipeline/train_models.py src/pipeline/evaluate_models.py src/models/random_forest.py src/models/lightgbm_model.py src/models/catboost_model.py src/models/neural_network.py
git commit -m "fix: restore ingestion and safe data orchestration"
```

---

### Task 6: Make clean-environment installation deterministic

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`
- Create: `tests/test_repository_integrity.py`
- Modify: `README.md:58-71`

**Interfaces:**
- Consumes: Python 3.12.
- Produces: a fully pinned runtime environment and a fully pinned test/lint layer.

- [ ] **Step 1: Write a failing runtime-import test**

```python
@pytest.mark.parametrize(
    "module_name",
    [
        "src.ingestion.validate_raw_data",
        "src.preprocessing.split_dataset",
        "src.features.lm_features",
        "src.evaluation.cross_validation",
        "src.models.xgboost_model",
        "demo.generate_xgboost_demo",
    ],
)
def test_public_module_imports_without_side_effects(module_name: str, capsys) -> None:
    importlib.import_module(module_name)
    captured = capsys.readouterr()
    assert captured.out == ""
```

- [ ] **Step 2: Pin verified runtime packages**

Replace `requirements.txt` with the locally verified Python 3.12 set:

```text
pandas==2.3.3
numpy==2.4.6
scipy==1.17.0
scikit-learn==1.9.0
statsmodels==0.14.6
matplotlib==3.10.8
seaborn==0.13.2
jupyter==1.1.1
pyarrow==20.0.0
ijson==3.5.0
optuna==4.9.0
xgboost==3.3.0
shap==0.52.0
joblib==1.5.3
```

Replace `requirements-dev.txt` with:

```text
-r requirements.txt
pytest==9.0.2
ruff==0.11.0
```

- [ ] **Step 3: Document clean setup**

Add exact PowerShell commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Document that CUDA-compatible XGBoost execution is optional and is never needed for the static demo or unit tests.

- [ ] **Step 4: Run current-environment checks**

Run: `python -m pytest tests/test_repository_integrity.py -q`

Expected: PASS because Tasks 1-5 have removed missing imports and top-level data reads.

Run: `python -m pip check`

Expected: `No broken requirements found.`

- [ ] **Step 5: Commit**

```powershell
git add requirements.txt requirements-dev.txt tests/test_repository_integrity.py README.md
git commit -m "build: pin complete Python dependencies"
```

---

### Task 7: Make the demo portable and correct its historical claims

**Files:**
- Modify: `.gitignore`
- Create: `scripts/export_demo_evidence.py`
- Create: `demo/artifacts/historical_evidence.json`
- Create: `demo/artifacts/shap_importance.csv`
- Modify: `demo/generate_xgboost_demo.py`
- Modify: `demo/open_demo.ps1`
- Modify: `tests/test_demo.py`

**Interfaces:**
- Consumes: existing persisted report artifacts only during the one-time export step.
- Produces: a versioned historical evidence bundle and a static page that explicitly identifies invalidated development validation.

- [ ] **Step 1: Write failing portable-demo tests**

```python
def test_demo_uses_versioned_evidence_not_reports_directory() -> None:
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    assert 'ROOT / "reports"' not in source
    assert 'ROOT / "demo" / "artifacts"' in source


def test_demo_discloses_temporal_invalidation(tmp_path: Path) -> None:
    output = tmp_path / "index.html"
    subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )
    page = output.read_text(encoding="utf-8")
    assert "development validation invalidated" in page.lower()
    assert "historical final result is nonconfirmatory" in page.lower()
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/test_demo.py -q`

Expected: FAIL because the generator currently reads ignored `reports/` artifacts and presents development as chronological.

- [ ] **Step 3: Narrow ignore rules**

Remove `src/evaluation/` from `.gitignore`. Keep generated `reports/` ignored, add explicit versioned evidence exceptions, and ignore logs:

```gitignore
reports/
logs/*.log
!demo/artifacts/
!demo/artifacts/*.json
!demo/artifacts/*.csv
```

Stop tracking logs without deleting local copies:

```powershell
git rm --cached logs/pipeline.log logs/training.log logs/feature_generation.log
```

- [ ] **Step 4: Export a minimal historical evidence bundle**

The exporter must validate required fields, compute source SHA-256 values, and write:

```json
{
  "schema_version": "historical-demo-evidence-v1",
  "development_validation_status": "invalidated_temporal_overlap",
  "historical_trial": 196,
  "historical_development_recall_at_5_percent": 0.262216,
  "frozen_result_status": "historical_nonconfirmatory",
  "historical_final_recall_at_5_percent": 0.1206896551724138,
  "historical_final_cases_found": 7,
  "historical_final_positive_labels": 58,
  "historical_final_brier_skill_score": -0.020106618445109747,
  "label_maturity_median_days": 1236.5,
  "source_sha256": {}
}
```

The word `historical` must appear in every old-result field name. The exporter must not import model modules or open Parquet files.

- [ ] **Step 5: Render invalidation prominently**

Change the first demo callout to state that the original development walk-forward result is invalid because reporting-year folds admitted later filings. Display the frozen metrics only as the historical outcome of that invalid selection pipeline. Keep the production-use prohibition and label-delay disclosure.

- [ ] **Step 6: Keep the one-command launch**

`demo/open_demo.ps1` must call only the generator and `Start-Process` on the generated local HTML. It must not check or regenerate report artifacts.

- [ ] **Step 7: Run checks**

Run: `python demo/generate_xgboost_demo.py --check`

Expected: `Validated versioned historical demo evidence.`

Run: `python -m pytest tests/test_demo.py -q`

Expected: PASS without a `reports/` dependency.

- [ ] **Step 8: Commit**

```powershell
git add .gitignore scripts/export_demo_evidence.py demo/artifacts demo/generate_xgboost_demo.py demo/open_demo.ps1 tests/test_demo.py src/evaluation/label_maturity.py
git commit -m "fix: ship portable invalidation-aware demo evidence"
```

---

### Task 8: Add a reproducible development temporal-audit command

**Files:**
- Create: `scripts/audit_temporal_folds.py`
- Modify: `tests/test_temporal.py`
- Create: `docs/audits/2026-08-17-temporal-validation-audit.md`

**Interfaces:**
- Consumes: a development-only feature parquet with `filing_date`, `reporting_date`, `filing_year`, and `fraudulent`.
- Produces: process exit 0 only when all evaluated folds are strictly ordered; JSON to stdout with row and overlap counts.

- [ ] **Step 1: Write failing CLI tests with a temporary Parquet fixture**

```python
def test_audit_exits_nonzero_for_overlapping_reporting_year_folds(tmp_path: Path) -> None:
    path = tmp_path / "dev.parquet"
    pd.DataFrame(
        {
            "filing_date": ["31-03-2018", "15-02-2018"],
            "reporting_date": ["31-12-2016", "31-12-2017"],
            "filing_year": [2016, 2017],
            "fraudulent": [0, 1],
        }
    ).to_parquet(path)
    completed = subprocess.run(
        [sys.executable, "scripts/audit_temporal_folds.py", "--input", str(path)],
        cwd=ROOT,
    )
    assert completed.returncode == 1
```

- [ ] **Step 2: Implement the audit**

The command must report:

```json
{
  "rows": 41748,
  "filing_year_mismatch_count": 0,
  "evaluated_folds": 21,
  "folds_with_temporal_overlap": 0,
  "max_overlapping_train_rows": 0,
  "status": "pass"
}
```

Return 1 if any parsed date is invalid, any stored `filing_year` differs from actual filing year, or any training filing is on/after the earliest filing in its test fold.

- [ ] **Step 3: Record the original measured evidence**

The audit document must record the verified pre-fix facts:

- 41,748 development rows;
- 30,582 filing-year mismatches (`73.2538%`);
- temporal overlap in all 21 evaluated folds;
- as many as 2,111 overlapping training rows in one fold;
- Trial 196 development selection and its 26.22% Recall@5% are invalidated;
- the 2019-2022 result remains historical and must not be rerun.

- [ ] **Step 4: Run synthetic checks**

Run: `python -m pytest tests/test_temporal.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/audit_temporal_folds.py tests/test_temporal.py docs/audits/2026-08-17-temporal-validation-audit.md
git commit -m "test: add strict temporal development audit"
```

---

### Task 9: Expand documentation and repository-wide regression coverage

**Files:**
- Modify: `README.md`
- Rewrite: `docs/methodology.md`
- Modify: `docs/model_card.md`
- Modify: `docs/future_work.md`
- Modify: `tests/test_repository_integrity.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the corrected APIs and statuses from Tasks 1-8.
- Produces: consistent claims, explicit commands, and a fast whole-repository validation suite.

- [ ] **Step 1: Add repository integrity assertions**

```python
def test_no_source_directory_is_gitignored() -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "src/evaluation/label_maturity.py"],
        cwd=ROOT,
    )
    assert completed.returncode == 1


def test_placeholder_test_files_are_gone() -> None:
    for name in ("test_ingestion.py", "test_preprocessing.py", "test_pipeline.py"):
        source = (ROOT / "tests" / name).read_text(encoding="utf-8")
        assert "def test_" in source
```

- [ ] **Step 2: Rewrite methodology around exact availability time**

Document:

1. actual `filing_date` defines fold membership;
2. `reporting_year` is descriptive/lookup metadata only;
3. every fold enforces `max(train filing_date) < min(test filing_date)`;
4. threshold-dependent metrics use a fixed 0.50 threshold;
5. Recall@5% and Precision@5% define the analyst queue;
6. Brier skill eligibility and PR-AUC tie-breaking remain unchanged;
7. artifacts are bound to data/configuration hashes;
8. AAER `dateTime` remains only a source-record timestamp;
9. the old final period is permanently closed.

- [ ] **Step 3: Correct every result claim**

Replace statements that call Trial 196 development validation “chronological” or “leakage-aware” with explicit invalidation language. Keep old numbers only in a historical-results table whose status column says `Invalidated temporal validation` or `Historical nonconfirmatory final result`.

- [ ] **Step 4: Correct future-work rules**

Replace “use the existing walk-forward folds” with “use strict filing-date folds generated by `src.evaluation.temporal`.” Preserve the prohibition on in-sample threshold-tuned F1 and add the separate approval requirement for expensive development experiments.

- [ ] **Step 5: Keep Ruff scoped to executable source and explicitly handle notebooks**

Add a Ruff exclusion for placeholder/archival notebooks or clean their imports in a separate formatting-only commit. The documented command must remain:

```powershell
python -m ruff check --no-cache src tests configs demo scripts
python -m ruff format --check --no-cache src tests configs demo scripts
```

- [ ] **Step 6: Run the fast full suite**

Run: `python -m pytest tests -q`

Expected: all tests pass without training and without reading the final-test parquet.

Run: `python -m ruff check --no-cache src tests configs demo scripts`

Expected: `All checks passed!`

Run: `python -m ruff format --check --no-cache src tests configs demo scripts`

Expected: all files already formatted.

- [ ] **Step 7: Commit**

```powershell
git add README.md docs/methodology.md docs/model_card.md docs/future_work.md tests/test_repository_integrity.py pyproject.toml
git commit -m "docs: correct evaluation and reproducibility claims"
```

---

### Task 10: Validate code fixes and prepare a separately approved development restart

**Files:**
- Modify: `README.md`
- Modify: `docs/model_card.md`
- Modify: `demo/artifacts/historical_evidence.json`
- Modify: `demo/index.html`

**Interfaces:**
- Consumes: all corrected code and tests from Tasks 1-9.
- Produces: a verified repair commit and exact, approval-gated commands for development-only feature rebuilding and research restart.

- [ ] **Step 1: Run non-model validation**

Run:

```powershell
python -m compileall -q src configs demo scripts
python -m pytest tests -q
python -m ruff check --no-cache src tests configs demo scripts
python -m ruff format --check --no-cache src tests configs demo scripts
python demo/generate_xgboost_demo.py --check
git diff --check
```

Expected: every command exits 0; no command imports or fits XGBoost; no command accesses the test-feature parquet.

- [ ] **Step 2: Review changed-file scope**

Run: `git status --short`

Expected: only files listed in this plan plus preexisting user changes. Confirm `logs/pipeline.log` is no longer tracked and `src/evaluation/label_maturity.py` is trackable.

- [ ] **Step 3: Stop and request approval for the development-only rebuild**

Do not run this command automatically. After explicit user approval, provide the user this exact command to run locally:

```powershell
python -c "from src.features.lm_features import engineer_development_features; print(engineer_development_features())"
```

This may read the development split and LM source CSV, but it must not read or write `test_features.parquet`.

- [ ] **Step 4: Audit the rebuilt development features**

After the user reports that the rebuild completed, run:

```powershell
python scripts/audit_temporal_folds.py --input data/processed/features/trainval_features.parquet
```

Expected JSON fields:

```json
{
  "filing_year_mismatch_count": 0,
  "folds_with_temporal_overlap": 0,
  "max_overlapping_train_rows": 0,
  "status": "pass"
}
```

If any field differs, do not start modeling.

- [ ] **Step 5: Stop and request separate approval for a new expensive experiment**

The old Optuna database and candidate review are invalid because their objective folds used reporting years. Do not reuse them. After explicit user approval, instruct the user to run the new v3 development experiment with:

```powershell
python -c "from src.models.xgboost_model import run_xgboost; run_xgboost(optimize=True, calibrate=True, run_shap=False)"
```

Expected outcomes are either:

- `status: selected` under `reports/models/xgboost_lm_text_surface_probability_v3_strict_time/`, or
- `status: no_eligible_candidate`, with no model or SHAP artifact.

This command must never call the historical final-test evaluator.

- [ ] **Step 6: Keep SHAP behind its own approval**

Only if the new development review selects an eligible candidate and the user separately approves SHAP, provide:

```powershell
python -c "from src.models.xgboost_model import run_selected_xgboost_shap; run_selected_xgboost_shap()"
```

- [ ] **Step 7: Do not produce a replacement frozen-test number**

Do not run `evaluate_selected_xgboost_on_test()`. The v3 model may report development-only results until a genuinely new, preregistered, label-mature future holdout exists. Keep the historical 2019-2022 metrics visually separated from v3 development results.

- [ ] **Step 8: Regenerate and verify the static page**

Run: `python demo/generate_xgboost_demo.py`

Expected: `demo/index.html` retains the invalidation warning and archived historical evidence. It must not imply that a new v3 model was tested on 2019-2022.

- [ ] **Step 9: Commit the verified repair state**

```powershell
git add README.md docs/model_card.md demo/artifacts/historical_evidence.json demo/index.html
git commit -m "chore: finalize repository integrity repair"
```

---

## Completion Criteria

- Development folds are generated from actual filing dates and every fold passes the strict date-order invariant.
- `filing_year` always means actual filing year; `reporting_year` is separate.
- Candidate review uses fixed-threshold classification metrics and unchanged queue/calibration selection rules.
- New artifacts contain verifiable development-data, schema, configuration, and contract fingerprints.
- The historical final test is programmatically and procedurally closed.
- Future prediction outputs have stable source-record identifiers and atomic completion manifests.
- Raw validation imports and streams correctly.
- There is one import-safe, data-only pipeline entrypoint and no advertised placeholder model.
- A clean Python 3.12 environment has complete pinned dependencies.
- The demo works from tracked evidence without `reports/`, training, or sealed-test access.
- README, methodology, model card, demo, and audit document all disclose the temporal invalidation.
- Fast tests, Ruff, formatting, compile checks, and demo validation pass.
- No expensive experiment, candidate review, SHAP run, or final-test access occurs without its explicit approval gate.
