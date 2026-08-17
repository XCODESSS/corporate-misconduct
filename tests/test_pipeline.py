"""Data-pipeline safety tests."""

import pytest
from src.pipeline.run_pipeline import STAGES, run_data_pipeline


def test_pipeline_requires_acknowledgement_before_split() -> None:
    with pytest.raises(RuntimeError, match="acknowledge-test-write"):
        run_data_pipeline("split", acknowledge_test_write=False)


def test_pipeline_contains_no_model_or_final_test_stage() -> None:
    names = {name for name, _ in STAGES}
    forbidden = {"optuna", "xgboost", "candidate-review", "shap", "final-test"}
    assert not names & forbidden
