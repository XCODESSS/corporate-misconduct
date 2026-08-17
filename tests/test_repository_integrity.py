"""Clean-import and repository-contract checks."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    [
        "src.ingestion.validate_raw_data",
        "src.preprocessing.split_dataset",
        "src.features.lm_features",
        "src.evaluation.cross_validation",
        "src.models.xgboost_model",
    ],
)
def test_public_module_imports_without_stdout(module_name: str, capsys) -> None:
    importlib.import_module(module_name)
    assert capsys.readouterr().out == ""


def test_runtime_requirements_cover_specialized_imports() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("pyarrow", "ijson", "optuna", "xgboost", "shap", "joblib"):
        assert f"{package}==" in requirements


def test_placeholder_tests_have_real_test_functions() -> None:
    for name in ("test_ingestion.py", "test_preprocessing.py", "test_pipeline.py"):
        assert "def test_" in (ROOT / "tests" / name).read_text(encoding="utf-8")
