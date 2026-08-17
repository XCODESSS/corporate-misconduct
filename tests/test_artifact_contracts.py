"""Artifact provenance and one-shot output tests."""

import json
from pathlib import Path

import pytest
from src.evaluation.one_shot import OneShotEvaluationWriter
from src.models.xgboost_model import evaluate_selected_xgboost_on_test
from src.utils.fingerprints import sha256_json
from src.utils.record_ids import stable_filing_id


def test_sha256_json_is_key_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_stable_filing_id_is_deterministic() -> None:
    first = stable_filing_id("0000001750", "2019-07-18", "2018-12-31", "10-K")
    second = stable_filing_id("1750", "18-07-2019", "31-12-2018", "10-K")
    assert first == second
    assert len(first) == 64


def test_historical_final_test_is_permanently_closed() -> None:
    with pytest.raises(RuntimeError, match="permanently closed"):
        evaluate_selected_xgboost_on_test()


def test_one_shot_writer_blocks_second_start(tmp_path: Path) -> None:
    writer = OneShotEvaluationWriter(tmp_path / "evaluation")
    writer.begin({"contract": "fixture"})
    manifest = writer.commit_text_files({"summary.json": json.dumps({"ok": True})})
    assert manifest["status"] == "complete"
    with pytest.raises(FileExistsError):
        writer.begin({"contract": "fixture"})
