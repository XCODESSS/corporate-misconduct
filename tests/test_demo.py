"""Fast regression checks for the artifact-only research demo."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "demo" / "generate_xgboost_demo.py"


def load_demo_module():
    """Load the standalone generator without invoking its command-line entrypoint."""
    spec = importlib.util.spec_from_file_location("xgboost_demo", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_payload_matches_versioned_historical_evidence() -> None:
    """The portable demo must use its tracked historical evidence bundle."""
    demo = load_demo_module()
    payload = demo.build_payload()
    evidence = json.loads(
        (ROOT / "demo" / "artifacts" / "historical_evidence.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["trial"] == 196
    assert payload["development_validation_status"] == "invalidated_temporal_overlap"
    assert payload["frozen_result_status"] == "historical_nonconfirmatory"
    assert payload["final_test"] == evidence["final_test"]
    assert payload["final_test"]["fraud_found"] == 7
    assert payload["final_test"]["fraud_cases"] == 58


def test_demo_generation_is_fast_and_contains_required_cautions(tmp_path: Path) -> None:
    """The page is generated locally from artifacts and retains its research warning."""
    output = tmp_path / "research_demo.html"
    completed = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    page = output.read_text(encoding="utf-8")

    assert "Demo page written to" in completed.stdout
    assert "RESEARCH PROTOTYPE - NOT A DEPLOYMENT DECISION TOOL" in page
    assert "Recall@5% means:" in page
    assert "Development validation invalidated" in page
    assert "historical final result is nonconfirmatory" in page.lower()
    assert "label" in page.lower()
    assert '"fraud_found": 7' in page
    assert '"fraud_cases": 58' in page


def test_demo_source_does_not_invoke_training_or_sealed_evaluation() -> None:
    """The generator may read evidence, but cannot trigger model work."""
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    forbidden_calls = (
        "XGBoostBaseline(",
        "run_pipeline(",
        "review_top_candidates(",
        "run_final_test_evaluation(",
        "optuna.create_study(",
        ".fit(",
    )

    assert not [call for call in forbidden_calls if call in source]


def test_demo_source_does_not_depend_on_reports_directory() -> None:
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    build_payload_source = source[
        source.index("def build_payload") : source.index("def render_html")
    ]
    assert "REPORT_DIR" not in build_payload_source
    assert "EVIDENCE_DIR" in build_payload_source
