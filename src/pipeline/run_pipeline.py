"""Explicit, data-only project pipeline with safety gates."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from src.features.lm_features import engineer_development_features
from src.ingestion.ingest_firm_years import ingest_firm_years
from src.ingestion.ingest_labels import ingest_labels
from src.ingestion.merge_labels import merge_labels
from src.ingestion.validate_raw_data import validate_raw_data
from src.preprocessing.clean_text import clean_firm_year_mda_text
from src.preprocessing.deduplicate import deduplicate_dataset
from src.preprocessing.normalize import normalize_dataset
from src.preprocessing.quality_checks import quality_check_dataset
from src.preprocessing.split_dataset import split_dataset

Stage = tuple[str, Callable[[], object]]
STAGES: tuple[Stage, ...] = (
    ("validate-raw", validate_raw_data),
    ("ingest-firm-years", ingest_firm_years),
    ("ingest-labels", ingest_labels),
    ("merge-labels", merge_labels),
    ("clean-text", clean_firm_year_mda_text),
    ("normalize", normalize_dataset),
    ("quality-check", quality_check_dataset),
    ("deduplicate", deduplicate_dataset),
    ("split", split_dataset),
    ("development-features", engineer_development_features),
)


def run_data_pipeline(through: str, acknowledge_test_write: bool = False) -> list[str]:
    names = [name for name, _ in STAGES]
    if through not in names:
        raise ValueError(f"Unknown pipeline stage: {through}")
    selected = STAGES[: names.index(through) + 1]
    if any(name == "split" for name, _ in selected) and not acknowledge_test_write:
        raise RuntimeError(
            "Stages at or beyond split require --acknowledge-test-write because "
            "the splitter writes held-out files."
        )
    completed: list[str] = []
    for name, function in selected:
        function()
        completed.append(name)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List data stages.")
    group.add_argument("--through", choices=[name for name, _ in STAGES])
    parser.add_argument("--acknowledge-test-write", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(name for name, _ in STAGES))
        return
    run_data_pipeline(args.through, args.acknowledge_test_write)


if __name__ == "__main__":
    main()
