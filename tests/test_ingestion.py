"""Raw FinNLP loader regression tests."""

from pathlib import Path

from src.ingestion.load_finnlp_dataset import FinNLPDatasetLoader


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
