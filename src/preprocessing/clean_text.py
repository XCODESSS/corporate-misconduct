"""Clean raw MD&A text while preserving firm-year metadata.

This module is the first preprocessing stage. It streams raw firm-year records,
cleans only the ``mda`` field, skips records with empty MD&A text, and writes a
batched Parquet dataset for downstream preprocessing and feature generation.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import configs.settings as settings
from src.ingestion.load_finnlp_dataset import FinNLPDatasetLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_OUTPUT_FILE = (
    settings.INTERIM_CLEANED_DIR / "cleaned_firm_years.parquet"
)


@dataclass(slots=True)
class TextCleaningResult:
    """Summary of the MD&A cleaning run."""

    input_records: int = 0
    written_records: int = 0
    skipped_empty_mda: int = 0
    output_path: Path = DEFAULT_OUTPUT_FILE


class MdaTextCleaner:
    """Stream firm-year records and clean only the MD&A text field."""

    def __init__(
        self,
        loader: FinNLPDatasetLoader | None = None,
        output_path: Path | str = DEFAULT_OUTPUT_FILE,
        batch_size: int = 10_000,
    ) -> None:
        self.loader = loader or FinNLPDatasetLoader()
        self.output_path = Path(output_path)
        self.batch_size = batch_size

    def clean_to_parquet(self) -> TextCleaningResult:
        """Clean raw firm-year MD&A text and write batched Parquet output."""

        logger.info("Starting MD&A text cleaning.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        result = TextCleaningResult(output_path=self.output_path)
        writer: pq.ParquetWriter | None = None
        batch: list[dict[str, Any]] = []

        try:
            for record in self.loader.stream_firm_years():
                result.input_records += 1
                cleaned_record = self._clean_record(record)

                if cleaned_record is None:
                    result.skipped_empty_mda += 1
                    self._log_skipped_empty_mda(record, result.input_records)
                    continue

                batch.append(cleaned_record)

                if len(batch) >= self.batch_size:
                    writer = self._write_batch(batch, writer)
                    result.written_records += len(batch)
                    batch.clear()

            if batch:
                writer = self._write_batch(batch, writer)
                result.written_records += len(batch)
                batch.clear()
        finally:
            if writer is not None:
                writer.close()

        logger.info(
            "MD&A cleaning complete | input=%d | written=%d | skipped=%d",
            result.input_records,
            result.written_records,
            result.skipped_empty_mda,
        )
        logger.info("Cleaned firm-year dataset saved to %s", self.output_path)
        return result

    def _clean_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Return a metadata-preserving record with cleaned MD&A text."""

        raw_mda = record.get("mda")
        cleaned_mda = clean_mda_text(raw_mda)

        if not cleaned_mda:
            return None

        cleaned_record = dict(record)
        cleaned_record["mda"] = cleaned_mda
        return cleaned_record

    def _write_batch(
        self,
        batch: list[dict[str, Any]],
        writer: pq.ParquetWriter | None,
    ) -> pq.ParquetWriter:
        """Write one batch to Parquet and return the active writer."""

        schema = writer.schema if writer else None
        table = self._records_to_table(batch, schema)

        if writer is None:
            writer = pq.ParquetWriter(
                self.output_path,
                table.schema,
                compression="snappy",
            )

        writer.write_table(table)
        return writer

    @staticmethod
    def _records_to_table(
        records: list[dict[str, Any]],
        schema: pa.Schema | None,
    ) -> pa.Table:
        """Convert a batch of records to an Arrow table."""

        frame = pd.DataFrame.from_records(records)

        if schema is not None:
            for field in schema:
                if field.name not in frame.columns:
                    frame[field.name] = None
            frame = frame.select_dtypes(exclude=["object"]).join(
                frame.select_dtypes(include=["object"]).astype("string")
            )[schema.names]
            return pa.Table.from_pandas(
                frame,
                schema=schema,
                preserve_index=False,
            )

        object_columns = frame.select_dtypes(include=["object"]).columns
        frame[object_columns] = frame[object_columns].astype("string")
        return pa.Table.from_pandas(frame, preserve_index=False)

    @staticmethod
    def _log_skipped_empty_mda(
        record: dict[str, Any],
        record_number: int,
    ) -> None:
        """Log skipped records without logging the MD&A body."""

        logger.warning(
            "Skipping firm-year record with empty MD&A | record=%d | cik=%s | "
            "filing_date=%s",
            record_number,
            record.get("cik"),
            record.get("filing_date"),
        )


def clean_mda_text(value: Any) -> str:
    """Clean a single MD&A text value without tokenizing or stemming.

    The cleaner decodes HTML entities, removes HTML tags, normalizes Unicode,
    removes control characters, and collapses whitespace. It preserves sentence
    punctuation, numbers, percentages, currency values, and other financial
    information present in the original text.
    """

    if value is None:
        return ""

    text = str(value)
    if not text.strip():
        return ""

    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _remove_control_characters(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _remove_control_characters(text: str) -> str:
    """Remove Unicode control characters while preserving word separation."""

    cleaned_characters: list[str] = []

    for character in text:
        if unicodedata.category(character)[0] == "C":
            cleaned_characters.append(" ")
        else:
            cleaned_characters.append(character)

    return "".join(cleaned_characters)


def stream_cleaned_records(
    loader: FinNLPDatasetLoader | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield cleaned records while skipping observations with empty MD&A."""

    active_loader = loader or FinNLPDatasetLoader()

    for record in active_loader.stream_firm_years():
        cleaned_mda = clean_mda_text(record.get("mda"))

        if not cleaned_mda:
            logger.warning(
                "Skipping firm-year record with empty MD&A | cik=%s | "
                "filing_date=%s",
                record.get("cik"),
                record.get("filing_date"),
            )
            continue

        cleaned_record = dict(record)
        cleaned_record["mda"] = cleaned_mda
        yield cleaned_record


def clean_firm_year_mda_text(
    output_path: Path | str = DEFAULT_OUTPUT_FILE,
    batch_size: int = 10_000,
) -> TextCleaningResult:
    """Clean raw firm-year MD&A text and write the interim Parquet file."""

    cleaner = MdaTextCleaner(output_path=output_path, batch_size=batch_size)
    return cleaner.clean_to_parquet()


if __name__ == "__main__":
    clean_firm_year_mda_text()

