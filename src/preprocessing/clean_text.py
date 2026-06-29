"""
Clean raw MD&A text while preserving firm-year metadata.

Responsibilities
----------------
- Read labeled_firm_years.parquet.
- Clean only the mda field.
- Skip records with empty MD&A text.
- Preserve all metadata columns including fraudulent label.
- Write cleaned dataset to Parquet.

This module DOES NOT

- normalize text
- deduplicate records
- engineer features
- modify fraud labels
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import configs.settings as settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

HTML_TAG_RE   = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


class MdaTextCleaner:
    """
    Clean MD&A text in the labeled firm-year dataset.
    """

    INPUT_FILE = (
        settings.INTERIM_DIR
        / "labeled_firm_years.parquet"
    )

    OUTPUT_FILE = (
        settings.INTERIM_CLEANED_DIR
        / "cleaned_firm_years.parquet"
    )

    def __init__(
        self,
        batch_size: int = 10_000,
    ) -> None:

        self.batch_size    = batch_size
        self.rows_read     = 0
        self.rows_written  = 0
        self.rows_skipped  = 0

    # ============================================================
    # Text Cleaning
    # ============================================================

    @staticmethod
    def clean_mda_text(value: Any) -> str:

        if value is None:
            return ""

        return str(value)

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """
        Replace Unicode control characters with spaces.
        """

        return "".join(
            " " if unicodedata.category(ch)[0] == "C" else ch
            for ch in text
        )

    # ============================================================
    # Record Processing
    # ============================================================

    def process_record(
    self,
    record: dict[str, Any],
) -> dict[str, Any]:
        """
        Clean a single record.

        Every record is preserved.
        Empty MD&A documents are handled later in
        quality_checks.py.
        """

        cleaned_record = dict(record)

        cleaned_record["mda"] = self.clean_mda_text(
            record.get("mda")
        )

        return cleaned_record
    # ============================================================
    # Batch Writing
    # ============================================================

    def write_batch(
        self,
        batch: list[dict[str, Any]],
        writer: pq.ParquetWriter | None,
    ) -> pq.ParquetWriter:
        """
        Write one batch to Parquet.
        """

        import pandas as pd

        frame = pd.DataFrame.from_records(batch)

        object_cols = frame.columns[
            frame.dtypes == "object"
        ]

        if len(object_cols) > 0:
            frame[object_cols] = (
                frame[object_cols]
                .astype("string")
            )

        if writer is not None:
            table = pa.Table.from_pandas(
                frame,
                schema=writer.schema,
                preserve_index=False,
            )
        else:
            table = pa.Table.from_pandas(
                frame,
                preserve_index=False,
            )
            writer = pq.ParquetWriter(
                self.OUTPUT_FILE,
                table.schema,
                compression="snappy",
            )

        writer.write_table(table)

        return writer

    # ============================================================
    # Pipeline
    # ============================================================

    def run(self) -> None:
        """
        Stream labeled_firm_years.parquet,
        clean MD&A text, write cleaned output.
        """

        logger.info("=" * 70)
        logger.info("Starting MD&A text cleaning...")

        self.OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        parquet = pq.ParquetFile(self.INPUT_FILE)
        writer: pq.ParquetWriter | None = None
        batch: list[dict[str, Any]] = []

        try:

            for arrow_batch in parquet.iter_batches(
                batch_size=self.batch_size
            ):

                records = (
                    pa.Table
                    .from_batches([arrow_batch])
                    .to_pylist()
                )

                for record in records:

                    self.rows_read += 1

                    cleaned = self.process_record(record)

                    if cleaned is None:
                        continue

                    batch.append(cleaned)

                    if len(batch) >= self.batch_size:
                        writer = self.write_batch(batch, writer)
                        self.rows_written += len(batch)
                        batch.clear()

            if batch:
                writer = self.write_batch(batch, writer)
                self.rows_written += len(batch)
                batch.clear()

        finally:

            if writer is not None:
                writer.close()

        logger.info(
            "Cleaning complete | read=%d | written=%d | skipped=%d",
            self.rows_read,
            self.rows_written,
            self.rows_skipped,
        )

        logger.info(
            "Output: %s",
            self.OUTPUT_FILE,
        )

        logger.info("=" * 70)


# ============================================================
# Public API
# ============================================================

def clean_firm_year_mda_text() -> None:
    MdaTextCleaner().run()


def main() -> None:
    clean_firm_year_mda_text()


if __name__ == "__main__":
    main()