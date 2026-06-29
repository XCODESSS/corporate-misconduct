"""
Quality checks for the modeling dataset.

Responsibilities
----------------
- Remove MD&A sections with fewer than 200 words.
- Remove amended filings.
- Remove filings before 1993.
- Produce a filtered dataset.
- Generate a quality check report.

This module DOES NOT

- clean text
- normalize text
- deduplicate
- merge labels
- generate features
"""

from __future__ import annotations


import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import configs.settings as settings

from src.utils.logger import get_logger



f = pq.ParquetFile(r"D:\coperate-misconduct-warning\data\interim\cleaned\deduplicated_firm_years.parquet")
batch = next(f.iter_batches(batch_size=5))

rows = pa.Table.from_batches([batch]).to_pylist()
for r in rows:
    print(r.get("filing_type"), "|", r.get("filing_date"), "|", len((r.get("mda") or "").split()))
logger = get_logger(__name__)


class QualityChecker:
    """
    Apply dataset quality filters prior to modeling.
    """

    INPUT_FILE = (
        settings.INTERIM_CLEANED_DIR
        / "deduplicated_firm_years.parquet"
    )

    OUTPUT_FILE = (
        settings.INTERIM_CLEANED_DIR
        / "quality_checked_firm_years.parquet"
    )

    REPORT_FILE = (
        settings.INTERIM_VALIDATED_DIR
        / "quality_check_report.json"
    )

    MIN_WORDS = 200

    EXCLUDED_FILINGS = {
        "10-K/A",
        "10-K405/A",
        "10-KT/A",
    }

    def __init__(self) -> None:

        self.rows_read = 0
        self.rows_written = 0

        self.short_mda_removed = 0
        self.amended_removed = 0
        self.pre1993_removed = 0

    # ============================================================
    # Utility Functions
    # ============================================================

    @staticmethod
    def word_count(text: str | None) -> int:
        """
        Count words in an MD&A section.
        """

        if not text:
            return 0

        return len(
            re.findall(
                r"\b\w+\b",
                text,
            )
        )

    @staticmethod
    def parse_year(
        date_value: str | None,
    ) -> int | None:
        """
        Extract year from filing_date.
        """

        if not date_value:
            return None

        formats = (
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y",
        )

        for fmt in formats:

            try:

                return datetime.strptime(
                    date_value,
                    fmt,
                ).year

            except ValueError:

                continue

        return None

    # ============================================================
    # Filtering Rules
    # ============================================================

    def should_keep(
        self,
        record: dict[str, Any],
    ) -> bool:

        keep = True

        mda = record.get("mda", "")

        if self.word_count(mda) < self.MIN_WORDS:

            self.short_mda_removed += 1

            keep = False

        filing_type = (
            record.get("filing_type", "")
            .strip()
            .upper()
        )

        if filing_type in self.EXCLUDED_FILINGS:

            self.amended_removed += 1

            keep = False

        year = self.parse_year(
            record.get("filing_date")
        )

        if year is None or year < 1993:

            self.pre1993_removed += 1

            keep = False

        return keep
        # ============================================================
    # Batch Processing
    # ============================================================

    def process_batch(
        self,
        batch: pa.RecordBatch,
    ) -> pa.Table:
        """
        Apply quality filters to a single batch.
        """

        records = (
            pa.Table
            .from_batches([batch])
            .to_pylist()
        )

        filtered_records = []

        for record in records:

            self.rows_read += 1

            if not self.should_keep(record):

                continue

            filtered_records.append(record)

            self.rows_written += 1

        return pa.Table.from_pylist(filtered_records)

    # ============================================================
    # Pipeline
    # ============================================================

    def run(self) -> Path:
        """
        Execute the quality-check pipeline.
        """

        logger.info("=" * 70)

        logger.info(
            "Starting quality checks..."
        )

        logger.info(
            "Reading deduplicated dataset..."
        )

        parquet = pq.ParquetFile(
            self.INPUT_FILE
        )

        writer = None

        batch_number = 0

        try:

            for batch in parquet.iter_batches(
                batch_size=256
            ):

                batch_number += 1

                table = self.process_batch(
                    batch
                )

                if table.num_rows > 0:

                    if writer is None:

                        writer = pq.ParquetWriter(
                            where=self.OUTPUT_FILE,
                            schema=table.schema,
                            compression="snappy",
                        )

                    writer.write_table(
                        table
                    )

                logger.info(
                    (
                        "Batch %d | "
                        "Read=%d | "
                        "Written=%d"
                    ),
                    batch_number,
                    self.rows_read,
                    self.rows_written,
                )

        finally:

            if writer is not None:

                writer.close()

        logger.info(
            "Quality checks complete."
        )

        self.validate_output()

        self.write_report()

        return self.OUTPUT_FILE

    # ============================================================
    # Validation
    # ============================================================

    def validate_output(
        self,
    ) -> None:
        """
        Validate the filtered dataset.
        """

        logger.info(
            "Validating filtered dataset..."
        )

        parquet = pq.ParquetFile(
            self.OUTPUT_FILE
        )

        rows = 0

        for batch in parquet.iter_batches(
            batch_size=512
        ):

            rows += batch.num_rows

            records = (
                pa.Table
                .from_batches([batch])
                .to_pylist()
            )

            for record in records:

                # Word count

                if (
                    self.word_count(
                        record.get("mda", "")
                    )
                    < self.MIN_WORDS
                ):

                    raise ValueError(
                        "Short MD&A found "
                        "after filtering."
                    )

                # Filing type

                filing_type = (
                    record.get(
                        "filing_type",
                        "",
                    )
                    .strip()
                    .upper()
                )

                if (
                    filing_type
                    in self.EXCLUDED_FILINGS
                ):

                    raise ValueError(
                        "Amended filing "
                        "found after filtering."
                    )

                # Filing year

                year = self.parse_year(
                    record.get(
                        "filing_date"
                    )
                )

                if (
                    year is None
                    or year < 1993
                ):

                    raise ValueError(
                        "Pre-1993 filing "
                        "found after filtering."
                    )

        if rows != self.rows_written:

            raise ValueError(
                (
                    "Output row count "
                    "does not match "
                    "written row count."
                )
            )

        logger.info(
            "Validation successful."
        )
        # ============================================================
    # Reporting
    # ============================================================

    def report(self) -> dict[str, Any]:
        """
        Build the quality-check report.
        """

        total_removed = (
            self.rows_read
            - self.rows_written
        )

        return {

            "input_records":
                self.rows_read,

            "output_records":
                self.rows_written,

            "total_removed":
                total_removed,

            "removed_short_mda":
                self.short_mda_removed,

            "removed_amended_filings":
                self.amended_removed,

            "removed_pre1993":
                self.pre1993_removed,

            "minimum_word_count":
                self.MIN_WORDS,

            "excluded_filing_types":
                sorted(
                    self.EXCLUDED_FILINGS
                ),

        }

    def write_report(self) -> None:
        """
        Save the quality-check report.
        """

        logger.info(
            "Writing quality report..."
        )

        with open(
            self.REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.report(),
                file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "Quality report written to:"
        )

        logger.info(
            "%s",
            self.REPORT_FILE,
        )

    # ============================================================
    # Summary
    # ============================================================

    def summary(
        self,
    ) -> dict[str, int]:

        return {

            "rows_read":
                self.rows_read,

            "rows_written":
                self.rows_written,

            "removed_short_mda":
                self.short_mda_removed,

            "removed_amended":
                self.amended_removed,

            "removed_pre1993":
                self.pre1993_removed,

            "total_removed":
                self.rows_read
                - self.rows_written,

        }

    def log_summary(
        self,
    ) -> None:

        logger.info(
            "=" * 70
        )

        logger.info(
            "Quality Check Summary"
        )

        logger.info(
            "=" * 70
        )

        stats = self.summary()

        for key, value in stats.items():

            logger.info(
                "%s: %s",
                key,
                value,
            )

        logger.info(
            "=" * 70
        )
    # ============================================================
# Public API
# ============================================================


def quality_check_dataset() -> Path:
    """
    Execute the quality-check pipeline.

    Returns
    -------
    Path
        Path to the quality-checked dataset.
    """

    checker = QualityChecker()

    output = checker.run()

    checker.log_summary()

    return output


def main() -> None:
    """
    CLI entry point.
    """

    output = quality_check_dataset()

    logger.info(
        "Quality-checked dataset written to:"
    )

    logger.info(
        "%s",
        output,
    )


if __name__ == "__main__":
    main()