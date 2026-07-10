"""
Remove duplicate firm-year filings.

Responsibilities
----------------
- Read the normalized dataset.
- Detect duplicate filings.
- Preserve one canonical record.
- Generate a deduplication report.
- Write a deduplicated dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import configs.settings as settings
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    INPUT_FILE = settings.INTERIM_CLEANED_DIR / "quality_checked_firm_years.parquet"
    OUTPUT_FILE = settings.INTERIM_CLEANED_DIR / "deduplicated_firm_years.parquet"
    REPORT_FILE = settings.INTERIM_VALIDATED_DIR / "deduplication_report.json"

    def __init__(self) -> None:
        self.rows_read = 0
        self.rows_written = 0
        self.duplicate_rows = 0
        self.unique_keys: set[tuple] = set()
        self.duplicate_examples: list[dict[str, Any]] = []

    @staticmethod
    def build_key(record: dict[str, Any]) -> tuple:
        return (record.get("cik"), record.get("filing_date"))

    def is_duplicate(self, record: dict[str, Any]) -> bool:
        key = self.build_key(record)
        if key in self.unique_keys:
            self.duplicate_rows += 1
            if len(self.duplicate_examples) < 25:
                self.duplicate_examples.append(
                    {"cik": record.get("cik"), "filing_date": record.get("filing_date")}
                )
            return True
        self.unique_keys.add(key)
        return False

    def process_batch(self, batch: pa.RecordBatch) -> pa.Table:
        records = pa.Table.from_batches([batch]).to_pylist()
        unique = []
        for record in records:
            self.rows_read += 1
            if self.is_duplicate(record):
                continue
            unique.append(record)
            self.rows_written += 1
        return pa.Table.from_pylist(unique)

    def validate_output(self) -> None:
        logger.info("Validating output...")
        parquet = pq.ParquetFile(self.OUTPUT_FILE)
        rows = 0
        seen = set()
        for batch in parquet.iter_batches(batch_size=512):
            rows += batch.num_rows
            for record in pa.Table.from_batches([batch]).to_pylist():
                key = self.build_key(record)
                if key in seen:
                    raise ValueError("Duplicate found after deduplication.")
                seen.add(key)
        if rows != self.rows_written:
            raise ValueError(f"Row mismatch: expected {self.rows_written}, got {rows}")
        logger.info("Validation successful.")

    def report(self) -> dict[str, Any]:
        pct = (self.duplicate_rows / self.rows_read * 100) if self.rows_read else 0
        return {
            "input_records": self.rows_read,
            "output_records": self.rows_written,
            "duplicate_records": self.duplicate_rows,
            "duplicate_percentage": round(pct, 4),
            "duplicate_key": ["cik", "filing_date"],
            "sample_duplicate_records": self.duplicate_examples,
        }

    def write_report(self) -> None:
        with open(self.REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.report(), f, indent=4, ensure_ascii=False)

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Deduplication Summary")
        logger.info("Rows read: %d", self.rows_read)
        logger.info("Rows written: %d", self.rows_written)
        logger.info("Duplicates: %d", self.duplicate_rows)
        logger.info("=" * 60)

    def run(self) -> Path:
        logger.info("Reading normalized dataset...")
        parquet = pq.ParquetFile(self.INPUT_FILE)
        writer = None
        try:
            for batch_no, batch in enumerate(
                parquet.iter_batches(batch_size=4096), start=1
            ):
                table = self.process_batch(batch)
                if table.num_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            self.OUTPUT_FILE, table.schema, compression="snappy"
                        )
                    writer.write_table(table)
                logger.info(
                    "Batch %d | Read=%d Written=%d Duplicates=%d",
                    batch_no,
                    self.rows_read,
                    self.rows_written,
                    self.duplicate_rows,
                )
        finally:
            if writer:
                writer.close()
        self.validate_output()
        self.write_report()
        return self.OUTPUT_FILE


def deduplicate_dataset() -> Path:
    d = Deduplicator()
    out = d.run()
    d.log_summary()
    return out


def main() -> None:
    logger.info("=" * 70)
    logger.info("Starting deduplication pipeline...")
    output = deduplicate_dataset()
    logger.info("Deduplicated dataset written to: %s", output)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
