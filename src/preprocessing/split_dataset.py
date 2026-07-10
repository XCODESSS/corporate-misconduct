"""
Dataset splitting.

Responsibilities
----------------
- Read the quality-checked dataset.
- Create a temporal train/validation pool and a held-out test set.
- Save both splits.
- Generate a split report.

Split strategy
--------------
Train/Validation pool : filing_year < 2013
    Used for walk-forward cross-validation during modeling.
    Walk-forward folds are generated dynamically in src/models/cross_validation.py.

Test set : filing_year >= 2013
    Held out. Never touched during training or tuning.
    Used only for final evaluation.

This module DOES NOT

- clean text
- normalize text
- engineer features
- train models
- implement walk-forward folds (that belongs in models/cross_validation.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import configs.settings as settings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Temporally split the quality-checked dataset into
    a train/validation pool and a held-out test set.
    """

    INPUT_FILE = settings.INTERIM_CLEANED_DIR / "deduplicated_firm_years.parquet"

    OUTPUT_DIR = settings.DATASETS_DIR

    TRAINVAL_FILE = OUTPUT_DIR / "trainval_dataset.parquet"

    TEST_FILE = OUTPUT_DIR / "test_dataset.parquet"

    REPORT_FILE = settings.INTERIM_VALIDATED_DIR / "split_report.json"

    TEST_CUTOFF_YEAR = 2019

    def __init__(self) -> None:
        self.total_records = 0
        self.trainval_records = 0
        self.test_records = 0

    # ============================================================
    # Load
    # ============================================================

    def load_dataset(self) -> pa.Table:
        logger.info("Reading quality-checked dataset...")

        table = pq.read_table(self.INPUT_FILE)

        self.total_records = table.num_rows

        logger.info(
            "Loaded %d records.",
            self.total_records,
        )

        return table

    # ============================================================
    # Split
    # ============================================================

    def split_dataset(
        self,
        table: pa.Table,
    ) -> tuple[pa.Table, pa.Table]:
        """
        Split dataset by filing year.

        No shuffling. No randomness. Temporal only.

        Note
        ----
        table.to_pandas() loads all 65k records into memory.
        Acceptable at this scale. For 500k+ records, consider
        a streaming Arrow-native approach.
        """

        df = table.to_pandas()

        df["_year"] = pd.to_datetime(
            df["filing_date"],
            dayfirst=True,
            errors="coerce",
        ).dt.year

        trainval_df = df[df["_year"] < self.TEST_CUTOFF_YEAR].drop(columns=["_year"])

        test_df = df[df["_year"] >= self.TEST_CUTOFF_YEAR].drop(columns=["_year"])

        trainval_table = pa.Table.from_pandas(
            trainval_df,
            preserve_index=False,
        )

        test_table = pa.Table.from_pandas(
            test_df,
            preserve_index=False,
        )

        self.trainval_records = trainval_table.num_rows
        self.test_records = test_table.num_rows

        logger.info(
            "Train/Val pool=%d | Test=%d",
            self.trainval_records,
            self.test_records,
        )

        return trainval_table, test_table

    # ============================================================
    # Save
    # ============================================================

    def save_dataset(
        self,
        table: pa.Table,
        output_file: Path,
    ) -> None:
        pq.write_table(
            table,
            output_file,
            compression="snappy",
        )

        logger.info(
            "Saved %s (%d rows)",
            output_file.name,
            table.num_rows,
        )

    def save_splits(
        self,
        trainval_table: pa.Table,
        test_table: pa.Table,
    ) -> None:
        logger.info("Saving splits...")

        self.save_dataset(
            trainval_table,
            self.TRAINVAL_FILE,
        )

        self.save_dataset(
            test_table,
            self.TEST_FILE,
        )

        logger.info("All splits saved.")

    # ============================================================
    # Validation
    # ============================================================

    def validate_splits(
        self,
        trainval_table: pa.Table,
        test_table: pa.Table,
    ) -> None:
        logger.info("Validating splits...")

        total = trainval_table.num_rows + test_table.num_rows

        if total != self.total_records:
            raise ValueError(
                f"Split validation failed. "
                f"Expected {self.total_records} "
                f"but found {total}."
            )

        logger.info("Validation successful.")

    # ============================================================
    # Report
    # ============================================================

    def report(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "trainval_records": self.trainval_records,
            "test_records": self.test_records,
            "test_cutoff_year": self.TEST_CUTOFF_YEAR,
            "split_method": "temporal",
            "walk_forward_note": (
                "Walk-forward CV folds are generated dynamically "
                "in src/evaluation/cross_validation.py over trainval_dataset.parquet"
            ),
            "trainval_ratio": round(self.trainval_records / self.total_records, 4),
            "test_ratio": round(self.test_records / self.total_records, 4),
        }

    def write_report(self) -> None:
        logger.info("Writing split report...")

        with open(
            self.REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.report(),
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "Split report written to: %s",
            self.REPORT_FILE,
        )

    # ============================================================
    # Summary
    # ============================================================

    def log_summary(self) -> None:
        logger.info("=" * 70)
        logger.info("Dataset Split Summary")
        logger.info("=" * 70)

        for key, value in self.report().items():
            logger.info("%s: %s", key, value)

        logger.info("=" * 70)

    # ============================================================
    # Pipeline
    # ============================================================

    def run(self) -> None:
        logger.info("=" * 70)
        logger.info("Starting dataset split pipeline...")

        table = self.load_dataset()

        trainval_table, test_table = self.split_dataset(table)

        self.validate_splits(trainval_table, test_table)

        self.save_splits(trainval_table, test_table)

        self.write_report()

        self.log_summary()

        logger.info("Dataset splitting complete.")
        logger.info("=" * 70)


# ============================================================
# Public API
# ============================================================


def split_dataset() -> None:
    DatasetSplitter().run()


def main() -> None:
    split_dataset()


if __name__ == "__main__":
    main()
