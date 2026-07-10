"""
Ingest AAER fraud periods.

Responsibilities
----------------
- Load aaer_mark5.csv.
- Parse fraud periods.
- Validate fraud period data.
- Create a clean fraud periods dataset.
- Save fraud periods as Parquet.
- Generate an ingestion report.

This module DOES NOT

- merge labels
- clean text
- normalize text
- deduplicate firm-years
"""

from __future__ import annotations

import calendar
import json
from typing import Any

import configs.settings as settings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudPeriodsIngestion:
    """
    Build a clean fraud-period dataset from the
    AAER Mark 5 dataset.
    """

    INPUT_FILE = settings.FINNLP_DIR / "aaer_mark5.csv"

    OUTPUT_FILE = settings.INTERIM_DIR / "fraud_periods.parquet"

    REPORT_FILE = settings.INTERIM_VALIDATED_DIR / "fraud_periods_report.json"

    COLUMN_NAMES = [
        "id",
        "dateTime",
        "aaerNo",
        "releaseNo",
        "respondents",
        "respondentsText",
        "urls",
        "fraud_start",
        "fraud_end",
        "cik",
        "revoked",
        "certainty_start",
        "certainty_end",
        "17a",
        "17a2",
        "17a3",
        "17b",
        "5a",
        "5b1",
        "5c",
        "10b",
        "13a",
        "12b20",
        "12b25",
        "13a1",
        "13a10",
        "13a11",
        "13a13",
        "13a14",
        "13a16",
        "13b2A",
        "13b2B",
        "13a15",
        "13b5",
        "14a",
        "14c",
        "19a",
        "30A",
        "100a2",
        "100b",
        "105c7B",
        "corruption",
        "amis",
        "fsf",
    ]

    OUTPUT_COLUMNS = [
        "cik",
        "fraud_start",
        "fraud_end",
        "certainty_start",
        "certainty_end",
    ]

    def __init__(self) -> None:
        self.rows_read = 0
        self.rows_written = 0

        self.missing_cik = 0
        self.missing_start = 0
        self.missing_end = 0

        self.invalid_periods = 0
        self.duplicate_periods = 0

    # ============================================================
    # Loading
    # ============================================================

    def load_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Load the AAER dataset.
        """

        logger.info("Loading AAER dataset...")

        dataframe = pd.read_csv(
            self.INPUT_FILE,
            sep=";",
            skiprows=2,
            header=None,
            names=self.COLUMN_NAMES,
            dtype=str,
            encoding="utf-8",
        )

        self.rows_read = len(dataframe)

        logger.info(
            "Loaded %d AAER records.",
            self.rows_read,
        )

        return dataframe

    # ============================================================
    # Date Parsing
    # ============================================================

    @staticmethod
    def parse_start_date(value: str) -> pd.Timestamp:
        """
        Convert MM-YYYY into the first day of the month.
        Returns NaT for missing or malformed values.
        """

        if pd.isna(value) or "-" not in str(value):
            return pd.NaT

        try:
            month, year = value.strip().split("-")
            return pd.Timestamp(
                year=int(year),
                month=int(month),
                day=1,
            )
        except (ValueError, TypeError):
            return pd.NaT

    @staticmethod
    def parse_end_date(value: str) -> pd.Timestamp:
        """
        Convert MM-YYYY into the last day of the month.
        Returns NaT for missing or malformed values.
        """

        if pd.isna(value) or "-" not in str(value):
            return pd.NaT

        try:
            month, year = value.strip().split("-")
            last_day = calendar.monthrange(
                int(year),
                int(month),
            )[1]
            return pd.Timestamp(
                year=int(year),
                month=int(month),
                day=last_day,
            )
        except (ValueError, TypeError):
            return pd.NaT

    # ============================================================
    # Preparation
    # ============================================================

    def prepare_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare the fraud periods dataset.
        """

        logger.info("Preparing fraud periods...")

        dataframe = dataframe.copy()

        dataframe["cik"] = dataframe["cik"].astype(str).str.strip()

        dataframe["fraud_start"] = dataframe["fraud_start"].apply(self.parse_start_date)

        dataframe["fraud_end"] = dataframe["fraud_end"].apply(self.parse_end_date)

        dataframe["certainty_start"] = (
            pd.to_numeric(
                dataframe["certainty_start"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        dataframe["certainty_end"] = (
            pd.to_numeric(
                dataframe["certainty_end"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        dataframe = dataframe[self.OUTPUT_COLUMNS]

        logger.info("Fraud periods prepared.")

        return dataframe

    # ============================================================
    # Validation
    # ============================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate fraud periods.
        """

        logger.info("Validating fraud periods...")

        self.missing_cik = dataframe["cik"].replace("", pd.NA).isna().sum()

        self.missing_start = dataframe["fraud_start"].isna().sum()

        self.missing_end = dataframe["fraud_end"].isna().sum()

        dataframe = dataframe.dropna(
            subset=[
                "cik",
                "fraud_start",
                "fraud_end",
            ]
        )

        invalid_mask = dataframe["fraud_start"] > dataframe["fraud_end"]

        self.invalid_periods = invalid_mask.sum()

        dataframe = dataframe.loc[~invalid_mask]

        self.duplicate_periods = dataframe.duplicated(
            subset=[
                "cik",
                "fraud_start",
                "fraud_end",
            ]
        ).sum()

        dataframe = dataframe.drop_duplicates(
            subset=[
                "cik",
                "fraud_start",
                "fraud_end",
            ]
        )

        self.rows_written = len(dataframe)

        logger.info("Validation successful.")

        return dataframe
        # ============================================================

    # Save Dataset
    # ============================================================

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Save the fraud periods dataset.
        """

        logger.info("Writing fraud periods dataset...")

        self.OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        table = pa.Table.from_pandas(
            dataframe,
            preserve_index=False,
        )

        pq.write_table(
            table,
            self.OUTPUT_FILE,
            compression="snappy",
        )

        logger.info("Fraud periods written to:")

        logger.info(
            "%s",
            self.OUTPUT_FILE,
        )

    # ============================================================
    # Reporting
    # ============================================================

    def report(
        self,
    ) -> dict[str, Any]:
        """
        Build the ingestion report.
        """

        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "missing_cik": int(self.missing_cik),
            "missing_fraud_start": int(self.missing_start),
            "missing_fraud_end": int(self.missing_end),
            "invalid_periods": int(self.invalid_periods),
            "duplicate_periods": int(self.duplicate_periods),
            "output_columns": self.OUTPUT_COLUMNS,
        }

    def write_report(
        self,
    ) -> None:
        """
        Write the ingestion report.
        """

        logger.info("Writing fraud period report...")

        self.REPORT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
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
                default=str,
            )

        logger.info("Fraud period report written to:")

        logger.info(
            "%s",
            self.REPORT_FILE,
        )

    # ============================================================
    # Summary
    # ============================================================

    def log_summary(
        self,
    ) -> None:
        """
        Log an ingestion summary.
        """

        logger.info("=" * 70)
        logger.info("Fraud Period Ingestion Summary")
        logger.info("=" * 70)

        for key, value in self.report().items():
            logger.info(
                "%s: %s",
                key,
                value,
            )

        logger.info("=" * 70)
        # ============================================================

    # Pipeline
    # ============================================================

    def run(
        self,
    ) -> None:
        """
        Execute the fraud period ingestion pipeline.
        """

        logger.info("=" * 70)

        logger.info("Starting fraud period ingestion...")

        dataframe = self.load_dataset()

        dataframe = self.prepare_dataset(
            dataframe,
        )

        dataframe = self.validate_dataset(
            dataframe,
        )

        self.save_dataset(
            dataframe,
        )

        self.write_report()

        self.log_summary()

        logger.info("Fraud period ingestion completed successfully.")

        logger.info("=" * 70)


# ============================================================
# Public API
# ============================================================


def ingest_labels() -> None:
    """
    Execute the fraud period ingestion pipeline.
    """

    pipeline = FraudPeriodsIngestion()

    pipeline.run()


def main() -> None:
    """
    CLI entry point.
    """

    ingest_labels()


if __name__ == "__main__":
    main()
