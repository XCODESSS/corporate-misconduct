"""
Ingest FinNLP firm-year datasets.

Responsibilities
----------------
- Load firm_years.json.
- Load firm_years_labels.json.
- Validate both datasets.
- Ensure schemas match.
- Concatenate datasets.
- Save a combined Parquet dataset.
- Generate an ingestion report.

This module DOES NOT

- clean text
- normalize text
- deduplicate records
- create fraud labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import configs.settings as settings
import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FirmYearsIngestion:
    """
    Ingest and combine the FinNLP firm-year datasets.
    """

    FIRM_YEARS_FILE = settings.FINNLP_DIR / "firm_years.json"

    LABEL_FIRM_YEARS_FILE = settings.FINNLP_DIR / "firm_years_labels.json"

    OUTPUT_FILE = settings.INTERIM_DIR / "firm_years_combined.parquet"

    REPORT_FILE = settings.INTERIM_VALIDATED_DIR / "firm_years_ingestion_report.json"

    REQUIRED_COLUMNS = [
        "cik",
        "name",
        "filing_type",
        "filing_date",
        "reporting_date",
        "fye",
        "mda",
    ]

    def __init__(self) -> None:
        self.firm_year_rows = 0
        self.label_rows = 0
        self.total_rows = 0

        self.schema_match = False

    # ============================================================
    # Loading
    # ============================================================

    def load_dataset(
        self,
        dataset_path: Path,
    ) -> pd.DataFrame:
        """
        Load a FinNLP JSON dataset using streaming.

        Uses ijson to stream records one at a time,
        avoiding loading the full 4.9GB file into memory
        before DataFrame construction.

        Parameters
        ----------
        dataset_path
            Path to the JSON file.

        Returns
        -------
        pandas.DataFrame
        """
        logger.info(
            "Loading %s",
            dataset_path.name,
        )

        records = []

        with open(
            dataset_path,
            "rb",
        ) as file:
            for record in ijson.items(
                file,
                "item",
            ):
                records.append(record)

        dataframe = pd.DataFrame.from_records(
            records,
        )

        logger.info(
            "Loaded %d records.",
            len(dataframe),
        )

        return dataframe

    # ============================================================
    # Validation
    # ============================================================

    def validate_required_columns(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """
        Ensure all required columns exist.
        """

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"{dataset_name} is missing columns: {missing_columns}")

        logger.info(
            "%s contains all required columns.",
            dataset_name,
        )

    def validate_schema(
        self,
        firm_years: pd.DataFrame,
        label_firm_years: pd.DataFrame,
    ) -> None:
        """
        Compare schemas of both datasets.

        Does NOT raise on mismatch.

        Reason: pd.concat handles column alignment
        automatically by filling missing columns with NaN.
        A hard crash here would be too aggressive for
        datasets from the same authors that may have
        minor column differences.

        Instead, differences are logged as warnings
        so the ingestion report captures them.
        """

        left = set(firm_years.columns)
        right = set(label_firm_years.columns)

        only_in_left = sorted(left - right)
        only_in_right = sorted(right - left)

        if only_in_left or only_in_right:
            logger.warning("Schema difference detected.")

            logger.warning(
                "Only in firm_years: %s",
                only_in_left,
            )

            logger.warning(
                "Only in labels: %s",
                only_in_right,
            )

            logger.warning("pd.concat will fill missing columns with NaN.")

            self.schema_match = False

        else:
            logger.info(
                "Schema validation successful. Both datasets have identical columns."
            )

            self.schema_match = True

    # ============================================================
    # Combine Datasets
    # ============================================================

    def combine_datasets(
        self,
        firm_years: pd.DataFrame,
        label_firm_years: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Concatenate both firm-year datasets.

        Notes
        -----
        No deduplication is performed here.

        Deduplication is handled later by
        src/preprocessing/deduplicate.py.
        """

        logger.info("Combining firm-year datasets...")

        self.firm_year_rows = len(firm_years)
        self.label_rows = len(label_firm_years)

        combined = pd.concat(
            [
                firm_years,
                label_firm_years,
            ],
            axis=0,
            ignore_index=True,
        )

        self.total_rows = len(combined)

        logger.info(
            "Combined dataset contains %d records.",
            self.total_rows,
        )

        return combined

    # ============================================================
    # Save Dataset
    # ============================================================

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Save the combined dataset as Parquet.
        """

        logger.info("Writing combined dataset...")

        table = pa.Table.from_pandas(
            dataframe,
            preserve_index=False,
        )

        pq.write_table(
            table,
            self.OUTPUT_FILE,
            compression="snappy",
        )

        logger.info("Combined dataset written to:")

        logger.info(
            "%s",
            self.OUTPUT_FILE,
        )

    # ============================================================
    # Reporting
    # ============================================================

    def report(self) -> dict[str, Any]:
        """
        Build the ingestion report.
        """

        return {
            "firm_year_records": self.firm_year_rows,
            "label_firm_year_records": self.label_rows,
            "combined_records": self.total_rows,
            "schema_match": self.schema_match,
            "deduplication_performed": False,
            "deduplication_stage": "src/preprocessing/deduplicate.py",
        }

    def write_report(
        self,
    ) -> None:
        """
        Write the ingestion report.
        """

        logger.info("Writing ingestion report...")

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

        logger.info("Ingestion report written to:")

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
        logger.info("Firm-Year Ingestion Summary")
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
        Execute the ingestion pipeline.
        """

        logger.info("=" * 70)

        logger.info("Starting firm-year ingestion...")

        # --------------------------------------------------------
        # Load datasets
        # --------------------------------------------------------

        firm_years = self.load_dataset(
            self.FIRM_YEARS_FILE,
        )

        label_firm_years = self.load_dataset(
            self.LABEL_FIRM_YEARS_FILE,
        )

        # --------------------------------------------------------
        # Validate datasets
        # --------------------------------------------------------

        self.validate_required_columns(
            firm_years,
            "firm_years.json",
        )

        self.validate_required_columns(
            label_firm_years,
            "firm_years_labels.json",
        )

        self.validate_schema(
            firm_years,
            label_firm_years,
        )

        # --------------------------------------------------------
        # Combine datasets
        # --------------------------------------------------------

        combined = self.combine_datasets(
            firm_years,
            label_firm_years,
        )

        # --------------------------------------------------------
        # Save outputs
        # --------------------------------------------------------

        self.save_dataset(
            combined,
        )

        self.write_report()

        self.log_summary()

        logger.info("Firm-year ingestion completed successfully.")

        logger.info("=" * 70)

    # ============================================================


# Public API
# ============================================================


def ingest_firm_years() -> None:
    """
    Execute the firm-year ingestion pipeline.
    """

    pipeline = FirmYearsIngestion()

    pipeline.run()


def main() -> None:
    """
    CLI entry point.
    """

    ingest_firm_years()


if __name__ == "__main__":
    main()
