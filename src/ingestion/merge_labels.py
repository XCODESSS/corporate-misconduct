"""
Merge fraud labels into the combined firm-year dataset.

Responsibilities
----------------
- Load combined firm-years.
- Load fraud periods.
- Normalize CIKs.
- Parse reporting dates.
- Create fraud labels.
- Save labeled dataset.
- Generate merge report.

This module DOES NOT

- clean text
- normalize text
- engineer features
- split datasets
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import configs.settings as settings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LabelMerger:
    """
    Merge AAER fraud periods into the combined
    firm-year dataset.
    """

    FIRM_YEARS_FILE = settings.INTERIM_DIR / "firm_years_combined.parquet"

    FRAUD_PERIODS_FILE = settings.INTERIM_DIR / "fraud_periods.parquet"

    OUTPUT_FILE = settings.INTERIM_DIR / "labeled_firm_years.parquet"

    REPORT_FILE = settings.INTERIM_VALIDATED_DIR / "merge_report.json"

    def __init__(self) -> None:
        self.total_firm_years = 0
        self.fraudulent_count = 0
        self.clean_count = 0

        self.fraud_rate = 0.0

        self.matched_ciks = 0
        self.unmatched_ciks = 0

        self.firm_years_with_missing_reporting_date = 0
        self.multiple_fraud_period_matches = 0

    # ============================================================
    # Loading
    # ============================================================

    def load_firm_years(
        self,
    ) -> pd.DataFrame:
        """
        Load combined firm-years.
        """

        logger.info("Loading combined firm-years...")

        table = pq.read_table(
            self.FIRM_YEARS_FILE,
        )

        dataframe = table.to_pandas()

        self.total_firm_years = len(dataframe)

        logger.info(
            "Loaded %d firm-years.",
            self.total_firm_years,
        )

        return dataframe

    def load_fraud_periods(
        self,
    ) -> pd.DataFrame:
        """
        Load fraud periods.
        """

        logger.info("Loading fraud periods...")

        table = pq.read_table(
            self.FRAUD_PERIODS_FILE,
        )

        dataframe = table.to_pandas()

        logger.info(
            "Loaded %d fraud periods.",
            len(dataframe),
        )

        return dataframe

    # ============================================================
    # Preparation
    # ============================================================

    @staticmethod
    def normalize_cik(
        value: Any,
    ) -> str:
        """
        Normalize CIK to a zero-padded
        10-digit string.
        """

        if pd.isna(value):
            return ""

        value = str(value).strip()

        if value.endswith(".0"):
            value = value[:-2]

        return value.zfill(10)

    def prepare_datasets(
        self,
        firm_years: pd.DataFrame,
        fraud_periods: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare datasets for merging.
        """

        logger.info("Preparing datasets...")

        firm_years = firm_years.copy()
        fraud_periods = fraud_periods.copy()

        firm_years["cik"] = firm_years["cik"].apply(self.normalize_cik)

        fraud_periods["cik"] = fraud_periods["cik"].apply(self.normalize_cik)

        firm_years["reporting_date"] = pd.to_datetime(
            firm_years["reporting_date"],
            dayfirst=True,
            errors="coerce",
        )

        self.firm_years_with_missing_reporting_date = (
            firm_years["reporting_date"].isna().sum()
        )

        logger.info("Datasets prepared.")

        return (
            firm_years,
            fraud_periods,
        )

    # ============================================================
    # Fraud Lookup
    # ============================================================

    def build_lookup(
        self,
        fraud_periods: pd.DataFrame,
    ) -> defaultdict[str, list]:
        """
        Build a CIK-indexed fraud lookup.
        """

        logger.info("Building fraud lookup...")

        fraud_lookup = defaultdict(list)

        for _, row in fraud_periods.iterrows():
            fraud_lookup[row["cik"]].append(
                (
                    row["fraud_start"],
                    row["fraud_end"],
                    row["certainty_start"],
                    row["certainty_end"],
                )
            )

        logger.info(
            "Fraud lookup built for %d CIKs.",
            len(fraud_lookup),
        )

        return fraud_lookup
        # ============================================================

    # Fiscal Year Utilities
    # ============================================================

    @staticmethod
    def fiscal_year_end(
        reporting_date: pd.Timestamp,
        fye: Any,
    ) -> pd.Timestamp:
        """
        Compute the fiscal year-end date for the
        reporting period.

        Examples
        --------
        fye = 1231
        reporting_date = 2018-03-15

        -> 2018-12-31

        fye = 0630
        reporting_date = 2018-11-10

        -> 2019-06-30
        """

        if pd.isna(reporting_date) or pd.isna(fye):
            return pd.NaT

        try:
            fye = str(fye).zfill(4)

            month = int(fye[:2])
            day = int(fye[2:])

            fiscal_end = pd.Timestamp(
                year=reporting_date.year,
                month=month,
                day=day,
            )

            if reporting_date > fiscal_end:
                fiscal_end = fiscal_end.replace(year=fiscal_end.year + 1)

            return fiscal_end

        except Exception:
            return pd.NaT

    # ============================================================
    # Fraud Labeling
    # ============================================================

    def label_dataset(
        self,
        firm_years: pd.DataFrame,
        fraud_lookup: defaultdict[str, list],
    ) -> pd.DataFrame:
        firm_years = firm_years.copy()
        firm_years["fraudulent"] = 0
        firm_years["matched_fraud_start"] = pd.NaT
        firm_years["matched_fraud_end"] = pd.NaT
        firm_years["certainty_start"] = pd.NA
        firm_years["certainty_end"] = pd.NA

        cik_arr = firm_years["cik"].to_numpy()
        reporting_arr = firm_years["reporting_date"].to_numpy()
        fye_arr = firm_years["fye"].to_numpy()
        matched_ciks = set()
        fraud_ciks = set(fraud_lookup.keys())
        firm_ciks = set(firm_years["cik"].unique())
        multiple_matches = np.zeros(len(firm_years), dtype=int)

        for cik, periods in fraud_lookup.items():
            cik_mask = cik_arr == cik

            if not cik_mask.any():
                continue

            matched_ciks.add(cik)
            indices = np.where(cik_mask)[0]

            for idx in indices:
                reporting_date = (
                    pd.Timestamp(reporting_arr[idx])
                    if not pd.isna(reporting_arr[idx])
                    else pd.NaT
                )
                fiscal_end = self.fiscal_year_end(reporting_date, fye_arr[idx])

                matches = []

                for fraud_start, fraud_end, cert_start, cert_end in periods:
                    reporting_match = (
                        pd.notna(reporting_date)
                        and fraud_start <= reporting_date <= fraud_end
                    )

                    fiscal_match = (
                        pd.notna(fiscal_end) and fraud_start <= fiscal_end <= fraud_end
                    )

                    if reporting_match or fiscal_match:
                        matches.append((fraud_start, fraud_end, cert_start, cert_end))

                if not matches:
                    continue

                if len(matches) > 1:
                    multiple_matches[idx] += 1

                fraudulent_column = firm_years.columns.get_loc("fraudulent")
                firm_years.iat[idx, fraudulent_column] = 1
                firm_years.iat[
                    idx, firm_years.columns.get_loc("matched_fraud_start")
                ] = matches[0][0]
                firm_years.iat[idx, firm_years.columns.get_loc("matched_fraud_end")] = (
                    matches[0][1]
                )
                firm_years.iat[idx, firm_years.columns.get_loc("certainty_start")] = (
                    matches[0][2]
                )
                firm_years.iat[idx, firm_years.columns.get_loc("certainty_end")] = (
                    matches[0][3]
                )

        self.multiple_fraud_period_matches = int(multiple_matches.sum())
        self.matched_ciks = len(matched_ciks)
        self.unmatched_ciks = len(fraud_ciks - firm_ciks)
        self.fraudulent_count = int(firm_years["fraudulent"].sum())
        self.clean_count = len(firm_years) - self.fraudulent_count
        self.fraud_rate = round(self.fraudulent_count / len(firm_years), 6)

        logger.info("Fraud labeling complete.")
        return firm_years

    # ============================================================
    # Validation
    # ============================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate the labeled dataset.
        """

        logger.info("Validating labeled dataset...")

        required_columns = [
            "fraudulent",
            "matched_fraud_start",
            "matched_fraud_end",
            "certainty_start",
            "certainty_end",
        ]

        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        fraud_values = set(dataframe["fraudulent"].unique())

        if not fraud_values.issubset({0, 1}):
            raise ValueError("fraudulent column must only contain 0 or 1.")

        if int(dataframe["fraudulent"].sum()) != self.fraudulent_count:
            raise ValueError("Fraud count mismatch.")

        logger.info("Validation successful.")

    # ============================================================
    # Save Dataset
    # ============================================================

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Save the labeled dataset.
        """

        logger.info("Writing labeled dataset...")

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

        logger.info("Labeled dataset written to:")

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
        Build the merge report.
        """

        return {
            "total_firm_years": self.total_firm_years,
            "fraudulent_count": self.fraudulent_count,
            "clean_count": self.clean_count,
            "fraud_rate": self.fraud_rate,
            "matched_ciks": self.matched_ciks,
            "unmatched_ciks": self.unmatched_ciks,
            "firm_years_with_missing_reporting_date": int(
                self.firm_years_with_missing_reporting_date
            ),
            "multiple_fraud_period_matches": self.multiple_fraud_period_matches,
        }

    def write_report(
        self,
    ) -> None:
        """
        Write the merge report.
        """

        logger.info("Writing merge report...")

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

        logger.info("Merge report written to:")

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
        Log merge summary.
        """

        logger.info("=" * 70)
        logger.info("Fraud Label Merge Summary")
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
        Execute the fraud label merge pipeline.
        """

        logger.info("=" * 70)
        logger.info("Starting fraud label merge...")

        # --------------------------------------------------------
        # Load datasets
        # --------------------------------------------------------

        firm_years = self.load_firm_years()

        fraud_periods = self.load_fraud_periods()

        # --------------------------------------------------------
        # Prepare datasets
        # --------------------------------------------------------

        (
            firm_years,
            fraud_periods,
        ) = self.prepare_datasets(
            firm_years,
            fraud_periods,
        )

        # --------------------------------------------------------
        # Build fraud lookup
        # --------------------------------------------------------

        fraud_lookup = self.build_lookup(
            fraud_periods,
        )

        # --------------------------------------------------------
        # Merge labels
        # --------------------------------------------------------

        labeled_dataset = self.label_dataset(
            firm_years,
            fraud_lookup,
        )

        # --------------------------------------------------------
        # Validate
        # --------------------------------------------------------

        self.validate_dataset(
            labeled_dataset,
        )

        # --------------------------------------------------------
        # Save outputs
        # --------------------------------------------------------

        self.save_dataset(
            labeled_dataset,
        )

        self.write_report()

        self.log_summary()

        logger.info("Fraud label merge completed successfully.")

        logger.info("=" * 70)


# ============================================================
# Public API
# ============================================================


def merge_labels() -> None:
    """
    Execute the fraud label merge pipeline.
    """

    pipeline = LabelMerger()

    pipeline.run()


def main() -> None:
    """
    CLI entry point.
    """

    merge_labels()


if __name__ == "__main__":
    main()
