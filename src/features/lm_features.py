"""
Loughran-McDonald feature engineering.

Responsibilities
----------------
- Read train/validation and test datasets.
- Build the required (CIK, filing_year) key set.
- Stream the Loughran-McDonald summary file.
- Keep only LM records actually needed.
- Compute normalized linguistic density features.
- Save feature-enhanced datasets.
- Generate a feature engineering report.

Design
------
This implementation intentionally avoids pandas merge().

The LM summary file contains roughly 1.25 million filings.
Instead of joining against the entire dataset, we:

1. Read train/test datasets.
2. Build the set of required (CIK, filing_year) keys.
3. Stream the LM CSV in chunks.
4. Store ONLY matching keys.
5. Process one dataset at a time.
6. Immediately save output and free memory.

Peak RAM stays low even on 16 GB machines.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import configs.settings as settings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LMFeatureEngineer:

    TRAINVAL_FILE = (
        settings.DATASETS_DIR
        / "trainval_dataset.parquet"
    )

    TEST_FILE = (
        settings.DATASETS_DIR
        / "test_dataset.parquet"
    )

    LM_SUMMARY_FILE = (
        settings.RAW_DIR
        / "lm"
        / "Loughran-McDonald_10X_Summaries_1993-2025.csv"
    )

    OUTPUT_DIR = settings.FEATURES_DIR

    TRAINVAL_OUTPUT = (
        OUTPUT_DIR
        / "trainval_features.parquet"
    )

    TEST_OUTPUT = (
        OUTPUT_DIR
        / "test_features.parquet"
    )

    REPORT_FILE = (
        settings.INTERIM_VALIDATED_DIR
        / "lm_feature_report.json"
    )

    CHUNK_SIZE = 100_000

    LM_COLUMNS = [
        "CIK",
        "FILING_DATE",
        "N_Words",
        "N_Negative",
        "N_Positive",
        "N_Uncertainty",
        "N_Litigious",
        "N_StrongModal",
        "N_WeakModal",
        "N_Constraining",
    ]

    DEFAULT_FEATURES = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    def __init__(self) -> None:

        self.train_records = 0
        self.test_records = 0

        self.train_matched = 0
        self.test_matched = 0

        self.train_unmatched = 0
        self.test_unmatched = 0

    # ============================================================
    # Loading
    # ============================================================

    def load_dataset(
        self,
        path: Path,
        name: str,
    ) -> pd.DataFrame:

        logger.info(
            "Reading %s...",
            name,
        )

        table = pq.read_table(path)

        df = table.to_pandas()

        logger.info(
            "Loaded %d %s records.",
            len(df),
            name,
        )

        return df

    # ============================================================
    # Key Normalization
    # ============================================================

    @staticmethod
    def normalize_cik(
        series: pd.Series,
    ) -> pd.Series:

        return (
            series
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(10)
        )

    @staticmethod
    def extract_year(
        series: pd.Series,
    ) -> pd.Series:

        return pd.to_datetime(
            series,
            errors="coerce",
        ).dt.year

    def prepare_dataset(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df["cik"] = self.normalize_cik(
            df["cik"]
        )

        df["filing_year"] = self.extract_year(
            df["reporting_date"]
        )

        missing = df["filing_year"].isna().sum()

        if missing:

            raise ValueError(
                f"{missing} rows have invalid reporting_date."
            )

        return df

    # ============================================================
    # Required Lookup Keys
    # ============================================================

    @staticmethod
    def build_required_keys(
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> set[tuple[str, int]]:
        """
        Build the set of (CIK, filing_year) pairs that
        actually exist in the modeling datasets.

        Only these keys will be loaded from the
        Loughran-McDonald summaries.
        """

        train_keys = set(
            zip(
                train["cik"],
                train["filing_year"],
            )
        )

        test_keys = set(
            zip(
                test["cik"],
                test["filing_year"],
            )
        )

        keys = train_keys | test_keys

        logger.info(
            "Unique required LM keys: %d",
            len(keys),
        )

        return keys
        # ============================================================
    # Build LM Lookup (Streaming)
    # ============================================================

    def build_lm_lookup(
        self,
        required_keys: set[tuple[str, int]],
    ) -> dict[tuple[str, int], tuple]:
        """
        Stream the LM CSV and build a lookup containing
        ONLY the (CIK, filing_year) pairs required by the
        modeling datasets.

        This avoids loading the entire 1.25M-row dataset
        into memory.
        """

        logger.info("=" * 70)
        logger.info("Building LM lookup...")
        logger.info("Streaming LM CSV in %d-row chunks...", self.CHUNK_SIZE)

        lookup: dict[tuple[str, int], tuple] = {}

        total_rows = 0
        matched_rows = 0
        chunk_number = 0

        for chunk in pd.read_csv(
            self.LM_SUMMARY_FILE,
            usecols=self.LM_COLUMNS,
            dtype={"CIK": str},
            chunksize=self.CHUNK_SIZE,
            low_memory=False,
        ):

            chunk_number += 1
            total_rows += len(chunk)

            chunk["cik"] = self.normalize_cik(
                chunk["CIK"]
            )

            chunk["filing_year"] = pd.to_datetime(
                chunk["FILING_DATE"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            ).dt.year

            chunk = chunk.dropna(
                subset=["filing_year"]
            )

            chunk["filing_year"] = (
                chunk["filing_year"]
                .astype(int)
            )

            for row in chunk.itertuples(index=False):

                key = (
                    row.cik,
                    row.filing_year,
                )

                if key not in required_keys:
                    continue

                # Keep only the first occurrence.
                if key in lookup:
                    continue

                lookup[key] = (
                    row.N_Words,
                    row.N_Negative,
                    row.N_Positive,
                    row.N_Uncertainty,
                    row.N_Litigious,
                    row.N_StrongModal,
                    row.N_WeakModal,
                    row.N_Constraining,
                )

                matched_rows += 1

            logger.info(
                (
                    "Chunk %d | "
                    "Rows=%d | "
                    "Lookup=%d"
                ),
                chunk_number,
                total_rows,
                len(lookup),
            )

            del chunk
            gc.collect()

        logger.info("=" * 70)
        logger.info(
            "LM lookup complete."
        )
        logger.info(
            "LM rows scanned: %d",
            total_rows,
        )
        logger.info(
            "Lookup entries: %d",
            len(lookup),
        )
        logger.info(
            "Matched LM rows: %d",
            matched_rows,
        )
        logger.info("=" * 70)

        return lookup

    # ============================================================
    # Feature Attachment
    # ============================================================

    def attach_features(
        self,
        df: pd.DataFrame,
        lookup: dict[
            tuple[str, int],
            tuple,
        ],
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Attach LM features using dictionary lookup.

        No merge.
        No join.
        O(number of firm-years).
        """

        logger.info(
            "Attaching LM features to %s...",
            dataset_name,
        )

        feature_rows = []

        matched = 0

        for row in df.itertuples(index=False):

            key = (
                row.cik,
                row.filing_year,
            )

            values = lookup.get(
                key,
                self.DEFAULT_FEATURES,
            )

            if values is not self.DEFAULT_FEATURES:
                matched += 1

            feature_rows.append(values)

        feature_df = pd.DataFrame(
            feature_rows,
            columns=[
                "N_Words",
                "N_Negative",
                "N_Positive",
                "N_Uncertainty",
                "N_Litigious",
                "N_StrongModal",
                "N_WeakModal",
                "N_Constraining",
            ],
        )

        df = pd.concat(
            [
                df.reset_index(drop=True),
                feature_df,
            ],
            axis=1,
        )

        denominator = (
            df["N_Words"]
            .replace(0, np.nan)
        )

        df["negative_density"] = (
            df["N_Negative"]
            / denominator
        )

        df["positive_density"] = (
            df["N_Positive"]
            / denominator
        )

        df["uncertainty_density"] = (
            df["N_Uncertainty"]
            / denominator
        )

        df["litigious_density"] = (
            df["N_Litigious"]
            / denominator
        )

        df["strong_modal_density"] = (
            df["N_StrongModal"]
            / denominator
        )

        df["weak_modal_density"] = (
            df["N_WeakModal"]
            / denominator
        )

        df["constraining_density"] = (
            df["N_Constraining"]
            / denominator
        )

        density_columns = [
            "negative_density",
            "positive_density",
            "uncertainty_density",
            "litigious_density",
            "strong_modal_density",
            "weak_modal_density",
            "constraining_density",
        ]

        df[density_columns] = (
            df[density_columns]
            .fillna(0.0)
        )

        total = len(df)
        unmatched = total - matched

        if dataset_name == "trainval":

            self.train_records = total
            self.train_matched = matched
            self.train_unmatched = unmatched

        else:

            self.test_records = total
            self.test_matched = matched
            self.test_unmatched = unmatched

        logger.info(
            "%s | matched=%d unmatched=%d",
            dataset_name,
            matched,
            unmatched,
        )

        return df
        # ============================================================
    # Saving
    # ============================================================

    @staticmethod
    def save_dataset(
        df: pd.DataFrame,
        output_file: Path,
    ) -> None:
        """
        Save feature-engineered dataset.
        """

        logger.info(
            "Writing %s...",
            output_file.name,
        )

        table = pa.Table.from_pandas(
            df,
            preserve_index=False,
        )

        pq.write_table(
            table,
            output_file,
            compression="snappy",
        )

        logger.info(
            "Saved %d rows.",
            len(df),
        )

    # ============================================================
    # Reporting
    # ============================================================

    def report(self) -> dict[str, Any]:

        return {

            "train_records":
                self.train_records,

            "train_matched":
                self.train_matched,

            "train_unmatched":
                self.train_unmatched,

            "train_match_rate":
                round(
                    self.train_matched
                    / self.train_records,
                    4,
                ),

            "test_records":
                self.test_records,

            "test_matched":
                self.test_matched,

            "test_unmatched":
                self.test_unmatched,

            "test_match_rate":
                round(
                    self.test_matched
                    / self.test_records,
                    4,
                ),
        }

    def write_report(self) -> None:

        logger.info(
            "Writing feature report..."
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

    def log_summary(self) -> None:

        logger.info("=" * 70)
        logger.info(
            "LM Feature Engineering Summary"
        )
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

    def process_dataset(
        self,
        input_file: Path,
        output_file: Path,
        lookup: dict[
            tuple[str, int],
            tuple,
        ],
        dataset_name: str,
    ) -> None:
        """
        Process a single dataset.

        The dataframe is immediately released after
        writing to keep peak RAM usage low.
        """

        logger.info("=" * 70)
        logger.info(
            "Processing %s dataset...",
            dataset_name,
        )

        df = self.load_dataset(
            input_file,
            dataset_name,
        )

        df = self.prepare_dataset(
            df,
        )

        df = self.attach_features(
            df,
            lookup,
            dataset_name,
        )

        self.save_dataset(
            df,
            output_file,
        )

        del df
        gc.collect()

        logger.info(
            "%s dataset complete.",
            dataset_name,
        )

    def run(self) -> None:

        logger.info("=" * 70)
        logger.info(
            "Starting LM feature engineering..."
        )

        self.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Reading datasets to build required key set..."
        )

        train = self.prepare_dataset(
            self.load_dataset(
                self.TRAINVAL_FILE,
                "trainval",
            )
        )

        test = self.prepare_dataset(
            self.load_dataset(
                self.TEST_FILE,
                "test",
            )
        )

        required_keys = self.build_required_keys(
            train,
            test,
        )

        del train
        del test
        gc.collect()

        lookup = self.build_lm_lookup(
            required_keys,
        )

        del required_keys
        gc.collect()

        self.process_dataset(
            self.TRAINVAL_FILE,
            self.TRAINVAL_OUTPUT,
            lookup,
            "trainval",
        )

        self.process_dataset(
            self.TEST_FILE,
            self.TEST_OUTPUT,
            lookup,
            "test",
        )

        del lookup
        gc.collect()

        self.write_report()

        self.log_summary()

        logger.info(
            "LM feature engineering completed successfully."
        )

        logger.info("=" * 70)


# ============================================================
# Public API
# ============================================================

def engineer_lm_features() -> None:

    LMFeatureEngineer().run()


def main() -> None:

    engineer_lm_features()


if __name__ == "__main__":

    main()