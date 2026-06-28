"""
Data ingestion utilities for the FinNLP Corporate Misconduct dataset.

Responsibilities
----------------
- Load the FinNLP dataset.
- Stream the large JSON dataset.
- Load fraud labels.
- Load AAER mappings.
- Expose reusable loading functions.

This module DOES NOT:
- clean text
- validate data
- engineer features
- modify raw data
"""

from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
)

import ijson

import configs.settings as settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Containers
# =============================================================================


@dataclass(slots=True)
class DatasetPaths:
    """
    Stores all raw dataset paths.

    The paths are resolved from configs.settings
    to avoid hardcoded locations.
    """

    firm_years: Path
    labels: Path
    aaer: Path


@dataclass(slots=True)
class DatasetSchema:
    """
    Lightweight description of the dataset structure.

    This is intentionally minimal.

    Validation of schema consistency belongs to
    validate_raw_data.py.
    """

    root_type: str
    top_level_keys: list[str]
    sample_record_keys: list[str]


# =============================================================================
# Main Loader
# =============================================================================


class FinNLPDatasetLoader:
    """
    Loads raw FinNLP data.

    Notes
    -----
    This class performs ingestion only.

    It intentionally does NOT

    - clean records
    - validate contents
    - engineer features

    Those responsibilities belong to later stages
    of the pipeline.
    """

    def __init__(self) -> None:
        self.paths = DatasetPaths(
        firm_years=settings.FIRM_YEARS_FILE,
        labels=settings.LABELS_FILE,
        aaer=settings.AAER_FILE,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _check_exists(path: Path) -> None:
        """
        Verify that a required file exists.

        Raises
        ------
        FileNotFoundError
            If the file cannot be found.
        """
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    def verify_files(self) -> None:
        """
        Ensure all required dataset files exist.

        This checks only file presence.
        Content validation is handled elsewhere.
        """
        logger.info("Checking raw dataset files...")

        self._check_exists(self.paths.firm_years)
        self._check_exists(self.paths.labels)
        self._check_exists(self.paths.aaer)

        logger.info("All required dataset files were found.")
    
    # -------------------------------------------------------------------------
    # Schema Discovery
    # -------------------------------------------------------------------------

    def detect_schema(self) -> DatasetSchema:
        """
        Inspect the raw JSON structure without loading the
        entire dataset into memory.

        Returns
        -------
        DatasetSchema
            Lightweight description of the dataset.

        Notes
        -----
        This method intentionally performs only inspection.
        No validation or preprocessing occurs here.
        """

        self._check_exists(self.paths.firm_years)

        logger.info("Detecting FinNLP dataset schema...")

        with self.paths.firm_years.open(
            "rb"
        ) as file:

            parser = ijson.parse(file)

            root_type: Optional[str] = None
            top_level_keys: list[str] = []
            sample_record_keys: list[str] = []

            for prefix, event, value in parser:

                # -----------------------------
                # Root object
                # -----------------------------
                if prefix == "" and event == "start_map":
                    root_type = "object"

                elif prefix == "" and event == "start_array":
                    root_type = "array"

                # -----------------------------
                # Top-level keys
                # -----------------------------
                if (
                    root_type == "object"
                    and prefix == ""
                    and event == "map_key"
                ):
                    top_level_keys.append(value)

                # -----------------------------
                # First record keys
                # -----------------------------
                if (
                    root_type == "array"
                    and prefix == "item"
                    and event == "map_key"
                ):
                    sample_record_keys.append(value)

                elif (
                    root_type == "object"
                    and "." in prefix
                    and event == "map_key"
                ):
                    sample_record_keys.append(value)

                # Stop once we've learned enough
                if (
                    root_type is not None
                    and len(sample_record_keys) >= 25
                ):
                    break

        schema = DatasetSchema(
            root_type=root_type or "unknown",
            top_level_keys=sorted(set(top_level_keys)),
            sample_record_keys=sorted(set(sample_record_keys)),
        )

        logger.info(
            "Detected schema | root=%s | top_keys=%d | record_keys=%d",
            schema.root_type,
            len(schema.top_level_keys),
            len(schema.sample_record_keys),
        )

        return schema
    # -------------------------------------------------------------------------
    # Labels Schema Discovery
    # -------------------------------------------------------------------------

    def inspect_labels_schema(self) -> None:
        """
        Inspect the labels JSON without making assumptions
        about its structure.
        """

        self._check_exists(self.paths.labels)

        logger.info("Inspecting labels dataset...")

        with self.paths.labels.open("rb") as file:

            parser = ijson.parse(file)

            root_type = None
            sample_keys = []

            for prefix, event, value in parser:

                if prefix == "" and event == "start_array":
                    root_type = "array"

                elif prefix == "" and event == "start_map":
                    root_type = "object"

                if event == "map_key":
                    sample_keys.append(value)

                if len(sample_keys) >= 30:
                    break

        logger.info("Labels root type: %s", root_type)
        logger.info(
            "Labels sample keys: %s",
            sorted(set(sample_keys))
        )

    # -------------------------------------------------------------------------
    # AAER Schema Discovery
    # -------------------------------------------------------------------------

    def inspect_aaer_schema(self) -> None:
        """
        Inspect the AAER CSV header.
        """

        self._check_exists(self.paths.aaer)

        logger.info("Inspecting AAER mapping...")

        with self.paths.aaer.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.reader(file)

            header = next(reader)

        logger.info(
            "AAER columns: %s",
            header
        )
    def inspect_first_label_record(self) -> None:
        with self.paths.labels.open("rb") as file:
            first = next(ijson.items(file, "item"))
        logger.info("First label record: %s", first)
    
    def inspect_first_aaer_record(self) -> None:
        with self.paths.aaer.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(
                file,
                delimiter=";"
            )

            first = next(reader)

        logger.info("First AAER record: %s", first)

    # -------------------------------------------------------------------------
    # Streaming Dataset
    # -------------------------------------------------------------------------

    def stream_firm_years(self) -> Iterator[Dict[str, Any]]:
        """
        Stream firm-year observations one record at a time.

        Yields
        ------
        dict
            One firm-year observation.

        Notes
        -----
        This function never loads the complete dataset
        into memory.
        """

        schema = self.detect_schema()

        logger.info("Streaming firm-year observations...")

        with self.paths.firm_years.open("rb") as file:

            if schema.root_type == "array":

                yield from ijson.items(file, "item")

            elif schema.root_type == "object":

                parser = ijson.kvitems(file, "")

                for _, record in parser:
                    yield record

            else:

                raise ValueError(
                    f"Unsupported dataset structure: {schema.root_type}"
                )

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def iter_sample(
        self,
        n: int = 5,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield the first n observations.

        Useful for exploration and testing while keeping
        memory usage constant.
        """

        count = 0

        for record in self.stream_firm_years():

            yield record

            count += 1

            if count >= n:
                break
    
        # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    def load_labels(self) -> Dict[str, Any]:
        """
        Load the fraud labels file.

        Returns
        -------
        Dict[str, Any]
            Parsed label data.

        Notes
        -----
        The structure of the labels file is intentionally
        preserved exactly as provided. Interpretation and
        validation belong to later pipeline stages.
        """

        self._check_exists(self.paths.labels)

        logger.info("Loading label dataset...")

        with self.paths.labels.open(
            "r",
            encoding="utf-8",
        ) as file:

            labels = json.load(file)

        logger.info("Label dataset loaded successfully.")

        return labels

    # -------------------------------------------------------------------------
    # AAER Mapping
    # -------------------------------------------------------------------------

    def load_aaer_mapping(self) -> list[Dict[str, str]]:
        """
        Load the AAER mapping CSV.

        Returns
        -------
        list[dict]
            One dictionary per CSV row.
        """

        self._check_exists(self.paths.aaer)

        logger.info("Loading AAER mapping...")

        rows: list[Dict[str, str]] = []

        with self.paths.aaer.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:
                rows.append(row)

        logger.info(
            "Loaded %d AAER records.",
            len(rows),
        )

        return rows

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def get_dataset_paths(self) -> DatasetPaths:
        """
        Return resolved dataset paths.
        """

        return self.paths

    def get_schema(self) -> DatasetSchema:
        """
        Convenience wrapper around detect_schema().
        """

        return self.detect_schema()
    
    
if __name__ == "__main__":

    loader = FinNLPDatasetLoader()

    loader.verify_files()

    schema = loader.detect_schema()

    logger.info("Firm Years Schema: %s", schema)

    loader.inspect_labels_schema()

    loader.inspect_aaer_schema()

    loader.inspect_first_label_record()

    loader.inspect_first_aaer_record()

    sample = next(loader.stream_firm_years())

    logger.info(
        "First record keys: %s",
        list(sample.keys())
    )