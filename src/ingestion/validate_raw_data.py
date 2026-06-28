"""Validate raw FinNLP datasets without mutating source data.

The validator checks file availability, schemas, row-level integrity, and
cross-dataset consistency before preprocessing or feature engineering begins.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import ijson

import configs.settings as settings
from src.ingestion.load_finnlp_dataset import FinNLPDatasetLoader
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


JsonDict = dict[str, Any]
Identifier = tuple[str, str, str, str]


@dataclass(slots=True)
class DatasetValidationSummary:
    """Validation metrics for one raw dataset."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    empty_records: int = 0
    duplicate_identifiers: int = 0
    duplicate_ids: int = 0
    duplicate_filings: int = 0
    invalid_filing_types: int = 0
    malformed_dates: int = 0
    malformed_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    schema_fields: list[str] = field(default_factory=list)
    required_missing: list[str] = field(default_factory=list)
    additional_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationResult:
    """Container for raw validation output and compatibility counters."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_mda: int = 0
    missing_cik: int = 0
    missing_name: int = 0
    duplicate_records: int = 0
    malformed_dates: int = 0
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_summary: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    cross_dataset: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class RawDataValidator:
    """Validate raw Corporate Misconduct datasets in read-only mode."""

    REQUIRED_FIRM_YEAR_FIELDS = {
        "cik",
        "name",
        "filing_date",
        "reporting_date",
        "mda",
    }
    REQUIRED_AAER_COLUMNS = {
        "id",
        "cik",
        "fraud_start",
        "fraud_end",
    }
    VALID_FILING_TYPES = {
        "10-K",
        "10-K/A",
        "10-K405",
        "10-K405/A",
        "10-KT",
        "10-KT/A",
    }

    def __init__(self) -> None:
        self.loader = FinNLPDatasetLoader()
        self.paths = self.loader.get_dataset_paths()
        self.result = ValidationResult()
        self._firm_identifiers: set[Identifier] = set()
        self._firm_ciks: set[str] = set()
        self._label_identifiers: set[Identifier] = set()
        self._label_ciks: set[str] = set()
        self._aaer_ciks: set[str] = set()
        self._firm_fields: set[str] = set()
        self._label_fields: set[str] = set()

    def validate(self) -> ValidationResult:
        """Run the complete raw data validation pipeline."""

        logger.info("Starting raw data validation.")
        self.validate_files()
        self.validate_firm_years()
        self.validate_labels()
        self.validate_aaer()
        self.validate_cross_dataset_consistency()
        self.save_report()
        logger.info("Raw data validation completed.")
        return self.result

    def save_report(self) -> Path:
        """Persist the raw validation report as JSON."""

        output_path = (
            settings.INTERIM_VALIDATED_DIR / "raw_validation_report.json"
        )
        save_json(self.to_report(), output_path)
        logger.info("Validation report saved to %s", output_path)
        return output_path

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable validation report."""

        labels = self.result.datasets.get("labels", {})
        aaer = self.result.datasets.get("aaer", {})
        firm_years = self.result.datasets.get("firm_years", {})
        dataset_totals = self._dataset_metric_totals("total_records")
        dataset_valid = self._dataset_metric_totals("valid_records")
        dataset_invalid = self._dataset_metric_totals("invalid_records")

        return {
            "total_records": self.result.total_records,
            "valid_records": self.result.valid_records,
            "invalid_records": self.result.invalid_records,
            "record_counts": dataset_totals,
            "aggregate_total_records": sum(dataset_totals.values()),
            "aggregate_valid_records": sum(dataset_valid.values()),
            "aggregate_invalid_records": sum(dataset_invalid.values()),
            "duplicate_counts": {
                "firm_year_duplicate_filings": self.result.duplicate_records,
                "label_duplicate_identifiers": labels.get(
                    "duplicate_identifiers",
                    0,
                ),
                "aaer_duplicate_ids": aaer.get("duplicate_ids", 0),
            },
            "missing_values": {
                "firm_years": firm_years.get("missing_values", {}),
                "labels": labels.get("missing_values", {}),
                "aaer": aaer.get("missing_values", {}),
            },
            "schema_summary": self.result.schema_summary,
            "files": self.result.files,
            "datasets": self.result.datasets,
            "cross_dataset": self.result.cross_dataset,
            "warnings": self.result.warnings,
            "errors": self.result.errors,
        }

    def validate_files(self) -> None:
        """Verify required raw files exist, are readable, and have sizes."""

        logger.info("Validating required raw files.")
        required_files = {
            "firm_years": self.paths.firm_years,
            "labels": self.paths.labels,
            "aaer": self.paths.aaer,
        }

        for dataset_name, path in required_files.items():
            self.result.files[dataset_name] = self._validate_file(path)

    def _validate_file(self, path: Path) -> dict[str, Any]:
        """Return file validation metadata for a raw dataset path."""

        file_result: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "readable": False,
            "size_bytes": 0,
        }

        if not path.exists():
            message = f"Missing required file: {path}"
            file_result["error"] = message
            self.result.errors.append(message)
            logger.error(message)
            return file_result

        if not path.is_file():
            message = f"Required dataset path is not a file: {path}"
            file_result["error"] = message
            self.result.errors.append(message)
            logger.error(message)
            return file_result

        file_result["size_bytes"] = path.stat().st_size

        try:
            with path.open("rb") as raw_file:
                raw_file.read(1)
            file_result["readable"] = True
        except OSError as exc:
            message = f"Required file is not readable: {path} ({exc})"
            file_result["error"] = message
            self.result.errors.append(message)
            logger.error(message)

        return file_result

    def validate_firm_years(self) -> None:
        """Stream and validate firm-year records."""

        logger.info("Validating firm-year records.")
        summary = DatasetValidationSummary()

        root_type = self._detect_json_root_type(self.paths.firm_years)
        if root_type != "array":
            summary.errors.append(
                f"firm_years.json must be a JSON array, found {root_type}."
            )

        seen_filings: set[Identifier] = set()

        try:
            for record in self.loader.stream_firm_years():
                row_errors = self._validate_firm_year_record(
                    record=record,
                    summary=summary,
                    seen_filings=seen_filings,
                )
                if row_errors:
                    summary.invalid_records += 1
                else:
                    summary.valid_records += 1
        except Exception as exc:
            message = f"Unable to stream firm_years.json: {exc}"
            summary.errors.append(message)
            logger.exception(message)

        summary.schema_fields = sorted(self._firm_fields)
        summary.required_missing = sorted(
            self.REQUIRED_FIRM_YEAR_FIELDS.difference(self._firm_fields)
        )
        self._record_missing_schema_errors(
            dataset_name="firm_years",
            missing=summary.required_missing,
            summary=summary,
        )
        self._store_dataset_summary("firm_years", summary)
        self._sync_firm_year_compatibility_counters(summary)

    def _validate_firm_year_record(
        self,
        record: Any,
        summary: DatasetValidationSummary,
        seen_filings: set[Identifier],
    ) -> list[str]:
        """Validate one firm-year record and update aggregate state."""

        summary.total_records += 1
        row_errors: list[str] = []

        if not isinstance(record, dict) or self._is_empty_record(record):
            summary.empty_records += 1
            row_errors.append("empty_record")
            return row_errors

        self._firm_fields.update(str(key) for key in record.keys())
        identifier = self._record_identifier(record)
        cik = identifier[0]

        if cik:
            self._firm_ciks.add(cik)
        if all(identifier):
            self._firm_identifiers.add(identifier)

        row_errors.extend(
            self._validate_required_values(
                record=record,
                required_fields=self.REQUIRED_FIRM_YEAR_FIELDS,
                summary=summary,
            )
        )

        if not self._is_valid_date(record.get("filing_date"), "%d-%m-%Y"):
            summary.malformed_dates += 1
            row_errors.append("malformed_filing_date")

        if not self._is_valid_date(record.get("reporting_date"), "%d-%m-%Y"):
            summary.malformed_dates += 1
            row_errors.append("malformed_reporting_date")

        filing_type = str(record.get("filing_type", "")).strip()
        if filing_type not in self.VALID_FILING_TYPES:
            summary.invalid_filing_types += 1
            row_errors.append("invalid_filing_type")

        if all(identifier):
            if identifier in seen_filings:
                summary.duplicate_filings += 1
                summary.duplicate_identifiers += 1
                row_errors.append("duplicate_filing")
            else:
                seen_filings.add(identifier)

        return row_errors

    def validate_labels(self) -> None:
        """Stream and validate firm-year labels."""

        logger.info("Validating label records.")
        summary = DatasetValidationSummary()

        root_type = self._detect_json_root_type(self.paths.labels)
        if root_type != "array":
            summary.errors.append(
                "firm_years_labels.json must be a JSON array, "
                f"found {root_type}."
            )

        seen_identifiers: set[Identifier] = set()

        try:
            for record in self._stream_json_array(self.paths.labels):
                row_errors = self._validate_label_record(
                    record=record,
                    summary=summary,
                    seen_identifiers=seen_identifiers,
                )
                if row_errors:
                    summary.invalid_records += 1
                else:
                    summary.valid_records += 1
        except Exception as exc:
            message = f"Unable to stream firm_years_labels.json: {exc}"
            summary.errors.append(message)
            logger.exception(message)

        summary.schema_fields = sorted(self._label_fields)
        summary.required_missing = sorted(
            self.REQUIRED_FIRM_YEAR_FIELDS.difference(self._label_fields)
        )
        summary.additional_fields = sorted(
            self._label_fields.difference(self._firm_fields)
        )
        self._record_missing_schema_errors(
            dataset_name="labels",
            missing=summary.required_missing,
            summary=summary,
        )
        if summary.additional_fields:
            summary.warnings.append(
                "Labels contain fields not present in firm_years.json: "
                f"{summary.additional_fields}"
            )
        self._store_dataset_summary("labels", summary)

    def _validate_label_record(
        self,
        record: Any,
        summary: DatasetValidationSummary,
        seen_identifiers: set[Identifier],
    ) -> list[str]:
        """Validate one label record and update aggregate label state."""

        summary.total_records += 1
        row_errors: list[str] = []

        if not isinstance(record, dict) or self._is_empty_record(record):
            summary.empty_records += 1
            row_errors.append("empty_record")
            return row_errors

        self._label_fields.update(str(key) for key in record.keys())
        identifier = self._record_identifier(record)
        cik = identifier[0]

        if cik:
            self._label_ciks.add(cik)
        if all(identifier):
            self._label_identifiers.add(identifier)

        row_errors.extend(
            self._validate_required_values(
                record=record,
                required_fields=self.REQUIRED_FIRM_YEAR_FIELDS,
                summary=summary,
            )
        )

        if all(identifier):
            if identifier in seen_identifiers:
                summary.duplicate_identifiers += 1
                row_errors.append("duplicate_identifier")
            else:
                seen_identifiers.add(identifier)

        return row_errors

    def validate_aaer(self) -> None:
        """Validate the semicolon-delimited AAER CSV file."""

        logger.info("Validating AAER records.")
        summary = DatasetValidationSummary()
        seen_ids: set[str] = set()

        try:
            with self.paths.aaer.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as csv_file:
                reader = csv.DictReader(csv_file, delimiter=";")
                fieldnames = [
                    field.strip()
                    for field in reader.fieldnames or []
                ]
                summary.schema_fields = sorted(fieldnames)
                summary.required_missing = sorted(
                    self.REQUIRED_AAER_COLUMNS.difference(fieldnames)
                )
                self._record_missing_schema_errors(
                    dataset_name="aaer",
                    missing=summary.required_missing,
                    summary=summary,
                )

                for row in reader:
                    row_errors = self._validate_aaer_row(
                        row=row,
                        summary=summary,
                        seen_ids=seen_ids,
                    )
                    if row_errors:
                        summary.invalid_records += 1
                    else:
                        summary.valid_records += 1
        except Exception as exc:
            message = f"Unable to read aaer_mark5.csv: {exc}"
            summary.errors.append(message)
            logger.exception(message)

        self._store_dataset_summary("aaer", summary)

    def _validate_aaer_row(
        self,
        row: dict[str | None, Any],
        summary: DatasetValidationSummary,
        seen_ids: set[str],
    ) -> list[str]:
        """Validate one AAER CSV row and update aggregate AAER state."""

        summary.total_records += 1
        row_errors: list[str] = []

        if self._is_empty_record(row):
            summary.empty_records += 1
            row_errors.append("empty_row")
            return row_errors

        if None in row:
            summary.malformed_rows += 1
            row_errors.append("malformed_row")

        row_errors.extend(
            self._validate_required_values(
                record=row,
                required_fields=self.REQUIRED_AAER_COLUMNS,
                summary=summary,
            )
        )

        aaer_id = str(row.get("id", "")).strip()
        if aaer_id:
            if aaer_id in seen_ids:
                summary.duplicate_ids += 1
                row_errors.append("duplicate_id")
            else:
                seen_ids.add(aaer_id)

        cik = str(row.get("cik", "")).strip()
        if cik:
            self._aaer_ciks.add(cik)

        if not self._is_valid_date(row.get("fraud_start"), "%m-%Y"):
            summary.malformed_dates += 1
            row_errors.append("malformed_fraud_start")

        if not self._is_valid_date(row.get("fraud_end"), "%m-%Y"):
            summary.malformed_dates += 1
            row_errors.append("malformed_fraud_end")

        return row_errors

    def validate_cross_dataset_consistency(self) -> None:
        """Validate consistency and overlap between raw datasets."""

        logger.info("Validating cross-dataset consistency.")
        orphan_label_identifiers = self._label_identifiers.difference(
            self._firm_identifiers
        )
        orphan_label_ciks = self._label_ciks.difference(self._firm_ciks)
        orphan_aaer_ciks = self._aaer_ciks.difference(self._firm_ciks)

        self.result.cross_dataset = {
            "orphan_labels": len(orphan_label_identifiers),
            "orphan_label_ciks": len(orphan_label_ciks),
            "orphan_aaer_entries": len(orphan_aaer_ciks),
            "unmatched_ciks": {
                "labels_not_in_firm_years": len(orphan_label_ciks),
                "aaer_not_in_firm_years": len(orphan_aaer_ciks),
                "firm_years_not_in_labels": len(
                    self._firm_ciks.difference(self._label_ciks)
                ),
                "firm_years_not_in_aaer": len(
                    self._firm_ciks.difference(self._aaer_ciks)
                ),
            },
            "summary_overlap_statistics": {
                "firm_year_unique_ciks": len(self._firm_ciks),
                "label_unique_ciks": len(self._label_ciks),
                "aaer_unique_ciks": len(self._aaer_ciks),
                "firm_label_cik_overlap": len(
                    self._firm_ciks.intersection(self._label_ciks)
                ),
                "firm_aaer_cik_overlap": len(
                    self._firm_ciks.intersection(self._aaer_ciks)
                ),
                "label_aaer_cik_overlap": len(
                    self._label_ciks.intersection(self._aaer_ciks)
                ),
                "firm_label_identifier_overlap": len(
                    self._firm_identifiers.intersection(
                        self._label_identifiers
                    )
                ),
            },
        }

        if orphan_label_identifiers:
            self.result.warnings.append(
                "Found label identifiers not present in firm_years.json: "
                f"{len(orphan_label_identifiers)}"
            )
        if orphan_aaer_ciks:
            self.result.warnings.append(
                "Found AAER CIKs not present in firm_years.json: "
                f"{len(orphan_aaer_ciks)}"
            )

    @staticmethod
    def _detect_json_root_type(path: Path) -> str:
        """Detect whether a JSON file starts with an array or object."""

        with path.open("rb") as json_file:
            for prefix, event, _ in ijson.parse(json_file):
                if prefix == "" and event == "start_array":
                    return "array"
                if prefix == "" and event == "start_map":
                    return "object"
        return "unknown"

    @staticmethod
    def _stream_json_array(path: Path) -> Iterable[JsonDict]:
        """Yield records from a JSON array without loading the full file."""

        with path.open("rb") as json_file:
            yield from ijson.items(json_file, "item")

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Return True when a scalar value is absent or blank."""

        return value is None or (isinstance(value, str) and not value.strip())

    @classmethod
    def _is_empty_record(cls, record: Any) -> bool:
        """Return True when a record has no meaningful values."""

        if not isinstance(record, dict) or not record:
            return True
        return all(cls._is_missing(value) for value in record.values())

    @staticmethod
    def _is_valid_date(value: Any, date_format: str) -> bool:
        """Validate a date string against an exact expected format."""

        if not isinstance(value, str) or not value.strip():
            return False

        try:
            parsed = datetime.strptime(value.strip(), date_format)
        except ValueError:
            return False

        return parsed.strftime(date_format) == value.strip()

    @classmethod
    def _record_identifier(cls, record: dict[Any, Any]) -> Identifier:
        """Build a stable filing identifier from a raw firm-year record."""

        return (
            cls._normalise_identifier_value(record.get("cik")),
            cls._normalise_identifier_value(record.get("filing_date")),
            cls._normalise_identifier_value(record.get("reporting_date")),
            cls._normalise_identifier_value(record.get("filing_type")),
        )

    @staticmethod
    def _normalise_identifier_value(value: Any) -> str:
        """Normalize values used in cross-dataset identifiers."""

        return "" if value is None else str(value).strip()

    @classmethod
    def _validate_required_values(
        cls,
        record: dict[Any, Any],
        required_fields: set[str],
        summary: DatasetValidationSummary,
    ) -> list[str]:
        """Validate required field presence and update missing counters."""

        row_errors: list[str] = []
        missing_values = defaultdict(int, summary.missing_values)

        for field_name in sorted(required_fields):
            if cls._is_missing(record.get(field_name)):
                missing_values[field_name] += 1
                row_errors.append(f"missing_{field_name}")

        summary.missing_values = dict(missing_values)
        return row_errors

    @staticmethod
    def _record_missing_schema_errors(
        dataset_name: str,
        missing: list[str],
        summary: DatasetValidationSummary,
    ) -> None:
        """Append dataset-level schema errors for missing required fields."""

        if not missing:
            return

        summary.errors.append(
            f"{dataset_name} missing required fields/columns: {missing}"
        )

    def _store_dataset_summary(
        self,
        dataset_name: str,
        summary: DatasetValidationSummary,
    ) -> None:
        """Store one summary and bubble warnings/errors to top level."""

        summary_dict = self._summary_to_dict(summary)
        self.result.datasets[dataset_name] = summary_dict
        self.result.schema_summary[dataset_name] = {
            "fields": summary.schema_fields,
            "required_missing": summary.required_missing,
            "additional_fields": summary.additional_fields,
        }
        self.result.warnings.extend(summary.warnings)
        self.result.errors.extend(summary.errors)

    def _dataset_metric_totals(self, metric_name: str) -> dict[str, int]:
        """Return one integer metric for every validated dataset."""

        return {
            dataset_name: int(dataset_summary.get(metric_name, 0))
            for dataset_name, dataset_summary in self.result.datasets.items()
        }

    @staticmethod
    def _summary_to_dict(summary: DatasetValidationSummary) -> dict[str, Any]:
        """Convert a dataset summary to JSON-serializable data."""

        return {
            "total_records": summary.total_records,
            "valid_records": summary.valid_records,
            "invalid_records": summary.invalid_records,
            "empty_records": summary.empty_records,
            "duplicate_identifiers": summary.duplicate_identifiers,
            "duplicate_ids": summary.duplicate_ids,
            "duplicate_filings": summary.duplicate_filings,
            "invalid_filing_types": summary.invalid_filing_types,
            "malformed_dates": summary.malformed_dates,
            "malformed_rows": summary.malformed_rows,
            "missing_values": dict(sorted(summary.missing_values.items())),
            "schema_fields": summary.schema_fields,
            "required_missing": summary.required_missing,
            "additional_fields": summary.additional_fields,
            "warnings": summary.warnings,
            "errors": summary.errors,
        }

    def _sync_firm_year_compatibility_counters(
        self,
        summary: DatasetValidationSummary,
    ) -> None:
        """Populate legacy top-level counters from firm-year validation."""

        self.result.total_records = summary.total_records
        self.result.valid_records = summary.valid_records
        self.result.invalid_records = summary.invalid_records
        self.result.duplicate_records = summary.duplicate_filings
        self.result.malformed_dates = summary.malformed_dates
        self.result.missing_cik = summary.missing_values.get("cik", 0)
        self.result.missing_name = summary.missing_values.get("name", 0)
        self.result.missing_mda = summary.missing_values.get("mda", 0)


def validate_raw_data() -> ValidationResult:
    """Validate all raw datasets and write the raw validation report."""

    validator = RawDataValidator()
    return validator.validate()


if __name__ == "__main__":
    validate_raw_data()


