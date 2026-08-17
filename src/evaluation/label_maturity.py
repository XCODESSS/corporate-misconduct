# ruff: noqa: E501
"""Audit the observed delay between filings and AAER fraud-label records.

The AAER Mark 5 source contains a ``dateTime`` field.  Its formal semantics are
not documented in the local source bundle, so this module deliberately calls it
an *AAER source-record timestamp*.  It is useful evidence about label maturity,
but it must not be represented as the SEC enforcement or publication date
without independent source documentation.

This audit is read-only with respect to modelling inputs.  It never trains a
model and never alters the sealed final-test artifacts.
"""

from __future__ import annotations

import calendar
import json
from typing import Any

import configs.settings as settings
import pandas as pd
import pyarrow.parquet as pq
from src.ingestion.ingest_labels import FraudPeriodsIngestion
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LabelMaturityAudit:
    """Quantify the observed availability lag for final-test fraud labels."""

    RAW_AAER_FILE = settings.FINNLP_DIR / "aaer_mark5.csv"
    TEST_FEATURES_FILE = settings.FEATURES_DIR / "test_features.parquet"
    OUTPUT_DIR = settings.REPORTS_DIR / "label_maturity"

    START_YEAR = 2019
    END_YEAR = 2022
    MATURITY_PERCENTILES = (0.90, 0.95)

    SUMMARY_FILE = OUTPUT_DIR / "label_maturity_summary.json"
    YEARLY_FILE = OUTPUT_DIR / "label_maturity_by_filing_year.csv"
    LABEL_LEVEL_FILE = OUTPUT_DIR / "label_maturity_positive_labels.csv"
    REPORT_FILE = OUTPUT_DIR / "label_maturity_report.md"

    TEST_COLUMNS = [
        "cik",
        "filing_date",
        "fraudulent",
        "matched_fraud_start",
        "matched_fraud_end",
    ]

    @staticmethod
    def normalize_cik(value: Any) -> str:
        """Return the canonical zero-padded CIK used by the label merger."""
        if pd.isna(value):
            return ""

        normalized = str(value).strip()
        if normalized.endswith(".0"):
            normalized = normalized[:-2]
        return normalized.zfill(10)

    @staticmethod
    def parse_start_date(value: Any) -> pd.Timestamp:
        """Parse the AAER source's MM-YYYY fraud-period start."""
        if pd.isna(value) or "-" not in str(value):
            return pd.NaT
        try:
            month, year = str(value).strip().split("-")
            return pd.Timestamp(year=int(year), month=int(month), day=1)
        except (TypeError, ValueError):
            return pd.NaT

    @staticmethod
    def parse_end_date(value: Any) -> pd.Timestamp:
        """Parse the AAER source's MM-YYYY fraud-period end."""
        if pd.isna(value) or "-" not in str(value):
            return pd.NaT
        try:
            month, year = str(value).strip().split("-")
            return pd.Timestamp(
                year=int(year),
                month=int(month),
                day=calendar.monthrange(int(year), int(month))[1],
            )
        except (TypeError, ValueError):
            return pd.NaT

    @staticmethod
    def parse_source_record_timestamps(values: pd.Series) -> pd.Series:
        """Parse mixed AAER timestamp formats into timezone-naive timestamps."""
        try:
            timestamps = pd.to_datetime(
                values,
                errors="coerce",
                utc=True,
                format="mixed",
            )
        except TypeError:  # pragma: no cover - compatibility with older pandas
            timestamps = pd.to_datetime(values, errors="coerce", utc=True)
        return timestamps.dt.tz_convert(None)

    def load_period_lookup(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load AAER periods with their earliest source-record timestamp."""
        raw = pd.read_csv(
            self.RAW_AAER_FILE,
            sep=";",
            skiprows=2,
            header=None,
            names=FraudPeriodsIngestion.COLUMN_NAMES,
            dtype=str,
            encoding="utf-8",
        )

        raw["cik_key"] = raw["cik"].map(self.normalize_cik)
        raw["fraud_start_key"] = raw["fraud_start"].map(self.parse_start_date)
        raw["fraud_end_key"] = raw["fraud_end"].map(self.parse_end_date)
        raw["source_record_timestamp"] = self.parse_source_record_timestamps(
            raw["dateTime"]
        )

        required = ["cik_key", "fraud_start_key", "fraud_end_key"]
        raw = raw.dropna(subset=required)

        periods = (
            raw.groupby(required, as_index=False)
            .agg(
                aaer_source_record_first=("source_record_timestamp", "min"),
                aaer_source_record_last=("source_record_timestamp", "max"),
                source_record_count=("id", "size"),
            )
            .sort_values(required)
            .reset_index(drop=True)
        )

        metadata = {
            "raw_aaer_rows": int(len(raw)),
            "unique_fraud_periods": int(len(periods)),
            "source_record_timestamp_min": raw["source_record_timestamp"].min(),
            "source_record_timestamp_max": raw["source_record_timestamp"].max(),
            "missing_source_record_timestamps": int(
                raw["source_record_timestamp"].isna().sum()
            ),
        }
        return periods, metadata

    def load_final_period_labels(self) -> pd.DataFrame:
        """Load final-period filings using the same day-first parsing as modelling."""
        filings = pq.read_table(
            self.TEST_FEATURES_FILE,
            columns=self.TEST_COLUMNS,
        ).to_pandas()

        filings["filing_date"] = pd.to_datetime(
            filings["filing_date"],
            dayfirst=True,
            errors="coerce",
        )
        if filings["filing_date"].isna().any():
            raise ValueError("Test features contain invalid filing dates.")

        filings = filings.loc[
            filings["filing_date"].dt.year.between(self.START_YEAR, self.END_YEAR)
        ].copy()
        filings["filing_year"] = filings["filing_date"].dt.year
        filings["cik_key"] = filings["cik"].map(self.normalize_cik)
        filings["matched_fraud_start"] = pd.to_datetime(
            filings["matched_fraud_start"], errors="coerce"
        )
        filings["matched_fraud_end"] = pd.to_datetime(
            filings["matched_fraud_end"], errors="coerce"
        )
        return filings

    @staticmethod
    def attach_source_timestamps(
        positive_filings: pd.DataFrame,
        periods: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach AAER record timestamps to positive filing labels exactly."""
        labeled = positive_filings.merge(
            periods,
            left_on=["cik_key", "matched_fraud_start", "matched_fraud_end"],
            right_on=["cik_key", "fraud_start_key", "fraud_end_key"],
            how="left",
            validate="many_to_one",
        )
        if labeled["aaer_source_record_first"].isna().any():
            missing = int(labeled["aaer_source_record_first"].isna().sum())
            raise ValueError(
                f"{missing} positive filing labels could not be linked to an AAER period."
            )

        labeled["observed_label_lag_days"] = (
            labeled["aaer_source_record_first"] - labeled["filing_date"]
        ).dt.days
        return labeled

    def summarize(
        self,
        filings: pd.DataFrame,
        positive_labels: pd.DataFrame,
        source_metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Build machine-readable audit findings and yearly evidence."""
        reference_date = pd.Timestamp(source_metadata["source_record_timestamp_max"])
        if pd.isna(reference_date):
            raise ValueError("AAER source has no usable source-record timestamps.")

        yearly_rows: list[dict[str, Any]] = []
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            year_filings = filings.loc[filings["filing_year"] == year]
            year_labels = positive_labels.loc[positive_labels["filing_year"] == year]
            year_end = pd.Timestamp(year=year, month=12, day=31)
            yearly_rows.append(
                {
                    "filing_year": year,
                    "filings": int(len(year_filings)),
                    "positive_labels": int(len(year_labels)),
                    "recorded_fraud_rate": float(
                        year_labels.shape[0] / len(year_filings)
                    ),
                    "labels_known_by_filing_year_end": int(
                        (year_labels["aaer_source_record_first"] <= year_end).sum()
                    ),
                    "labels_recorded_after_filing_year_end": int(
                        (year_labels["aaer_source_record_first"] > year_end).sum()
                    ),
                    "median_observed_label_lag_days": float(
                        year_labels["observed_label_lag_days"].median()
                    ),
                    "p90_observed_label_lag_days": float(
                        year_labels["observed_label_lag_days"].quantile(0.90)
                    ),
                    "max_observed_label_lag_days": int(
                        year_labels["observed_label_lag_days"].max()
                    ),
                }
            )
        yearly = pd.DataFrame(yearly_rows)

        lag_days = positive_labels["observed_label_lag_days"]
        maturity_thresholds: dict[str, Any] = {}
        for percentile in self.MATURITY_PERCENTILES:
            lag = int(round(lag_days.quantile(percentile)))
            cutoff = reference_date - pd.Timedelta(days=lag)
            maturity_thresholds[f"p{int(percentile * 100)}"] = {
                "observed_lag_days": lag,
                "reference_date": reference_date,
                "filings_on_or_before": cutoff,
                "full_calendar_years_covered": [
                    year
                    for year in range(self.START_YEAR, self.END_YEAR + 1)
                    if pd.Timestamp(year=year, month=12, day=31) <= cutoff
                ],
            }

        labels_known_by_year_end = {
            str(year): int(
                (
                    positive_labels["aaer_source_record_first"]
                    <= pd.Timestamp(year, 12, 31)
                ).sum()
            )
            for year in range(self.START_YEAR, int(reference_date.year) + 1)
        }

        summary = {
            "audit_scope": {
                "filing_years": [self.START_YEAR, self.END_YEAR],
                "test_feature_file": str(self.TEST_FEATURES_FILE),
                "source_file": str(self.RAW_AAER_FILE),
                "source_timestamp_definition": (
                    "AAER Mark 5 dateTime field. Its formal source semantics were not "
                    "available locally; it is treated only as an AAER source-record timestamp."
                ),
            },
            "source": source_metadata,
            "coverage": {
                "final_period_filings": int(len(filings)),
                "positive_filing_labels": int(len(positive_labels)),
                "positive_labels_linked_to_aaer_period": int(len(positive_labels)),
                "unique_linked_fraud_periods": int(
                    len(
                        positive_labels[
                            ["cik_key", "matched_fraud_start", "matched_fraud_end"]
                        ].drop_duplicates()
                    )
                ),
            },
            "observed_label_delay": {
                "all_positive_labels_recorded_after_filing": bool((lag_days > 0).all()),
                "median_days": float(lag_days.median()),
                "p90_days": float(lag_days.quantile(0.90)),
                "p95_days": float(lag_days.quantile(0.95)),
                "max_days": int(lag_days.max()),
                "labels_recorded_after_2022_12_31": int(
                    (
                        positive_labels["aaer_source_record_first"]
                        > pd.Timestamp(2022, 12, 31)
                    ).sum()
                ),
                "labels_known_by_source_year_end": labels_known_by_year_end,
            },
            "empirical_maturity_thresholds": maturity_thresholds,
            "interpretation": {
                "strict_certification": (
                    "No year is certified fully mature: the audit observes only labels that "
                    "exist in the source and cannot prove that undiscovered fraud is absent."
                ),
                "conditional_p90_conclusion": (
                    "A filing period is conditionally mature at the 90th-percentile threshold "
                    "only when its end date is on or before the stated cutoff."
                ),
                "evaluation_rule": (
                    "Do not use 2020-2022 as a final decision holdout without an explicit "
                    "label-maturity policy or independently documented enforcement/publication dates."
                ),
            },
        }
        return summary, yearly

    def write_outputs(
        self,
        summary: dict[str, Any],
        yearly: pd.DataFrame,
        positive_labels: pd.DataFrame,
    ) -> None:
        """Save audit evidence without changing model data or results."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with self.SUMMARY_FILE.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, default=str)

        yearly.to_csv(self.YEARLY_FILE, index=False)
        positive_labels[
            [
                "cik",
                "filing_date",
                "filing_year",
                "matched_fraud_start",
                "matched_fraud_end",
                "aaer_source_record_first",
                "aaer_source_record_last",
                "observed_label_lag_days",
            ]
        ].to_csv(self.LABEL_LEVEL_FILE, index=False)

        p90 = summary["empirical_maturity_thresholds"]["p90"]
        p95 = summary["empirical_maturity_thresholds"]["p95"]
        with self.REPORT_FILE.open("w", encoding="utf-8") as file:
            file.write("# Label-Maturity Audit\n\n")
            file.write("## Conclusion\n\n")
            file.write(
                "The final-period labels are delayed relative to filings. This is expected "
                "for an enforcement-derived label, but it means 2020–2022 cannot be treated "
                "as fully mature labels without a stated maturity policy.\n\n"
            )
            file.write("## Evidence\n\n")
            file.write(
                f"- {summary['coverage']['positive_filing_labels']} positive filing labels "
                "were linked exactly to AAER fraud periods.\n"
            )
            file.write(
                f"- All were recorded after their filing date; median observed delay: "
                f"{summary['observed_label_delay']['median_days']:.0f} days.\n"
            )
            file.write(
                f"- {summary['observed_label_delay']['labels_recorded_after_2022_12_31']} "
                "of those labels first appear after 2022-12-31.\n"
            )
            file.write(
                f"- Empirical p90 lag: {p90['observed_lag_days']} days; filings on or before "
                f"{pd.Timestamp(p90['filings_on_or_before']).date()} meet this conditional threshold.\n"
            )
            file.write(
                f"- Empirical p95 lag: {p95['observed_lag_days']} days; filings on or before "
                f"{pd.Timestamp(p95['filings_on_or_before']).date()} meet this conditional threshold.\n\n"
            )
            file.write("## Important limitation\n\n")
            file.write(
                "`dateTime` is retained as an AAER source-record timestamp, not asserted to be "
                "an SEC enforcement or publication date. This audit measures observed source "
                "availability, not a legally authoritative event date.\n\n"
            )
            file.write("## Recommended policy\n\n")
            file.write(
                "Treat 2019 as conditionally mature at the empirical p90 threshold. Treat "
                "2020–2022 as maturity-sensitive and do not tune or make final performance "
                "claims from them without a documented delay policy or external enforcement dates.\n"
            )

    def run(self) -> dict[str, Any]:
        """Run the read-only audit and return its JSON-serializable summary."""
        logger.info("Starting label-maturity audit.")
        periods, source_metadata = self.load_period_lookup()
        filings = self.load_final_period_labels()
        positive_labels = filings.loc[filings["fraudulent"] == 1].copy()
        positive_labels = self.attach_source_timestamps(positive_labels, periods)
        summary, yearly = self.summarize(filings, positive_labels, source_metadata)
        self.write_outputs(summary, yearly, positive_labels)
        logger.info("Label-maturity audit complete: %s", self.REPORT_FILE)
        return summary


def run_label_maturity_audit() -> dict[str, Any]:
    """Public API for the read-only label-maturity audit."""
    return LabelMaturityAudit().run()
