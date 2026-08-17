"""Export a small, versioned snapshot from historical report artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.fingerprints import sha256_file  # noqa: E402

EXPERIMENT = "xgboost_lm_text_surface_probability_v2"
REPORT_DIR = ROOT / "reports" / "models" / EXPERIMENT
OUTPUT_DIR = ROOT / "demo" / "artifacts"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    review_path = REPORT_DIR / "candidate_review_full_refit.json"
    final_path = REPORT_DIR / "final_test" / "final_test_summary.json"
    maturity_path = ROOT / "reports" / "label_maturity" / "label_maturity_summary.json"
    shap_path = REPORT_DIR / "shap" / "shap_importance.csv"
    review, final, maturity = map(read_json, (review_path, final_path, maturity_path))
    selected = next(
        row
        for row in review["candidates"]
        if row["trial_number"] == review["selected_trial_number"]
    )
    metrics = selected["metrics"]
    evidence = {
        "schema_version": "historical-demo-evidence-v1",
        "experiment": EXPERIMENT,
        "trial": selected["trial_number"],
        "selection_rule": review["selection_rule"],
        "development_validation_status": "invalidated_temporal_overlap",
        "frozen_result_status": "historical_nonconfirmatory",
        "development": {
            "folds": metrics["n_folds"],
            "recall_at_5": metrics["recall_at_5_percent"]["mean"],
            "precision_at_5": metrics["precision_at_5_percent"]["mean"],
            "pr_auc": metrics["pr_auc"]["mean"],
            "brier_skill": metrics["brier_skill_score"]["mean"],
        },
        "final_test": {
            "period": final["test_period"],
            "filings": final["test_n"],
            "fraud_cases": final["test_fraud_n"],
            "reviewed": final["review_n_at_5_percent"],
            "fraud_found": round(final["recall_at_5_percent"] * final["test_fraud_n"]),
            "recall_at_5": final["recall_at_5_percent"],
            "precision_at_5": final["precision_at_5_percent"],
            "roc_auc": final["roc_auc"],
            "pr_auc": final["pr_auc"],
            "brier_skill": final["brier_skill_score"],
            "calibrated": final["calibrated"],
            "threshold": final["decision_threshold"],
            "threshold_tuned_on_test": final["threshold_tuned_on_test"],
        },
        "label_maturity": {
            "median_days": maturity["observed_label_delay"]["median_days"],
            "after_2022": maturity["observed_label_delay"][
                "labels_recorded_after_2022_12_31"
            ],
            "positive_labels": maturity["coverage"]["positive_filing_labels"],
            "p90_cutoff": maturity["empirical_maturity_thresholds"]["p90"][
                "filings_on_or_before"
            ],
        },
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (review_path, final_path, maturity_path, shap_path)
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "historical_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    with shap_path.open(encoding="utf-8", newline="") as source:
        rows = sorted(
            csv.DictReader(source),
            key=lambda row: float(row["mean_abs_shap"]),
            reverse=True,
        )[:10]
    with (OUTPUT_DIR / "shap_importance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=["feature", "mean_abs_shap"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported historical demo evidence to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
