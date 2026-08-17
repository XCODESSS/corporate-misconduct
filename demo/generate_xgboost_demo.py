"""Build a static, artifact-driven XGBoost research-demo page.

This script does not train, score, tune, or reopen the sealed test set. It only
reads existing artifacts and creates ``demo/index.html`` for local presentation.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "xgboost_lm_text_surface_probability_v2"
REPORT_DIR = ROOT / "reports" / "models" / EXPERIMENT
LABEL_MATURITY_DIR = ROOT / "reports" / "label_maturity"

REQUIRED_REVIEW_FIELDS = {
    "status",
    "selected_trial_number",
    "selection_rule",
    "candidates",
}
REQUIRED_FINAL_TEST_FIELDS = {
    "test_period",
    "test_n",
    "test_fraud_n",
    "review_n_at_5_percent",
    "recall_at_5_percent",
    "precision_at_5_percent",
    "pr_auc",
    "brier_skill_score",
    "calibrated",
    "decision_threshold",
    "threshold_tuned_on_test",
}
REQUIRED_MATURITY_FIELDS = {
    "observed_label_delay",
    "coverage",
    "empirical_maturity_thresholds",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read an expected JSON artifact with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(f"Required demo artifact is missing: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def require_fields(
    artifact_name: str, payload: dict[str, Any], required_fields: set[str]
) -> None:
    """Validate the small, explicit artifact contract consumed by the demo."""
    missing = sorted(required_fields.difference(payload))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{artifact_name} is missing required field(s): {joined}")


def load_shap_importance(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Load the top SHAP features used in the selected-model explanation."""
    if not path.exists():
        raise FileNotFoundError(f"Required SHAP artifact is missing: {path}")

    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows or not {"feature", "mean_abs_shap"}.issubset(rows[0]):
        raise ValueError(
            "SHAP importance must contain non-empty feature and mean_abs_shap columns."
        )

    ranked = sorted(rows, key=lambda row: float(row["mean_abs_shap"]), reverse=True)
    return [
        {
            "feature": row["feature"],
            "importance": float(row["mean_abs_shap"]),
        }
        for row in ranked[:limit]
    ]


def select_candidate(review: dict[str, Any]) -> dict[str, Any]:
    """Return the persisted winner and reject incomplete review artifacts."""
    if review.get("status") != "selected":
        raise ValueError("Candidate review does not contain a selected model.")

    trial_number = review["selected_trial_number"]
    for candidate in review["candidates"]:
        if candidate["trial_number"] == trial_number:
            return candidate
    raise ValueError(f"Selected trial {trial_number} is absent from candidate review.")


def build_payload() -> dict[str, Any]:
    """Assemble only persisted evidence for the presentation page."""
    review = read_json(REPORT_DIR / "candidate_review_full_refit.json")
    final_test = read_json(REPORT_DIR / "final_test" / "final_test_summary.json")
    maturity = read_json(LABEL_MATURITY_DIR / "label_maturity_summary.json")
    require_fields("Candidate review", review, REQUIRED_REVIEW_FIELDS)
    require_fields("Final-test summary", final_test, REQUIRED_FINAL_TEST_FIELDS)
    require_fields("Label-maturity summary", maturity, REQUIRED_MATURITY_FIELDS)
    selected = select_candidate(review)
    require_fields("Selected candidate", selected, {"trial_number", "metrics"})
    development = selected["metrics"]
    require_fields(
        "Development metrics",
        development,
        {
            "n_folds",
            "years_evaluated",
            "recall_at_5_percent",
            "precision_at_5_percent",
            "pr_auc",
            "brier_skill_score",
        },
    )
    for metric_name in (
        "recall_at_5_percent",
        "precision_at_5_percent",
        "pr_auc",
        "brier_skill_score",
    ):
        require_fields(
            f"Development metric {metric_name}", development[metric_name], {"mean"}
        )
    require_fields(
        "Observed label delay",
        maturity["observed_label_delay"],
        {"median_days", "p90_days", "labels_recorded_after_2022_12_31"},
    )
    require_fields("Label coverage", maturity["coverage"], {"positive_filing_labels"})
    require_fields(
        "P90 label maturity",
        maturity["empirical_maturity_thresholds"]["p90"],
        {"filings_on_or_before"},
    )

    return {
        "experiment": EXPERIMENT,
        "trial": selected["trial_number"],
        "selection_rule": review["selection_rule"],
        "development": {
            "folds": development["n_folds"],
            "years": development["years_evaluated"],
            "recall_at_5": development["recall_at_5_percent"]["mean"],
            "precision_at_5": development["precision_at_5_percent"]["mean"],
            "pr_auc": development["pr_auc"]["mean"],
            "brier_skill": development["brier_skill_score"]["mean"],
        },
        "final_test": {
            "period": final_test["test_period"],
            "filings": final_test["test_n"],
            "fraud_cases": final_test["test_fraud_n"],
            "reviewed": final_test["review_n_at_5_percent"],
            "fraud_found": round(
                final_test["recall_at_5_percent"] * final_test["test_fraud_n"]
            ),
            "recall_at_5": final_test["recall_at_5_percent"],
            "precision_at_5": final_test["precision_at_5_percent"],
            "roc_auc": final_test["roc_auc"],
            "pr_auc": final_test["pr_auc"],
            "brier_skill": final_test["brier_skill_score"],
            "calibrated": final_test["calibrated"],
            "threshold": final_test["decision_threshold"],
            "threshold_tuned_on_test": final_test["threshold_tuned_on_test"],
        },
        "label_maturity": {
            "median_days": maturity["observed_label_delay"]["median_days"],
            "p90_days": maturity["observed_label_delay"]["p90_days"],
            "after_2022": maturity["observed_label_delay"][
                "labels_recorded_after_2022_12_31"
            ],
            "positive_labels": maturity["coverage"]["positive_filing_labels"],
            "p90_cutoff": maturity["empirical_maturity_thresholds"]["p90"][
                "filings_on_or_before"
            ],
        },
        "shap": load_shap_importance(REPORT_DIR / "shap" / "shap_importance.csv"),
        "artifacts": [
            {
                "label": "Candidate-selection review",
                "path": "../reports/models/xgboost_lm_text_surface_probability_v2/"
                "candidate_review_full_refit.json",
            },
            {
                "label": "Frozen-test metrics",
                "path": "../reports/models/xgboost_lm_text_surface_probability_v2/"
                "final_test/final_test_summary.json",
            },
            {
                "label": "Label-maturity audit",
                "path": "../reports/label_maturity/label_maturity_summary.json",
            },
            {
                "label": "SHAP feature importance",
                "path": "../reports/models/xgboost_lm_text_surface_probability_v2/"
                "shap/shap_importance.csv",
            },
        ],
    }


def render_html(payload: dict[str, Any]) -> str:
    """Return a self-contained page; no server, model, or external CDN is needed."""
    data = json.dumps(payload).replace("</", "<\\/")
    title = html.escape("Corporate Misconduct Warning - XGBoost Research Demo")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ --ink:#102a43; --muted:#627d98; --surface:#f5f7fa; --line:#d9e2ec; --blue:#1769aa; --red:#b42318; --green:#067647; --gold:#b54708; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--surface); font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.5; }}
    main {{ max-width:1120px; margin:auto; padding:32px 20px 56px; }}
    header {{ padding:14px 0 28px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(1.9rem,4vw,3rem); letter-spacing:-.04em; }}
    h2 {{ margin:0 0 12px; font-size:1.2rem; }}
    h3 {{ margin:0; font-size:.95rem; color:var(--muted); font-weight:600; }}
    p {{ margin:8px 0; }}
    .lede {{ max-width:760px; color:var(--muted); font-size:1.08rem; }}
    .badge {{ display:inline-block; margin-bottom:12px; padding:5px 9px; border-radius:999px; background:#fff3cd; color:#664d03; font-size:.82rem; font-weight:700; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
    .card, section {{ background:white; border:1px solid var(--line); border-radius:12px; box-shadow:0 1px 2px rgb(16 42 67 / .04); }}
    .card {{ padding:16px; }}
    .card strong {{ display:block; margin-top:7px; font-size:1.55rem; letter-spacing:-.03em; }}
    section {{ margin-top:18px; padding:22px; }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.94rem; }}
    th,td {{ padding:10px 8px; text-align:left; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-weight:600; }}
    .positive {{ color:var(--green); }} .negative {{ color:var(--red); }} .warning {{ color:var(--gold); }}
    .callout {{ border-left:4px solid var(--red); background:#fff6f5; padding:13px 15px; border-radius:4px; }}
    .flow {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }}
    .flow div {{ padding:10px; text-align:center; background:#edf6ff; border-radius:8px; font-size:.87rem; font-weight:600; }}
    .bar-row {{ display:grid; grid-template-columns:180px 1fr 64px; gap:10px; align-items:center; margin:8px 0; font-size:.88rem; }}
    .bar {{ height:11px; background:#e6edf5; border-radius:99px; overflow:hidden; }}
    .bar span {{ display:block; height:100%; background:var(--blue); border-radius:99px; }}
    code {{ background:#eef2f6; padding:2px 4px; border-radius:4px; }}
    footer {{ margin-top:20px; color:var(--muted); font-size:.84rem; }}
    @media(max-width:700px) {{ .two {{ grid-template-columns:1fr; }} .flow {{ grid-template-columns:1fr 1fr; }} .bar-row {{ grid-template-columns:120px 1fr 54px; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <span class="badge">RESEARCH PROTOTYPE - NOT A DEPLOYMENT DECISION TOOL</span>
      <h1>Corporate Misconduct Warning</h1>
      <p class="lede">A leakage-aware XGBoost research pipeline that ranks filings for a fixed analyst review queue and reports what survived, and what failed, on an untouched future period.</p>
    </header>
    <div id="app"></div>
    <footer>Generated from persisted project artifacts. This page does not train a model, score new companies, or reopen the sealed test set.</footer>
  </main>
  <script>
    const d = {data};
    const pct = value => `${{(value * 100).toFixed(2)}}%`;
    const dec = value => Number(value).toFixed(4);
    const date = value => new Date(value).toLocaleDateString('en-CA');
    const cls = value => value >= 0 ? 'positive' : 'negative';
    const dev = d.development, test = d.final_test, maturity = d.label_maturity;
    const maxShap = Math.max(...d.shap.map(item => item.importance));
    const shap = d.shap.map(item => `<div class="bar-row"><span>${{item.feature}}</span><div class="bar"><span style="width:${{100 * item.importance / maxShap}}%"></span></div><span>${{item.importance.toFixed(3)}}</span></div>`).join('');
    document.querySelector('#app').innerHTML = `
      <section class="callout"><strong>Headline result:</strong> Trial ${{d.trial}} selected by walk-forward validation found ${{test.fraud_found}} of ${{test.fraud_cases}} recorded fraud cases in the top 5% review queue on ${{test.period}} filings. But its final Brier skill was <span class="negative">${{dec(test.brier_skill)}}</span>, so the calibrated probabilities did not beat the naive fraud-rate baseline.</section>
      <section><h2>How the research workflow works</h2><div class="flow"><div>Chronological training</div><div>Walk-forward validation</div><div>Top-5% analyst queue</div><div>One-time final test</div></div></section>
      <section><h2>Model selection: development period</h2><p>Selected Trial ${{d.trial}} with the pre-defined rule: ${{d.selection_rule}}.</p><div class="grid"><div class="card"><h3>Walk-forward folds</h3><strong>${{dev.folds}}</strong></div><div class="card"><h3>Recall at 5%</h3><strong>${{pct(dev.recall_at_5)}}</strong></div><div class="card"><h3>Precision at 5%</h3><strong>${{pct(dev.precision_at_5)}}</strong></div><div class="card"><h3>Development Brier skill</h3><strong class="${{cls(dev.brier_skill)}}">${{dec(dev.brier_skill)}}</strong></div></div></section>
      <section><h2>Frozen final evaluation: ${{test.period}}</h2><div class="grid"><div class="card"><h3>Filings / fraud cases</h3><strong>${{test.filings.toLocaleString()}} / ${{test.fraud_cases}}</strong></div><div class="card"><h3>Top-5% review queue</h3><strong>${{test.reviewed.toLocaleString()}}</strong></div><div class="card"><h3>Recall at 5%</h3><strong>${{pct(test.recall_at_5)}}</strong></div><div class="card"><h3>Precision at 5%</h3><strong>${{pct(test.precision_at_5)}}</strong></div><div class="card"><h3>PR-AUC</h3><strong>${{dec(test.pr_auc)}}</strong></div><div class="card"><h3>Brier skill</h3><strong class="${{cls(test.brier_skill)}}">${{dec(test.brier_skill)}}</strong></div></div><p><strong>Recall@5% means:</strong> rank all filings by the model score, send only the highest-scoring 5% to analyst review, then measure what share of the recorded misconduct cases appears in that queue. It is a ranking metric, not a claim that the displayed probabilities are reliable.</p><p>Calibration was used; the classification threshold was fixed at ${{test.threshold.toFixed(2)}} and was not tuned on the test set.</p></section>
      <section class="two"><div><h2>Why later labels need caution</h2><p>Every observed positive label first appeared after its filing date. The median observed delay was <strong>${{Math.round(maturity.median_days).toLocaleString()}} days</strong>.</p><p><strong>${{maturity.after_2022}} of ${{maturity.positive_labels}}</strong> final-period labels first appeared after 2022 ended.</p><p class="warning">At the empirical p90 delay threshold, only filings on or before ${{date(maturity.p90_cutoff)}} are conditionally mature.</p></div><div><h2>Top explanatory features</h2><p>Mean absolute SHAP value from the selected development model.</p>${{shap}}</div></section>
      <section><h2>Interpretation</h2><ul><li><strong>Working pipeline:</strong> Candidate selection, calibration, ranking metrics, SHAP, and a sealed final evaluation are reproducible.</li><li><strong>Clear result:</strong> Historical development performance did not carry into the later period with trustworthy probability calibration.</li><li><strong>Next research step:</strong> improve label provenance and test a new feature family on development data only; do not tune against the frozen final test.</li></ul><p><strong>Evidence files:</strong> ${{d.artifacts.map(item => `<a href="${{item.path}}">${{item.label}}</a>`).join(' &middot; ')}}</p></section>`;
  </script>
</body>
</html>"""


def main() -> None:
    """Generate the default local presentation page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "demo" / "index.html",
        help="Destination HTML file (default: demo/index.html).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate source artifacts without writing an HTML file.",
    )
    args = parser.parse_args()

    payload = build_payload()
    if args.check:
        print(
            f"Validated demo artifacts for Trial {payload['trial']} "
            f"({payload['final_test']['period']})."
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(f"Demo page written to {args.output}")


if __name__ == "__main__":
    main()
