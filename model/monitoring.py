"""
Module 7 - Monitoring with Evidently.

This script compares:
- reference data: artifacts/training_reference_sample.csv
- current data: artifacts/scoring_log.csv

Outputs:
- reports/data_drift_report.html
- reports/monitoring_summary.json

If labeled current data is available:
- artifacts/labeled_scoring_log.csv
then it also creates:
- reports/classification_performance_report.html
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently.presets import *
from evidently import Report
from evidently.metrics import *
from data_prep import FEATURE_COLUMNS


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPORT_DIR = Path(__file__).parent / "reports"

REFERENCE_PATH = ARTIFACT_DIR / "training_reference_sample.csv"
CURRENT_LOG_PATH = ARTIFACT_DIR / "scoring_log.csv"
LABELED_CURRENT_PATH = ARTIFACT_DIR / "labeled_scoring_log.csv"

DATA_DRIFT_REPORT_PATH = REPORT_DIR / "data_drift_report.html"
CLASSIFICATION_REPORT_PATH = REPORT_DIR / "classification_performance_report.html"
MONITORING_SUMMARY_PATH = REPORT_DIR / "monitoring_summary.json"


def load_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current scoring logs."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_PATH}. "
            "Run `python train.py --data <csv_path>` first."
        )

    if not CURRENT_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Current scoring log not found: {CURRENT_LOG_PATH}. "
            "Start the API and call /score or /score_batch first."
        )

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_LOG_PATH)

    missing_ref = [col for col in FEATURE_COLUMNS if col not in reference.columns]
    missing_cur = [col for col in FEATURE_COLUMNS if col not in current.columns]

    if missing_ref:
        raise ValueError(f"Missing reference feature columns: {missing_ref}")

    if missing_cur:
        raise ValueError(f"Missing current feature columns: {missing_cur}")

    return reference, current


def prediction_drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Create a simple business-friendly prediction drift summary."""
    summary = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
    }

    if "fraud_probability" in reference.columns and "fraud_probability" in current.columns:
        summary.update(
            {
                "reference_avg_fraud_probability": float(
                    reference["fraud_probability"].mean()
                ),
                "current_avg_fraud_probability": float(
                    current["fraud_probability"].mean()
                ),
                "reference_p95_fraud_probability": float(
                    reference["fraud_probability"].quantile(0.95)
                ),
                "current_p95_fraud_probability": float(
                    current["fraud_probability"].quantile(0.95)
                ),
            }
        )

    if "prediction" in reference.columns and "prediction" in current.columns:
        summary.update(
            {
                "reference_review_rate_pct": float(reference["prediction"].mean() * 100),
                "current_review_rate_pct": float(current["prediction"].mean() * 100),
            }
        )

    return summary


def run_data_drift_report() -> dict:
    """Generate Evidently data drift report."""
    from evidently import Report
    # from evidently.metrics import *
    # from evidently.presets import *

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reference, current = load_reference_current()

    reference_features = reference[FEATURE_COLUMNS].copy()
    current_features = current[FEATURE_COLUMNS].copy()

    report = Report(metrics=[DataDriftPreset()])

    result = report.run(
        reference_data=reference_features,
        current_data=current_features,
    )

    result.save_html(str(DATA_DRIFT_REPORT_PATH))

    summary = prediction_drift_summary(reference, current)
    summary["data_drift_report_path"] = str(DATA_DRIFT_REPORT_PATH)

    MONITORING_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote data drift report -> {DATA_DRIFT_REPORT_PATH}")
    print(f"Wrote monitoring summary -> {MONITORING_SUMMARY_PATH}")

    return summary


def run_classification_performance_report() -> dict:
    """
    Generate classification performance report if delayed labels are available.

    Required file:
        artifacts/labeled_scoring_log.csv

    Required columns:
        - isFraud
        - prediction
    """
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset

    if not LABELED_CURRENT_PATH.exists():
        raise FileNotFoundError(
            f"Labeled current file not found: {LABELED_CURRENT_PATH}. "
            "Skip this step until actual labels are available."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    current = pd.read_csv(LABELED_CURRENT_PATH)

    required_cols = FEATURE_COLUMNS + ["isFraud", "prediction"]
    missing = [col for col in required_cols if col not in current.columns]

    if missing:
        raise ValueError(f"Missing columns in labeled current data: {missing}")

    column_mapping = ColumnMapping()
    column_mapping.target = "isFraud"
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = FEATURE_COLUMNS

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=None,
        current_data=current[required_cols],
        column_mapping=column_mapping,
    )
    report.save_html(str(CLASSIFICATION_REPORT_PATH))

    summary = {
        "current_rows": int(len(current)),
        "classification_report_path": str(CLASSIFICATION_REPORT_PATH),
    }

    print(f"Wrote classification performance report -> {CLASSIFICATION_REPORT_PATH}")

    return summary


def main() -> None:
    run_data_drift_report()

    if LABELED_CURRENT_PATH.exists():
        run_classification_performance_report()
    else:
        print(
            "Skipping classification performance report: "
            "artifacts/labeled_scoring_log.csv not found."
        )


if __name__ == "__main__":
    main()