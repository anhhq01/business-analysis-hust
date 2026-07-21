"""
Check whether the project is using the full Kaggle PaySim data or a small mock file.

Run from the repository root:
    python data_balancing/data_source_check.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
FULL_PAYSIM_MIN_BYTES = 400_000_000
MOCK_MAX_BYTES = 50_000_000

RAW_CANDIDATES = [
    ROOT / "feature_engineering" / "fraud-detection" / "data" / "raw" / "online_fraud_detection.csv",
    ROOT / "feature_engineering" / "fraud-detection" / "data" / "raw" / "PS_20174392719_1491204439457_log.csv",
]
ZIP_CANDIDATES = [
    ROOT / "feature_engineering" / "fraud-detection" / "data" / "raw" / "online-payments-fraud-detection-dataset.zip",
]
CLEANED_DATA = (
    ROOT
    / "feature_engineering"
    / "fraud-detection"
    / "data"
    / "processed"
    / "transactions_cleaned.parquet"
)


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def inspect_csv(path: Path) -> dict:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    columns: list[str] = []
    row_sample = 0
    schema_ok = False
    if exists and size:
        try:
            sample = pd.read_csv(path, nrows=5)
            columns = sample.columns.tolist()
            row_sample = len(sample)
            schema_ok = columns == EXPECTED_RAW_COLUMNS
        except Exception as exc:  # pragma: no cover - displayed to user
            columns = [f"read_error: {exc}"]
    if not exists:
        status = "missing"
    elif schema_ok and size >= FULL_PAYSIM_MIN_BYTES:
        status = "full_kaggle_candidate"
    elif schema_ok and size <= MOCK_MAX_BYTES:
        status = "mock_or_small_sample"
    elif schema_ok:
        status = "partial_or_unknown_size"
    else:
        status = "schema_mismatch"
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": exists,
        "size_bytes": size,
        "size": format_size(size),
        "status": status,
        "schema_ok": schema_ok,
        "sample_rows_read": row_sample,
        "columns": ", ".join(columns[:12]),
    }


def inspect_zip(path: Path) -> dict:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": exists,
        "size_bytes": size,
        "size": format_size(size),
        "status": "downloaded_zip" if exists else "missing",
    }


def inspect_cleaned_data(path: Path = CLEANED_DATA) -> dict:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    rows = None
    fraud_rows = None
    fraud_rate = None
    if exists and size:
        try:
            df = pd.read_parquet(path, columns=["isFraud"])
            rows = len(df)
            fraud_rows = int(df["isFraud"].sum())
            fraud_rate = float(df["isFraud"].mean())
        except Exception:
            rows = None
    status = "missing"
    if exists and rows is not None:
        status = "full_like" if rows >= 1_000_000 else "mock_or_sample"
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": exists,
        "size_bytes": size,
        "size": format_size(size),
        "rows": rows,
        "fraud_rows": fraud_rows,
        "fraud_rate": fraud_rate,
        "status": status,
    }


def build_data_source_report() -> dict:
    raw = [inspect_csv(path) for path in RAW_CANDIDATES]
    zips = [inspect_zip(path) for path in ZIP_CANDIDATES]
    cleaned = inspect_cleaned_data()
    full_candidates = [item for item in raw if item["status"] == "full_kaggle_candidate"]
    active_raw = raw[0]
    active_is_full = active_raw["status"] == "full_kaggle_candidate"
    recommendation = (
        "Active raw file looks like full Kaggle PaySim."
        if active_is_full
        else "Active feature_engineering raw file is not full Kaggle. Copy or rename a full PaySim CSV to feature_engineering/fraud-detection/data/raw/online_fraud_detection.csv, then rerun synthesis and cleaning."
    )
    return {
        "kaggle_cli_available": shutil.which("kaggle") is not None,
        "active_raw": active_raw,
        "active_raw_is_full": active_is_full,
        "full_raw_candidates": full_candidates,
        "raw_candidates": raw,
        "zip_candidates": zips,
        "cleaned_data": cleaned,
        "recommendation": recommendation,
    }


def main() -> None:
    report = build_data_source_report()
    print("Kaggle CLI:", "available" if report["kaggle_cli_available"] else "missing")
    print("Active raw:", report["active_raw"]["path"], report["active_raw"]["status"], report["active_raw"]["size"])
    print("Cleaned data:", report["cleaned_data"]["path"], report["cleaned_data"]["status"], report["cleaned_data"]["size"])
    if report["cleaned_data"]["rows"] is not None:
        print(
            "Cleaned rows:",
            f"{report['cleaned_data']['rows']:,}",
            "fraud:",
            f"{report['cleaned_data']['fraud_rows']:,}",
            "fraud_rate:",
            f"{report['cleaned_data']['fraud_rate']:.4%}",
        )
    print("Recommendation:", report["recommendation"])
    print("\nRaw candidates:")
    for item in report["raw_candidates"]:
        print("-", item["path"], "|", item["status"], "|", item["size"], "| schema_ok=", item["schema_ok"])
    print("\nZip candidates:")
    for item in report["zip_candidates"]:
        print("-", item["path"], "|", item["status"], "|", item["size"])


if __name__ == "__main__":
    main()
