# src/cleaning.py

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from config import ROOT, ENRICHED_DATA_PATH, DATA_PROCESSED

CLEANED_DATA_PATH = DATA_PROCESSED / "transactions_cleaned.parquet"
DOCS = ROOT / "docs"
CLEANING_REPORT_PATH = DOCS / "cleaning_decisions.md"

REQUIRED_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
    "account_age_days", "home_billing_country", "home_device_id",
    "device_id", "is_new_device", "browser_fingerprint",
    "shipping_billing_mismatch", "failed_payment_attempts",
    "ip_country", "ip_billing_distance_km",
    "hour_of_day", "is_night",
]

BINARY_COLUMNS = [
    "isFraud",
    "isFlaggedFraud",
    "is_new_device",
    "shipping_billing_mismatch",
    "is_night",
]


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing_cols


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}

    report["rows_before"] = len(df)
    report["columns_before"] = df.shape[1]
    report["missing_values_before"] = int(df.isna().sum().sum())
    report["duplicate_rows_before"] = int(df.duplicated().sum())

    df = df.copy()

    # 1. Standardize categorical text
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    df["home_billing_country"] = df["home_billing_country"].astype(str).str.strip().str.upper()
    df["ip_country"] = df["ip_country"].astype(str).str.strip().str.upper()

    # 2. Ensure numeric columns
    numeric_cols = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "account_age_days",
        "failed_payment_attempts",
        "ip_billing_distance_km",
        "hour_of_day",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Ensure binary columns are valid 0/1
    for col in BINARY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid_binary = ~df[col].isin([0, 1])
        report[f"invalid_binary_{col}"] = int(invalid_binary.sum())

    # 4. Remove exact duplicate rows if any
    df = df.drop_duplicates()

    # 5. Invalid amount handling
    df["invalid_amount_flag"] = (df["amount"] < 0).astype(int)
    df["amount_zero_flag"] = (df["amount"] == 0).astype(int)

    # Negative amount is invalid for this business setting
    negative_amount_rows = int((df["amount"] < 0).sum())
    report["negative_amount_rows_removed"] = negative_amount_rows
    df = df[df["amount"] >= 0].copy()

    # 6. Invalid distance/account age handling
    df["invalid_ip_distance_flag"] = (df["ip_billing_distance_km"] < 0).astype(int)
    df["invalid_account_age_flag"] = (df["account_age_days"] < 1).astype(int)

    report["invalid_ip_distance_rows"] = int(df["invalid_ip_distance_flag"].sum())
    report["invalid_account_age_rows"] = int(df["invalid_account_age_flag"].sum())

    # Do not remove these automatically; mark them for review.
    # In current synthetic generator, these should normally be 0.

    # 7. Destination account helper for later modules
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

    # 8. Final report
    report["rows_after"] = len(df)
    report["columns_after"] = df.shape[1]
    report["missing_values_after"] = int(df.isna().sum().sum())
    report["duplicate_rows_after"] = int(df.duplicated().sum())

    return df, report


def write_cleaning_report(report: dict, path: Path = CLEANING_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Module 3 — Data Cleaning Decisions",
        "",
        "## Summary",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]

    for key, value in report.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Cleaning Decisions",
        "",
        "| Issue | Decision | Reason |",
        "|---|---|---|",
        "| Exact duplicate rows | Removed if present | Duplicate rows may bias model training. |",
        "| `amount < 0` | Removed | Negative transaction amount is invalid for payment fraud detection. |",
        "| `amount = 0` | Kept with `amount_zero_flag` | Zero amount may be system-generated or rule-related; do not delete without evidence. |",
        "| Extreme amount values | Kept | Large transactions may be valid fraud signals. Use `log_amount` in feature engineering. |",
        "| `isFlaggedFraud` | Kept in cleaned data, excluded from default model features | It is a weak existing rule flag and may bias evaluation. |",
        "| Raw identifiers | Kept for traceability, excluded from default model features | High-cardinality IDs are not directly model-ready. |",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> pd.DataFrame:
    print(f"Reading enriched data: {ENRICHED_DATA_PATH}")
    df = pd.read_parquet(ENRICHED_DATA_PATH)

    missing_cols = validate_required_columns(df)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    cleaned, report = clean_transactions(df)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(CLEANED_DATA_PATH, index=False)
    write_cleaning_report(report)

    print(f"Wrote cleaned data -> {CLEANED_DATA_PATH}")
    print(f"Wrote cleaning report -> {CLEANING_REPORT_PATH}")

    return cleaned


if __name__ == "__main__":
    run()