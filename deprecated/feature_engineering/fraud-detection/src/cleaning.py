"""
MODULE 3 - Data Cleaning.

Input:
    data/processed/transactions_enriched.parquet

Output:
    data/processed/transactions_cleaned.parquet
    docs/cleaning_decisions.md

This module validates schema, standardizes categories, checks invalid values,
removes exact duplicates, and documents cleaning decisions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, ENRICHED_DATA_PATH, DATA_PROCESSED, COUNTRIES


CLEANED_DATA_PATH = DATA_PROCESSED / "transactions_cleaned.parquet"
DOCS = ROOT / "docs"
CLEANING_REPORT_PATH = DOCS / "cleaning_decisions.md"


REQUIRED_COLUMNS = [
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
    "account_age_days",
    "home_billing_country",
    "home_device_id",
    "device_id",
    "is_new_device",
    "browser_fingerprint",
    "shipping_billing_mismatch",
    "failed_payment_attempts",
    "ip_country",
    "ip_billing_distance_km",
    "hour_of_day",
    "is_night",
]

BINARY_COLUMNS = [
    "isFraud",
    "isFlaggedFraud",
    "is_new_device",
    "shipping_billing_mismatch",
    "is_night",
]

ALLOWED_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

ALLOWED_COUNTRIES = set(COUNTRIES)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise an error if required input columns are missing."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean enriched transaction data and return cleaned data + report metrics."""
    report = {
        "rows_before": len(df),
        "columns_before": df.shape[1],
        "missing_values_before": int(df.isna().sum().sum()),
        "duplicate_rows_before": int(df.duplicated().sum()),
        "fraud_count_before": int(df["isFraud"].sum()),
        "fraud_rate_before_pct": float(df["isFraud"].mean() * 100),
    }

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Standardize categorical columns
    # ------------------------------------------------------------------
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    df["home_billing_country"] = (
        df["home_billing_country"].astype(str).str.strip().str.upper()
    )
    df["ip_country"] = df["ip_country"].astype(str).str.strip().str.upper()

    # ------------------------------------------------------------------
    # 2. Convert numeric columns safely
    # ------------------------------------------------------------------
    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "account_age_days",
        "failed_payment_attempts",
        "ip_billing_distance_km",
        "hour_of_day",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 3. Validate binary columns
    # ------------------------------------------------------------------
    for col in BINARY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = ~df[col].isin([0, 1])
        report[f"invalid_binary_{col}"] = int(invalid_mask.sum())

    # ------------------------------------------------------------------
    # 4. Remove exact duplicate rows
    # ------------------------------------------------------------------
    df = df.drop_duplicates()

    # ------------------------------------------------------------------
    # 5. Amount checks
    # ------------------------------------------------------------------
    df["amount_zero_flag"] = (df["amount"] == 0).astype(int)
    df["invalid_amount_flag"] = (df["amount"] < 0).astype(int)

    negative_amount_rows = int((df["amount"] < 0).sum())
    report["negative_amount_rows_removed"] = negative_amount_rows

    # Negative transaction amount is invalid in this business context.
    df = df[df["amount"] >= 0].copy()

    # ------------------------------------------------------------------
    # 6. Synthetic numeric validity checks
    # ------------------------------------------------------------------
    df["invalid_ip_distance_flag"] = (df["ip_billing_distance_km"] < 0).astype(int)
    df["invalid_account_age_flag"] = (df["account_age_days"] < 1).astype(int)

    report["invalid_ip_distance_rows"] = int(df["invalid_ip_distance_flag"].sum())
    report["invalid_account_age_rows"] = int(df["invalid_account_age_flag"].sum())

    # ------------------------------------------------------------------
    # 7. Helper flag for later feature engineering
    # ------------------------------------------------------------------
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

    # ------------------------------------------------------------------
    # 8. Final report
    # ------------------------------------------------------------------
    report["rows_after"] = len(df)
    report["columns_after"] = df.shape[1]
    report["missing_values_after"] = int(df.isna().sum().sum())
    report["duplicate_rows_after"] = int(df.duplicated().sum())
    report["fraud_count_after"] = int(df["isFraud"].sum())
    report["fraud_rate_after_pct"] = float(df["isFraud"].mean() * 100)

    return df, report


def write_cleaning_report(report: dict, path: Path = CLEANING_REPORT_PATH) -> None:
    """Write a Markdown report documenting before/after checks and decisions."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Module 3 — Data Cleaning Decisions",
        "",
        "## Before / After Summary",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]

    for key, value in report.items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Cleaning Decisions",
            "",
            "| Issue | Decision | Reason |",
            "|---|---|---|",
            "| Exact duplicate rows | Removed if present | Duplicate rows can bias model training and EDA summaries. |",
            "| `amount < 0` | Removed | Negative transaction amount is invalid for payment transactions. |",
            "| `amount = 0` | Kept with `amount_zero_flag` | Zero-amount transactions may be system-generated or rule-related; they should not be deleted without evidence. |",
            "| Extreme amount values | Kept | Large transactions may contain fraud signal; use `log_amount` later in feature engineering. |",
            "| `isFlaggedFraud` | Kept in cleaned data but excluded from default model features later | It is an existing rule flag, not a raw behavioural signal. |",
            "| Raw IDs: `nameOrig`, `nameDest`, `device_id`, `browser_fingerprint` | Kept for traceability | These are high-cardinality identifiers and should not be used directly as default model features. |",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> pd.DataFrame:
    """Run Module 3 end-to-end."""
    print(f"Reading enriched data: {ENRICHED_DATA_PATH}")
    df = pd.read_parquet(ENRICHED_DATA_PATH)

    validate_required_columns(df)

    cleaned_df, report = clean_transactions(df)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(CLEANED_DATA_PATH, index=False)

    write_cleaning_report(report)

    print(f"Wrote cleaned data -> {CLEANED_DATA_PATH}")
    print(f"Wrote cleaning report -> {CLEANING_REPORT_PATH}")
    print(f"Cleaned shape: {cleaned_df.shape}")

    return cleaned_df


if __name__ == "__main__":
    run()