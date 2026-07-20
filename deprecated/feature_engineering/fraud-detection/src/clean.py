"""
MODULE 3 - Data cleaning.

Exposes a single reusable `clean()` function so the SAME logic runs in the EDA/
training notebooks and later in the deployment API - no train/serve skew.

Design decisions (documented in notebooks/03_cleaning.ipynb with before/after):
  * PaySim is structurally clean (no missing, no duplicates) - we still guard
    against all three, so the function is safe on any incoming batch.
  * Zero balances are often MISSING, not real zeros: PaySim does not track
    merchant balances, so many rows show oldbalanceDest==newbalanceDest==0 while
    amount>0. We DO NOT impute these; we expose them as explicit flags
    (`dest_balance_unknown`, `orig_balance_unknown`) so the model can use the
    "unknown" pattern instead of trusting a fake 0.
  * Outliers in `amount` are RETAINED deliberately - extreme amounts are
    fraud-informative and tree models are robust to them. We only remove
    genuinely invalid rows (amount<=0, out-of-range step).
  * `isFlaggedFraud` is dropped: it is a post-hoc rule flag with ~0 recall
    (16/8213 on the full data) and is not a legitimate real-time input.
  * Type scoping: fraud occurs only in TRANSFER/CASH_OUT. Scoping to those types
    roughly doubles the working fraud rate and shrinks the data ~2.3x. This is a
    documented modelling choice; at serving time, other types are auto-approved
    (they are never fraud) rather than scored.
  * ID columns are KEPT here - Module 4 needs `nameOrig` etc. for velocity
    features. They are dropped just before modelling, not during cleaning.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import FRAUD_TYPES

STEP_MIN, STEP_MAX = 1, 743
LEAKAGE_COLS = ["isFlaggedFraud"]
# analysis-only columns that must never ride into the modelling dataset
SCRATCH_COLS = ["amount_decile", "log_amount"]


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dest_balance_unknown"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)
    df["orig_balance_unknown"] = (
        (df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0) & (df["amount"] > 0)
    ).astype(int)
    return df


def clean(df: pd.DataFrame,
          scope_types: bool = True,
          drop_leakage: bool = True) -> tuple[pd.DataFrame, dict]:
    """Clean a transaction batch. Returns (clean_df, report)."""
    report: dict = {"rows_in": len(df)}
    df = df.copy()

    # 1) duplicates
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    report["duplicates_removed"] = dups

    # 2) missing values in core fields (PaySim has none; guard anyway)
    core = ["step", "type", "amount", "isFraud"]
    report["missing_core_removed"] = int(df[core].isna().any(axis=1).sum())
    df = df.dropna(subset=core)

    # 3) implausible amounts and invalid steps.
    #    PaySim has no negative amounts, so `amount<=0` only ever catches
    #    amount==0 - and those rows are LABELLED FRAUD (probe/test transactions).
    #    In a problem with ~8k positives we must NOT discard them: we remove only
    #    genuinely invalid negatives and expose amount==0 as a flag instead.
    report["negative_amount_removed"] = int((df["amount"] < 0).sum())
    df = df[df["amount"] >= 0]
    df["amount_is_zero"] = (df["amount"] == 0).astype(int)
    report["amount_is_zero_flagged"] = int(df["amount_is_zero"].sum())
    report["invalid_step_removed"] = int(((df["step"] < STEP_MIN) | (df["step"] > STEP_MAX)).sum())
    df = df[(df["step"] >= STEP_MIN) & (df["step"] <= STEP_MAX)]

    # 4) expose zero-balance-as-missing as explicit flags (no imputation)
    df = add_quality_flags(df)
    report["dest_balance_unknown"] = int(df["dest_balance_unknown"].sum())
    report["orig_balance_unknown"] = int(df["orig_balance_unknown"].sum())

    # 5) scope to fraud-bearing transaction types (documented modelling choice)
    if scope_types:
        before = len(df)
        df = df[df["type"].isin(FRAUD_TYPES)].copy()
        report["rows_dropped_by_type_scope"] = before - len(df)

    # 6) drop leakage column(s) and any analysis-only scratch columns
    dropped = []
    if drop_leakage:
        for c in LEAKAGE_COLS:
            if c in df.columns:
                df = df.drop(columns=c)
                dropped.append(c)
    for c in SCRATCH_COLS:
        if c in df.columns:
            df = df.drop(columns=c)
            dropped.append(c)
    report["columns_dropped"] = dropped

    report["rows_out"] = len(df)
    report["fraud_out"] = int(df["isFraud"].sum())
    report["fraud_rate_out_pct"] = round(df["isFraud"].mean() * 100, 4)
    return df, report


if __name__ == "__main__":
    from config import ENRICHED_DATA_PATH, DATA_PROCESSED
    df = pd.read_parquet(ENRICHED_DATA_PATH)
    clean_df, rep = clean(df)
    out = DATA_PROCESSED / "transactions_clean.parquet"
    clean_df.to_parquet(out, index=False)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"Wrote {out}")
