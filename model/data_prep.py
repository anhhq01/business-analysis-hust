"""
Shared data preparation and feature engineering.

This module is imported by BOTH the training script (Module 5) and the
serving API (Module 6) so that a transaction is transformed exactly the
same way at train time and at inference time. Keeping this logic in one
place is the single most important safeguard against train/serve skew.

The base features follow directly from the EDA in
``data_cleaning/EDA.ipynb`` (log-amount transform, fraud concentrated in
TRANSFER / CASH_OUT) and add a few light risk signals in the spirit of
Module 4 (balance-error checks, time-of-day, amount ratios).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Transaction types present in the Kaggle "Online Payments Fraud
# Detection" dataset. Fixing the order makes one-hot encoding stable
# between training and serving.
TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Raw columns the model consumes from a single transaction record.
RAW_INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Final, ordered list of engineered feature columns fed to the model.
FEATURE_COLUMNS = [
    "log_amount",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "orig_balance_delta",
    "dest_balance_delta",
    "amount_to_oldOrg_ratio",
    "zero_dest_balance",
    "hour_of_day",
] + [f"type_{t}" for t in TYPE_CATEGORIES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction rows into the engineered feature matrix.

    Works for a full training frame or a single-row inference frame.
    Returns a DataFrame with exactly ``FEATURE_COLUMNS`` in order.
    """
    out = pd.DataFrame(index=df.index)

    amount = df["amount"].astype(float)
    old_org = df["oldbalanceOrg"].astype(float)
    new_org = df["newbalanceOrig"].astype(float)
    old_dest = df["oldbalanceDest"].astype(float)
    new_dest = df["newbalanceDest"].astype(float)

    # Amount signals (log transform justified by the right-skew in EDA).
    out["log_amount"] = np.log1p(amount)
    out["amount"] = amount
    out["oldbalanceOrg"] = old_org
    out["newbalanceOrig"] = new_org
    out["oldbalanceDest"] = old_dest
    out["newbalanceDest"] = new_dest

    # Balance-consistency errors. For a clean transfer the origin balance
    # should drop by ``amount``; a non-zero error is a strong fraud signal.
    out["errorBalanceOrig"] = new_org + amount - old_org
    out["errorBalanceDest"] = old_dest + amount - new_dest
    out["orig_balance_delta"] = old_org - new_org
    out["dest_balance_delta"] = new_dest - old_dest

    # How large is the transfer relative to what the account held?
    out["amount_to_oldOrg_ratio"] = amount / (old_org + 1.0)

    # Destination account never held or received money (drained/mule).
    out["zero_dest_balance"] = (
        ((old_dest == 0) & (new_dest == 0)).astype(int)
    )

    # Time-of-day pattern. ``step`` is hours since the simulation start.
    out["hour_of_day"] = (df["step"].astype(int) % 24)

    # One-hot encode transaction type against the fixed category set.
    type_series = df["type"].astype(str).str.upper()
    for t in TYPE_CATEGORIES:
        out[f"type_{t}"] = (type_series == t).astype(int)

    # Replace any inf/NaN produced by the arithmetic above.
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out[FEATURE_COLUMNS]


def load_raw(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load the base Kaggle CSV and apply the light cleaning from the EDA.

    The EDA found no missing values or duplicates, so cleaning here is
    limited to enforcing dtypes and dropping identifier columns the model
    must not learn from (``nameOrig`` / ``nameDest``).
    """
    df = pd.read_csv(csv_path)

    # Drop leakage-prone / identifier columns if present.
    for col in ["nameOrig", "nameDest"]:
        if col in df.columns:
            df = df.drop(columns=col)

    if max_rows is not None and len(df) > max_rows:
        # Stratified down-sample to keep both fraud and non-fraud, so
        # training stays fast on modest hardware without losing the
        # minority class.
        frac = max_rows / len(df)
        df = (
            df.groupby("isFraud", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=42))
            .reset_index(drop=True)
        )

    return df
