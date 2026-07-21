"""
MODULE 4 - Feature engineering (reusable).

Two responsibilities, both importable by training AND serving:
  * engineer_features(df): construct risk-signal features. All account-history
    ("velocity") features are CAUSAL - each row uses only earlier transactions
    (sorted by `step`), so they never peek at the future and are reproducible at
    serving time given the account's history.
  * build_preprocessor(...): a sklearn ColumnTransformer (scale numerics,
    one-hot categoricals) that is FIT ON TRAIN ONLY and saved, so the identical
    transform is reloaded by the API. No train/serve skew.

Leakage discipline enforced here and in build_features.py:
  - split BEFORE any resampling; SMOTE/class-weights touch the train fold only
  - the scaler/encoder are fit on train only
  - velocity features look strictly backwards (causal), mirroring production
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# high-cardinality ids / raw strings used to BUILD features, then dropped
DROP_AFTER_FE = ["nameOrig", "nameDest", "device_id", "home_device_id",
                 "browser_fingerprint", "home_billing_country", "ip_country", "step"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("step", kind="stable").copy()

    # --- balance / error signals (strongest raw signals from EDA) ---
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["origin_emptied"] = ((df["newbalanceOrig"] == 0) &
                            (np.isclose(df["oldbalanceOrg"], df["amount"]))).astype(int)
    df["amount_to_oldbalanceOrg"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    # --- amount ---
    df["log_amount"] = np.log1p(df["amount"])

    # --- geo mismatch flag (complements ip_billing_distance_km) ---
    df["ip_country_mismatch"] = (df["ip_country"] != df["home_billing_country"]).astype(int)

    # --- CAUSAL velocity features (only prior transactions per account) ---
    g_orig = df.groupby("nameOrig", sort=False)
    df["orig_prior_txn_count"] = g_orig.cumcount()                       # 0 = first ever
    prev_step = g_orig["step"].shift(1)
    df["orig_time_since_last"] = (df["step"] - prev_step)
    df["is_first_orig_txn"] = df["orig_time_since_last"].isna().astype(int)
    df["orig_time_since_last"] = df["orig_time_since_last"].fillna(-1)

    g_dest = df.groupby("nameDest", sort=False)
    df["dest_prior_txn_count"] = g_dest.cumcount()
    # cumulative amount received BEFORE this row (exclude current)
    df["dest_prior_amount_sum"] = g_dest["amount"].cumsum() - df["amount"]

    return df


# ---- column groups for the preprocessor -----------------------------------
NUMERIC_FEATURES = [
    "amount", "log_amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
    "amount_to_oldbalanceOrg", "account_age_days", "failed_payment_attempts",
    "ip_billing_distance_km", "hour_of_day",
    "orig_prior_txn_count", "orig_time_since_last", "dest_prior_txn_count",
    "dest_prior_amount_sum",
]
BINARY_FEATURES = [
    "is_new_device", "shipping_billing_mismatch", "is_night", "origin_emptied",
    "dest_balance_unknown", "orig_balance_unknown", "amount_is_zero",
    "ip_country_mismatch", "is_first_orig_txn",
]
CATEGORICAL_FEATURES = ["type"]
TARGET = "isFraud"


def model_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, pass binaries through, one-hot the type column."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = list(NUMERIC_FEATURES) + list(BINARY_FEATURES)
    ohe = preprocessor.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return names
