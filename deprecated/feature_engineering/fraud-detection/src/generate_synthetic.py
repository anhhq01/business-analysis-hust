"""
MODULE 1 - Synthetic contextual/behavioural data generation.

The Kaggle PaySim base has no customer, device, or geo context - only an
account id (nameOrig). This script generates a realistic contextual layer and
joins it to every transaction, producing data/processed/transactions_enriched.parquet.

DESIGN PRINCIPLES
-----------------
1. Two levels of data, correctly keyed:
     * account-level fields (account_age_days, home country, home device) are
       STABLE per nameOrig - the same account always gets the same profile.
     * transaction-level fields (device used this time, ip distance, failed
       attempts, mismatch) VARY per row.
2. Realistic, NOT leaking, correlation with fraud. Each risky signal is drawn
   from a "risky" distribution with a HIGHER probability for fraud rows and a
   LOW base-rate probability for legit rows. No single feature separates the
   classes - the model has to combine them, and false positives / missed fraud
   remain possible (which is the whole business trade-off).
3. Fully seeded and deterministic given the input file.

Run:  python src/generate_synthetic.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import (RAW_DATA_PATH, ENRICHED_DATA_PATH, SEED,
                    COUNTRIES, COUNTRY_WEIGHTS)


# ---------------------------------------------------------------------------
# Account-level profiles (stable per nameOrig)
# ---------------------------------------------------------------------------
def build_account_profiles(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # one profile per unique origin account; age skews younger for accounts
    # that are ever involved in fraud (mild, account-level, heavy overlap kept)
    acct = df.groupby("nameOrig")["isFraud"].max().reset_index()
    uniq = acct["nameOrig"].to_numpy()
    n = len(uniq)
    base_age = rng.gamma(shape=2.0, scale=180.0, size=n)
    younger = np.where(acct["isFraud"].to_numpy() == 1,
                       rng.uniform(0.25, 0.75, size=n), 1.0)
    account_age_days = (base_age * younger).clip(1, 3650).astype(int)
    home_country = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    home_device_id = np.array([f"dev_{d:08x}" for d in rng.integers(0, 16**8, size=n)])
    return pd.DataFrame({
        "nameOrig": uniq,
        "account_age_days": account_age_days,
        "home_billing_country": home_country,
        "home_device_id": home_device_id,
    })


# ---------------------------------------------------------------------------
# Transaction-level contextual fields (vary per row, biased by fraud label)
# ---------------------------------------------------------------------------
def add_transaction_context(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    fraud = df["isFraud"].to_numpy() == 1

    def biased_bool(p_fraud: float, p_legit: float) -> np.ndarray:
        thr = np.where(fraud, p_fraud, p_legit)
        return rng.random(n) < thr

    # New device / device used this transaction
    is_new_device = biased_bool(0.70, 0.08)
    new_dev_ids = np.array([f"dev_{d:08x}" for d in rng.integers(0, 16**8, size=n)])
    device_id = np.where(is_new_device, new_dev_ids, df["home_device_id"].to_numpy())

    # Browser fingerprint derived from the device (stable per device, with noise)
    browser_fingerprint = np.array([f"fp_{abs(hash(d)) % (16**10):010x}" for d in device_id])

    # Shipping vs billing address mismatch
    shipping_billing_mismatch = biased_bool(0.55, 0.10)

    # Failed payment attempts before this transaction (fraud tries more)
    failed_payment_attempts = rng.poisson(np.where(fraud, 2.5, 0.3)).clip(0, 12)

    # Geo: IP country differs from billing more often for fraud; distance follows
    ip_foreign = biased_bool(0.60, 0.07)
    ip_country = np.where(
        ip_foreign, rng.choice(COUNTRIES, size=n), df["home_billing_country"].to_numpy()
    )
    same_country = ip_country == df["home_billing_country"].to_numpy()
    ip_billing_distance_km = np.where(
        same_country,
        rng.exponential(120.0, size=n),                 # domestic: small
        rng.exponential(2500.0, size=n) + 300.0,        # foreign: large
    ).round(1)

    # Time-of-day derived from PaySim's `step` (hours). Not injected - derived.
    hour_of_day = (df["step"].to_numpy() % 24).astype(int)
    is_night = ((hour_of_day >= 0) & (hour_of_day <= 5)).astype(int)

    out = df.copy()
    out["device_id"] = device_id
    out["is_new_device"] = is_new_device.astype(int)
    out["browser_fingerprint"] = browser_fingerprint
    out["shipping_billing_mismatch"] = shipping_billing_mismatch.astype(int)
    out["failed_payment_attempts"] = failed_payment_attempts.astype(int)
    out["ip_country"] = ip_country
    out["ip_billing_distance_km"] = ip_billing_distance_km
    out["hour_of_day"] = hour_of_day
    out["is_night"] = is_night
    return out


def generate(raw_path=RAW_DATA_PATH, out_path=ENRICHED_DATA_PATH, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    print(f"Reading base data: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  {len(df):,} rows, {df['isFraud'].sum():,} fraud "
          f"({df['isFraud'].mean()*100:.3f}%)")

    profiles = build_account_profiles(df, rng)
    df = df.merge(profiles, on="nameOrig", how="left")
    df = add_transaction_context(df, rng)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote enriched data -> {out_path}  ({df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    generate()
