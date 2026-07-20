"""
Generate a small PaySim-SHAPED sample for testing the pipeline without the
full 6.3M-row Kaggle file.

It reproduces the structural facts that matter for our EDA/modelling:
  * columns & dtypes identical to the real PaySim CSV
  * fraud occurs ONLY in TRANSFER and CASH_OUT
  * fraud drains the origin account: oldbalanceOrg == amount, newbalanceOrig == 0
  * destinations starting with 'M' are merchants and are never defrauded

This is a TEST FIXTURE, not the real data. Do not use it for final results.
"""
import numpy as np
import pandas as pd
from config import DATA_RAW, SEED

TYPES = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
TYPE_P = [0.34, 0.35, 0.22, 0.083, 0.007]


def make_mock_paysim(n=100_000, fraud_rate=0.0013, seed=SEED):
    rng = np.random.default_rng(seed)
    step = rng.integers(1, 744, size=n)
    ttype = rng.choice(TYPES, size=n, p=TYPE_P)
    amount = rng.gamma(shape=1.3, scale=120_000, size=n).round(2)

    nameOrig = np.array([f"C{rng.integers(10**8, 10**9)}" for _ in range(n)])
    # ~15% of destinations are merchants (prefix M), rest customers (prefix C)
    is_merchant = rng.random(n) < 0.15
    nameDest = np.where(
        is_merchant,
        [f"M{rng.integers(10**8, 10**9)}" for _ in range(n)],
        [f"C{rng.integers(10**8, 10**9)}" for _ in range(n)],
    )

    oldbalanceOrg = rng.gamma(1.5, 200_000, size=n).round(2)
    newbalanceOrig = np.maximum(oldbalanceOrg - amount, 0).round(2)
    oldbalanceDest = rng.gamma(1.2, 300_000, size=n).round(2)
    newbalanceDest = (oldbalanceDest + amount).round(2)

    isFraud = np.zeros(n, dtype=int)
    # eligible fraud rows: TRANSFER/CASH_OUT to a customer (not merchant)
    eligible = np.isin(ttype, ["TRANSFER", "CASH_OUT"]) & (~is_merchant)
    n_fraud = int(n * fraud_rate)
    fraud_idx = rng.choice(np.where(eligible)[0], size=n_fraud, replace=False)
    isFraud[fraud_idx] = 1

    # inject the balance "tell" for fraud rows: account fully drained
    oldbalanceOrg[fraud_idx] = amount[fraud_idx]
    newbalanceOrig[fraud_idx] = 0.0

    isFlaggedFraud = np.zeros(n, dtype=int)  # PaySim's rule flags almost nothing
    isFlaggedFraud[(ttype == "TRANSFER") & (amount > 200_000)] = 0  # keep ~0

    df = pd.DataFrame({
        "step": step, "type": ttype, "amount": amount,
        "nameOrig": nameOrig, "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig, "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
        "isFraud": isFraud, "isFlaggedFraud": isFlaggedFraud,
    })
    return df


if __name__ == "__main__":
    out = DATA_RAW / "online_fraud_detection.csv"
    make_mock_paysim().to_csv(out, index=False)
    print(f"wrote mock PaySim -> {out}")
