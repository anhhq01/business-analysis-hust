"""Builds notebooks/02_eda.ipynb (Module 2). Run, then execute with nbconvert."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Module 2 — Exploratory Data Analysis
Fraud detection on PaySim + synthetic contextual layer (`transactions_enriched.parquet`).

**Goals (per rubric):** quantify class imbalance; compare fraud rates across
transaction types and time; analyse distributions, outliers and correlations;
and visualise fraud patterns (amount, the balance "tell", geo/device signals).""")

code("""import sys
from pathlib import Path
# robust path setup: works whether cwd is repo root or notebooks/
here = Path.cwd()
root = here if (here / "src").exists() else here.parent
sys.path.insert(0, str(root / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ENRICHED_DATA_PATH, FIGURES

sns.set_theme(style="whitegrid")
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

df = pd.read_parquet(ENRICHED_DATA_PATH)
print(f"{len(df):,} rows x {df.shape[1]} columns")""")

md("## 1. Data integrity")
code("""df.info()
print("\\nmissing values:", int(df.isna().sum().sum()))
print("duplicate rows:", int(df.duplicated().sum()))""")

md("""## 2. Class imbalance — stated properly
The single most important number in the project. (The earlier notebook's
`value_counts(normalize=True)` was hidden by float formatting — here it is raw.)""")
code("""n = len(df)
n_fraud = int(df["isFraud"].sum())
print(f"Total transactions : {n:,}")
print(f"Fraud transactions : {n_fraud:,}")
print(f"Fraud rate         : {n_fraud/n*100:.4f}%  (~1 in {n//max(n_fraud,1):,})")
print(f"Imbalance ratio    : {(n-n_fraud)/max(n_fraud,1):,.0f} legit : 1 fraud")""")

md("""## 3. Fraud by transaction type
Fraud is concentrated entirely in `TRANSFER` and `CASH_OUT`; the other types
never contain fraud. This justifies scoping features/modelling to those types.""")
code("""by_type = (df.groupby("type")
             .agg(transactions=("isFraud", "size"),
                  fraud=("isFraud", "sum"),
                  fraud_rate_pct=("isFraud", lambda s: s.mean()*100))
             .sort_values("fraud_rate_pct", ascending=False))
by_type["fraud_share_pct"] = by_type["fraud"] / by_type["fraud"].sum() * 100
display(by_type)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
by_type["fraud_rate_pct"].plot.bar(ax=ax[0], color="crimson")
ax[0].set_title("Fraud rate by type (%)"); ax[0].set_ylabel("%")
by_type["transactions"].plot.bar(ax=ax[1], color="steelblue")
ax[1].set_title("Volume by type"); ax[1].set_ylabel("transactions")
plt.tight_layout(); plt.savefig(FIGURES/"fraud_by_type.png", dpi=150, bbox_inches="tight"); plt.show()""")

md("""## 4. Transaction amount — distribution and split by class
Amount is highly right-skewed, so we work on `log1p`. The key question is
whether fraud sits at different amounts than legit traffic.""")
code("""df["log_amount"] = np.log1p(df["amount"])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for label, sub in df.groupby("isFraud"):
    ax[0].hist(sub["log_amount"], bins=60, density=True, alpha=0.5,
               label=("fraud" if label else "legit"))
ax[0].set_title("log(amount+1) density by class"); ax[0].set_xlabel("log(amount+1)"); ax[0].legend()

sns.boxplot(data=df, x="isFraud", y="log_amount", ax=ax[1])
ax[1].set_xticklabels(["legit", "fraud"]); ax[1].set_title("log(amount+1) by class")
plt.tight_layout(); plt.savefig(FIGURES/"amount_by_class.png", dpi=150, bbox_inches="tight"); plt.show()

# fraud rate across amount deciles
df["amount_decile"] = pd.qcut(df["amount"], 10, labels=False, duplicates="drop")
display(df.groupby("amount_decile")["isFraud"].mean().mul(100).round(4).rename("fraud_rate_pct"))""")

md("""## 5. The balance "tell" — strongest raw signal
In PaySim, fraud drains the origin account. We build two consistency-error
features and inspect them by class:

- `errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig`
- `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`""")
code("""df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["origin_emptied"] = ((df["newbalanceOrig"] == 0) &
                        (np.isclose(df["oldbalanceOrg"], df["amount"]))).astype(int)

print("mean error-balance by class:")
display(df.groupby("isFraud")[["errorBalanceOrig", "errorBalanceDest"]].mean())
print("\\nP(origin fully emptied) by class:")
display(df.groupby("isFraud")["origin_emptied"].mean())""")

md("""## 6. Time-of-day pattern
`hour_of_day` is derived from `step` (`step % 24`). We compare fraud **rate** by
hour rather than counts, since volume varies through the day.""")
code("""hourly = df.groupby("hour_of_day")["isFraud"].mean().mul(100)
plt.figure(figsize=(11, 4))
hourly.plot(marker="o", color="crimson")
plt.title("Fraud rate by hour of day (%)"); plt.xlabel("hour"); plt.ylabel("fraud rate %")
plt.xticks(range(0, 24)); plt.tight_layout()
plt.savefig(FIGURES/"fraud_by_hour.png", dpi=150, bbox_inches="tight"); plt.show()""")

md("""## 7. Synthetic contextual signals vs fraud
Each engineered contextual signal should show a clear but non-perfect lift for
fraud — real signal, no leakage.""")
code("""fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# binary flags -> fraud rate
for a, col in zip(ax[0], ["is_new_device", "shipping_billing_mismatch"]):
    df.groupby(col)["isFraud"].mean().mul(100).plot.bar(ax=a, color="crimson")
    a.set_title(f"Fraud rate by {col} (%)"); a.set_ylabel("%")

# failed attempts -> fraud rate
df.groupby("failed_payment_attempts")["isFraud"].mean().mul(100).plot.bar(
    ax=ax[1][0], color="darkorange")
ax[1][0].set_title("Fraud rate by failed_payment_attempts (%)")

# ip distance buckets -> fraud rate
df["ip_dist_bucket"] = pd.cut(df["ip_billing_distance_km"],
                              [-1, 100, 500, 1500, 5000, 1e9],
                              labels=["<100", "100-500", "500-1.5k", "1.5k-5k", "5k+"])
df.groupby("ip_dist_bucket")["isFraud"].mean().mul(100).plot.bar(
    ax=ax[1][1], color="seagreen")
ax[1][1].set_title("Fraud rate by IP-billing distance (%)")
plt.tight_layout(); plt.savefig(FIGURES/"synthetic_signals.png", dpi=150, bbox_inches="tight"); plt.show()""")

md("## 8. Correlation with fraud (numeric features)")
code("""num = ["amount", "log_amount", "oldbalanceOrg", "newbalanceOrig",
       "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
       "account_age_days", "is_new_device", "shipping_billing_mismatch",
       "failed_payment_attempts", "ip_billing_distance_km", "is_night", "isFraud"]
corr = df[num].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False)
plt.title("Correlation matrix (numeric)")
plt.tight_layout(); plt.savefig(FIGURES/"correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.show()

print("Correlation with isFraud (sorted):")
display(corr["isFraud"].drop("isFraud").sort_values(key=np.abs, ascending=False))""")

md("""## 9. Two documentation findings
- **`isFlaggedFraud` is near-useless** — PaySim's built-in rule flags almost
  nothing, so it carries no signal and is a candidate to drop.
- **Merchants (`nameDest` starting `M`) are never defrauded** — fraud only
  targets customer destinations.""")
code("""print("isFlaggedFraud value counts:")
display(df["isFlaggedFraud"].value_counts())
print("fraud rate when isFlaggedFraud==1 vs 0:")
display(df.groupby("isFlaggedFraud")["isFraud"].mean())

df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(int)
print("\\nfraud rate by destination type (merchant vs customer):")
display(df.groupby("dest_is_merchant")["isFraud"].mean().mul(100).rename("fraud_rate_pct"))""")

md("""## 10. EDA takeaways (feed into Modules 3–5)
1. **Extreme imbalance (~0.13%)** → use AUC-PR / precision-recall + a cost metric; resample only on the training fold.
2. **Fraud only in TRANSFER & CASH_OUT** → scope modelling to these; huge search-space reduction.
3. **Balance tell** (`errorBalanceOrig/Dest`, `origin_emptied`) is the strongest raw signal → keep as features.
4. **Synthetic signals** (new device, mismatch, failed attempts, IP distance, younger accounts) all lift fraud rate without leaking → keep.
5. **Drop `isFlaggedFraud`** (no signal). Consider restricting to non-merchant destinations.
6. **Amount is right-skewed** → use `log_amount`.""")

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_eda.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
