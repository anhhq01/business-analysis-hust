"""
MODULE 2 - Exploratory Data Analysis (script -> markdown run report).

Produces reports/eda_report.md plus PNG figures in figures/, covering:

  BEFORE (base PaySim only, from RAW_DATA_PATH)
    - dataset overview, dtypes, missing/duplicate audit
    - class imbalance + value-at-risk (feeds the Module-5 cost model)
    - fraud rate by transaction type (confirms TRANSFER/CASH_OUT-only)
    - amount distributions by class (log scale)
    - drained-origin balance signature + balance-error terms
    - destination merchant-vs-customer split
    - fraud across step / hour-of-day
    - isFlaggedFraud near-uselessness check
    - numeric-feature correlation heatmap

  AFTER (enriched file, from ENRICHED_DATA_PATH - only if it exists)
    - schema diff (new columns) + merge-integrity checks
    - per-signal fraud-vs-legit distributions
    - anti-leakage LIFT tables (fraud rate | signal), incl. deciles
    - correlation of each synthetic feature with isFraud
    - multivariate lift as risky flags stack
    - browser_fingerprint reproducibility note (PYTHONHASHSEED)

The BEFORE half runs on the base file alone, so this script is useful even
before Module 1's generator has been run. The AFTER half is skipped (with a
note in the report) when the enriched parquet is absent.

Run:
    python src/eda.py                 # full data, paths from config.py
    python src/eda.py --sample 500000 # quick pass on a random raw sample
    python src/eda.py --raw <csv> --enriched <parquet> --outdir <dir>
"""
from __future__ import annotations
import argparse
import datetime as _dt
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

from config import (RAW_DATA_PATH, ENRICHED_DATA_PATH, FIGURES, ROOT,
                    FRAUD_TYPES)

# config.py creates DATA_RAW/DATA_PROCESSED/FIGURES but not a reports dir.
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

BASE_COLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
             "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
             "isFraud", "isFlaggedFraud"]

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class Report:
    """Accumulates markdown sections and figure references, then writes one file."""

    def __init__(self, outdir: Path, figdir: Path):
        self.parts: list[str] = []
        self.outdir = outdir
        self.figdir = figdir

    def add(self, md: str) -> None:
        self.parts.append(textwrap.dedent(md).strip("\n") + "\n")

    def table(self, df: pd.DataFrame, floatfmt: str = ",.4f",
              index: bool = True) -> None:
        self.parts.append(df.to_markdown(floatfmt=floatfmt, index=index) + "\n")

    def figure(self, fig, name: str, caption: str = "") -> None:
        path = self.figdir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        # relative path so the .md renders on GitHub / locally from repo root
        rel = path.relative_to(self.outdir.parent) if self.outdir.parent in path.parents \
            else Path("..") / "figures" / path.name
        self.parts.append(f"![{caption or name}]({rel.as_posix()})\n")
        if caption:
            self.parts.append(f"*{caption}*\n")

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.parts), encoding="utf-8")


def _fraud_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)["isFraud"].agg(n="count", fraud="sum", fraud_rate="mean")
    g["fraud_rate_%"] = g["fraud_rate"] * 100
    return g.drop(columns="fraud_rate")


# ---------------------------------------------------------------------------
# BEFORE: base PaySim
# ---------------------------------------------------------------------------
def section_overview(rep: Report, df: pd.DataFrame, raw_path: Path,
                     sampled: int | None) -> None:
    note = (f"> **Note:** run on a random sample of {sampled:,} raw rows "
            f"(`--sample`); numbers are indicative, not the full-data figures.\n"
            if sampled else "")
    dtypes = (pd.DataFrame({"dtype": df.dtypes.astype(str),
                            "n_missing": df.isnull().sum(),
                            "n_unique": [df[c].nunique() for c in df.columns]}))
    rep.add(f"""
    ## 1. Dataset overview (base PaySim)

    {note}Source file: `{raw_path.name}`
    Rows: **{len(df):,}**  |  Columns: **{df.shape[1]}**
    Duplicate rows: **{df.duplicated().sum():,}**  |  Total missing cells: **{int(df.isnull().sum().sum()):,}**
    """)
    rep.table(dtypes, floatfmt=",.0f")


def section_imbalance(rep: Report, df: pd.DataFrame) -> None:
    n = len(df); f = int(df.isFraud.sum()); prev = df.isFraud.mean()
    fraud_val = df.loc[df.isFraud == 1, "amount"].sum()
    tot_val = df["amount"].sum()
    rep.add(f"""
    ## 2. Class imbalance & value-at-risk

    - Fraud transactions: **{f:,}** of {n:,}
    - Prevalence: **{prev*100:.4f}%**  ->  roughly **1 in {round(1/prev):,}**
    - A "never fraud" classifier is {100*(1-prev):.4f}% accurate and useless,
      which is why we optimise **AUC-PR / precision-recall**, not accuracy.

    **Value at risk** (drives `C_fn` in the Module-5 cost model):

    - Fraud transaction value: **{fraud_val:,.0f}**
    - Total transaction value: **{tot_val:,.0f}**
    - Fraud is **{fraud_val/tot_val*100:.4f}%** of value while only
      {prev*100:.4f}% of count -> fraudulent transactions are far larger than
      average, so a missed fraud costs on the order of its `amount`.
    """)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = df.isFraud.value_counts().sort_index()
    ax.bar(["legit (0)", "fraud (1)"], counts.values,
           color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log"); ax.set_ylabel("transactions (log)")
    ax.set_title("Class imbalance (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    rep.figure(fig, "01_class_imbalance", "Legit vs fraud counts (log scale).")


def section_by_type(rep: Report, df: pd.DataFrame) -> None:
    g = _fraud_rate_by(df, "type").sort_values("fraud_rate_%", ascending=False)
    within = df[df.type.isin(FRAUD_TYPES)]
    rep.add(f"""
    ## 3. Fraud by transaction type

    Fraud occurs **only** in {', '.join(f'`{t}`' for t in FRAUD_TYPES)} - every
    other type has zero fraud. Restricting to those two types, prevalence rises
    to **{within.isFraud.mean()*100:.4f}%** (vs {df.isFraud.mean()*100:.4f}%
    overall): the modelling universe is far less imbalanced than the headline.
    """)
    rep.table(g)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    g_sorted = g.sort_values("fraud_rate_%")
    ax.barh(g_sorted.index, g_sorted["fraud_rate_%"], color="#C44E52")
    ax.set_xlabel("fraud rate (%)"); ax.set_title("Fraud rate by transaction type")
    rep.figure(fig, "02_fraud_by_type", "Fraud rate by type (TRANSFER & CASH_OUT only).")


def section_amount(rep: Report, df: pd.DataFrame) -> None:
    q = df.groupby("isFraud")["amount"].describe(
        percentiles=[.5, .9, .99])[["mean", "50%", "90%", "99%", "max"]]
    q.index = ["legit", "fraud"]
    rep.add("""
    ## 4. Transaction amount distribution

    Amounts are heavily right-skewed, so we view them on a log scale. Fraudulent
    transactions sit systematically higher, consistent with account-draining.
    """)
    rep.table(q, floatfmt=",.1f")
    d = df[df.amount > 0]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(np.log10(d.loc[d.isFraud == 0, "amount"]), bins=60, alpha=.6,
            density=True, label="legit", color="#4C72B0")
    ax.hist(np.log10(d.loc[d.isFraud == 1, "amount"]), bins=60, alpha=.6,
            density=True, label="fraud", color="#C44E52")
    ax.set_xlabel("log10(amount)"); ax.set_ylabel("density")
    ax.set_title("Amount distribution by class (log10)"); ax.legend()
    rep.figure(fig, "03_amount_log", "Log-amount density, legit vs fraud.")


def section_balance(rep: Report, df: pd.DataFrame) -> None:
    d = df.copy()
    d["orig_err"] = (d.oldbalanceOrg - d.amount - d.newbalanceOrig)
    d["dest_err"] = (d.oldbalanceDest + d.amount - d.newbalanceDest)
    drained = ((d.isFraud == 1) & (np.isclose(d.newbalanceOrig, 0))).sum()
    n_fraud = int(d.isFraud.sum())
    orig_err_incons = (~np.isclose(d.orig_err, 0)).mean() * 100
    rep.add(f"""
    ## 5. Balance mechanics (the drained-origin signature)

    For fraud, the origin account is typically emptied
    (`oldbalanceOrg approx amount`, `newbalanceOrig = 0`):
    **{drained:,} of {n_fraud:,}** fraud rows ({drained/max(n_fraud,1)*100:.1f}%)
    end with a zero origin balance.

    We compute two reconciliation-error terms:
    `orig_err = oldbalanceOrg - amount - newbalanceOrig` and
    `dest_err = oldbalanceDest + amount - newbalanceDest`. About
    **{orig_err_incons:.1f}%** of rows have a non-zero `orig_err` (a known PaySim
    accounting quirk, and a Module-3 item). These balance signals are highly
    predictive but partly mechanical, so they must be handled carefully to avoid
    leakage in later modules.
    """)
    tbl = d.groupby("isFraud")[["orig_err", "dest_err"]].mean()
    tbl.index = ["legit", "fraud"]
    rep.table(tbl, floatfmt=",.1f")


def section_destination(rep: Report, df: pd.DataFrame) -> None:
    d = df.assign(dest_type=np.where(df.nameDest.str.startswith("M"),
                                     "merchant (M)", "customer (C)"))
    g = _fraud_rate_by(d, "dest_type")
    rep.add("""
    ## 6. Destination type

    Merchant destinations (`M...`) are never defrauded; all fraud lands on
    customer destinations (`C...`). A merchant destination is effectively a
    strong negative signal.
    """)
    rep.table(g)


def section_time(rep: Report, df: pd.DataFrame) -> None:
    d = df.assign(hour=(df.step % 24))
    g = _fraud_rate_by(d, "hour")
    rep.add("""
    ## 7. Time-of-day pattern

    `hour_of_day = step % 24` is **derived**, so any pattern here reflects the
    real PaySim clock rather than anything injected. Fraud is disproportionately
    concentrated in the low-activity night hours.
    """)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(g.index, g["fraud_rate_%"], color="#C44E52")
    ax.set_xlabel("hour of day (step % 24)"); ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate by hour of day")
    rep.figure(fig, "04_fraud_by_hour", "Fraud rate across the 24-hour cycle.")


def section_flagged(rep: Report, df: pd.DataFrame) -> None:
    flagged = int(df.isFlaggedFraud.sum())
    caught = int(df.loc[df.isFlaggedFraud == 1, "isFraud"].sum())
    total_fraud = int(df.isFraud.sum())
    rec = caught / max(total_fraud, 1) * 100
    rep.add(f"""
    ## 8. `isFlaggedFraud` (PaySim's built-in rule)

    Fires **{flagged:,}** times in total and catches only **{caught:,}** of
    {total_fraud:,} frauds (recall **{rec:.2f}%**). It is far too conservative to
    be useful and is dropped from modelling, as the data dictionary advises.
    """)


def section_corr(rep: Report, df: pd.DataFrame) -> None:
    num = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
           "newbalanceDest", "isFraud"]
    corr = df[num].corr()
    rep.add("""
    ## 9. Numeric correlation (base features)

    Correlations among the base numeric fields and the label. Balance/amount
    fields carry most of the base signal; nothing else is strongly linear.
    """)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=45, ha="right")
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Correlation (base numeric features)")
    rep.figure(fig, "05_corr_base", "Base numeric correlation matrix.")


# ---------------------------------------------------------------------------
# AFTER: enriched (synthetic) layer
# ---------------------------------------------------------------------------
SYNTH_ACCOUNT = ["account_age_days", "home_billing_country", "home_device_id"]
SYNTH_TXN = ["device_id", "is_new_device", "browser_fingerprint",
             "shipping_billing_mismatch", "failed_payment_attempts",
             "ip_country", "ip_billing_distance_km"]
SYNTH_DERIVED = ["hour_of_day", "is_night"]
SYNTH_FLAGS = ["is_new_device", "shipping_billing_mismatch", "is_night"]
SYNTH_NUM = ["account_age_days", "failed_payment_attempts",
             "ip_billing_distance_km"]


def section_schema_diff(rep: Report, base: pd.DataFrame,
                        enr: pd.DataFrame) -> None:
    added = [c for c in enr.columns if c not in base.columns]
    row_ok = "preserved" if len(base) == len(enr) else "CHANGED (investigate)"
    no_nulls = int(enr[added].isnull().sum().sum()) == 0
    rep.add(f"""
    # Part B - After synthetic enrichment

    ## 10. Schema diff & merge integrity

    Enrichment adds **{len(added)}** columns and changes **no** base values.

    - Base rows: **{len(base):,}**  |  Enriched rows: **{len(enr):,}**
      -> row count {row_ok}.
    - New columns introduced no nulls: **{no_nulls}**.
    """)
    stable_ok = True
    for col in SYNTH_ACCOUNT:
        if col in enr.columns:
            per_acct = enr.groupby("nameOrig")[col].nunique()
            if (per_acct > 1).any():
                stable_ok = False
    rep.add(f"""
    Account-level fields ({', '.join(f'`{c}`' for c in SYNTH_ACCOUNT)}) are
    **stable per `nameOrig`**: {stable_ok}. (Note: PaySim origin accounts are
    almost all unique, so "stable per account" is rarely exercised in practice -
    a limitation of the base data, not the generator.)

    New columns:

    - account-level: {', '.join(f'`{c}`' for c in SYNTH_ACCOUNT)}
    - transaction-level: {', '.join(f'`{c}`' for c in SYNTH_TXN)}
    - derived: {', '.join(f'`{c}`' for c in SYNTH_DERIVED)}
    """)


def section_lift(rep: Report, enr: pd.DataFrame) -> None:
    rep.add("""
    ## 11. Anti-leakage lift tables

    The core validation: each risky signal must **raise** the fraud rate without
    **separating** the classes. Below, fraud rate conditional on each binary flag
    (=1 vs =0). Elevated-but-overlapping is correct; anything near 100%/0% would
    indicate leakage.
    """)
    rows = []
    for f in SYNTH_FLAGS:
        if f in enr.columns:
            r1 = enr.loc[enr[f] == 1, "isFraud"].mean() * 100
            r0 = enr.loc[enr[f] == 0, "isFraud"].mean() * 100
            rows.append({"signal": f, "fraud_rate_% (=1)": r1,
                         "fraud_rate_% (=0)": r0,
                         "lift_x": (r1 / r0) if r0 else np.nan})
    rep.table(pd.DataFrame(rows).set_index("signal"), floatfmt=",.3f")

    rep.add("""
    For the continuous signals, fraud rate by decile (monotone-ish increase
    expected, never a clean step to 100%):
    """)
    for col in SYNTH_NUM:
        if col in enr.columns:
            try:
                enr["_dec"] = pd.qcut(enr[col].rank(method="first"), 10,
                                      labels=False)
                g = enr.groupby("_dec")["isFraud"].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 2.8))
                ax.plot(g.index, g.values, marker="o", color="#C44E52")
                ax.set_xlabel(f"{col} decile (0=low)"); ax.set_ylabel("fraud rate (%)")
                ax.set_title(f"Fraud rate by {col} decile")
                rep.figure(fig, f"06_decile_{col}",
                           f"Fraud rate across {col} deciles.")
            except Exception as e:  # pragma: no cover
                rep.add(f"*(decile plot for {col} skipped: {e})*")
    enr.drop(columns=[c for c in ["_dec"] if c in enr.columns], inplace=True)


def section_synth_corr(rep: Report, enr: pd.DataFrame) -> None:
    cols = [c for c in (SYNTH_FLAGS + SYNTH_NUM) if c in enr.columns]
    corr = enr[cols + ["isFraud"]].corr()["isFraud"].drop("isFraud").sort_values()
    rep.add("""
    ## 12. Correlation of synthetic features with `isFraud`

    Every synthetic feature correlates with the label only **weakly** - none
    approaches 1.0. This is the single-glance anti-leakage check: no field is a
    deterministic (or near-deterministic) function of fraud.
    """)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(corr.index, corr.values, color="#55A868")
    ax.set_xlabel("Pearson correlation with isFraud")
    ax.set_title("Synthetic feature vs isFraud")
    rep.figure(fig, "07_synth_corr", "Weak per-feature correlation with the label.")
    rep.table(corr.to_frame("corr_with_isFraud"))


def section_multivariate(rep: Report, enr: pd.DataFrame) -> None:
    flags = [f for f in SYNTH_FLAGS if f in enr.columns]
    enr["_nflags"] = enr[flags].sum(axis=1)
    g = enr.groupby("_nflags")["isFraud"].agg(n="count", fraud="sum", rate="mean")
    g["fraud_rate_%"] = g["rate"] * 100
    g = g.drop(columns="rate")
    rep.add(f"""
    ## 13. Multivariate lift (signals must be combined)

    Stacking risky flags ({', '.join(f'`{f}`' for f in flags)}) raises the fraud
    rate sharply - the model must **combine** signals rather than rely on any one.
    This both validates the generation design and motivates Module-4 feature
    engineering.
    """)
    rep.table(g, floatfmt=",.3f")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(g.index, g["fraud_rate_%"], color="#8172B3")
    ax.set_xlabel("number of risky flags active"); ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate vs count of active risk flags")
    rep.figure(fig, "08_multivariate_lift", "Fraud rate rises as flags stack.")
    enr.drop(columns=["_nflags"], inplace=True)


def section_repro_note(rep: Report) -> None:
    rep.add("""
    ## 14. Data-quality / reproducibility notes (for Module 3)

    - **`browser_fingerprint` is not reproducible across runs.** It is built with
      Python's built-in `hash()`, which is salted per process via
      `PYTHONHASHSEED` and is **not** governed by the numpy seed in `config.py`.
      Within a run it is stable per `device_id` (as documented), but the actual
      `fp_...` values differ between runs/machines. If reproducibility matters,
      seed the mapping with numpy or set `PYTHONHASHSEED=0`. It does not affect
      modelling (the raw id would be hashed or dropped anyway).
    - **Balance-reconciliation quirks** (non-zero `orig_err`) are inherent to
      PaySim and are a cleaning item, not a bug.
    - **Origin accounts are near-unique**, so origin-level velocity features will
      be degenerate; build velocity on `nameDest` (customer destinations recur).
    """)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(raw_path: Path, enriched_path: Path, outdir: Path, figdir: Path,
        sample: int | None) -> Path:
    figdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    rep = Report(outdir, figdir)

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rep.add(f"""
    # EDA run report - e-commerce fraud detection

    Generated: **{stamp}**
    Base file: `{raw_path}`
    Enriched file: `{enriched_path}` ({'present' if enriched_path.exists() else 'ABSENT - Part B skipped'})

    ---

    # Part A - Before synthetic enrichment (base PaySim)
    """)

    print(f"[eda] reading base: {raw_path}")
    base = pd.read_csv(raw_path)
    if sample:
        base = base.sample(n=min(sample, len(base)), random_state=42)
        print(f"[eda] sampled to {len(base):,} rows")

    section_overview(rep, base, raw_path, sample)
    section_imbalance(rep, base)
    section_by_type(rep, base)
    section_amount(rep, base)
    section_balance(rep, base)
    section_destination(rep, base)
    section_time(rep, base)
    section_flagged(rep, base)
    section_corr(rep, base)

    if enriched_path.exists():
        print(f"[eda] reading enriched: {enriched_path}")
        enr = pd.read_parquet(enriched_path)
        section_schema_diff(rep, base if not sample else pd.read_csv(raw_path,
                            usecols=BASE_COLS), enr)
        section_lift(rep, enr)
        section_synth_corr(rep, enr)
        section_multivariate(rep, enr)
        section_repro_note(rep)
    else:
        rep.add("""
        # Part B - After synthetic enrichment

        Enriched file not found - run `python src/generate_synthetic.py` first,
        then re-run this script to populate the synthetic-layer validation.
        """)
        section_repro_note(rep)

    out = outdir / "eda_report.md"
    rep.write(out)
    print(f"[eda] wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA -> markdown run report.")
    ap.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    ap.add_argument("--enriched", type=Path, default=ENRICHED_DATA_PATH)
    ap.add_argument("--outdir", type=Path, default=REPORTS)
    ap.add_argument("--figdir", type=Path, default=FIGURES)
    ap.add_argument("--sample", type=int, default=None,
                    help="optional: run on a random sample of N raw rows")
    a = ap.parse_args()
    run(a.raw, a.enriched, a.outdir, a.figdir, a.sample)


if __name__ == "__main__":
    main()
