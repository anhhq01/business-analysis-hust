"""
Monitoring utilities for realtime/batch fraud scoring.

This file is intentionally separate from training. It compares an incoming
transaction window against a saved reference distribution and writes a drift
report that can be used in a Streamlit dashboard, scheduled job or API monitor.

Example:
    python data_balancing/monitor.py --current feature_engineering/fraud-detection/data/processed/transactions_cleaned.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from balancing_experiments import (
    ANALYSIS_SAMPLE_ROWS,
    DEFAULT_CLEANED_DATA,
    FIGURE_DIR,
    OUTPUT_DIR,
    ROOT,
    TARGET,
    add_balancing_features,
    log,
    split_features,
)

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
import matplotlib.pyplot as plt
import seaborn as sns


REFERENCE_PATH = OUTPUT_DIR / "monitoring_reference.parquet"
DRIFT_REPORT_PATH = OUTPUT_DIR / "monitoring_drift_report.csv"
DRIFT_SUMMARY_PATH = OUTPUT_DIR / "monitoring_summary.md"
DRIFT_FIGURE_PATH = FIGURE_DIR / "monitoring_drift_summary.png"

NUMERIC_MONITOR_COLUMNS = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "amount_to_oldOrg_ratio",
    "account_age_days",
    "failed_payment_attempts",
    "ip_billing_distance_km",
    "hour_of_day",
    "time_since_prev_orig",
    "tx_count_prev_orig",
    "amount_vs_prev_mean_orig",
]
CATEGORICAL_MONITOR_COLUMNS = [
    "type",
    "home_billing_country",
    "ip_country",
    "is_new_device",
    "shipping_billing_mismatch",
    "is_night",
    "ip_country_mismatch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor data drift for fraud features.")
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--current", type=Path, default=DEFAULT_CLEANED_DATA)
    parser.add_argument(
        "--build-reference",
        action="store_true",
        help="Create/update the monitoring reference from --current.",
    )
    parser.add_argument(
        "--source-max-rows",
        type=int,
        default=0,
        help="Optional early sample before feature engineering. Useful for quick monitoring tests.",
    )
    parser.add_argument(
        "--max-reference-rows",
        type=int,
        default=100_000,
        help="Reference sample size. Use 0 to keep all rows.",
    )
    parser.add_argument(
        "--max-current-rows",
        type=int,
        default=50_000,
        help="Incoming/current window sample size. Use 0 to keep all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psi-warning", type=float, default=0.10)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--ks-alert", type=float, default=0.10)
    return parser.parse_args()


def sample_frame(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df
    if TARGET in df.columns:
        parts = [
            group.sample(
                n=min(len(group), max(1, int(round(max_rows * len(group) / len(df))))),
                random_state=seed,
            )
            for _, group in df.groupby(TARGET, sort=False)
        ]
        return pd.concat(parts, ignore_index=True)
    return df.sample(n=max_rows, random_state=seed)


def load_feature_frame(
    path: Path,
    max_rows: int,
    seed: int,
    source_max_rows: int = 0,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    df = sample_frame(df, source_max_rows, seed)
    df = add_balancing_features(df)
    if max_rows and len(df) > max_rows:
        df = sample_frame(df, max_rows, seed)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def save_reference(
    data_path: Path,
    reference_path: Path,
    max_rows: int,
    seed: int,
    source_max_rows: int,
) -> None:
    reference = load_feature_frame(data_path, max_rows, seed, source_max_rows)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(reference_path, index=False)
    log(f"Saved monitoring reference: {reference_path} ({len(reference):,} rows)")


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        quantiles = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), bins + 1)
    if np.unique(quantiles).size <= 1:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.astype(str).value_counts(normalize=True)
    cur_dist = current.astype(str).value_counts(normalize=True)
    keys = ref_dist.index.union(cur_dist.index)
    ref = ref_dist.reindex(keys, fill_value=0.0)
    cur = cur_dist.reindex(keys, fill_value=0.0)
    return float((cur - ref).abs().sum() / 2.0)


def status_from_scores(row: dict, psi_warning: float, psi_alert: float, ks_alert: float) -> str:
    if row["metric_type"] == "numeric":
        if row["psi"] >= psi_alert or row["ks_statistic"] >= ks_alert:
            return "alert"
        if row["psi"] >= psi_warning:
            return "warning"
        return "ok"
    if row["category_distribution_shift"] >= psi_alert:
        return "alert"
    if row["category_distribution_shift"] >= psi_warning:
        return "warning"
    return "ok"


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_warning: float,
    psi_alert: float,
    ks_alert: float,
) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        col for col in NUMERIC_MONITOR_COLUMNS if col in reference.columns and col in current.columns
    ]
    categorical_cols = [
        col
        for col in CATEGORICAL_MONITOR_COLUMNS
        if col in reference.columns and col in current.columns
    ]

    for col in numeric_cols:
        ref = pd.to_numeric(reference[col], errors="coerce")
        cur = pd.to_numeric(current[col], errors="coerce")
        ks = ks_2samp(ref, cur)
        row = {
            "feature": col,
            "metric_type": "numeric",
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "reference_std": float(ref.std()),
            "current_std": float(cur.std()),
            "psi": population_stability_index(ref, cur),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "category_distribution_shift": np.nan,
        }
        row["status"] = status_from_scores(row, psi_warning, psi_alert, ks_alert)
        rows.append(row)

    for col in categorical_cols:
        row = {
            "feature": col,
            "metric_type": "categorical",
            "reference_mean": np.nan,
            "current_mean": np.nan,
            "reference_std": np.nan,
            "current_std": np.nan,
            "psi": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "category_distribution_shift": categorical_drift(reference[col], current[col]),
        }
        row["status"] = status_from_scores(row, psi_warning, psi_alert, ks_alert)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["status", "feature"], ascending=[True, True])


def plot_drift_summary(report: pd.DataFrame, path: Path) -> None:
    if report.empty:
        return
    plot_df = report.copy()
    plot_df["score"] = plot_df["psi"].fillna(plot_df["category_distribution_shift"])
    plot_df = plot_df.sort_values("score", ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=plot_df, x="score", y="feature", hue="status", dodge=False)
    ax.set_title("Top Drift Signals")
    ax.set_xlabel("PSI or categorical distribution shift")
    ax.set_ylabel("")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def write_summary(report: pd.DataFrame, path: Path) -> None:
    counts = report["status"].value_counts().to_dict() if not report.empty else {}
    top = report.copy()
    top["score"] = top["psi"].fillna(top["category_distribution_shift"])
    top = top.sort_values("score", ascending=False).head(10)
    lines = [
        "# Monitoring Drift Summary",
        "",
        "## Status Counts",
        "",
        f"- alert: {counts.get('alert', 0)}",
        f"- warning: {counts.get('warning', 0)}",
        f"- ok: {counts.get('ok', 0)}",
        "",
        "## Recommended Realtime Monitoring",
        "",
        "- Log every scored transaction with timestamp, model version, score, threshold and final action.",
        "- Track rolling fraud-score distribution by hour/day and alert when PSI >= 0.25.",
        "- Track high-signal features: amount ratios, balance errors, failed attempts, new device, IP distance, country mismatch and transaction type.",
        "- Monitor prediction volume: review queue size, auto-block count, approval count and score percentiles.",
        "- When labels arrive later, monitor delayed precision/recall and fraud value captured.",
        "- Add graph features later: account-device-IP-country bipartite links, shared device count, shared destination count and connected component risk.",
        "",
        "## Top Drift Features",
        "",
    ]
    if top.empty:
        lines.append("No monitored features found.")
    else:
        lines.extend(top.to_string(index=False).splitlines())
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.build_reference or not args.reference.exists():
        save_reference(
            args.current,
            args.reference,
            args.max_reference_rows,
            args.seed,
            args.source_max_rows,
        )
        if args.build_reference:
            return

    log(f"Reading reference: {args.reference}")
    reference = pd.read_parquet(args.reference)
    log(f"Reading current window: {args.current}")
    current = load_feature_frame(
        args.current,
        args.max_current_rows,
        args.seed,
        args.source_max_rows,
    )

    report = compute_drift_report(
        reference,
        current,
        args.psi_warning,
        args.psi_alert,
        args.ks_alert,
    )
    report.to_csv(DRIFT_REPORT_PATH, index=False)
    plot_drift_summary(report, DRIFT_FIGURE_PATH)
    write_summary(report, DRIFT_SUMMARY_PATH)

    log(f"Wrote drift report: {DRIFT_REPORT_PATH}")
    log(f"Wrote drift summary: {DRIFT_SUMMARY_PATH}")
    log(f"Wrote drift figure: {DRIFT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
