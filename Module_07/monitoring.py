"""
Module 7 - model monitoring scaffold.

This script rebuilds the deployed behavioural feature set, scores an incoming
slice with the saved RandomForest model, tracks drift and rolling performance,
and exports a simple Evidently dashboard.

Run from the repo root:
    python Module_07/monitoring.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import COUNTRIES, ENRICHED_DATA_PATH, FRAUD_TYPES, ROOT


MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "Module_07" / "outputs"
EVENTS_LOG = ROOT / "Module_07" / "inputs" / "module6_scored_events.jsonl"
REFERENCE_MAX_STEP = 354
DEFAULT_ROLLING_WINDOW = 168
EPS = 1.0


@dataclass(frozen=True)
class TriggerConfig:
    drift_feature_fraction: float = 0.30
    precision_multiplier: float = 0.70
    recall_drop: float = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 7 monitoring scaffold")
    parser.add_argument("--enriched", type=Path, default=ENRICHED_DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.joblib")
    parser.add_argument("--policy", type=Path, default=MODELS_DIR / "decision_threshold.json")
    parser.add_argument("--features", type=Path, default=MODELS_DIR / "feature_list.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-max-step", type=int, default=REFERENCE_MAX_STEP)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--dashboard-max-rows", type=int, default=100000)
    parser.add_argument("--events-log", type=Path, default=EVENTS_LOG)
    return parser.parse_args()


def load_artifacts(model_path: Path, policy_path: Path, features_path: Path):
    model = joblib.load(model_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    meta = json.loads(features_path.read_text(encoding="utf-8"))
    deployed_features = policy.get("features") or meta["features"]
    return model, policy, meta, deployed_features


def required_raw_columns() -> list[str]:
    return [
        "step",
        "type",
        "amount",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "account_age_days",
        "is_new_device",
        "shipping_billing_mismatch",
        "failed_payment_attempts",
        "ip_billing_distance_km",
        "is_night",
        "ip_country",
        "home_billing_country",
        "hour_of_day",
    ]


def build_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["type"].isin(FRAUD_TYPES)].sort_values("step", kind="stable").copy()

    df["log_amount"] = np.log10(df["amount"] + EPS)
    df["dest_balance_err"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["drained_origin"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    df["amount_to_oldbalanceOrg"] = df["amount"] / (df["oldbalanceOrg"] + EPS)
    df["dest_balances_masked"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)
    ).astype(int)

    df["ip_country_mismatch"] = (df["ip_country"] != df["home_billing_country"]).astype(int)
    df["log_ip_distance"] = np.log10(df["ip_billing_distance_km"] + 1.0)
    df["young_account"] = (df["account_age_days"] <= 90).astype(int)
    df["high_failed_attempts"] = (df["failed_payment_attempts"] >= 3).astype(int)

    dest_group = df.groupby("nameDest", sort=False)
    df["dest_prior_count"] = dest_group.cumcount()
    prev_step = dest_group["step"].shift(1)
    df["dest_steps_since_last"] = (df["step"] - prev_step).fillna(-1)
    df["dest_first_seen"] = prev_step.isna().astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)

    for prefix, column in (("home", "home_billing_country"), ("ip", "ip_country")):
        dummies = pd.get_dummies(df[column], prefix=prefix).astype(int)
        df = pd.concat([df, dummies], axis=1)
        for country in COUNTRIES:
            dummy_name = f"{prefix}_{country}"
            if dummy_name not in df.columns:
                df[dummy_name] = 0

    return df


def score_monitoring_frame(
    enriched_path: Path,
    model,
    deployed_features: list[str],
    threshold: float,
) -> pd.DataFrame:
    raw = pd.read_parquet(enriched_path, columns=required_raw_columns())
    feature_df = build_behavioural_features(raw)

    missing = [feature for feature in deployed_features if feature not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing deployed features: {missing}")

    X = feature_df[deployed_features].copy()
    scores = model.predict_proba(X)[:, 1]
    pred = (scores >= threshold).astype(int)

    monitor = X.copy()
    monitor["target"] = feature_df["isFraud"].astype(int).to_numpy()
    monitor["prediction"] = pred
    monitor["prediction_proba"] = scores
    monitor["amount_raw"] = feature_df["amount"].to_numpy()
    return monitor


def split_reference_current(df: pd.DataFrame, reference_max_step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = df[df["step"] <= reference_max_step].copy()
    current = df[df["step"] > reference_max_step].copy()
    if reference.empty or current.empty:
        raise ValueError(
            "Reference/current split is empty. Check the input data and --reference-max-step."
        )
    return reference, current


def build_monitoring_frame_from_module6_events(
    events_log_path: Path,
    model,
    deployed_features: list[str],
    threshold: float,
) -> pd.DataFrame:
    if not events_log_path.exists():
        return pd.DataFrame()

    rows = []
    with events_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    raw = raw[raw["type"].isin(FRAUD_TYPES)].copy()
    if raw.empty:
        return pd.DataFrame()

    raw["step"] = pd.to_numeric(raw.get("step"), errors="coerce").fillna(744).astype(int)
    raw["hour_of_day"] = pd.to_numeric(raw.get("hour_of_day"), errors="coerce")
    raw["hour_of_day"] = raw["hour_of_day"].fillna(raw["step"] % 24).astype(int)
    raw["is_night"] = ((raw["hour_of_day"] >= 0) & (raw["hour_of_day"] <= 5)).astype(int)

    defaults = {
        "nameDest": "UNKNOWN_DEST",
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    for col, default in defaults.items():
        if col not in raw.columns:
            raw[col] = default
        raw[col] = raw[col].fillna(default)

    numeric_cols = [
        "amount",
        "account_age_days",
        "failed_payment_attempts",
        "ip_billing_distance_km",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    bool_cols = ["is_new_device", "shipping_billing_mismatch"]
    for col in bool_cols:
        raw[col] = raw[col].fillna(0).astype(int)

    if "actual_label" in raw.columns:
        raw["isFraud"] = pd.to_numeric(raw["actual_label"], errors="coerce")
    else:
        raw["isFraud"] = np.nan

    feature_df = build_behavioural_features(raw)
    missing = [feature for feature in deployed_features if feature not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing deployed features from module6 events: {missing}")

    X = feature_df[deployed_features].copy()
    monitor = X.copy()

    raw_selected = raw.loc[feature_df.index].copy()
    if "fraud_probability" in raw_selected.columns:
        scores = pd.to_numeric(raw_selected["fraud_probability"], errors="coerce").to_numpy()
    else:
        scores = np.full(len(raw_selected), np.nan)
    missing_score = np.isnan(scores)
    if missing_score.any():
        scores[missing_score] = model.predict_proba(X.iloc[missing_score])[:, 1]

    if "decision" in raw_selected.columns:
        pred = raw_selected["decision"].isin(["REVIEW", "BLOCK"]).astype(int).to_numpy()
    else:
        pred = (scores >= threshold).astype(int)

    monitor["target"] = pd.to_numeric(raw_selected["isFraud"], errors="coerce")
    monitor["prediction"] = pred
    monitor["prediction_proba"] = scores
    monitor["amount_raw"] = pd.to_numeric(raw_selected["amount"], errors="coerce").to_numpy()
    return monitor


def is_binary_feature(series_a: pd.Series, series_b: pd.Series) -> bool:
    values = pd.Index(pd.concat([series_a, series_b], ignore_index=True).dropna().unique())
    return len(values) <= 2 and values.isin([0, 1]).all()


def compute_drift_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    from scipy.stats import chi2_contingency, ks_2samp

    rows: list[dict[str, float | str | bool]] = []
    for feature in features:
        ref = reference[feature].dropna()
        cur = current[feature].dropna()
        if ref.empty or cur.empty:
            continue

        if is_binary_feature(ref, cur):
            ref_counts = ref.astype(int).value_counts().reindex([0, 1], fill_value=0)
            cur_counts = cur.astype(int).value_counts().reindex([0, 1], fill_value=0)
            table = np.array([ref_counts.to_numpy(), cur_counts.to_numpy()])
            p_value = float(chi2_contingency(table)[1]) if table.sum() else 1.0
            score = float(abs(cur.mean() - ref.mean()))
            method = "chi2_binary"
        else:
            score, p_value = ks_2samp(ref, cur)
            score = float(score)
            p_value = float(p_value)
            method = "ks_numeric"

        rows.append(
            {
                "feature": feature,
                "method": method,
                "score": score,
                "p_value": p_value,
                "reference_mean": float(ref.mean()),
                "current_mean": float(cur.mean()),
                "drifted": bool(p_value < alpha),
            }
        )

    return pd.DataFrame(rows).sort_values(["drifted", "score"], ascending=[False, False])


def rolling_classification_metrics(current: pd.DataFrame, window_steps: int) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

    rows: list[dict[str, float | int]] = []
    scored = current[current["target"].notna()].copy()
    if scored.empty:
        return pd.DataFrame()
    scored["target"] = scored["target"].astype(int)
    step_values = np.sort(scored["step"].unique())

    for end_step in step_values:
        start_step = end_step - window_steps + 1
        window = scored[(scored["step"] >= start_step) & (scored["step"] <= end_step)]
        y_true = window["target"]
        y_pred = window["prediction"]
        y_score = window["prediction_proba"]
        if window.empty:
            continue

        ap = float("nan")
        if y_true.nunique() > 1:
            ap = float(average_precision_score(y_true, y_score))

        rows.append(
            {
                "window_start_step": int(start_step),
                "window_end_step": int(end_step),
                "n_transactions": int(len(window)),
                "n_fraud": int(y_true.sum()),
                "n_flagged": int(y_pred.sum()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "average_precision": ap,
                "fraud_rate": float(y_true.mean()),
            }
        )

    return pd.DataFrame(rows)


def build_evidently_dashboard(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    output_path: Path,
    max_rows: int,
) -> None:
    # Import Evidently lazily so the rest of monitoring can run without it.
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    ref_sample = reference.sample(min(len(reference), max_rows), random_state=42).copy()
    cur_sample = current.sample(min(len(current), max_rows), random_state=42).copy()
    data_columns = features + ["target", "prediction", "prediction_proba"]

    # Normalize types so Evidently/Sklearn never sees mixed binary/continuous labels.
    for frame in (ref_sample, cur_sample):
        frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
        frame["prediction_proba"] = pd.to_numeric(frame["prediction_proba"], errors="coerce")

    definition = DataDefinition(
        numerical_columns=features + ["prediction_proba"],
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas="prediction_proba",
                pos_label=1,
            )
        ],
    )

    reference_dataset = Dataset.from_pandas(ref_sample[data_columns], data_definition=definition)
    current_dataset = Dataset.from_pandas(cur_sample[data_columns], data_definition=definition)

    ref_labeled = ref_sample[
        ref_sample["target"].isin([0, 1])
        & ref_sample["prediction"].isin([0, 1])
        & ref_sample["prediction_proba"].notna()
    ]
    cur_labeled = cur_sample[
        cur_sample["target"].isin([0, 1])
        & cur_sample["prediction"].isin([0, 1])
        & cur_sample["prediction_proba"].notna()
    ]

    has_ref_classes = ref_labeled["target"].nunique() >= 2
    has_cur_classes = cur_labeled["target"].nunique() >= 2
    has_enough_labeled_rows = len(ref_labeled) >= 2 and len(cur_labeled) >= 2

    metrics = [DataDriftPreset()]
    metadata = {"module": "Module 7", "variant": "behavioural"}
    if has_ref_classes and has_cur_classes and has_enough_labeled_rows:
        metrics.append(ClassificationPreset())
        metadata["classification_report"] = "enabled"
    else:
        metadata["classification_report"] = (
            "skipped: insufficient valid labeled rows/classes in current/reference data"
        )

    report = Report(metrics=metrics, metadata=metadata)
    try:
        # Evidently may trigger benign numpy warnings on constant/near-constant
        # slices; suppress only this known noise pattern to keep logs readable.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
                module=r"numpy\\.lib\\._function_base_impl",
            )
            snapshot = report.run(current_data=current_dataset, reference_data=reference_dataset)
    except ValueError as e:
        # Some tiny/edge batches can still fail legacy classification metrics.
        # Retry with drift-only so monitoring never crashes in production.
        fallback = Report(
            metrics=[DataDriftPreset()],
            metadata={
                **metadata,
                "classification_report": f"skipped due to Evidently error: {e}",
            },
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
                module=r"numpy\\.lib\\._function_base_impl",
            )
            snapshot = fallback.run(current_data=current_dataset, reference_data=reference_dataset)
    snapshot.save_html(str(output_path))


def evaluate_triggers(
    drift_summary: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    baseline_metrics: dict,
    config: TriggerConfig,
) -> dict:
    latest = rolling_metrics.iloc[-1].to_dict() if not rolling_metrics.empty else {}
    drift_fraction = float(drift_summary["drifted"].mean()) if not drift_summary.empty else 0.0
    baseline_precision = float(baseline_metrics["precision"])
    baseline_recall = float(baseline_metrics["recall"])

    has_perf = not rolling_metrics.empty
    latest_precision = float(latest.get("precision", float("nan")))
    latest_recall = float(latest.get("recall", float("nan")))

    triggers = {
        "drift_fraction": {
            "value": drift_fraction,
            "threshold": config.drift_feature_fraction,
            "breached": drift_fraction > config.drift_feature_fraction,
        },
        "precision_drop": {
            "value": latest_precision,
            "threshold": baseline_precision * config.precision_multiplier,
            "breached": (latest_precision < baseline_precision * config.precision_multiplier)
            if has_perf
            else False,
        },
        "recall_drop": {
            "value": latest_recall,
            "threshold": baseline_recall - config.recall_drop,
            "breached": (latest_recall < baseline_recall - config.recall_drop)
            if has_perf
            else False,
        },
        "scheduled_retrain": {
            "policy": "review monthly even without drift, because fraud patterns evolve",
            "breached": False,
        },
    }

    return {
        "baseline_metrics": baseline_metrics,
        "latest_window_metrics": latest,
        "has_labelled_performance": has_perf,
        "triggers": triggers,
        "should_retrain": any(item.get("breached", False) for item in triggers.values()),
    }


def write_summary_markdown(
    output_path: Path,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_summary: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    trigger_summary: dict,
) -> None:
    latest = rolling_metrics.tail(1)
    top_drift = drift_summary.head(10)
    lines = [
        "# Module 7 Monitoring Summary",
        "",
        "## Dataset split",
        "",
        f"- Reference rows: {len(reference):,}",
        f"- Current rows: {len(current):,}",
        f"- Drifted features: {int(drift_summary['drifted'].sum())} / {len(drift_summary)}",
        "",
        "## Latest rolling metrics",
        "",
    ]
    if not latest.empty:
        lines.append(latest.to_markdown(index=False, floatfmt=",.4f"))
    else:
        lines.append("No rolling metrics were produced.")

    lines.extend([
        "",
        "## Top drifted features",
        "",
    ])
    if not top_drift.empty:
        lines.append(top_drift.to_markdown(index=False, floatfmt=",.4f"))
    else:
        lines.append("No drift summary rows were produced.")

    lines.extend([
        "",
        "## Retraining decision",
        "",
        f"- Should retrain: {trigger_summary['should_retrain']}",
    ])
    for name, rule in trigger_summary["triggers"].items():
        if "threshold" in rule:
            lines.append(
                f"- {name}: value={rule['value']:.4f}, threshold={rule['threshold']:.4f}, breached={rule['breached']}"
            )
        else:
            lines.append(f"- {name}: {rule['policy']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, policy, _meta, deployed_features = load_artifacts(args.model, args.policy, args.features)
    threshold = float(policy["threshold"])

    monitored = score_monitoring_frame(args.enriched, model, deployed_features, threshold)
    reference, historical_current = split_reference_current(monitored, args.reference_max_step)

    current = historical_current
    current_source = "historical_enriched"
    events_current = build_monitoring_frame_from_module6_events(
        args.events_log, model, deployed_features, threshold
    )
    if not events_current.empty:
        current = events_current
        current_source = "module6_events"

    drift_summary = compute_drift_summary(reference, current, deployed_features)
    rolling_metrics = rolling_classification_metrics(current, args.rolling_window)

    dashboard_path = args.output_dir / "monitoring_dashboard.html"
    build_evidently_dashboard(
        reference=reference,
        current=current,
        features=deployed_features,
        output_path=dashboard_path,
        max_rows=args.dashboard_max_rows,
    )

    trigger_summary = evaluate_triggers(
        drift_summary=drift_summary,
        rolling_metrics=rolling_metrics,
        baseline_metrics=policy["metrics_at_threshold"],
        config=TriggerConfig(),
    )

    drift_summary.to_csv(args.output_dir / "drift_summary.csv", index=False)
    rolling_metrics.to_csv(args.output_dir / "rolling_metrics.csv", index=False)
    (args.output_dir / "trigger_summary.json").write_text(
        json.dumps(trigger_summary, indent=2, default=float),
        encoding="utf-8",
    )
    write_summary_markdown(
        args.output_dir / "monitor_summary.md",
        reference,
        current,
        drift_summary,
        rolling_metrics,
        trigger_summary,
    )

    print(f"[monitor] dashboard -> {dashboard_path}")
    print(f"[monitor] drift      -> {args.output_dir / 'drift_summary.csv'}")
    print(f"[monitor] rolling    -> {args.output_dir / 'rolling_metrics.csv'}")
    print(f"[monitor] triggers   -> {args.output_dir / 'trigger_summary.json'}")
    print(f"[monitor] current source -> {current_source}")


if __name__ == "__main__":
    main()