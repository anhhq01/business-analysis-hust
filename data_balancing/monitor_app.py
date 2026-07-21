"""
Streamlit dashboard for data balancing results and realtime fraud monitoring.

Run from the repository root:
    streamlit run data_balancing/monitor_app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

from balancing_experiments import (
    ARTIFACT_DIR,
    DEFAULT_CLEANED_DATA,
    DROP_COLUMNS,
    OUTPUT_DIR,
    TARGET,
    add_balancing_features,
)
from data_source_check import build_data_source_report
from monitor import (
    CATEGORICAL_MONITOR_COLUMNS,
    NUMERIC_MONITOR_COLUMNS,
    REFERENCE_PATH,
    categorical_drift,
    compute_drift_report,
    load_feature_frame,
    population_stability_index,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_PATH = ARTIFACT_DIR / "best_balancing_model.joblib"
CHECKPOINT_MANIFEST_PATH = ARTIFACT_DIR / "model_checkpoints.csv"
METADATA_PATH = OUTPUT_DIR / "run_metadata.json"
CV_RESULTS_PATH = OUTPUT_DIR / "cv_aucpr_by_fold.csv"
RETRAIN_LOG_PATH = OUTPUT_DIR / "retrain_from_dashboard.log"
RETRAIN_STATE_PATH = OUTPUT_DIR / "retrain_state.json"

STATUS_COLORS = {
    "ok": "#16A34A",
    "warning": "#D97706",
    "alert": "#DC2626",
    "Fraud": "#C2410C",
    "Normal": "#2563EB",
}
DRIFT_COLORS = {
    "ok": "#7CB342",
    "warning": "#F2A93B",
    "alert": "#D96055",
}
SYNTHESIS_MODES = [
    "Normal traffic",
    "Account takeover",
    "High-value cashout",
    "Bot burst",
    "Foreign IP wave",
    "Mixed attack",
]
STREAM_PRESETS = {
    "Demo dễ quan sát": {
        "mode": "Mixed attack",
        "events_per_tick": 3,
        "speed": 60,
        "target_fraud_rate": 0.10,
        "intensity": 0.20,
        "max_points": 160,
    },
    "Traffic bình thường": {
        "mode": "Normal traffic",
        "events_per_tick": 2,
        "speed": 40,
        "target_fraud_rate": 0.01,
        "intensity": 0.00,
        "max_points": 120,
    },
    "Tấn công rõ": {
        "mode": "Mixed attack",
        "events_per_tick": 5,
        "speed": 120,
        "target_fraud_rate": 0.25,
        "intensity": 0.45,
        "max_points": 220,
    },
}
STATUS_LABELS = {
    "ok": "Ổn định",
    "warning": "Cần theo dõi",
    "alert": "Lệch mạnh",
}
TIME_RANGE_OPTIONS = {
    "5 phút": pd.Timedelta(minutes=5),
    "30 phút": pd.Timedelta(minutes=30),
    "1 giờ": pd.Timedelta(hours=1),
    "1 ngày": pd.Timedelta(days=1),
    "1 tuần": pd.Timedelta(weeks=1),
    "1 tháng": pd.Timedelta(days=30),
    "1 năm": pd.Timedelta(days=365),
    "Từ lúc bắt đầu": None,
}


st.set_page_config(
    page_title="Fraud Balancing Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)


def path(name: str) -> Path:
    return OUTPUT_DIR / name


@st.cache_data(show_spinner=False)
def read_csv(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def read_json(file_path: str) -> dict:
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_text(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return ""
    return p.read_text(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def load_model(file_path: str):
    p = Path(file_path)
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_reference_rows(max_rows: int) -> pd.DataFrame:
    if REFERENCE_PATH.exists():
        return pd.read_parquet(REFERENCE_PATH).head(max_rows)
    if DEFAULT_CLEANED_DATA.exists():
        return load_feature_frame(DEFAULT_CLEANED_DATA, max_rows=max_rows, seed=42)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data_source_status() -> dict:
    return build_data_source_report()


def load_outputs() -> dict[str, pd.DataFrame | dict | str]:
    return {
        "model_results": read_csv(str(path("balancing_model_results.csv"))),
        "checkpoints": read_csv(str(CHECKPOINT_MANIFEST_PATH)),
        "review_results": read_csv(str(path("fraud_review_results.csv"))),
        "drift_report": read_csv(str(path("monitoring_drift_report.csv"))),
        "feature_scores": read_csv(str(path("feature_target_scores.csv"))),
        "permutation": read_csv(str(path("model_permutation_importance.csv"))),
        "dropped_features": read_csv(str(path("correlated_features_dropped.csv"))),
        "cv_results": read_csv(str(CV_RESULTS_PATH)),
        "metadata": read_json(str(METADATA_PATH)),
        "balancing_report": read_text(str(path("balancing_report.md"))),
        "monitoring_summary": read_text(str(path("monitoring_summary.md"))),
    }


def best_threshold(model_results: pd.DataFrame, default: float = 0.5) -> float:
    if model_results.empty or "operating_threshold" not in model_results.columns:
        return default
    best = model_results.sort_values(["auc_pr", "f1"], ascending=False).iloc[0]
    return float(best["operating_threshold"])


def available_checkpoints(data: dict[str, pd.DataFrame | dict | str]) -> pd.DataFrame:
    checkpoints = data.get("checkpoints", pd.DataFrame())
    if isinstance(checkpoints, pd.DataFrame) and not checkpoints.empty:
        out = checkpoints.copy()
    else:
        results = data.get("model_results", pd.DataFrame())
        if not isinstance(results, pd.DataFrame) or results.empty:
            return pd.DataFrame()
        out = results.copy()
        out["label"] = out["strategy"] + "__" + out["model"]
        if "checkpoint_path" not in out.columns:
            out["checkpoint_path"] = ""

    if "label" not in out.columns:
        out["label"] = out["strategy"] + "__" + out["model"]
    out["resolved_path"] = out["checkpoint_path"].apply(resolve_checkpoint_path)
    out["exists"] = out["resolved_path"].apply(lambda p: Path(p).exists() if p else False)

    if MODEL_PATH.exists():
        best_row = out.sort_values(["auc_pr", "f1"], ascending=False).head(1).copy()
        if best_row.empty:
            best_row = pd.DataFrame(
                [
                    {
                        "label": "best_balancing_model",
                        "strategy": "best",
                        "model": "best",
                        "auc_pr": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "business_cost": np.nan,
                        "operating_threshold": 0.5,
                        "checkpoint_path": str(MODEL_PATH.relative_to(ROOT)),
                        "resolved_path": str(MODEL_PATH),
                        "exists": True,
                    }
                ]
            )
        else:
            best_row["label"] = "best__" + best_row["label"].astype(str)
            best_row["checkpoint_path"] = str(MODEL_PATH.relative_to(ROOT))
            best_row["resolved_path"] = str(MODEL_PATH)
            best_row["exists"] = True
        out = pd.concat([best_row, out], ignore_index=True)

    return out.drop_duplicates(subset=["label"], keep="first")


def resolve_checkpoint_path(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    p = Path(str(value))
    if p.is_absolute():
        return str(p)
    return str(ROOT / p)


def selected_checkpoint_config(checkpoints: pd.DataFrame, selected_label: str) -> dict:
    if checkpoints.empty:
        return {
            "label": "fallback_risk_score",
            "path": "",
            "threshold": 0.5,
            "exists": False,
            "row": {},
        }
    row = checkpoints[checkpoints["label"] == selected_label]
    if row.empty:
        row = checkpoints.head(1)
    record = row.iloc[0].to_dict()
    threshold = record.get("operating_threshold", 0.5)
    if pd.isna(threshold):
        threshold = 0.5
    return {
        "label": str(record.get("label", "")),
        "path": str(record.get("resolved_path", "")),
        "threshold": float(threshold),
        "exists": bool(record.get("exists", False)),
        "row": record,
    }


def threat_level(max_drift: float, alert_features: int, warning_features: int) -> str:
    if alert_features >= 1 or max_drift >= 0.25:
        return "alert"
    if warning_features >= 1 or max_drift >= 0.10:
        return "warning"
    return "ok"


def status_label(status: str) -> str:
    return STATUS_LABELS.get(str(status), str(status))


def format_cell(value, max_len: int = 72) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = f"{value:,.4f}" if abs(value) < 10_000 else f"{value:,.0f}"
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def drift_action(row: pd.Series) -> str:
    status = str(row.get("status", "ok"))
    metric_type = str(row.get("metric_type", ""))
    if status == "alert":
        return "Ưu tiên kiểm tra feature này và xem sample transaction mới."
    if status == "warning":
        return "Theo dõi thêm vài phút hoặc tăng buffer để xác nhận."
    if metric_type == "numeric":
        return "Chưa cần xử lý, phân phối numeric đang gần reference."
    return "Chưa cần xử lý, tỉ trọng category đang gần reference."


def drift_reason(row: pd.Series) -> str:
    metric_type = str(row.get("metric_type", ""))
    if metric_type == "numeric":
        psi = row.get("psi")
        ks = row.get("ks_statistic")
        return f"PSI={psi:.3f}, KS={ks:.3f}" if pd.notna(psi) and pd.notna(ks) else "So sánh phân phối numeric."
    shift = row.get("category_distribution_shift")
    return f"Category shift={shift:.3f}" if pd.notna(shift) else "So sánh tỉ trọng category."


def prepare_drift_view(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()
    out = report.copy()
    out["drift_score"] = out["psi"].fillna(out["category_distribution_shift"])
    out["display_score"] = out["drift_score"].clip(lower=0, upper=1)
    priority = {"alert": 0, "warning": 1, "ok": 2}
    out["priority"] = out["status"].map(priority).fillna(3)
    out["Mức ưu tiên"] = out["status"].map(status_label)
    out["Trạng thái"] = out["Mức ưu tiên"]
    out["Ý nghĩa"] = out.apply(drift_reason, axis=1)
    out["Nên làm gì"] = out.apply(drift_action, axis=1)
    out["Điểm drift"] = out["drift_score"].round(3)
    out["Mức lệch hiển thị"] = out["display_score"].round(3)
    out["Feature"] = out["feature"]
    out["Loại"] = out["metric_type"].map({"numeric": "Số", "categorical": "Nhóm"}).fillna(out["metric_type"])
    return out.sort_values(["priority", "drift_score"], ascending=[True, False])


def apply_transaction_filters(
    scored: pd.DataFrame,
    time_range_label: str,
    search_text: str,
    class_filter: list[str],
    type_filter: list[str],
    truth_filter: str,
    date_range,
) -> pd.DataFrame:
    out = scored.copy()
    out["arrival_time"] = pd.to_datetime(out["arrival_time"], errors="coerce")

    window = TIME_RANGE_OPTIONS[time_range_label]
    if window is not None and out["arrival_time"].notna().any():
        end_time = out["arrival_time"].max()
        out = out[out["arrival_time"] >= end_time - window]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            dates = out["arrival_time"].dt.date
            out = out[(dates >= start_date) & (dates <= end_date)]
    elif date_range:
        dates = out["arrival_time"].dt.date
        out = out[dates == date_range]

    if search_text.strip():
        needle = search_text.strip().lower()
        search_cols = [col for col in ["tx_id", "type", "nameOrig", "nameDest", "ip_country", "home_billing_country"] if col in out]
        mask = pd.Series(False, index=out.index)
        for col in search_cols:
            mask |= out[col].astype(str).str.lower().str.contains(needle, na=False)
        out = out[mask]

    if class_filter:
        out = out[out["predicted_class"].isin(class_filter)]
    if type_filter and "type" in out.columns:
        out = out[out["type"].astype(str).isin(type_filter)]
    if truth_filter != "Tất cả" and "true_label" in out.columns:
        target_value = 1 if truth_filter == "Fraud thật" else 0
        labels = pd.to_numeric(out["true_label"], errors="coerce").fillna(-1).astype(int)
        out = out[labels == target_value]
    if "event_index" in out.columns:
        return out.sort_values("event_index", ascending=True)
    return out.sort_values("arrival_time", ascending=True)


def transaction_table_view(scored: pd.DataFrame) -> pd.DataFrame:
    if "event_index" not in scored.columns:
        scored = scored.copy()
        scored["event_index"] = np.arange(len(scored), dtype=int)
    view = scored[
        ["event_index", "tx_id", "arrival_time", "predicted_class", "fraud_score", "latency_ms", "type", "amount", "true_label"]
    ].copy()
    return view.rename(
        columns={
            "event_index": "Row",
            "tx_id": "Transaction",
            "arrival_time": "Thời điểm đến",
            "predicted_class": "Class dự đoán",
            "fraud_score": "Fraud score",
            "latency_ms": "Latency ms",
            "type": "Loại GD",
            "amount": "Amount",
            "true_label": "Fraud thật",
        }
    )


def is_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def read_retrain_state() -> dict:
    if not RETRAIN_STATE_PATH.exists() or RETRAIN_STATE_PATH.stat().st_size == 0:
        return {}
    try:
        return json.loads(RETRAIN_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_retrain_state(state: dict) -> None:
    RETRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def stop_retrain_job(pid: int | None) -> bool:
    if not is_process_running(pid):
        return False
    try:
        os.kill(int(pid), 15)
        return True
    except OSError:
        return False


def start_retrain_job(
    max_rows: int,
    split: str,
    strategies: list[str],
    models: list[str],
    run_cv: bool,
    cv_folds: int,
    fraud_review_size: int,
) -> int:
    cmd = [
        sys.executable,
        "data_balancing/balancing_experiments.py",
        "--max-rows",
        str(max_rows),
        "--split",
        split,
        "--n-jobs",
        "4",
        "--xgboost-device",
        "auto",
        "--fraud-review-size",
        str(fraud_review_size),
        "--strategies",
        *strategies,
        "--models",
        *models,
    ]
    if run_cv:
        cmd.extend(["--run-cv", "--cv-folds", str(cv_folds)])
    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = RETRAIN_LOG_PATH.open("a", encoding="utf-8")
    log_file.write("\n\n=== Dashboard retrain ===\n")
    log_file.write(" ".join(cmd) + "\n")
    log_file.flush()
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_retrain_state(
        {
            "pid": int(process.pid),
            "cmd": cmd,
            "started_at": pd.Timestamp.now().isoformat(),
            "status": "running",
            "strategies": strategies,
            "models": models,
            "max_rows": max_rows,
            "split": split,
            "run_cv": run_cv,
            "cv_folds": cv_folds,
            "fraud_review_size": fraud_review_size,
        }
    )
    return int(process.pid)


def read_log_tail(file_path: Path, max_lines: int = 80) -> str:
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def pad_transaction_page(page_df: pd.DataFrame, page_size: int = 50) -> pd.DataFrame:
    out = page_df.copy()
    missing = max(0, page_size - len(out))
    if missing:
        blank = pd.DataFrame([{col: "" for col in out.columns} for _ in range(missing)])
        out = pd.concat([out, blank], ignore_index=True)
    out = out.head(page_size)
    for col in out.columns:
        out[col] = out[col].map(lambda value: "" if pd.isna(value) else format_cell(value))
    return out


def scroll_to_detail() -> None:
    st.html(
        """
<script>
const doc = window.parent.document;
const target = doc.getElementById("transaction-detail-anchor");
if (target) {
  target.scrollIntoView({behavior: "smooth", block: "start"});
}
</script>
"""
    )


def clicked_tx_from_plotly_events(events: list[dict], fig: go.Figure) -> str | None:
    if not events:
        return None
    event = events[0]
    curve_number = event.get("curveNumber", event.get("curve_number", 0))
    point_number = event.get("pointNumber", event.get("pointIndex", event.get("point_number", None)))
    try:
        trace = fig.data[int(curve_number)]
        customdata = getattr(trace, "customdata", None)
        if customdata is not None and point_number is not None:
            value = customdata[int(point_number)]
            if isinstance(value, (list, tuple, np.ndarray)):
                return str(value[0])
            return str(value)
    except Exception:
        return None
    return None


def apply_synthesis_mode(
    batch: pd.DataFrame,
    mode: str,
    intensity: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = batch.copy()
    if out.empty or mode == "Normal traffic":
        return out

    n_attack = max(1, int(round(len(out) * intensity)))
    attack_idx = rng.choice(out.index.to_numpy(), size=min(n_attack, len(out)), replace=False)

    modes = ["Account takeover", "High-value cashout", "Bot burst", "Foreign IP wave"]
    active_mode = mode
    if mode == "Mixed attack":
        for idx in attack_idx:
            active_mode = rng.choice(modes)
            out = apply_synthesis_mode(out.loc[[idx]], active_mode, 1.0, rng).combine_first(out)
        return finalize_engineered_amounts(out)

    if active_mode == "Account takeover":
        for col in ["is_new_device", "shipping_billing_mismatch", "ip_country_mismatch"]:
            if col in out.columns:
                out.loc[attack_idx, col] = 1
        if "failed_payment_attempts" in out.columns:
            out.loc[attack_idx, "failed_payment_attempts"] = rng.integers(3, 10, size=len(attack_idx))
        if "ip_billing_distance_km" in out.columns:
            out.loc[attack_idx, "ip_billing_distance_km"] = (
                pd.to_numeric(out.loc[attack_idx, "ip_billing_distance_km"], errors="coerce").fillna(0)
                * rng.uniform(4, 12, size=len(attack_idx))
                + 500
            )

    elif active_mode == "High-value cashout":
        amount_factor = rng.uniform(4, 16, size=len(attack_idx))
        if "amount" in out.columns:
            out.loc[attack_idx, "amount"] = (
                pd.to_numeric(out.loc[attack_idx, "amount"], errors="coerce").fillna(0) * amount_factor
            )
        if "type" in out.columns:
            out.loc[attack_idx, "type"] = "CASH_OUT"
        for col in ["orig_balance_delta", "dest_balance_delta", "amount_to_oldOrg_ratio"]:
            if col in out.columns:
                out.loc[attack_idx, col] = (
                    pd.to_numeric(out.loc[attack_idx, col], errors="coerce").fillna(0)
                    * rng.uniform(3, 10, size=len(attack_idx))
                )

    elif active_mode == "Bot burst":
        if "time_since_prev_orig" in out.columns:
            out.loc[attack_idx, "time_since_prev_orig"] = rng.integers(0, 2, size=len(attack_idx))
        if "tx_count_prev_orig" in out.columns:
            out.loc[attack_idx, "tx_count_prev_orig"] = (
                pd.to_numeric(out.loc[attack_idx, "tx_count_prev_orig"], errors="coerce").fillna(0)
                + rng.integers(10, 80, size=len(attack_idx))
            )
        if "failed_payment_attempts" in out.columns:
            out.loc[attack_idx, "failed_payment_attempts"] = (
                pd.to_numeric(out.loc[attack_idx, "failed_payment_attempts"], errors="coerce").fillna(0)
                + rng.integers(2, 8, size=len(attack_idx))
            )

    elif active_mode == "Foreign IP wave":
        if "ip_country_mismatch" in out.columns:
            out.loc[attack_idx, "ip_country_mismatch"] = 1
        if "ip_billing_distance_km" in out.columns:
            out.loc[attack_idx, "ip_billing_distance_km"] = rng.uniform(3000, 16000, size=len(attack_idx))
        if "ip_country" in out.columns:
            out.loc[attack_idx, "ip_country"] = rng.choice(["RU", "CN", "NG", "BR"], size=len(attack_idx))

    return finalize_engineered_amounts(out)


def finalize_engineered_amounts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "amount" in out.columns:
        amount = pd.to_numeric(out["amount"], errors="coerce").fillna(0)
        out["log_amount"] = np.log1p(amount.clip(lower=0))
        if "oldbalanceOrg" in out.columns:
            old_org = pd.to_numeric(out["oldbalanceOrg"], errors="coerce").fillna(0)
            out["amount_to_oldOrg_ratio"] = amount / (old_org + 1.0)
        if "oldbalanceDest" in out.columns:
            old_dest = pd.to_numeric(out["oldbalanceDest"], errors="coerce").fillna(0)
            out["amount_to_oldDest_ratio"] = amount / (old_dest + 1.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def prepare_model_input(batch: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    engineered = add_balancing_features(batch)
    drop = [col for col in DROP_COLUMNS if col in engineered.columns]
    X = engineered.drop(columns=drop)
    feature_columns = metadata.get("feature_columns") or X.columns.tolist()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns].copy()
    for col in ["type", "home_billing_country", "ip_country"]:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().str.upper()
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def fallback_risk_score(batch: pd.DataFrame) -> np.ndarray:
    engineered = add_balancing_features(batch)
    score = np.zeros(len(engineered), dtype=float)
    if "amount_to_oldOrg_ratio" in engineered.columns:
        score += np.clip(pd.to_numeric(engineered["amount_to_oldOrg_ratio"], errors="coerce").fillna(0) / 10, 0, 0.35)
    if "failed_payment_attempts" in engineered.columns:
        score += np.clip(pd.to_numeric(engineered["failed_payment_attempts"], errors="coerce").fillna(0) / 12, 0, 0.25)
    if "ip_country_mismatch" in engineered.columns:
        score += 0.15 * pd.to_numeric(engineered["ip_country_mismatch"], errors="coerce").fillna(0).to_numpy()
    if "is_new_device" in engineered.columns:
        score += 0.10 * pd.to_numeric(engineered["is_new_device"], errors="coerce").fillna(0).to_numpy()
    if "type" in engineered.columns:
        score += 0.10 * engineered["type"].astype(str).str.upper().isin(["TRANSFER", "CASH_OUT"]).to_numpy()
    return np.clip(score, 0, 0.99)


def score_realtime_batch(
    batch: pd.DataFrame,
    model,
    metadata: dict,
    threshold: float,
    transactions_per_minute: int,
) -> pd.DataFrame:
    rows = []
    model_input = prepare_model_input(batch, metadata)
    start_time = pd.Timestamp.now().floor("s")
    spacing_seconds = 60 / max(transactions_per_minute, 1)

    for position, idx in enumerate(batch.index):
        single_X = model_input.loc[[idx]]
        start = time.perf_counter()
        if model is not None and hasattr(model, "predict_proba"):
            score = float(model.predict_proba(single_X)[:, 1][0])
            scorer = "model"
        else:
            score = float(fallback_risk_score(batch.loc[[idx]])[0])
            scorer = "fallback"
        latency_ms = (time.perf_counter() - start) * 1000
        original = batch.loc[idx].to_dict()
        tx_id = f"tx_{position + 1:04d}"
        rows.append(
            {
                "tx_id": tx_id,
                "arrival_time": start_time + pd.Timedelta(seconds=position * spacing_seconds),
                "fraud_score": score,
                "predicted_class": "Fraud" if score >= threshold else "Normal",
                "threshold": threshold,
                "latency_ms": latency_ms,
                "scorer": scorer,
                "true_label": int(original.get(TARGET, 0)) if TARGET in original else None,
                "amount": float(original.get("amount", 0) or 0),
                "type": original.get("type", ""),
                "source_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            }
        )

    scored = pd.DataFrame(rows)
    detail = batch.reset_index(drop=True).copy()
    detail["tx_id"] = scored["tx_id"].values
    return scored.merge(detail, on="tx_id", how="left", suffixes=("", "_raw"))


def sample_realtime_batch(
    reference: pd.DataFrame,
    n_transactions: int,
    target_fraud_rate: float,
    seed: int,
    generated_before: int,
) -> pd.DataFrame:
    if reference.empty:
        return pd.DataFrame()
    if TARGET not in reference.columns:
        replace = len(reference) < n_transactions
        return reference.sample(n=n_transactions, replace=replace, random_state=seed).copy()

    labels = pd.to_numeric(reference[TARGET], errors="coerce").fillna(0).astype(int)
    fraud_pool = reference[labels == 1]
    normal_pool = reference[labels == 0]
    if fraud_pool.empty or normal_pool.empty:
        replace = len(reference) < n_transactions
        return reference.sample(n=n_transactions, replace=replace, random_state=seed).copy()

    previous_target_fraud = int(round(generated_before * target_fraud_rate))
    next_target_fraud = int(round((generated_before + n_transactions) * target_fraud_rate))
    n_fraud = int(np.clip(next_target_fraud - previous_target_fraud, 0, n_transactions))
    n_normal = n_transactions - n_fraud

    parts = []
    if n_fraud:
        parts.append(
            fraud_pool.sample(
                n=n_fraud,
                replace=len(fraud_pool) < n_fraud,
                random_state=seed + 17,
            )
        )
    if n_normal:
        parts.append(
            normal_pool.sample(
                n=n_normal,
                replace=len(normal_pool) < n_normal,
                random_state=seed + 29,
            )
        )
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed + 41)
        .reset_index(drop=True)
        .copy()
    )


def append_realtime_transactions(
    reference: pd.DataFrame,
    mode: str,
    events_per_tick: int,
    intensity: float,
    target_fraud_rate: float,
    seed: int,
    model,
    metadata: dict,
    threshold: float,
    transactions_per_minute: int,
    max_points: int,
) -> None:
    counter = int(st.session_state.get("stream_counter", 0))
    rng_seed = seed + counter + 1
    batch = sample_realtime_batch(reference, events_per_tick, target_fraud_rate, rng_seed, counter)
    batch = apply_synthesis_mode(batch, mode, intensity, np.random.default_rng(rng_seed))
    scored = score_realtime_batch(batch, model, metadata, threshold, transactions_per_minute)

    tx_ids = [f"tx_{i:06d}" for i in range(counter + 1, counter + 1 + len(scored))]
    event_indices = list(range(counter, counter + len(scored)))
    scored["tx_id"] = tx_ids
    scored["event_index"] = event_indices
    batch = batch.reset_index(drop=True).copy()
    batch["tx_id"] = tx_ids
    batch["event_index"] = event_indices

    previous_scored = st.session_state.get("realtime_scored", pd.DataFrame())
    previous_batch = st.session_state.get("realtime_current_batch", pd.DataFrame())
    st.session_state["realtime_scored"] = (
        pd.concat([previous_scored, scored], ignore_index=True)
    )
    st.session_state["realtime_current_batch"] = (
        pd.concat([previous_batch, batch], ignore_index=True)
    )
    st.session_state["stream_counter"] = counter + len(scored)


def reset_realtime_stream(reference: pd.DataFrame, mode: str) -> None:
    st.session_state["realtime_scored"] = pd.DataFrame()
    st.session_state["realtime_current_batch"] = pd.DataFrame()
    st.session_state["realtime_reference"] = reference
    st.session_state["realtime_mode"] = mode
    st.session_state["stream_counter"] = 0
    st.session_state["selected_tx_id"] = None
    st.session_state["transaction_page"] = 1


def compute_current_drift(reference: pd.DataFrame, current: pd.DataFrame, config: dict) -> pd.DataFrame:
    if reference.empty or current.empty:
        return pd.DataFrame()
    return compute_drift_report(
        reference,
        current,
        config["psi_warning"],
        config["psi_alert"],
        config["ks_alert"],
    )


def cached_home_drift(reference: pd.DataFrame, current: pd.DataFrame, config: dict) -> pd.DataFrame:
    now = time.time()
    cached = st.session_state.get("home_drift_report", pd.DataFrame())
    last_at = float(st.session_state.get("home_drift_computed_at", 0.0))
    last_rows = int(st.session_state.get("home_drift_rows", 0))
    enough_time = now - last_at >= 5
    enough_rows = len(current) - last_rows >= 50
    if not cached.empty and not enough_time and not enough_rows:
        return cached
    report = compute_current_drift(reference, current, config)
    st.session_state["home_drift_report"] = report
    st.session_state["home_drift_computed_at"] = now
    st.session_state["home_drift_rows"] = len(current)
    return report


def render_realtime_state(reference: pd.DataFrame, config: dict, checkpoint_label: str) -> None:
    scored = st.session_state.get("realtime_scored", pd.DataFrame())
    current_batch = st.session_state.get("realtime_current_batch", pd.DataFrame())
    if scored.empty:
        st.info("Chọn preset rồi bấm Start để bắt đầu tạo transaction liên tục.")
        return

    drift_report = cached_home_drift(reference, current_batch, config)
    drift_view = prepare_drift_view(drift_report)
    max_drift = float(drift_view["drift_score"].max()) if not drift_view.empty else 0.0
    alert_count = int((drift_report["status"] == "alert").sum()) if not drift_report.empty else 0
    warning_count = int((drift_report["status"] == "warning").sum()) if not drift_report.empty else 0
    level = threat_level(max_drift, alert_count, warning_count)
    true_labels = pd.to_numeric(scored.get("true_label", pd.Series(dtype=float)), errors="coerce")
    true_fraud = int((true_labels == 1).sum())
    true_fraud_rate = true_fraud / len(scored) if len(scored) else 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Buffered", f"{len(scored):,}")
    m2.metric("True fraud", true_fraud, f"{true_fraud_rate:.1%}")
    m3.metric("Predicted fraud", int((scored["predicted_class"] == "Fraud").sum()))
    m4.metric("Avg latency", f"{scored['latency_ms'].mean():.2f} ms")
    m5.metric("P95 latency", f"{scored['latency_ms'].quantile(0.95):.2f} ms")
    m6.metric("Drift", status_label(level), f"max {max_drift:.3f}")
    st.caption(f"Scoring checkpoint: `{checkpoint_label}`")

    st.markdown("#### Bộ lọc transaction")
    f1, f2, f3, f4 = st.columns([1.2, 1.4, 1, 1])
    time_range_label = f1.selectbox("Khung thời gian", list(TIME_RANGE_OPTIONS.keys()), index=7)
    search_text = f2.text_input("Tìm theo mã/tên/IP/type", placeholder="VD: tx_000120, CASH_OUT, C123...")
    class_filter = f3.multiselect("Class dự đoán", ["Fraud", "Normal"], default=["Fraud", "Normal"])
    truth_filter = f4.selectbox("Fraud thật", ["Tất cả", "Fraud thật", "Normal thật"])

    type_options = sorted(scored["type"].dropna().astype(str).unique().tolist()) if "type" in scored else []
    min_date = pd.to_datetime(scored["arrival_time"], errors="coerce").min()
    max_date = pd.to_datetime(scored["arrival_time"], errors="coerce").max()
    g1, g2 = st.columns([1.4, 2.6])
    type_filter = g1.multiselect("Loại giao dịch", type_options, default=type_options)
    default_dates = (
        min_date.date() if pd.notna(min_date) else pd.Timestamp.now().date(),
        max_date.date() if pd.notna(max_date) else pd.Timestamp.now().date(),
    )
    date_range = g2.date_input("Ngày đến", value=default_dates)

    filtered = apply_transaction_filters(
        scored,
        time_range_label,
        search_text,
        class_filter,
        type_filter,
        truth_filter,
        date_range,
    )
    if filtered.empty:
        st.warning("Không có transaction nào khớp bộ lọc hiện tại.")
        return

    fig = px.scatter(
        filtered,
        x="arrival_time",
        y="fraud_score",
        color="predicted_class",
        hover_data=["tx_id", "type", "amount", "latency_ms", "true_label"],
        custom_data=["tx_id"],
        color_discrete_map=STATUS_COLORS,
        title="Transaction mới đến: score càng cao càng đáng review",
    )
    fig.add_hline(y=config["threshold"], line_dash="dash", line_color="#111827", annotation_text="threshold")
    fig.update_traces(marker={"size": 9, "opacity": 0.78})
    fig.update_layout(
        height=440,
        legend_title_text="Predicted class",
        clickmode="event+select",
        dragmode="select",
    )
    clicked_points = plotly_events(
        fig,
        click_event=True,
        select_event=True,
        hover_event=False,
        override_height=440,
        override_width="100%",
        key="incoming_score_chart_live",
    )
    st.caption("Click trực tiếp vào dot xanh/đỏ để chọn transaction. Stream vẫn tiếp tục chạy cho đến khi bấm Pause.")

    selected_tx = st.session_state.get("selected_tx_id")
    selection_changed = False
    clicked_tx = clicked_tx_from_plotly_events(clicked_points, fig)
    if clicked_tx:
        selected_tx = clicked_tx
        st.session_state["selected_tx_id"] = selected_tx
        selection_changed = True

    st.markdown("#### Bảng transaction")
    table_source = filtered.sort_values("event_index" if "event_index" in filtered.columns else "arrival_time", ascending=True)
    table_view = transaction_table_view(table_source)
    page_size = 50
    total_pages = max(1, int(np.ceil(len(table_view) / page_size)))
    if "transaction_page" not in st.session_state:
        st.session_state["transaction_page"] = 1
    current_page = int(st.session_state.get("transaction_page", 1))
    current_page = min(max(current_page, 1), total_pages)
    nav1, nav2, nav3, nav4 = st.columns([0.8, 0.8, 1.1, 2.3])
    if nav1.button("Prev", width="stretch", disabled=current_page <= 1):
        current_page -= 1
    if nav2.button("Next", width="stretch", disabled=current_page >= total_pages):
        current_page += 1
    current_page = nav3.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=current_page,
        step=1,
        key="transaction_page_input",
    )
    if nav4.button("Trang mới nhất", width="stretch"):
        current_page = total_pages
    st.session_state["transaction_page"] = int(current_page)
    nav4.caption(f"Page {current_page}/{total_pages} | 50 row/page cố định | tổng {len(table_view):,} row")

    start_row = (int(current_page) - 1) * page_size
    page_df = table_view.iloc[start_row : start_row + page_size].copy()
    page_df = pad_transaction_page(page_df, page_size=page_size)
    table_event = st.dataframe(
        page_df,
        width="stretch",
        height=520,
        hide_index=True,
        key=f"incoming_transaction_table_page_{current_page}",
        on_select="rerun",
        selection_mode="single-row",
        row_height=26,
    )
    try:
        selected_rows = table_event["selection"]["rows"]
        if selected_rows:
            candidate = str(page_df.iloc[selected_rows[0]]["Transaction"])
            if candidate:
                selected_tx = candidate
                st.session_state["selected_tx_id"] = selected_tx
                selection_changed = True
    except Exception:
        pass

    if selected_tx:
        st.success(f"Đang chọn transaction: {selected_tx}")
    st.caption(f"Đang hiển thị {len(filtered):,}/{len(scored):,} transaction theo bộ lọc.")
    st.markdown('<div id="transaction-detail-anchor"></div>', unsafe_allow_html=True)
    if selection_changed:
        scroll_to_detail()
    show_transaction_detail(scored, selected_tx)


@st.fragment(run_every=1.0)
def live_stream_fragment(
    reference: pd.DataFrame,
    mode: str,
    events_per_tick: int,
    intensity: float,
    target_fraud_rate: float,
    seed: int,
    model,
    metadata: dict,
    config: dict,
    speed: int,
    max_points: int,
    checkpoint_label: str,
) -> None:
    if st.session_state.get("stream_running", False):
        append_realtime_transactions(
            reference,
            mode,
            events_per_tick,
            intensity,
            target_fraud_rate,
            seed,
            model,
            metadata,
            config["threshold"],
            speed,
            max_points,
        )
    render_realtime_state(reference, config, checkpoint_label)


def status_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()
    summary = (
        report.groupby(["metric_type", "status"])
        .size()
        .reset_index(name="features")
        .sort_values(["metric_type", "status"])
    )
    return summary


def sidebar(data: dict[str, pd.DataFrame | dict | str]) -> dict:
    st.sidebar.title("Fraud Monitor")
    st.sidebar.caption("Realtime scoring, drift monitoring, and balancing results.")
    if st.sidebar.button("Reload artifacts"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Runtime config")
    config = {
        "reference_rows": st.sidebar.slider("Reference rows", 1_000, 100_000, 5_000, step=1_000),
        "psi_warning": st.sidebar.number_input("PSI warning", min_value=0.01, max_value=1.0, value=0.10, step=0.01),
        "psi_alert": st.sidebar.number_input("PSI alert", min_value=0.01, max_value=1.0, value=0.25, step=0.01),
        "ks_alert": st.sidebar.number_input("KS alert", min_value=0.01, max_value=1.0, value=0.10, step=0.01),
    }
    checkpoints = available_checkpoints(data)
    st.sidebar.divider()
    st.sidebar.subheader("Model checkpoint")
    if checkpoints.empty:
        st.sidebar.warning("No trained checkpoint found. Realtime monitor will use fallback risk score.")
        checkpoint_config = selected_checkpoint_config(checkpoints, "")
    else:
        labels = checkpoints["label"].tolist()
        selected_label = st.sidebar.selectbox("Checkpoint", labels, index=0)
        checkpoint_config = selected_checkpoint_config(checkpoints, selected_label)
        row = checkpoint_config["row"]
        st.sidebar.caption(
            " | ".join(
                [
                    f"AUC-PR={row.get('auc_pr', np.nan):.4f}" if pd.notna(row.get("auc_pr", np.nan)) else "AUC-PR=N/A",
                    f"Recall={row.get('recall', np.nan):.4f}" if pd.notna(row.get("recall", np.nan)) else "Recall=N/A",
                    f"F1={row.get('f1', np.nan):.4f}" if pd.notna(row.get("f1", np.nan)) else "F1=N/A",
                ]
            )
        )
        if not checkpoint_config["exists"]:
            st.sidebar.warning("Checkpoint file is missing. Run balancing_experiments.py again.")

    config["checkpoint"] = checkpoint_config
    config["threshold"] = st.sidebar.slider(
        "Score threshold",
        0.01,
        0.99,
        checkpoint_config["threshold"],
        step=0.01,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Artifacts")
    for label, file_path in {
        "Best model": MODEL_PATH,
        "Checkpoint manifest": CHECKPOINT_MANIFEST_PATH,
        "Run metadata": METADATA_PATH,
        "Reference": REFERENCE_PATH,
        "Model results": path("balancing_model_results.csv"),
    }.items():
        st.sidebar.write(f"{'OK' if file_path.exists() else 'Missing'} {label}")
    return config


def home_tab(data: dict[str, pd.DataFrame | dict | str], config: dict) -> None:
    st.subheader("Realtime Transaction Stream")
    st.caption("Luồng chính: chọn scenario, bấm Start, quan sát score/drift, rồi chọn transaction để xem chi tiết.")

    reference = load_reference_rows(config["reference_rows"])
    if reference.empty:
        st.warning("Chưa có reference data. Hãy chạy monitor.py --build-reference hoặc tạo cleaned data trước.")
        st.code("python data_balancing/monitor.py --build-reference --max-reference-rows 100000", language="bash")
        return

    checkpoint = config.get("checkpoint", {})
    model = load_model(checkpoint.get("path", "")) if checkpoint.get("exists") else None
    metadata = data["metadata"] if isinstance(data["metadata"], dict) else {}

    preset_name = st.selectbox("Scenario preset", list(STREAM_PRESETS.keys()))
    preset = STREAM_PRESETS[preset_name]
    c1, c2, c3, c4, c5 = st.columns(5)
    mode = c1.selectbox(
        "Mode dữ liệu đến",
        SYNTHESIS_MODES,
        index=SYNTHESIS_MODES.index(preset["mode"]),
        help="Chọn kiểu traffic realtime để giả lập.",
    )
    events_per_tick = c2.slider(
        "GD mỗi giây",
        1,
        20,
        preset["events_per_tick"],
        step=1,
        help="Số transaction mới append vào dashboard sau mỗi giây.",
    )
    speed = c3.slider(
        "Nhịp timestamp",
        10,
        300,
        preset["speed"],
        step=10,
        help="Tốc độ thời gian mô phỏng, không phải số dòng render mỗi giây.",
    )
    target_fraud_rate = c4.slider(
        "Tỉ lệ fraud",
        0.0,
        0.8,
        preset["target_fraud_rate"],
        step=0.01,
        help="Tỉ lệ fraud thật mong muốn trong stream.",
    )
    intensity = c5.slider(
        "Mức bất thường",
        0.0,
        0.9,
        preset["intensity"],
        step=0.05,
        help="Tỉ lệ transaction được inject pattern rủi ro để test drift/scoring.",
    )

    p1, p2, p3, p4 = st.columns([1, 1, 1, 2])
    seed = p1.number_input("Simulation seed", value=42, step=1)
    max_points = p2.slider("Buffer hiển thị", 50, 500, preset["max_points"], step=10)
    start_clicked = p3.button("Start", type="primary", width="stretch")
    stop_clicked = p4.button("Pause", width="stretch")

    q1, q2 = st.columns([1, 5])
    clear_clicked = q1.button("Reset", width="stretch")
    stream_signature = {
        "mode": mode,
        "seed": int(seed),
        "checkpoint": checkpoint.get("label", "fallback_risk_score"),
        "threshold": round(float(config["threshold"]), 6),
        "target_fraud_rate": round(float(target_fraud_rate), 6),
    }

    if clear_clicked:
        reset_realtime_stream(reference, mode)
        st.session_state["stream_running"] = False
        st.session_state["stream_signature"] = stream_signature
    if start_clicked:
        if st.session_state.get("stream_signature") != stream_signature:
            reset_realtime_stream(reference, mode)
        st.session_state["stream_running"] = True
        st.session_state["stream_signature"] = stream_signature
    if stop_clicked:
        st.session_state["stream_running"] = False

    running = bool(st.session_state.get("stream_running", False))
    q2.caption(
        f"Trạng thái: {'ĐANG CHẠY' if running else 'ĐANG DỪNG'} | "
        f"mode={mode} | append={events_per_tick} transaction(s)/second | "
        f"target fraud={target_fraud_rate:.1%} | buffer={max_points}"
    )

    live_stream_fragment(
        reference,
        mode,
        events_per_tick,
        intensity,
        target_fraud_rate,
        int(seed),
        model,
        metadata,
        config,
        speed,
        max_points,
        checkpoint.get("label", "fallback_risk_score"),
    )


def show_transaction_detail(scored: pd.DataFrame, tx_id: str) -> None:
    row = scored[scored["tx_id"] == tx_id]
    if row.empty:
        return
    row = row.iloc[0]
    with st.expander(f"Chi tiết transaction đang chọn: {tx_id}", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Class dự đoán", row["predicted_class"])
        c2.metric("Fraud score", f"{row['fraud_score']:.4f}")
        c3.metric("Latency", f"{row['latency_ms']:.2f} ms")
        c4.metric("Amount", f"{row['amount']:,.2f}")

        important = [
            ("type", "Loại giao dịch"),
            ("amount", "Số tiền"),
            ("oldbalanceOrg", "Số dư nguồn trước GD"),
            ("newbalanceOrig", "Số dư nguồn sau GD"),
            ("oldbalanceDest", "Số dư đích trước GD"),
            ("newbalanceDest", "Số dư đích sau GD"),
            ("account_age_days", "Tuổi account"),
            ("is_new_device", "Thiết bị mới"),
            ("shipping_billing_mismatch", "Mismatch shipping/billing"),
            ("failed_payment_attempts", "Số lần payment fail"),
            ("ip_country", "IP country"),
            ("home_billing_country", "Billing country"),
            ("ip_billing_distance_km", "Khoảng cách IP-billing"),
            ("ip_country_mismatch", "IP country mismatch"),
            ("time_since_prev_orig", "Thời gian từ GD trước"),
            ("tx_count_prev_orig", "Số GD trước đó"),
        ]
        details = pd.DataFrame(
            [
                {"Nhóm thông tin": label, "Feature": col, "Giá trị": format_cell(row[col])}
                for col, label in important
                if col in scored.columns
            ]
        )
        st.dataframe(details, width="stretch", hide_index=True)


def feature_drift_tab(config: dict) -> None:
    st.subheader("Feature Drift")
    st.caption("Trang này trả lời: dữ liệu đang đến có còn giống reference ban đầu không, và feature nào cần xem trước.")
    reference = st.session_state.get("realtime_reference", load_reference_rows(config["reference_rows"]))
    current = st.session_state.get("realtime_current_batch", pd.DataFrame())
    if reference.empty:
        st.warning("Chưa có reference data.")
        return
    if current.empty:
        st.info("Chưa có current incoming batch. Vào tab Home và bấm Start trước.")
        return

    report = compute_current_drift(reference, current, config)
    drift_view = prepare_drift_view(report)
    alert_count = int((drift_view["status"] == "alert").sum()) if not drift_view.empty else 0
    warning_count = int((drift_view["status"] == "warning").sum()) if not drift_view.empty else 0
    ok_count = int((drift_view["status"] == "ok").sum()) if not drift_view.empty else 0
    max_drift = float(drift_view["drift_score"].max()) if not drift_view.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Kết luận", status_label(threat_level(max_drift, alert_count, warning_count)), f"max {max_drift:.3f}")
    k2.metric("Lệch mạnh", alert_count)
    k3.metric("Cần theo dõi", warning_count)
    k4.metric("Ổn định", ok_count)

    with st.expander("Cách đọc nhanh", expanded=False):
        st.markdown(
            """
- Ưu tiên xem các dòng `Lệch mạnh` trước.
- `Điểm drift` càng cao nghĩa là current stream càng khác reference.
- Với feature số, dashboard dùng PSI/KS. Với feature nhóm, dashboard dùng mức đổi tỉ trọng category.
- Drift không tự kết luận fraud, nhưng báo rằng data vào model đã khác data ban đầu.
"""
        )

    st.markdown("### Feature cần chú ý trước")
    top = drift_view.head(12)
    st.dataframe(
        top[["Mức ưu tiên", "Feature", "Loại", "Điểm drift", "Mức lệch hiển thị", "Ý nghĩa", "Nên làm gì"]],
        width="stretch",
        hide_index=True,
    )
    st.caption("Biểu đồ bên dưới dùng mức lệch hiển thị từ 0 đến 1 để dễ nhìn. `Điểm drift` raw vẫn xem trong bảng.")

    fig = px.bar(
        top.sort_values("display_score", ascending=True),
        x="display_score",
        y="feature",
        color="Trạng thái",
        color_discrete_map={
            "Ổn định": DRIFT_COLORS["ok"],
            "Cần theo dõi": DRIFT_COLORS["warning"],
            "Lệch mạnh": DRIFT_COLORS["alert"],
        },
        orientation="h",
        labels={"display_score": "Mức lệch hiển thị", "feature": "Feature"},
        title="Feature đang khác reference nhiều nhất",
    )
    fig.add_vline(x=config["psi_warning"], line_dash="dash", line_color="#B7791F")
    fig.add_vline(x=config["psi_alert"], line_dash="dash", line_color="#B91C1C")
    fig.update_layout(
        height=440,
        legend_title_text="Trạng thái",
        xaxis_range=[0, 1.08],
        margin={"l": 20, "r": 40, "t": 70, "b": 40},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Xem từng feature")
    for _, row in drift_view.iterrows():
        feature = row["feature"]
        label = f"{status_label(row['status'])} | {feature} | điểm drift {row['drift_score']:.3f}"
        with st.expander(label, expanded=row["status"] != "ok"):
            show_feature_section(reference, current, feature, row)


def show_feature_section(reference: pd.DataFrame, current: pd.DataFrame, feature: str, row: pd.Series) -> None:
    st.caption(f"{row['Ý nghĩa']} | {row['Nên làm gì']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mức ưu tiên", row["Mức ưu tiên"])
    m2.metric("PSI", "N/A" if pd.isna(row["psi"]) else f"{row['psi']:.3f}")
    m3.metric("KS", "N/A" if pd.isna(row["ks_statistic"]) else f"{row['ks_statistic']:.3f}")
    m4.metric(
        "Category shift",
        "N/A" if pd.isna(row["category_distribution_shift"]) else f"{row['category_distribution_shift']:.3f}",
    )
    compare = pd.concat(
        [
            reference[[feature]].assign(window="reference"),
            current[[feature]].assign(window="current"),
        ],
        ignore_index=True,
    )
    if feature in NUMERIC_MONITOR_COLUMNS and pd.api.types.is_numeric_dtype(compare[feature]):
        ref_values = pd.to_numeric(reference[feature], errors="coerce")
        cur_values = pd.to_numeric(current[feature], errors="coerce")
        summary = pd.DataFrame(
            [
                {
                    "Window": "reference",
                    "Rows": int(ref_values.notna().sum()),
                    "Median": ref_values.median(),
                    "P95": ref_values.quantile(0.95),
                    "Mean": ref_values.mean(),
                },
                {
                    "Window": "current",
                    "Rows": int(cur_values.notna().sum()),
                    "Median": cur_values.median(),
                    "P95": cur_values.quantile(0.95),
                    "Mean": cur_values.mean(),
                },
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)
        fig = px.histogram(
            compare,
            x=feature,
            color="window",
            nbins=50,
            barmode="overlay",
            marginal="box",
            labels={feature: feature, "window": "Window"},
            title=f"Phân phối reference vs current: {feature}",
        )
    else:
        counts = compare.groupby(["window", feature]).size().reset_index(name="rows")
        totals = counts.groupby("window")["rows"].transform("sum")
        counts["share"] = counts["rows"] / totals
        pivot = counts.pivot_table(index=feature, columns="window", values="share", fill_value=0).reset_index()
        for col in ["reference", "current"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot["abs_shift"] = (pivot["current"] - pivot["reference"]).abs()
        top_shift = pivot.sort_values("abs_shift", ascending=False).head(10).rename(
            columns={
                feature: "Category",
                "reference": "Reference share",
                "current": "Current share",
                "abs_shift": "Abs shift",
            }
        )
        st.dataframe(top_shift, width="stretch", hide_index=True)
        fig = px.bar(
            counts,
            x=feature,
            y="share",
            color="window",
            barmode="group",
            labels={feature: feature, "share": "Tỉ trọng", "window": "Window"},
            title=f"Tỉ trọng category reference vs current: {feature}",
        )
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")


def models_tab(data: dict[str, pd.DataFrame | dict | str]) -> None:
    st.subheader("Nhiệm vụ 1: So sánh Balancing Methods Và Models")
    results = data["model_results"]
    review = data["review_results"]
    cv_results = data.get("cv_results", pd.DataFrame())
    metadata = data.get("metadata", {})
    if not isinstance(results, pd.DataFrame) or results.empty:
        st.warning("Chưa có balancing_model_results.csv.")
        st.code("python data_balancing/balancing_experiments.py --max-rows 50000", language="bash")
        return

    results = results.copy()
    results["model_label"] = results["strategy"].astype(str) + "__" + results["model"].astype(str)
    best = results.sort_values(["auc_pr", "f1"], ascending=False).iloc[0]
    min_test_fraud = int(pd.to_numeric(results.get("test_fraud_cases", pd.Series([0])), errors="coerce").fillna(0).min())

    st.markdown("### Độ tin cậy của run hiện tại")
    r1, r2, r3, r4 = st.columns(4)
    if isinstance(metadata, dict) and metadata:
        r1.metric("Train rows", f"{int(metadata.get('train_rows', 0)):,}")
        r2.metric("Test rows", f"{int(metadata.get('test_rows', 0)):,}")
        r3.metric("Train fraud rate", f"{float(metadata.get('train_fraud_rate', 0)):.4%}")
        r4.metric("Test fraud rate", f"{float(metadata.get('test_fraud_rate', 0)):.4%}")
    else:
        r1.metric("Train rows", "N/A")
        r2.metric("Test rows", "N/A")
        r3.metric("Train fraud rate", "N/A")
        r4.metric("Test fraud rate", "N/A")

    if min_test_fraud < 100:
        st.error(
            f"Run hiện tại chưa đủ tin cậy để kết luận model: test chỉ có {min_test_fraud:,} fraud case. "
            "Đây nhiều khả năng là smoke/sample run. Cần retrain full data và bật k-fold."
        )
    elif min_test_fraud < 500:
        st.warning(
            f"Test có {min_test_fraud:,} fraud case. Có thể đọc kết quả, nhưng nên xem thêm k-fold validation."
        )
    else:
        st.success(f"Test có {min_test_fraud:,} fraud case. Kết quả holdout đáng đọc hơn, vẫn nên xem k-fold để kiểm tra ổn định.")

    cv_available = isinstance(cv_results, pd.DataFrame) and not cv_results.empty
    coverage = pd.DataFrame(
        [
            {
                "Split": "Train",
                "Đang dùng để làm gì": "Fit model sau khi balancing train-only",
                "Metric hiện có": "Chưa lưu riêng trong artifact cũ",
                "Có nên kết luận?": "Không dùng train metric để kết luận production",
            },
            {
                "Split": "Validation / k-fold",
                "Đang dùng để làm gì": "Kiểm tra độ ổn định giữa các fold",
                "Metric hiện có": "AUC-PR theo fold" if cv_available else "Chưa có, cần Retrain bật k-fold",
                "Có nên kết luận?": "Dùng để xem model có ổn định hay không",
            },
            {
                "Split": "Holdout test",
                "Đang dùng để làm gì": "So sánh model trên dữ liệu chưa train",
                "Metric hiện có": "Fraud captured/missed, value captured, AUC-PR, precision/recall/F1",
                "Có nên kết luận?": "Chỉ đáng tin nếu đủ fraud case",
            },
        ]
    )
    st.dataframe(coverage, width="stretch", hide_index=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best checkpoint", f"{best['strategy']} / {best['model']}")
    if "fraud_cases_captured" in results.columns:
        c2.metric("Fraud captured", f"{int(best['fraud_cases_captured']):,}/{int(best['test_fraud_cases']):,}")
        c3.metric("Fraud missed", f"{int(best['fraud_cases_missed']):,}")
        c4.metric("Value captured", f"{best['fraud_value_capture_rate']:.2%}")
        c5.metric("Review queue", f"{int(best['review_queue_size']):,}")
    else:
        c2.metric("Recall", f"{best['recall']:.2%}")
        c3.metric("Precision", f"{best['precision']:.2%}")
        c4.metric("AUC-PR", f"{best['auc_pr']:.4f}")
        c5.metric("Business cost", f"{best['business_cost']:,.0f}")

    st.markdown("### Chọn một model để xem kỹ")
    selected_label = st.selectbox("Model checkpoint", results["model_label"].tolist())
    selected = results[results["model_label"] == selected_label].iloc[0]
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Strategy", str(selected["strategy"]))
    d2.metric("Model", str(selected["model"]))
    d3.metric("Fraud captured", f"{int(selected.get('fraud_cases_captured', 0)):,}/{int(selected.get('test_fraud_cases', 0)):,}")
    d4.metric("Fraud missed", f"{int(selected.get('fraud_cases_missed', 0)):,}")
    d5.metric("False alerts", f"{int(selected.get('false_alerts', 0)):,}")
    d6.metric("Threshold", f"{float(selected.get('operating_threshold', 0.5)):.2f}")
    st.dataframe(
        pd.DataFrame([selected]).drop(columns=["model_label"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )

    preferred_metrics = [
        "fraud_cases_captured",
        "fraud_case_capture_rate",
        "fraud_amount_captured",
        "fraud_amount_missed",
        "fraud_value_capture_rate",
        "fraud_cases_missed",
        "review_queue_size",
        "false_alerts",
        "review_fraud_hit_rate",
        "auc_pr",
        "precision",
        "recall",
        "f1",
        "business_cost",
        "fit_seconds",
    ]
    available_metrics = [metric for metric in preferred_metrics if metric in results.columns]
    st.markdown("### So sánh toàn bộ model")
    metric = st.selectbox("Metric", available_metrics)
    model_labels = results["model_label"].tolist()
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set2
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(model_labels)}
    fig = px.bar(
        results,
        x="strategy",
        y=metric,
        color="model_label",
        color_discrete_map=color_map,
        barmode="group",
        title=f"{metric} theo balancing strategy và model",
        labels={"model_label": "Model checkpoint"},
    )
    st.plotly_chart(fig, width="stretch")

    score_metrics = [m for m in ["auc_pr", "fraud_case_capture_rate", "fraud_value_capture_rate", "precision", "recall", "f1"] if m in results.columns]
    if score_metrics:
        long_scores = results.melt(
            id_vars=["model_label", "strategy", "model"],
            value_vars=score_metrics,
            var_name="metric",
            value_name="value",
        )
        fig_scores = px.line(
            long_scores,
            x="metric",
            y="value",
            color="model_label",
            markers=True,
            color_discrete_map=color_map,
            title="Các chỉ số chính trên holdout test",
            labels={"model_label": "Model checkpoint", "value": "Score"},
        )
        fig_scores.update_layout(height=420)
        st.plotly_chart(fig_scores, width="stretch")

    priority_cols = [
        "strategy",
        "model",
        "test_fraud_cases",
        "fraud_cases_captured",
        "fraud_cases_missed",
        "fraud_case_capture_rate",
        "fraud_amount_captured",
        "fraud_amount_missed",
        "fraud_value_capture_rate",
        "review_queue_size",
        "false_alerts",
        "review_fraud_hit_rate",
        "auc_pr",
        "precision",
        "recall",
        "f1",
        "business_cost",
        "operating_threshold",
    ]
    display_cols = [col for col in priority_cols if col in results.columns]
    remaining_cols = [col for col in results.columns if col not in display_cols]
    st.dataframe(results[display_cols + remaining_cols], width="stretch", hide_index=True)

    st.markdown("### K-fold validation")
    if isinstance(cv_results, pd.DataFrame) and not cv_results.empty:
        cv_results = cv_results.copy()
        cv_results["model_label"] = cv_results["strategy"].astype(str) + "__" + cv_results["model"].astype(str)
        folds = sorted(cv_results["fold"].unique().tolist()) if "fold" in cv_results.columns else []
        st.caption(f"K-fold ở đây là validation stability, không phải test set. Hiện có {len(folds)} fold: {folds}.")
        cv_summary = (
            cv_results.groupby(["strategy", "model", "model_label"])["auc_pr"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        st.dataframe(cv_summary, width="stretch", hide_index=True)
        fig_cv = px.line(
            cv_results,
            x="fold",
            y="auc_pr",
            color="model_label",
            markers=True,
            color_discrete_map=color_map,
            title="AUC-PR theo từng validation fold",
            labels={"model_label": "Model checkpoint", "auc_pr": "Validation AUC-PR"},
        )
        st.plotly_chart(fig_cv, width="stretch")
    else:
        st.warning("Chưa có k-fold validation. Bấm Retrain bên dưới và bật `Run k-fold validation` để sinh `cv_aucpr_by_fold.csv`.")

    st.markdown("### Retrain từ dashboard")
    retrain_state = read_retrain_state()
    running_pid = st.session_state.get("retrain_pid") or retrain_state.get("pid")
    retrain_running = is_process_running(running_pid)
    if retrain_state and not retrain_running and retrain_state.get("status") == "running":
        retrain_state["status"] = "finished_or_stopped"
        write_retrain_state(retrain_state)

    st.caption("Retrain chạy nền, từng combo strategy/model. Mặc định là chạy nhẹ để kiểm tra trước, không tự chạy full grid.")
    if retrain_running:
        st.info(
            f"Đang retrain nền, PID={running_pid}. "
            f"Bắt đầu: {retrain_state.get('started_at', 'N/A')}. "
            "Có thể xem log bên dưới hoặc bấm Stop."
        )
    elif retrain_state:
        st.success("Không có retrain nào đang chạy. Có thể Start retrain mới hoặc Reload artifacts để đọc kết quả mới nhất.")

    st.markdown("#### Bước 1: Chọn mức chạy")
    run_profile = st.radio(
        "Mức chạy",
        [
            "Kiểm tra nhanh - 10k rows",
            "Vừa sức - 50k rows",
            "Production - full data",
        ],
        index=1,
        horizontal=True,
    )
    max_rows = {
        "Kiểm tra nhanh - 10k rows": 10_000,
        "Vừa sức - 50k rows": 50_000,
        "Production - full data": 0,
    }[run_profile]
    p1, p2, p3 = st.columns(3)
    split = p1.selectbox("Cách chia train/test", ["stratified", "time"], index=0)
    run_cv = p2.checkbox("Chạy k-fold validation", value=run_profile != "Kiểm tra nhanh - 10k rows")
    cv_folds = p3.number_input("Số fold", min_value=2, max_value=10, value=5, step=1, disabled=not run_cv)

    st.markdown("#### Bước 2: Chọn model cần train")
    strategy_options = ["original", "undersampling", "smote", "class_weights"]
    model_options = ["logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting"]
    advanced_grid = st.checkbox("Chọn nhiều combo nâng cao", value=False)
    if advanced_grid:
        s_col, m_col = st.columns(2)
        selected_strategies = s_col.multiselect("Strategy", strategy_options, default=["original"])
        selected_models = m_col.multiselect("Model", model_options, default=["random_forest"])
    else:
        c1, c2 = st.columns(2)
        selected_strategies = [c1.selectbox("Strategy", strategy_options, index=0)]
        selected_models = [c2.selectbox("Model", model_options, index=1)]

    combo_count = len(selected_strategies) * len(selected_models)
    fraud_review_size = 3_000
    command_preview = [
        "python",
        "data_balancing/balancing_experiments.py",
        "--max-rows",
        str(max_rows),
        "--split",
        split,
        "--n-jobs",
        "4",
        "--xgboost-device",
        "auto",
        "--strategies",
        *selected_strategies,
        "--models",
        *selected_models,
    ]
    if run_cv:
        command_preview.extend(["--run-cv", "--cv-folds", str(int(cv_folds))])
    st.caption(f"Sẽ chạy {combo_count} combo tuần tự. Output sẽ merge vào scoreboard hiện có, không xoá model khác.")
    with st.expander("Xem command sẽ chạy", expanded=False):
        st.code(" ".join(command_preview), language="bash")

    st.markdown("#### Bước 3: Điều khiển")
    a1, a2, a3 = st.columns(3)
    start_disabled = retrain_running or not selected_strategies or not selected_models
    retrain_clicked = a1.button("Start retrain", type="primary", disabled=start_disabled, width="stretch")
    stop_clicked = a2.button("Stop retrain", disabled=not retrain_running, width="stretch")
    reload_clicked = a3.button("Reload artifacts", width="stretch")
    if stop_clicked and stop_retrain_job(running_pid):
        retrain_state["status"] = "stopped"
        write_retrain_state(retrain_state)
        st.warning("Đã gửi tín hiệu stop retrain. Chờ vài giây rồi reload status/log.")
    if reload_clicked:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    if retrain_clicked:
        pid = start_retrain_job(
            max_rows,
            split,
            selected_strategies,
            selected_models,
            run_cv,
            int(cv_folds),
            int(fraud_review_size),
        )
        st.session_state["retrain_pid"] = pid
        st.success(f"Đã start retrain nền, PID={pid}. Xem log bên dưới để biết đang chạy đến bước nào.")
    if RETRAIN_LOG_PATH.exists():
        with st.expander("Retrain log tail", expanded=True):
            st.code(read_log_tail(RETRAIN_LOG_PATH), language="text")

    if isinstance(review, pd.DataFrame) and not review.empty:
        st.subheader("Fraud-focused review set")
        st.dataframe(review, width="stretch", hide_index=True)


def importance_tab(data: dict[str, pd.DataFrame | dict | str]) -> None:
    st.subheader("Feature Importance")
    scores = data["feature_scores"]
    permutation = data["permutation"]
    dropped = data["dropped_features"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Mutual information")
        if isinstance(scores, pd.DataFrame) and not scores.empty:
            fig = px.bar(
                scores.head(25).sort_values("mutual_info"),
                x="mutual_info",
                y="feature",
                orientation="h",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Chưa có feature_target_scores.csv.")
    with c2:
        st.markdown("#### Permutation importance")
        if isinstance(permutation, pd.DataFrame) and not permutation.empty:
            fig = px.bar(
                permutation.head(25).sort_values("importance_mean"),
                x="importance_mean",
                y="feature",
                orientation="h",
                error_x="importance_std" if "importance_std" in permutation.columns else None,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Chưa có model_permutation_importance.csv.")

    st.markdown("#### Correlated features dropped")
    if isinstance(dropped, pd.DataFrame) and not dropped.empty:
        st.dataframe(dropped, width="stretch", hide_index=True)
    else:
        st.caption("Không có feature nào bị drop trong run hiện tại hoặc file chưa được sinh.")


def config_tab(data: dict[str, pd.DataFrame | dict | str], config: dict) -> None:
    st.subheader("Config Và Data Source Check")
    source = load_data_source_status()

    c1, c2, c3 = st.columns(3)
    c1.metric("Kaggle CLI", "Available" if source["kaggle_cli_available"] else "Missing")
    c2.metric("Active raw", source["active_raw"]["status"])
    cleaned = source["cleaned_data"]
    c3.metric("Cleaned rows", "N/A" if cleaned["rows"] is None else f"{cleaned['rows']:,}")

    if not source["active_raw_is_full"]:
        st.warning(source["recommendation"])
    else:
        st.success(source["recommendation"])

    st.markdown("#### Raw CSV candidates")
    st.dataframe(pd.DataFrame(source["raw_candidates"]), width="stretch", hide_index=True)

    st.markdown("#### Downloaded zip candidates")
    st.dataframe(pd.DataFrame(source["zip_candidates"]), width="stretch", hide_index=True)

    checkpoints = available_checkpoints(data)
    st.markdown("#### Trained model checkpoints")
    if checkpoints.empty:
        st.info("Chưa có checkpoint. Chạy balancing_experiments.py để train và lưu checkpoints.")
    else:
        cols = [
            "label",
            "strategy",
            "model",
            "auc_pr",
            "precision",
            "recall",
            "f1",
            "operating_threshold",
            "exists",
            "checkpoint_path",
        ]
        cols = [col for col in cols if col in checkpoints.columns]
        st.dataframe(checkpoints[cols], width="stretch", hide_index=True)

    st.markdown("#### Current dashboard config")
    serializable_config = dict(config)
    if "checkpoint" in serializable_config:
        serializable_config["checkpoint"] = {
            key: value
            for key, value in serializable_config["checkpoint"].items()
            if key != "row"
        }
    st.json(serializable_config)

    st.markdown("#### Commands")
    st.code(
        "\n".join(
            [
                "python data_balancing/data_source_check.py",
                "kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset -p feature_engineering/fraud-detection/data/raw --unzip",
                "ln -sf PS_20174392719_1491204439457_log.csv feature_engineering/fraud-detection/data/raw/online_fraud_detection.csv",
                "cd feature_engineering/fraud-detection",
                "PYTHONPATH=src python src/generate_synthetic.py",
                "PYTHONPATH=src python src/cleaning.py",
                "cd ../..",
                "python data_balancing/balancing_experiments.py --max-rows 0 --split stratified",
                "python data_balancing/monitor.py --build-reference --max-reference-rows 100000",
                "streamlit run data_balancing/monitor_app.py",
            ]
        ),
        language="bash",
    )


def guide_tab(data: dict[str, pd.DataFrame | dict | str]) -> None:
    st.subheader("Hướng Dẫn")
    st.markdown(
        """
### Nhiệm vụ 1: Giảm imbalance và đánh giá model

Chạy `balancing_experiments.py` để so sánh `original`, `undersampling`, `smote`
và `class_weights` trên nhiều model. Kết quả chính nằm trong
`data_balancing/outputs/balancing_model_results.csv`.

Script này cũng lưu checkpoint cho từng cặp strategy/model ở
`data_balancing/outputs/artifacts/checkpoints/` và ghi manifest
`data_balancing/outputs/artifacts/model_checkpoints.csv`.

### Nhiệm vụ 2: Monitor realtime data

Monitor không train lại model. Sidebar cho chọn checkpoint đã train, sau đó Home
tab tạo incoming stream, score từng transaction, đo latency và cho xem chi tiết
từng điểm dữ liệu mới đến.

Luồng đọc dashboard nên đi theo thứ tự:

1. Vào `Home`, chọn `Scenario preset`, bấm `Start`.
2. Dùng `Khung thời gian` để xem 5 phút, 30 phút, 1 giờ, 1 ngày, 1 tuần,
   1 tháng, 1 năm hoặc từ lúc bắt đầu.
3. Filter theo mã transaction, account/name, IP, type, class dự đoán, fraud thật
   hoặc ngày đến.
4. Nhìn `True fraud`, `Predicted fraud`, `Latency` và `Drift`.
5. Click điểm trên chart hoặc click row trong bảng để xem feature chi tiết.
6. Sang `Feature Drift`, xem bảng `Feature cần chú ý trước`.
7. Mở từng feature đang `Lệch mạnh` để xem current khác reference ở đâu.

Feature Drift không kết luận fraud trực tiếp. Nó báo rằng dữ liệu realtime đang
khác dữ liệu reference ban đầu, nên model có thể đang bị đưa vào vùng dữ liệu lạ.
Biểu đồ top drift dùng mức lệch hiển thị 0-1 để dễ nhìn; điểm drift raw vẫn nằm
trong bảng.

### Các mode synthesis realtime

- `Normal traffic`: lấy mẫu từ reference, không inject bất thường.
- `Account takeover`: tăng new device, mismatch, failed attempts, IP distance.
- `High-value cashout`: tăng amount và chuyển type về `CASH_OUT`.
- `Bot burst`: giảm time gap, tăng transaction count và failed attempts.
- `Foreign IP wave`: tăng IP country mismatch và billing distance.
- `Mixed attack`: trộn nhiều pattern trong cùng stream.

### Lệnh dashboard đúng trên Linux

```bash
streamlit run data_balancing/monitor_app.py
```

Không dùng `data_balancing\\monitor_app.py` trên Linux.
"""
    )

    report = data.get("balancing_report", "")
    monitor_summary = data.get("monitoring_summary", "")
    if report:
        with st.expander("Balancing report", expanded=False):
            st.markdown(report)
    if monitor_summary:
        with st.expander("Monitoring summary", expanded=False):
            st.markdown(monitor_summary)


def main() -> None:
    data = load_outputs()
    config = sidebar(data)

    st.title("Fraud Balancing And Realtime Monitoring")
    st.caption("Nhiệm vụ 1: balancing/model comparison. Nhiệm vụ 2: realtime scoring và drift monitoring.")

    pages = ["Home", "Feature Drift", "Models", "Importance", "Config", "Guide"]
    if hasattr(st, "segmented_control"):
        page = st.segmented_control("Page", pages, default="Home", key="active_dashboard_page")
    else:
        page = st.radio("Page", pages, horizontal=True, key="active_dashboard_page")

    if page == "Home":
        home_tab(data, config)
    elif page == "Feature Drift":
        feature_drift_tab(config)
    elif page == "Models":
        models_tab(data)
    elif page == "Importance":
        importance_tab(data)
    elif page == "Config":
        config_tab(data, config)
    elif page == "Guide":
        guide_tab(data)


if __name__ == "__main__":
    main()
