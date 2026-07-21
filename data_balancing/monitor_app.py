"""
Streamlit dashboard for fraud balancing and drift monitoring outputs.

Run from the repository root:
    streamlit run data_balancing/monitor_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from monitor import (
    NUMERIC_MONITOR_COLUMNS,
    REFERENCE_PATH,
    categorical_drift,
    compute_drift_report,
    load_feature_frame,
    population_stability_index,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data_balancing" / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


st.set_page_config(
    page_title="Fraud Monitoring",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def path(name: str) -> Path:
    return OUTPUT_DIR / name


def figure_path(name: str) -> Path:
    return FIGURE_DIR / name


@st.cache_data(show_spinner=False)
def read_csv(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def read_text(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return ""
    return p.read_text(encoding="utf-8")


def show_missing(file_name: str) -> None:
    st.info(f"`{file_name}` has not been generated yet.")


def metric_card(label: str, value, help_text: str | None = None) -> None:
    st.metric(label, value if value is not None else "N/A", help=help_text)


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "model_results": read_csv(str(path("balancing_model_results.csv"))),
        "review_results": read_csv(str(path("fraud_review_results.csv"))),
        "drift_report": read_csv(str(path("monitoring_drift_report.csv"))),
        "feature_scores": read_csv(str(path("feature_target_scores.csv"))),
        "permutation": read_csv(str(path("model_permutation_importance.csv"))),
        "dropped_features": read_csv(str(path("correlated_features_dropped.csv"))),
    }


@st.cache_data(show_spinner=False)
def load_reference_rows(max_rows: int) -> pd.DataFrame:
    if REFERENCE_PATH.exists():
        return pd.read_parquet(REFERENCE_PATH).head(max_rows)
    cleaned = ROOT / "feature_engineering" / "fraud-detection" / "data" / "processed" / "transactions_cleaned.parquet"
    if not cleaned.exists():
        return pd.DataFrame()
    return load_feature_frame(cleaned, max_rows=max_rows, seed=42, source_max_rows=max_rows)


def inject_attack(batch: pd.DataFrame, scenario: str, intensity: float, rng: np.random.Generator) -> pd.DataFrame:
    attacked = batch.copy()
    if attacked.empty:
        return attacked

    n_attack = max(1, int(round(len(attacked) * intensity)))
    attack_idx = rng.choice(attacked.index.to_numpy(), size=min(n_attack, len(attacked)), replace=False)

    if scenario == "Account takeover":
        for col in ["is_new_device", "shipping_billing_mismatch", "ip_country_mismatch"]:
            if col in attacked.columns:
                attacked.loc[attack_idx, col] = 1
        if "failed_payment_attempts" in attacked.columns:
            attacked.loc[attack_idx, "failed_payment_attempts"] = rng.integers(3, 9, size=len(attack_idx))
        if "ip_billing_distance_km" in attacked.columns:
            attacked.loc[attack_idx, "ip_billing_distance_km"] *= rng.uniform(4, 12, size=len(attack_idx))

    elif scenario == "High-value cashout":
        amount_factor = rng.uniform(4, 15, size=len(attack_idx))
        for col in ["amount", "orig_balance_delta", "dest_balance_delta"]:
            if col in attacked.columns:
                attacked.loc[attack_idx, col] *= amount_factor
        if "amount_to_oldOrg_ratio" in attacked.columns:
            attacked.loc[attack_idx, "amount_to_oldOrg_ratio"] *= rng.uniform(3, 10, size=len(attack_idx))
        if "type" in attacked.columns:
            attacked.loc[attack_idx, "type"] = "CASH_OUT"

    elif scenario == "Bot burst":
        if "time_since_prev_orig" in attacked.columns:
            attacked.loc[attack_idx, "time_since_prev_orig"] = rng.integers(0, 2, size=len(attack_idx))
        if "tx_count_prev_orig" in attacked.columns:
            attacked.loc[attack_idx, "tx_count_prev_orig"] += rng.integers(10, 80, size=len(attack_idx))
        if "failed_payment_attempts" in attacked.columns:
            attacked.loc[attack_idx, "failed_payment_attempts"] += rng.integers(2, 7, size=len(attack_idx))

    elif scenario == "Foreign IP wave":
        if "ip_country_mismatch" in attacked.columns:
            attacked.loc[attack_idx, "ip_country_mismatch"] = 1
        if "ip_billing_distance_km" in attacked.columns:
            attacked.loc[attack_idx, "ip_billing_distance_km"] = rng.uniform(3000, 16000, size=len(attack_idx))
        if "ip_country" in attacked.columns:
            attacked.loc[attack_idx, "ip_country"] = rng.choice(["RU", "CN", "NG", "BR"], size=len(attack_idx))

    if "log_amount" in attacked.columns and "amount" in attacked.columns:
        attacked["log_amount"] = np.log1p(pd.to_numeric(attacked["amount"], errors="coerce").clip(lower=0))
    return attacked.replace([np.inf, -np.inf], np.nan).fillna(0)


def simulate_attack_stream(
    reference: pd.DataFrame,
    scenario: str,
    batches: int,
    batch_size: int,
    intensity: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    last_batch = pd.DataFrame()
    numeric_cols = [col for col in NUMERIC_MONITOR_COLUMNS if col in reference.columns]
    focus_features = [
        "failed_payment_attempts",
        "ip_billing_distance_km",
        "amount_to_oldOrg_ratio",
        "time_since_prev_orig",
        "tx_count_prev_orig",
        "is_new_device",
        "ip_country_mismatch",
    ]
    focus_features = [col for col in focus_features if col in reference.columns]

    simulated_clock = pd.Timestamp("2026-07-21 09:00:00")
    for batch_no in range(1, batches + 1):
        replace = len(reference) < batch_size
        batch = reference.sample(n=batch_size, replace=replace, random_state=seed + batch_no).copy()
        attack_intensity = intensity * (batch_no / batches)
        batch = inject_attack(batch, scenario, attack_intensity, rng)
        last_batch = batch

        psi_scores = {
            col: population_stability_index(reference[col], batch[col])
            for col in numeric_cols
        }
        max_psi_feature = max(psi_scores, key=psi_scores.get) if psi_scores else ""
        max_psi = psi_scores[max_psi_feature] if psi_scores else 0.0
        alert_count = sum(score >= 0.25 for score in psi_scores.values())
        warning_count = sum(0.10 <= score < 0.25 for score in psi_scores.values())

        row = {
            "batch": batch_no,
            "time": simulated_clock + pd.Timedelta(minutes=5 * (batch_no - 1)),
            "attack_intensity": attack_intensity,
            "max_psi": max_psi,
            "max_psi_feature": max_psi_feature,
            "alert_features": alert_count,
            "warning_features": warning_count,
            "threat_level": threat_level(max_psi, alert_count, warning_count),
        }
        for col in focus_features:
            if pd.api.types.is_numeric_dtype(reference[col]):
                row[f"{col}_psi"] = population_stability_index(reference[col], batch[col])
            else:
                row[f"{col}_shift"] = categorical_drift(reference[col], batch[col])
        rows.append(row)

    return pd.DataFrame(rows), last_batch


def threat_level(max_psi: float, alerts: int, warnings: int) -> str:
    if alerts >= 3 or max_psi >= 0.50:
        return "critical"
    if alerts >= 1 or max_psi >= 0.25:
        return "alert"
    if warnings >= 1 or max_psi >= 0.10:
        return "warning"
    return "normal"


def threat_color(level: str) -> str:
    return {
        "critical": "#8B0000",
        "alert": "#D62728",
        "warning": "#F59E0B",
        "normal": "#2CA02C",
    }.get(level, "#6B7280")


def build_event_log(timeline: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame()
    events = timeline[
        ["time", "batch", "threat_level", "max_psi_feature", "max_psi", "alert_features", "warning_features"]
    ].copy()
    events["event"] = np.where(
        events["threat_level"].isin(["alert", "critical"]),
        "Drift alert",
        np.where(events["threat_level"] == "warning", "Early warning", "Normal traffic"),
    )
    return events.sort_values("batch", ascending=False)


def sidebar_status(data: dict[str, pd.DataFrame]) -> None:
    st.sidebar.title("Fraud Monitor")
    st.sidebar.caption("Reads generated outputs from `data_balancing/outputs`.")
    if st.sidebar.button("Refresh dashboard data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Artifacts")
    for file_name in [
        "balancing_model_results.csv",
        "monitoring_drift_report.csv",
        "model_permutation_importance.csv",
        "monitoring_summary.md",
    ]:
        exists = path(file_name).exists()
        st.sidebar.write(("OK " if exists else "Missing ") + file_name)


def overview_tab(data: dict[str, pd.DataFrame]) -> None:
    model_results = data["model_results"]
    drift = data["drift_report"]

    st.subheader("Operational Status")
    c1, c2, c3, c4 = st.columns(4)
    if not model_results.empty:
        best = model_results.sort_values(["auc_pr", "f1"], ascending=False).iloc[0]
        c1.metric("Best model", f"{best['strategy']} / {best['model']}")
        c2.metric("AUC-PR", f"{best['auc_pr']:.4f}")
        c3.metric("Recall", f"{best['recall']:.4f}")
        c4.metric("Business cost", f"{best['business_cost']:,.0f}")
    else:
        for col, label in zip(c1, ["Best model"]):
            col.metric(label, "N/A")
        c2.metric("AUC-PR", "N/A")
        c3.metric("Recall", "N/A")
        c4.metric("Business cost", "N/A")

    st.subheader("Drift Status")
    d1, d2, d3 = st.columns(3)
    if not drift.empty and "status" in drift.columns:
        counts = drift["status"].value_counts()
        d1.metric("Alert features", int(counts.get("alert", 0)))
        d2.metric("Warning features", int(counts.get("warning", 0)))
        d3.metric("OK features", int(counts.get("ok", 0)))
    else:
        d1.metric("Alert features", "N/A")
        d2.metric("Warning features", "N/A")
        d3.metric("OK features", "N/A")

    fig = figure_path("monitoring_drift_summary.png")
    if fig.exists():
        st.image(str(fig), use_container_width=True)
    else:
        show_missing("figures/monitoring_drift_summary.png")


def drift_tab(data: dict[str, pd.DataFrame]) -> None:
    drift = data["drift_report"]
    st.subheader("Data Drift")
    st.caption("Numeric features use PSI and KS statistic. Categorical features use distribution shift.")

    if drift.empty:
        show_missing("monitoring_drift_report.csv")
        st.code(
            "python data_balancing\\monitor.py --max-current-rows 50000",
            language="powershell",
        )
        return

    status_filter = st.multiselect(
        "Status",
        sorted(drift["status"].dropna().unique()),
        default=sorted(drift["status"].dropna().unique()),
    )
    filtered = drift[drift["status"].isin(status_filter)] if status_filter else drift
    score = filtered["psi"].fillna(filtered["category_distribution_shift"])
    plot_df = filtered.assign(drift_score=score).sort_values("drift_score", ascending=False)

    fig = px.bar(
        plot_df.head(25),
        x="drift_score",
        y="feature",
        color="status",
        orientation="h",
        title="Top Drift Signals",
        labels={"drift_score": "PSI or categorical shift", "feature": ""},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=650)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(plot_df, use_container_width=True, hide_index=True)


def model_tab(data: dict[str, pd.DataFrame]) -> None:
    results = data["model_results"]
    review = data["review_results"]

    st.subheader("Balancing and Model Results")
    if results.empty:
        show_missing("balancing_model_results.csv")
        return

    metric = st.selectbox("Metric", ["auc_pr", "precision", "recall", "f1", "business_cost"])
    fig = px.bar(
        results,
        x="strategy",
        y=metric,
        color="model",
        barmode="group",
        title=f"{metric} by balancing strategy and model",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.subheader("Fraud-Focused Review Set")
    st.caption("Auxiliary set for recall/fraud capture inspection, not the main production scorecard.")
    if review.empty:
        show_missing("fraud_review_results.csv")
    else:
        st.dataframe(review, use_container_width=True, hide_index=True)


def feature_tab(data: dict[str, pd.DataFrame]) -> None:
    scores = data["feature_scores"]
    permutation = data["permutation"]
    dropped = data["dropped_features"]

    st.subheader("Feature Importance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model-free feature signal**")
        if scores.empty:
            show_missing("feature_target_scores.csv")
        else:
            fig = px.bar(
                scores.head(20).sort_values("mutual_info"),
                x="mutual_info",
                y="feature",
                orientation="h",
                title="Mutual information with isFraud",
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Best-model dependency**")
        if permutation.empty:
            show_missing("model_permutation_importance.csv")
        else:
            fig = px.bar(
                permutation.head(20).sort_values("importance_mean"),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Permutation importance",
                error_x="importance_std" if "importance_std" in permutation.columns else None,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation-Based Feature Selection")
    if dropped.empty:
        st.caption("No highly correlated features were dropped in the latest run.")
    else:
        st.dataframe(dropped, use_container_width=True, hide_index=True)


def report_tab() -> None:
    st.subheader("Reports")
    report = read_text(str(path("balancing_report.md")))
    monitor = read_text(str(path("monitoring_summary.md")))

    if report:
        with st.expander("Balancing report", expanded=True):
            st.markdown(report)
    else:
        show_missing("balancing_report.md")

    if monitor:
        with st.expander("Monitoring summary", expanded=True):
            st.markdown(monitor)
    else:
        show_missing("monitoring_summary.md")

    st.subheader("Refresh Commands")
    st.code(
        "\n".join(
            [
                "python data_balancing\\balancing_experiments.py --max-rows 0 --split stratified",
                "python data_balancing\\monitor.py --build-reference --max-reference-rows 100000",
                "python data_balancing\\monitor.py --max-current-rows 50000",
            ]
        ),
        language="powershell",
    )


def attack_simulator_tab() -> None:
    st.subheader("Realtime Attack Simulator")
    st.caption(
        "Simulates incoming transaction batches, injects an attack pattern, and updates drift alerts per batch."
    )

    reference_rows = st.slider("Reference rows loaded", 1_000, 50_000, 5_000, step=1_000)
    reference = load_reference_rows(reference_rows)
    if reference.empty:
        show_missing("monitoring_reference.parquet")
        st.code("python data_balancing\\monitor.py --build-reference --max-reference-rows 100000")
        return

    c1, c2, c3, c4 = st.columns(4)
    scenario = c1.selectbox(
        "Attack scenario",
        ["Account takeover", "High-value cashout", "Bot burst", "Foreign IP wave"],
    )
    batches = c2.slider("Batches", 5, 60, 20)
    batch_size = c3.slider("Batch size", 100, 5_000, 500, step=100)
    intensity = c4.slider("Final attack share", 0.05, 0.80, 0.30, step=0.05)
    seed = st.number_input("Simulation seed", value=42, step=1)

    if st.button("Run realtime simulation", type="primary"):
        timeline, last_batch = simulate_attack_stream(
            reference,
            scenario,
            batches,
            batch_size,
            intensity,
            int(seed),
        )
        st.session_state["attack_timeline"] = timeline
        st.session_state["attack_last_batch"] = last_batch
        st.session_state["attack_reference"] = reference

    timeline = st.session_state.get("attack_timeline", pd.DataFrame())
    if timeline.empty:
        st.info("Choose a scenario and run the simulation.")
        return

    latest = timeline.iloc[-1]
    level = latest["threat_level"]
    st.markdown(
        f"""
        <div style="border-left: 8px solid {threat_color(level)}; padding: 12px 16px; background: #f8fafc; border-radius: 8px;">
            <div style="font-size: 13px; color: #475569;">Current simulated status</div>
            <div style="font-size: 30px; font-weight: 700; color: {threat_color(level)};">{level.upper()}</div>
            <div style="font-size: 14px; color: #334155;">Most threatened feature: <b>{latest['max_psi_feature']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current time", str(pd.to_datetime(latest["time"]).strftime("%H:%M")))
    m2.metric("Current batch", int(latest["batch"]))
    m3.metric("Max PSI", f"{latest['max_psi']:.3f}", latest["max_psi_feature"])
    m4.metric("Alert features", int(latest["alert_features"]))
    m5.metric("Warning features", int(latest["warning_features"]))

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(latest["max_psi"]),
            number={"valueformat": ".3f"},
            title={"text": "Threat Gauge: Max PSI"},
            gauge={
                "axis": {"range": [0, max(0.6, float(timeline["max_psi"].max()) * 1.1)]},
                "bar": {"color": threat_color(level)},
                "steps": [
                    {"range": [0, 0.10], "color": "#DCFCE7"},
                    {"range": [0.10, 0.25], "color": "#FEF3C7"},
                    {"range": [0.25, 0.50], "color": "#FEE2E2"},
                    {"range": [0.50, max(0.6, float(timeline["max_psi"].max()) * 1.1)], "color": "#FCA5A5"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 0.25},
            },
        )
    )
    gauge.update_layout(height=260, margin={"l": 20, "r": 20, "t": 40, "b": 10})

    fig = px.line(
        timeline,
        x="time",
        y="max_psi",
        markers=True,
        color="threat_level",
        color_discrete_map={
            "normal": "#2CA02C",
            "warning": "#F59E0B",
            "alert": "#D62728",
            "critical": "#8B0000",
        },
        hover_data=["batch", "max_psi_feature", "alert_features", "warning_features", "attack_intensity"],
        title="Realtime Drift Alert Timeline",
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="warning")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="alert")
    fig.update_layout(height=360)

    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.plotly_chart(fig, use_container_width=True)

    psi_cols = [col for col in timeline.columns if col.endswith("_psi") or col.endswith("_shift")]
    if psi_cols:
        long = timeline.melt(
            id_vars=["batch"],
            value_vars=psi_cols,
            var_name="feature_metric",
            value_name="drift_score",
        )
        fig2 = px.line(
            long,
            x="batch",
            y="drift_score",
            color="feature_metric",
            title="Focused Feature Drift During Attack",
        )
        fig2.add_hline(y=0.25, line_dash="dash", line_color="red")
        fig2.update_layout(height=420)

        heatmap_df = long.pivot(index="feature_metric", columns="batch", values="drift_score").fillna(0)
        heatmap = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Threat Heatmap: Feature Drift by Batch",
            labels={"x": "Batch", "y": "Feature", "color": "Drift"},
        )
        heatmap.update_layout(height=420)

        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(fig2, use_container_width=True)
        with h2:
            st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Threat Event Log")
    events = build_event_log(timeline)
    event_filter = st.multiselect(
        "Threat levels",
        ["critical", "alert", "warning", "normal"],
        default=["critical", "alert", "warning", "normal"],
    )
    if event_filter:
        events = events[events["threat_level"].isin(event_filter)]
    st.dataframe(events, use_container_width=True, hide_index=True)

    st.subheader("Threat Detail")
    reference = st.session_state.get("attack_reference", reference)
    last_batch = st.session_state.get("attack_last_batch", pd.DataFrame())
    latest_report = compute_drift_report(reference, last_batch, 0.10, 0.25, 0.10)
    latest_report = latest_report.copy()
    latest_report["score"] = latest_report["psi"].fillna(latest_report["category_distribution_shift"])
    latest_report = latest_report.sort_values("score", ascending=False)
    threat_features = latest_report["feature"].tolist()
    selected_feature = st.selectbox(
        "Select a threatening feature",
        threat_features,
        index=0 if threat_features else None,
    )
    if selected_feature:
        feature_row = latest_report[latest_report["feature"] == selected_feature].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Status", feature_row["status"])
        d2.metric("PSI", "N/A" if pd.isna(feature_row["psi"]) else f"{feature_row['psi']:.3f}")
        d3.metric("KS", "N/A" if pd.isna(feature_row["ks_statistic"]) else f"{feature_row['ks_statistic']:.3f}")
        d4.metric(
            "Category shift",
            "N/A"
            if pd.isna(feature_row["category_distribution_shift"])
            else f"{feature_row['category_distribution_shift']:.3f}",
        )

        compare = pd.concat(
            [
                reference[[selected_feature]].assign(window="reference"),
                last_batch[[selected_feature]].assign(window="current attack batch"),
            ],
            ignore_index=True,
        )
        if pd.api.types.is_numeric_dtype(compare[selected_feature]):
            detail_fig = px.histogram(
                compare,
                x=selected_feature,
                color="window",
                nbins=40,
                barmode="overlay",
                marginal="box",
                title=f"Reference vs Current: {selected_feature}",
            )
        else:
            detail = (
                compare.groupby(["window", selected_feature])
                .size()
                .reset_index(name="rows")
            )
            detail_fig = px.bar(
                detail,
                x=selected_feature,
                y="rows",
                color="window",
                barmode="group",
                title=f"Reference vs Current: {selected_feature}",
            )
        detail_fig.update_layout(height=420)
        st.plotly_chart(detail_fig, use_container_width=True)

    with st.expander("Latest simulated batch drift report"):
        st.dataframe(latest_report, use_container_width=True, hide_index=True)


def main() -> None:
    data = load_data()
    sidebar_status(data)

    st.title("Real-Time Fraud Monitoring Dashboard")
    st.caption("Model comparison, feature importance and data drift monitoring for payment fraud detection.")

    tabs = st.tabs(["Overview", "Realtime Attack", "Drift", "Models", "Features", "Reports"])
    with tabs[0]:
        overview_tab(data)
    with tabs[1]:
        attack_simulator_tab()
    with tabs[2]:
        drift_tab(data)
    with tabs[3]:
        model_tab(data)
    with tabs[4]:
        feature_tab(data)
    with tabs[5]:
        report_tab()


if __name__ == "__main__":
    main()
