"""
Module 6 - Model Deployment (fraud-analyst review queue demo).

A Streamlit interface that simulates the review queue a fraud analyst
works through. It does NOT load the model itself - it calls the FastAPI
scoring service (Module 6) over HTTP, exactly like a real front-end would
call the deployed model API. This keeps a clean separation between the
model service and the analyst UI.

Prerequisites:
    1. Start the API first:   uvicorn api:app --reload
    2. Then run this app:     streamlit run streamlit_app.py
       (from inside the `model/` folder)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

from data_prep import RAW_INPUT_COLUMNS, TYPE_CATEGORIES

# Default points at a locally running API; override with the API_URL env
# var (e.g. the Render/HF Spaces URL once deployed).
DEFAULT_API_URL = os.environ.get("API_URL", "https://business-analysis-hust.onrender.com/")

st.set_page_config(page_title="Fraud Review Queue", page_icon="🛡️", layout="wide")


def make_sample_queue(n: int, seed: int = 0) -> pd.DataFrame:
    """Generate a small batch of plausible transactions to review."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "step": rng.integers(1, 743, n),
            "type": rng.choice(TYPE_CATEGORIES, n, p=[0.22, 0.35, 0.01, 0.34, 0.08]),
            "amount": rng.lognormal(mean=10, sigma=1.5, size=n).round(2),
        }
    )
    # Build loosely consistent balances; some rows will look anomalous.
    df["oldbalanceOrg"] = (df["amount"] * rng.uniform(0.5, 2.0, n)).round(2)
    df["newbalanceOrig"] = np.maximum(0, df["oldbalanceOrg"] - df["amount"]).round(2)
    df["oldbalanceDest"] = (df["amount"] * rng.uniform(0, 3.0, n)).round(2)
    df["newbalanceDest"] = (df["oldbalanceDest"] + df["amount"] * rng.uniform(0, 1.2, n)).round(2)
    return df[RAW_INPUT_COLUMNS]


def check_api(api_url: str) -> tuple[bool, str]:
    """Ping the API health endpoint."""
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        r.raise_for_status()
        return True, r.json().get("status", "unknown")
    except requests.RequestException as exc:
        return False, str(exc)


def score_via_api(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    """Send the queue to the API's /score_batch endpoint and merge results."""
    payload = df.to_dict(orient="records")
    r = requests.post(f"{api_url}/score_batch", json=payload, timeout=30)
    r.raise_for_status()
    results = pd.DataFrame(r.json())

    scored = df.copy().reset_index(drop=True)
    scored["fraud_probability"] = results["fraud_probability"]
    scored["decision"] = results["decision"]
    return scored.sort_values("fraud_probability", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
st.title("🛡️ Fraud-Analyst Review Queue")
st.caption("Module 6 demo — transactions scored in real time by the FastAPI model service.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", DEFAULT_API_URL).rstrip("/")
    ok, info = check_api(api_url)
    if ok:
        st.success(f"API reachable ({info})")
    else:
        st.error("API not reachable. Start it with `uvicorn api:app --reload`.")
        st.caption(info)
    n = st.number_input("Transactions in queue", 5, 200, 25, 5)
    seed = st.number_input("Sample seed", 0, 9999, 0, 1)

if not ok:
    st.stop()

queue_raw = make_sample_queue(int(n), int(seed))

try:
    queue = score_via_api(queue_raw, api_url)
except requests.RequestException as exc:
    st.error(f"Scoring request failed: {exc}")
    st.stop()

flagged = int((queue["decision"] != "APPROVE").sum())
c1, c2, c3 = st.columns(3)
c1.metric("Transactions", len(queue))
c2.metric("Flagged for review", flagged)
c3.metric("Auto-approved", len(queue) - flagged)

st.subheader("Review queue (highest risk first)")
st.dataframe(
    queue.style.format({"amount": "{:,.2f}", "fraud_probability": "{:.2%}"}),
    use_container_width=True,
    height=420,
)

st.subheader("Inspect a single transaction")
idx = st.number_input("Row to inspect", 0, len(queue) - 1, 0, 1)
case = queue.iloc[int(idx)]
left, right = st.columns([2, 1])
with left:
    st.json(case[RAW_INPUT_COLUMNS].to_dict())
with right:
    st.metric("Fraud probability", f"{case['fraud_probability']:.2%}")
    if case["decision"] == "APPROVE":
        st.success(f"Decision: {case['decision']}")
    else:
        st.error(f"Decision: {case['decision']}")
    st.button("✅ Approve", key="approve")
    st.button("🚫 Confirm fraud", key="confirm")
