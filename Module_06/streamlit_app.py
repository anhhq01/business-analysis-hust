"""
MODULE 6 - Fraud-analyst review-queue demo (Streamlit).

Simulates the daily life of the platform's risk team on top of the deployed
scoring policy:

  1. LIVE STREAM   - generate a batch of synthetic incoming transactions
                     (legit + fraud-shaped, using the same biased-signal logic
                     as the Module 1 generator), score each one in real time.
  2. REVIEW QUEUE  - every REVIEW/BLOCK decision lands in the analyst queue,
                     sorted by score; the analyst confirms fraud or releases
                     the order, and the app tracks workload + $ saved against
                     the simulation's ground truth.
  3. WHAT-IF       - score a single hand-crafted transaction and see which
                     risk signals fire.

Scoring backends (sidebar):
  * Local model  - imports Module_06/scoring.py directly (default; this is the
                   mode to use on Streamlit Community Cloud / HF Spaces, where
                   only one process runs).
  * Remote API   - POSTs to the FastAPI service (Module_06/api/main.py), e.g.
                   a Render deployment, demonstrating the decoupled setup.

Run locally (from the repo root):
  uv run streamlit run Module_06/app/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Make the repo root importable regardless of working directory.
_p = Path(__file__).resolve()
for _cand in [_p, *_p.parents]:
    if (_cand / "scripts" / "config.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from Module_06.scoring import DEFAULT_STEP, FRAUD_TYPES, FraudScorer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants for the stream simulator (mirrors Module 1's generator logic:
# each risky signal fires with a HIGHER probability for fraud rows and a LOW
# base rate for legit rows - no single signal separates the classes).
# ---------------------------------------------------------------------------
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "NG", "RU", "CN", "VN"]
COUNTRY_WEIGHTS = [0.30, 0.15, 0.12, 0.10, 0.10, 0.07, 0.06, 0.04, 0.03, 0.03]
LEGIT_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
LEGIT_TYPE_W = [0.08, 0.35, 0.34, 0.01, 0.22]  # approx PaySim mix

st.set_page_config(page_title="Fraud Review Queue", page_icon="🚨", layout="wide")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts ...")
def get_local_scorer() -> FraudScorer:
    return FraudScorer()


def score_remote(api_url: str, txn: dict, block_threshold: float | None) -> dict:
    payload = {"transaction": txn,
               "options": {"update_state": True,
                           "block_threshold": block_threshold}}
    r = requests.post(f"{api_url.rstrip('/')}/score", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def score_txn(txn: dict) -> dict:
    """Dispatch to the selected backend; returns the scoring-result dict."""
    blk = st.session_state.block_threshold if st.session_state.enable_block else None
    if st.session_state.backend == "Remote API":
        return score_remote(st.session_state.api_url, txn, blk)
    return get_local_scorer().score(txn, block_threshold=blk)


# ---------------------------------------------------------------------------
# Synthetic incoming-transaction generator (demo stream with ground truth)
# ---------------------------------------------------------------------------
def simulate_batch(n: int, fraud_rate: float, rng: np.random.Generator) -> list[dict]:
    txns = []
    step = st.session_state.sim_step
    for _ in range(n):
        fraud = rng.random() < fraud_rate
        if fraud:
            txn_type = rng.choice(list(FRAUD_TYPES))
            amount = float(np.round(rng.lognormal(10.0, 1.2), 2))
            age = float(int(np.clip(rng.gamma(2.0, 180.0)
                                    * rng.uniform(0.25, 0.75), 1, 3650)))
        else:
            txn_type = rng.choice(LEGIT_TYPES, p=LEGIT_TYPE_W)
            amount = float(np.round(rng.lognormal(7.0, 1.5), 2))
            age = float(int(np.clip(rng.gamma(2.0, 180.0), 1, 3650)))

        def biased(p_fraud, p_legit):
            return bool(rng.random() < (p_fraud if fraud else p_legit))

        home = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
        foreign = biased(0.60, 0.07)
        ip = rng.choice(COUNTRIES) if foreign else home
        dist = float(np.round(rng.exponential(2500.0) + 300.0
                              if ip != home else rng.exponential(120.0), 1))
        # fraud reuses a small pool of collector ("mule") destinations so the
        # velocity features get exercised; legit mostly hits fresh accounts
        # with some popular merchants mixed in.
        if fraud and rng.random() < 0.35:
            dest = rng.choice(st.session_state.mule_pool)
        elif not fraud and rng.random() < 0.20:
            dest = rng.choice(st.session_state.merchant_pool)
        else:
            dest = f"C{rng.integers(10**8, 10**9)}"

        txns.append({
            "transaction_id": f"txn_{uuid.uuid4().hex[:10]}",
            "type": str(txn_type),
            "amount": amount,
            "step": int(step),
            "account_age_days": age,
            "is_new_device": biased(0.70, 0.08),
            "shipping_billing_mismatch": biased(0.55, 0.10),
            "failed_payment_attempts": int(min(rng.poisson(2.5 if fraud else 0.3), 12)),
            "ip_billing_distance_km": dist,
            "home_billing_country": str(home),
            "ip_country": str(ip),
            "nameDest": str(dest),
            "_true_fraud": fraud,  # ground truth, revealed only after analyst action
        })
        if rng.random() < 0.25:  # advance the clock every few transactions
            step += 1
    st.session_state.sim_step = step
    return txns


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("backend", "Local model")
    ss.setdefault("api_url", "http://127.0.0.1:8000")
    ss.setdefault("enable_block", False)
    ss.setdefault("block_threshold", 0.90)
    ss.setdefault("queue", [])       # pending analyst cases (REVIEW/BLOCK)
    ss.setdefault("resolved", [])    # actioned cases
    ss.setdefault("scored_log", [])  # every scored txn (for the stream table)
    ss.setdefault("sim_step", DEFAULT_STEP)
    ss.setdefault("rng_seed", 42)
    if "mule_pool" not in ss:
        rng = np.random.default_rng(ss.rng_seed)
        ss.mule_pool = [f"C{rng.integers(10**8, 10**9)}" for _ in range(5)]
        ss.merchant_pool = [f"M{rng.integers(10**8, 10**9)}" for _ in range(8)]


init_state()

# ---------------------------------------------------------------------------
# Sidebar - backend + policy knobs + session stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Scoring backend")
    st.session_state.backend = st.radio(
        "Backend", ["Local model", "Remote API"],
        help="Local = in-process model (use on Streamlit Cloud). "
             "Remote = FastAPI service (e.g. Render).")
    if st.session_state.backend == "Remote API":
        st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)

    st.header("Decision policy")
    if st.session_state.backend == "Local model":
        pol = get_local_scorer().policy
        st.caption(f"**{pol['model']}** ({pol['variant']} variant) — review "
                   f"threshold **{pol['threshold']}** from Module 5 cost "
                   "optimisation (persisted, not re-derived).")
    st.session_state.enable_block = st.checkbox(
        "Enable auto-block tier (demo)", st.session_state.enable_block,
        help="NOT part of the persisted Module 5 policy — a knob for the "
             "Module 8 'manual review vs auto-block' recommendation.")
    if st.session_state.enable_block:
        st.session_state.block_threshold = st.slider(
            "Auto-block above score", 0.50, 1.00, st.session_state.block_threshold, 0.01)

    st.header("Session stats")
    n_scored = len(st.session_state.scored_log)
    n_queue = len(st.session_state.queue)
    n_res = len(st.session_state.resolved)
    st.metric("Transactions scored", n_scored)
    st.metric("Queue depth", n_queue)
    st.metric("Cases resolved", n_res)
    if st.button("Reset session"):
        for k in ["queue", "resolved", "scored_log", "mule_pool"]:
            st.session_state.pop(k, None)
        st.session_state.sim_step = DEFAULT_STEP
        if st.session_state.backend == "Local model":
            get_local_scorer().dest_history.reset()
        st.rerun()

st.title("🚨 Real-Time Fraud Detection — Analyst Review Queue")
st.caption("Module 6 demo: the Module 5 behavioural RandomForest scoring live "
           "transactions at the cost-optimal threshold. The threshold is "
           "deliberately low (recall ≈ 0.999): missed fraud costs its full "
           "amount, a false alarm costs a fixed review fee, so the policy "
           "prefers reviewing many to missing one.")

tab_stream, tab_queue, tab_whatif = st.tabs(
    ["📡 Live stream", "🗂️ Review queue", "🔍 What-if scoring"])

# ---------------------------------------------------------------------------
# Tab 1 - live stream simulator
# ---------------------------------------------------------------------------
with tab_stream:
    c1, c2, c3 = st.columns([1, 1, 2])
    n_new = c1.number_input("Transactions per batch", 5, 200, 25, 5)
    fraud_rate = c2.slider("Simulated fraud rate", 0.0, 0.30, 0.08, 0.01,
                           help="Inflated vs the real ~0.13% so the demo queue "
                                "fills within a few clicks.")
    if c3.button("▶ Generate & score batch", type="primary"):
        rng = np.random.default_rng(int(time.time()))
        batch = simulate_batch(int(n_new), float(fraud_rate), rng)
        prog = st.progress(0.0, "scoring ...")
        failed = False
        for i, txn in enumerate(batch):
            truth = txn.pop("_true_fraud")
            try:
                res = score_txn(txn)
            except requests.RequestException as e:
                st.error(f"API call failed: {e}")
                failed = True
                break
            rec = {**txn, **res, "_true_fraud": truth,
                   "scored_at": time.strftime("%H:%M:%S")}
            st.session_state.scored_log.append(rec)
            if res["decision"] in ("REVIEW", "BLOCK"):
                st.session_state.queue.append(rec)
            prog.progress((i + 1) / len(batch))
        prog.empty()
        if not failed:
            st.rerun()  # refresh so the sidebar stats include this batch

    log = st.session_state.scored_log
    if log:
        df = pd.DataFrame(log)
        dec_counts = df["decision"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Auto-approved (rule)", int(dec_counts.get("AUTO_APPROVE", 0)),
                  help="PAYMENT / DEBIT / CASH_IN: fraud never occurs in these "
                       "types — approved without scoring (Module 4 policy).")
        m2.metric("Approved (scored)", int(dec_counts.get("APPROVE", 0)))
        m3.metric("Sent to review", int(dec_counts.get("REVIEW", 0)))
        m4.metric("Auto-blocked", int(dec_counts.get("BLOCK", 0)))

        show = df[["scored_at", "transaction_id", "type", "amount",
                   "fraud_probability", "decision"]].iloc[::-1].head(200)
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "fraud_probability": st.column_config.NumberColumn(
                             "score", format="%.4f"),
                         "amount": st.column_config.NumberColumn(format="$%.2f"),
                     })
    else:
        st.info("Generate a batch to see the incoming stream.")

# ---------------------------------------------------------------------------
# Tab 2 - analyst review queue
# ---------------------------------------------------------------------------
with tab_queue:
    queue = st.session_state.queue
    if not queue:
        st.success("Queue is empty — generate a batch in the Live stream tab.")
    else:
        queue.sort(key=lambda r: -(r["fraud_probability"] or 0))
        qdf = pd.DataFrame(queue)[["transaction_id", "fraud_probability", "decision",
                                   "type", "amount", "home_billing_country",
                                   "ip_country", "risk_signals"]]
        st.dataframe(qdf, use_container_width=True, hide_index=True,
                     column_config={
                         "fraud_probability": st.column_config.NumberColumn(
                             "score", format="%.4f"),
                         "amount": st.column_config.NumberColumn(format="$%.2f"),
                         "risk_signals": st.column_config.ListColumn("risk signals"),
                     })

        st.subheader("Work a case")
        case_id = st.selectbox("Transaction", [r["transaction_id"] for r in queue])
        case = next(r for r in queue if r["transaction_id"] == case_id)
        left, right = st.columns([2, 1])
        with left:
            st.metric("Fraud score", f"{case['fraud_probability']:.4f}",
                      delta=f"threshold {case['threshold']}")
            st.write(f"**{case['type']}** — **${case['amount']:,.2f}** — "
                     f"account age {case['account_age_days']:.0f} d — "
                     f"{case['failed_payment_attempts']} failed attempts — "
                     f"{case['home_billing_country']} → {case['ip_country']} "
                     f"({case['ip_billing_distance_km']:,.0f} km)")
            if case["risk_signals"]:
                for s in case["risk_signals"]:
                    st.write(f"- ⚠️ {s}")
            else:
                st.write("- no discrete risk flags; score driven by "
                         "combination of continuous signals")
        with right:
            confirm = st.button("🛑 Confirm fraud (block)", type="primary",
                                use_container_width=True)
            release = st.button("✅ Release (legitimate)", use_container_width=True)
            if confirm or release:
                case["analyst_decision"] = "fraud" if confirm else "legit"
                case["correct"] = case["analyst_decision"] == \
                    ("fraud" if case["_true_fraud"] else "legit")
                st.session_state.resolved.append(case)
                st.session_state.queue = [r for r in queue
                                          if r["transaction_id"] != case_id]
                st.rerun()

    res = st.session_state.resolved
    if res:
        st.divider()
        st.subheader("Resolved cases — outcome vs simulation ground truth")
        rdf = pd.DataFrame(res)
        caught = rdf[(rdf["analyst_decision"] == "fraud") & rdf["_true_fraud"]]
        missed = rdf[(rdf["analyst_decision"] == "legit") & rdf["_true_fraud"]]
        false_alarm = rdf[(rdf["analyst_decision"] == "fraud") & ~rdf["_true_fraud"]]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Fraud caught", len(caught))
        k2.metric("Fraud $ saved", f"${caught['amount'].sum():,.0f}")
        k3.metric("Fraud released in error", len(missed),
                  delta=f"-${missed['amount'].sum():,.0f}" if len(missed) else None,
                  delta_color="inverse")
        k4.metric("Good orders blocked", len(false_alarm))
        precision_q = rdf["_true_fraud"].mean()
        st.caption(f"Queue precision this session: {precision_q:.1%} of reviewed "
                   "cases were truly fraudulent. At the real prevalence this is "
                   "~1.3% (Module 5) — the cost model accepts many reviews to "
                   "avoid missing high-value fraud.")

# ---------------------------------------------------------------------------
# Tab 3 - what-if scoring
# ---------------------------------------------------------------------------
with tab_whatif:
    st.caption("Hand-craft a transaction and see the live decision.")
    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        w_type = c1.selectbox("Type", LEGIT_TYPES, index=0)
        w_amount = c1.number_input("Amount ($)", 1.0, 10_000_000.0, 8500.0, 100.0)
        w_age = c1.number_input("Account age (days)", 1, 3650, 30)
        w_failed = c2.number_input("Failed payment attempts", 0, 12, 0)
        w_dist = c2.number_input("IP-billing distance (km)", 0.0, 20000.0, 50.0)
        w_hour = c2.slider("Hour of day", 0, 23, 14)
        w_home = c3.selectbox("Billing country", COUNTRIES, index=0)
        w_ip = c3.selectbox("IP country", COUNTRIES, index=0)
        w_newdev = c3.checkbox("New device")
        w_mismatch = c3.checkbox("Shipping/billing mismatch")
        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        txn = {
            "transaction_id": f"whatif_{uuid.uuid4().hex[:8]}",
            "type": w_type, "amount": float(w_amount),
            "step": int(st.session_state.sim_step),
            "hour_of_day": int(w_hour),
            "account_age_days": float(w_age),
            "is_new_device": bool(w_newdev),
            "shipping_billing_mismatch": bool(w_mismatch),
            "failed_payment_attempts": int(w_failed),
            "ip_billing_distance_km": float(w_dist),
            "home_billing_country": w_home, "ip_country": w_ip,
            "nameDest": "WHATIF_DEST",
        }
        try:
            res = score_txn(txn)
        except requests.RequestException as e:
            st.error(f"API call failed: {e}")
        else:
            badge = {"AUTO_APPROVE": "🟢 AUTO-APPROVE (rule)",
                     "APPROVE": "🟢 APPROVE",
                     "REVIEW": "🟠 SEND TO REVIEW",
                     "BLOCK": "🔴 AUTO-BLOCK"}[res["decision"]]
            a, b = st.columns(2)
            a.metric("Decision", badge.split(maxsplit=1)[1],
                     help=res["reason"])
            score_txt = ("n/a (not scored)" if res["fraud_probability"] is None
                         else f"{res['fraud_probability']:.4f}")
            b.metric("Fraud score", score_txt,
                     delta=f"review threshold {res['threshold']}")
            st.write(f"**Reason:** {res['reason']}")
            for s in res["risk_signals"]:
                st.write(f"- ⚠️ {s}")
            st.caption(f"Scoring latency: {res['latency_ms']} ms")
