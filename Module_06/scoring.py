"""
MODULE 6 - Shared scoring engine (used by BOTH the FastAPI service and the
Streamlit demo, so the two surfaces can never disagree on feature logic).

Reproduces, for a single live transaction, the exact Module 4 feature pipeline
that produced the training matrices:

  raw transaction fields
    -> derived risk signals (geo, behavioural, cyclical time)
    -> causal destination velocity (prior count / steps since last seen)
    -> RobustScaler transform with the statistics FIT ON THE TRAINING FOLD
    -> the 37-column behavioural feature vector of models/decision_threshold.json
    -> RandomForest probability -> policy decision at the persisted threshold

Design notes / deliberate choices
---------------------------------
* Artifacts are loaded from models/ (best_model.joblib, scaler.joblib,
  decision_threshold.json). Nothing is re-derived at serve time: the deployed
  decision is exactly the Module 5 policy (auditable, stable).
* Transaction types outside the fraud universe (PAYMENT / DEBIT / CASH_IN) are
  AUTO-APPROVED by rule, never scored - fraud never occurs in these types in
  the data, and Module 4 recorded this as an explicit policy decision.
* Destination velocity needs state ("how often have we seen this nameDest
  before?"). Here it is an in-process dict - fine for a demo / single worker.
  In production this would be a shared store (e.g. Redis) keyed the same way.
  Callers that keep their own state can pass dest_prior_count /
  dest_steps_since_last explicitly to override the store.
* The scaler was fit on 18 continuous columns, some of which are PaySim
  balance ("mechanical") features that the deployed behavioural model does NOT
  use. We still build those columns (from optional balance inputs, else 0) so
  the scaler receives the exact column set it was fit on; their values cannot
  influence the model because they are dropped before predict_proba.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locate the repo root (same idiom as the Module 4/5 notebooks)
# ---------------------------------------------------------------------------
def find_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / "scripts" / "config.py").exists():
            return cand
    return p


ROOT = find_root()
MODELS_DIR = ROOT / "models"

# Fraud in PaySim only ever occurs in these two types (see scripts/config.py;
# re-declared here so the deployed service has no import beyond the artifacts).
FRAUD_TYPES = ("TRANSFER", "CASH_OUT")
ALL_TYPES = ("TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN")

# One step past the end of the PaySim horizon (steps 1-743). Used when a live
# request does not carry a step; scaled `step` extrapolates linearly, which is
# exactly what "a transaction later than all training data" should look like.
DEFAULT_STEP = 744

# Human-readable explanations for the review queue. Each entry:
#   (feature flag in the built row, label shown to the analyst)
RISK_SIGNAL_LABELS = [
    ("is_new_device", "transaction from a device never used by this account"),
    ("shipping_billing_mismatch", "shipping address differs from billing address"),
    ("ip_country_mismatch", "IP country differs from billing country"),
    ("young_account", "account is young (<= 90 days old)"),
    ("high_failed_attempts", "3+ failed payment attempts before this transaction"),
    ("is_night", "night-time transaction (00:00-05:59)"),
    ("dest_first_seen", "first transaction ever sent to this destination"),
]


class DestinationHistory:
    """Causal per-destination velocity state (thread-safe).

    Mirrors Module 4 exactly: for each row, `dest_prior_count` counts only
    EARLIER transactions to that nameDest, and `dest_steps_since_last` is the
    step gap to the previous sighting (-1 sentinel = never seen). The store is
    read BEFORE it is updated with the current transaction, so the features
    stay causal at serve time too.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, tuple[int, int]] = {}  # name -> (count, last_step)

    def peek(self, name_dest: str, step: int) -> tuple[int, float, int]:
        """Return (prior_count, steps_since_last, first_seen) without updating."""
        with self._lock:
            if name_dest in self._state:
                count, last_step = self._state[name_dest]
                return count, float(step - last_step), 0
        return 0, -1.0, 1

    def update(self, name_dest: str, step: int) -> None:
        with self._lock:
            count, _ = self._state.get(name_dest, (0, step))
            self._state[name_dest] = (count + 1, step)

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def __len__(self) -> int:
        return len(self._state)


class FraudScorer:
    """Loads the Module 4/5 artifacts once and scores raw transactions."""

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        models_dir = Path(models_dir)
        self.policy = json.loads((models_dir / "decision_threshold.json").read_text())
        self.model = joblib.load(models_dir / "best_model.joblib")
        bundle = joblib.load(models_dir / "scaler.joblib")
        self.scaler = bundle["scaler"]
        self.scaler_cols: list[str] = list(bundle["continuous"])
        self.features: list[str] = list(self.policy["features"])
        self.threshold: float = float(self.policy["threshold"])

        # Countries the model knows, recovered from its own one-hot columns so
        # the serving contract can never drift from the trained feature set.
        self.home_countries = sorted(c[len("home_"):] for c in self.features
                                     if c.startswith("home_"))
        self.ip_countries = sorted(c[len("ip_"):] for c in self.features
                                   if c.startswith("ip_"))
        self.dest_history = DestinationHistory()

    # -- feature construction ------------------------------------------------
    def build_features(self, txn: dict) -> pd.DataFrame:
        """Raw transaction dict -> one-row DataFrame in training column order.

        Expected keys (see api/main.py for the validated schema):
          type, amount, account_age_days, is_new_device,
          shipping_billing_mismatch, failed_payment_attempts,
          ip_billing_distance_km, home_billing_country, ip_country
        Optional: step, hour_of_day, nameDest, dest_prior_count,
          dest_steps_since_last, oldbalanceOrg, newbalanceOrig,
          oldbalanceDest, newbalanceDest
        """
        step = int(txn.get("step") or DEFAULT_STEP)
        # Module 1 derives hour from step; honour an explicit hour if given.
        hour = int(txn["hour_of_day"]) if txn.get("hour_of_day") is not None else step % 24

        home = str(txn["home_billing_country"])
        ip = str(txn["ip_country"])
        if home not in self.home_countries:
            raise ValueError(f"unknown home_billing_country '{home}' "
                             f"(model knows: {self.home_countries})")
        if ip not in self.ip_countries:
            raise ValueError(f"unknown ip_country '{ip}' "
                             f"(model knows: {self.ip_countries})")

        # Destination velocity: caller-supplied override wins, else the store.
        if txn.get("dest_prior_count") is not None:
            prior = int(txn["dest_prior_count"])
            since = float(txn.get("dest_steps_since_last", -1.0))
            first = int(since < 0 and prior == 0)
        else:
            prior, since, first = self.dest_history.peek(
                str(txn.get("nameDest") or "UNKNOWN_DEST"), step)

        amount = float(txn["amount"])
        dist = float(txn["ip_billing_distance_km"])
        age = float(txn["account_age_days"])
        failed = float(txn["failed_payment_attempts"])
        old_org = float(txn.get("oldbalanceOrg") or 0.0)
        new_org = float(txn.get("newbalanceOrig") or 0.0)
        old_dst = float(txn.get("oldbalanceDest") or 0.0)
        new_dst = float(txn.get("newbalanceDest") or 0.0)

        row = {
            # raw context (some only feed the scaler, not the model)
            "step": step,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dst,
            "newbalanceDest": new_dst,
            "account_age_days": age,
            "is_new_device": int(bool(txn["is_new_device"])),
            "shipping_billing_mismatch": int(bool(txn["shipping_billing_mismatch"])),
            "failed_payment_attempts": failed,
            "ip_billing_distance_km": dist,
            # Module 4 derived features, formulas copied verbatim
            "is_night": int(0 <= hour <= 5),
            "log_amount": np.log10(amount + 1.0),
            "orig_balance_err": old_org - amount - new_org,
            "dest_balance_err": old_dst + amount - new_dst,
            "amount_to_oldbalanceOrg": amount / (old_org + 1.0),
            "ip_country_mismatch": int(ip != home),
            "log_ip_distance": np.log10(dist + 1.0),
            "young_account": int(age <= 90),
            "high_failed_attempts": int(failed >= 3),
            "dest_prior_count": prior,
            "dest_steps_since_last": since,
            "dest_first_seen": first,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_transfer": int(txn["type"] == "TRANSFER"),
        }
        for c in self.home_countries:
            row[f"home_{c}"] = int(home == c)
        for c in self.ip_countries:
            row[f"ip_{c}"] = int(ip == c)

        df = pd.DataFrame([row])
        # Scale with the training-time statistics, on the exact fitted columns.
        df[self.scaler_cols] = self.scaler.transform(df[self.scaler_cols])
        return df

    # -- scoring -------------------------------------------------------------
    def score(self, txn: dict, update_state: bool = True,
              block_threshold: float | None = None) -> dict:
        """Score one raw transaction and apply the persisted decision policy.

        Decisions:
          AUTO_APPROVE - type outside the fraud universe (rule, never scored)
          APPROVE      - scored below the policy threshold
          REVIEW       - scored at/above the policy threshold -> analyst queue
          BLOCK        - optional demo tier: at/above `block_threshold`
                         (NOT part of the persisted Module 5 policy; a knob for
                         the Module 8 "manual review vs auto-block" discussion)
        """
        t0 = time.perf_counter()
        txn_type = str(txn["type"])
        if txn_type not in ALL_TYPES:
            raise ValueError(f"unknown transaction type '{txn_type}' (expected {ALL_TYPES})")

        if txn_type not in FRAUD_TYPES:
            return {
                "fraud_probability": None,
                "decision": "AUTO_APPROVE",
                "threshold": self.threshold,
                "risk_signals": [],
                "reason": (f"type={txn_type} is outside the fraud universe "
                           f"{list(FRAUD_TYPES)}; auto-approved by rule "
                           "(0 observed fraud in these types)"),
                "model": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # Resolve destination velocity HERE (raw values), so the analyst-facing
        # explanation can use the raw prior count - the scaled copy that feeds
        # the model is centred on the training median and can be negative.
        txn = dict(txn)
        step = int(txn.get("step") or DEFAULT_STEP)
        caller_state = txn.get("dest_prior_count") is not None
        if caller_state:
            raw_prior = int(txn["dest_prior_count"])
        else:
            raw_prior, since, _ = self.dest_history.peek(
                str(txn.get("nameDest") or "UNKNOWN_DEST"), step)
            txn["dest_prior_count"] = raw_prior
            txn["dest_steps_since_last"] = since

        feats_scaled = self.build_features(txn)
        # Unscaled flags for the analyst-facing explanations (scaling leaves the
        # 0/1 flags untouched, so we can read them straight off the built row).
        proba = float(self.model.predict_proba(feats_scaled[self.features])[0, 1])

        if block_threshold is not None and proba >= block_threshold:
            decision = "BLOCK"
        elif proba >= self.threshold:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        signals = [label for flag, label in RISK_SIGNAL_LABELS
                   if int(feats_scaled.iloc[0][flag]) == 1]
        if raw_prior > 0 and txn.get("nameDest"):
            signals.append(f"destination account already received {raw_prior} "
                           "prior transaction(s) (velocity)")

        # Update velocity state AFTER computing features (keeps them causal).
        if update_state and not caller_state:
            self.dest_history.update(str(txn.get("nameDest") or "UNKNOWN_DEST"),
                                     step)

        return {
            "fraud_probability": proba,
            "decision": decision,
            "threshold": self.threshold,
            "risk_signals": signals,
            "reason": (f"score {proba:.4f} vs policy threshold {self.threshold} "
                       f"({self.policy['model']}, {self.policy['variant']} variant)"),
            "model": f"{self.policy['model']} ({self.policy['variant']})",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


if __name__ == "__main__":  # smoke test: python Module_06/scoring.py
    scorer = FraudScorer()
    demo = {
        "type": "TRANSFER", "amount": 8500.0, "step": 744,
        "account_age_days": 20, "is_new_device": 1,
        "shipping_billing_mismatch": 1, "failed_payment_attempts": 4,
        "ip_billing_distance_km": 4200.0,
        "home_billing_country": "US", "ip_country": "NG",
        "nameDest": "C9999999",
    }
    out = scorer.score(demo)
    print(json.dumps(out, indent=2))
    sys.exit(0)
