"""
MODULE 6 - Real-time fraud-scoring API (FastAPI).

Endpoints
---------
  GET  /health        liveness + which model/threshold is loaded
  GET  /policy        the persisted Module 5 decision policy (verbatim)
  POST /score         score ONE transaction -> probability + decision + signals
  POST /score/batch   score a list of transactions (demo convenience)
  POST /state/reset   clear the in-memory destination-velocity store (demo)

Run locally (from the repo root):
  uv run uvicorn Module_06.api.main:app --reload --port 8000
Interactive docs: http://127.0.0.1:8000/docs

Render (free tier) start command:
  uvicorn Module_06.api.main:app --host 0.0.0.0 --port $PORT

The API is a thin, validated shell around Module_06/scoring.py - every feature
formula, the scaler statistics, and the operating threshold come from the
persisted Module 4/5 artifacts, never re-derived here.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make the repo root importable regardless of the working directory
# (same root-finding idiom as the notebooks).
_p = Path(__file__).resolve()
for _cand in [_p, *_p.parents]:
    if (_cand / "scripts" / "config.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from Module_06.scoring import ALL_TYPES, FraudScorer  # noqa: E402

app = FastAPI(
    title="E-commerce Fraud Scoring API",
    description=("Scores a payment transaction in real time with the Module 5 "
                 "behavioural RandomForest and the persisted cost-optimal "
                 "decision threshold."),
    version="1.0.0",
)

scorer = FraudScorer()  # loads model + scaler + policy once at startup

Country = Literal["BR", "CN", "DE", "FR", "GB", "IN", "NG", "RU", "US", "VN"]


class Transaction(BaseModel):
    """Raw transaction as the platform would see it at checkout time.

    Balance fields are optional: the deployed behavioural model deliberately
    ignores them (PaySim leakage - see Module 5), but they are accepted for
    audit completeness and for the scaler's fitted column set.
    """
    type: Literal["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
    amount: float = Field(gt=0, description="transaction amount ($)")
    account_age_days: float = Field(ge=0, le=3650)
    is_new_device: bool
    shipping_billing_mismatch: bool
    failed_payment_attempts: int = Field(ge=0, le=12)
    ip_billing_distance_km: float = Field(ge=0)
    home_billing_country: Country
    ip_country: Country

    # optional context
    transaction_id: Optional[str] = None
    step: Optional[int] = Field(default=None, ge=1,
                                description="PaySim-style hour index; defaults "
                                            "to one step past the data horizon")
    hour_of_day: Optional[int] = Field(default=None, ge=0, le=23)
    nameDest: Optional[str] = Field(default=None,
                                    description="destination account id, used "
                                                "for velocity state")
    dest_prior_count: Optional[int] = Field(default=None, ge=0,
                                            description="override: caller-side "
                                                        "velocity state")
    dest_steps_since_last: Optional[float] = None
    oldbalanceOrg: Optional[float] = Field(default=None, ge=0)
    newbalanceOrig: Optional[float] = Field(default=None, ge=0)
    oldbalanceDest: Optional[float] = Field(default=None, ge=0)
    newbalanceDest: Optional[float] = Field(default=None, ge=0)


class ScoreOptions(BaseModel):
    update_state: bool = Field(default=True,
                               description="record this txn in the velocity store")
    block_threshold: Optional[float] = Field(
        default=None, gt=0, le=1,
        description="optional auto-block tier ON TOP of the persisted review "
                    "threshold (demo knob, not part of the Module 5 policy)")


class ScoreRequest(BaseModel):
    transaction: Transaction
    options: ScoreOptions = ScoreOptions()


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: Optional[float]
    decision: Literal["AUTO_APPROVE", "APPROVE", "REVIEW", "BLOCK"]
    threshold: float
    risk_signals: list[str]
    reason: str
    model: Optional[str]
    latency_ms: float


class BatchScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(max_length=1000)
    options: ScoreOptions = ScoreOptions()


def _score_one(txn: Transaction, opts: ScoreOptions) -> ScoreResponse:
    try:
        result = scorer.score(txn.model_dump(),
                              update_state=opts.update_state,
                              block_threshold=opts.block_threshold)
    except ValueError as e:  # unknown country/type -> client error, not a 500
        raise HTTPException(status_code=422, detail=str(e))
    return ScoreResponse(
        transaction_id=txn.transaction_id or f"txn_{uuid.uuid4().hex[:12]}",
        **result,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": scorer.policy["model"],
        "variant": scorer.policy["variant"],
        "threshold": scorer.threshold,
        "n_features": len(scorer.features),
        "velocity_store_size": len(scorer.dest_history),
    }


@app.get("/policy")
def policy():
    """The Module 5 decision policy, verbatim (auditable deployment)."""
    return scorer.policy


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    return _score_one(req.transaction, req.options)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(req: BatchScoreRequest):
    return [_score_one(t, req.options) for t in req.transactions]


@app.post("/state/reset")
def reset_state():
    """Demo helper: clear the in-memory destination-velocity store."""
    scorer.dest_history.reset()
    return {"status": "ok", "velocity_store_size": len(scorer.dest_history)}
