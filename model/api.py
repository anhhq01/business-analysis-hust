"""
Module 6 - Model Deployment (real-time scoring API).

Run locally:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from data_prep import build_features

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Lazily populated on startup so import stays cheap.
_MODEL = None
_META: dict = {}


def _load_artifacts() -> None:
    global _MODEL, _META
    model_path = ARTIFACT_DIR / "model.joblib"
    meta_path = ARTIFACT_DIR / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python train.py` (Module 5) first."
        )
    _MODEL = joblib.load(model_path)
    _META = json.loads(meta_path.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()  # load the model once when the service starts
    yield


app = FastAPI(
    title="E-Commerce Fraud Detection API",
    description="Real-time transaction fraud scoring (Business Analytics group project).",
    version="1.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    """One incoming transaction to score."""

    step: int = Field(1, ge=0, description="Hours since simulation start (time-of-day = step % 24)")
    type: str = Field(..., description="CASH_IN | CASH_OUT | DEBIT | PAYMENT | TRANSFER")
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 10,
                "type": "TRANSFER",
                "amount": 181000.0,
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }
    }


class ScoreResponse(BaseModel):
    fraud_probability: float
    threshold: float
    is_fraud: bool
    decision: str
    model_name: str


def _score_frame(df: pd.DataFrame) -> list[ScoreResponse]:
    """Score one or more transactions (shared by /score and /score_batch)."""
    features = build_features(df)
    proba = _MODEL.predict_proba(features)[:, 1]
    threshold = float(_META["operating_threshold"])
    model_name = _META.get("model_name", "unknown")

    responses = []
    for p in proba:
        p = float(p)
        is_fraud = p >= threshold
        responses.append(
            ScoreResponse(
                fraud_probability=round(p, 4),
                threshold=threshold,
                is_fraud=is_fraud,
                decision="BLOCK / REVIEW" if is_fraud else "APPROVE",
                model_name=model_name,
            )
        )
    return responses


@app.get("/health")
def health() -> dict:
    """Liveness probe for the cloud platform."""
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Score a single transaction and return a block/approve decision."""
    return _score_frame(pd.DataFrame([txn.model_dump()]))[0]


@app.post("/score_batch", response_model=list[ScoreResponse])
def score_batch(txns: list[Transaction]) -> list[ScoreResponse]:
    """Score a batch of transactions in one call (used by the review queue)."""
    return _score_frame(pd.DataFrame([t.model_dump() for t in txns]))
