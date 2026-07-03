"""
Module 5 - Model Development.

The best model, the chosen threshold and the metrics are written to
``artifacts/`` for the deployment step (Module 6).

Run:
    python train.py --data "path/to/your/online_fraud_detection.csv"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import FEATURE_COLUMNS, build_features, load_raw

# --- Business cost assumptions (Module 5: cost-based metric) -------------
# A missed fraud (false negative) loses the transaction amount.
# A false alarm (false positive) costs a fixed manual-review / friction
# cost. These numbers are assumptions the team should justify in the
# report; they drive the operating-threshold choice.
FALSE_POSITIVE_COST = 5.0  # currency units per legitimate order blocked/reviewed

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
DEFAULT_DATA = "D:/study/online_fraud_detection.csv"


def build_models(scale_pos_weight: float) -> dict:
    """Return the three candidate classifiers, all imbalance-aware."""
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    # XGBoost is optional - fall back gracefully if it is not installed.
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        )
    except ImportError:
        print("[warn] xgboost not installed - skipping (pip install xgboost).")

    return models


def choose_threshold(y_true, y_score, amounts) -> tuple[float, float]:
    """Pick the probability threshold that minimises total business cost.

    Cost = sum(amount of missed frauds) + FALSE_POSITIVE_COST * (# false alarms)
    Returns (best_threshold, best_cost).
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=float)

    best_threshold, best_cost = 0.5, np.inf
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_score >= threshold).astype(int)
        missed_fraud = (y_true == 1) & (pred == 0)
        false_alarm = (y_true == 0) & (pred == 1)
        cost = amounts[missed_fraud].sum() + FALSE_POSITIVE_COST * false_alarm.sum()
        if cost < best_cost:
            best_cost, best_threshold = cost, float(threshold)

    return best_threshold, float(best_cost)


def evaluate(name, model, X_test, y_test, amounts_test) -> dict:
    """Compute imbalanced-data metrics and the cost-based threshold."""
    y_score = model.predict_proba(X_test)[:, 1]

    threshold, cost = choose_threshold(y_test, y_score, amounts_test)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "model": name,
        "auc_pr": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "operating_threshold": threshold,
        "business_cost": cost,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5 - fraud model training")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to Kaggle CSV")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=600_000,
        help="Stratified row cap for faster training (None-like 0 = use all)",
    )
    args = parser.parse_args()

    max_rows = None if args.max_rows in (0, None) else args.max_rows

    print(f"[1/5] Loading data from {args.data} ...")
    df = load_raw(args.data, max_rows=max_rows)
    print(f"      rows={len(df):,}  fraud_rate={df['isFraud'].mean():.4%}")

    print("[2/5] Building features ...")
    X = build_features(df)
    y = df["isFraud"].astype(int)
    amounts = df["amount"].astype(float)

    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amounts, test_size=0.25, stratify=y, random_state=42
    )

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = build_models(scale_pos_weight)

    print("[3/5] Training & evaluating candidates ...")
    results = []
    fitted = {}
    for name, model in models.items():
        print(f"      -> {name}")
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append(evaluate(name, model, X_test, y_test, amt_test))

    results_df = pd.DataFrame(results).sort_values("auc_pr", ascending=False)
    print("\n=== Model comparison (sorted by AUC-PR) ===")
    print(results_df.to_string(index=False))

    # Select the best model by AUC-PR - the right headline metric for a
    # severely imbalanced problem.
    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    best_model = fitted[best_name]
    best_threshold = float(best_row["operating_threshold"])
    print(f"\n[4/5] Best model: {best_name} (threshold={best_threshold:.2f})")

    print("[5/5] Saving artifacts ...")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, ARTIFACT_DIR / "model.joblib")
    with open(ARTIFACT_DIR / "model_meta.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "operating_threshold": best_threshold,
                "feature_columns": FEATURE_COLUMNS,
                "false_positive_cost": FALSE_POSITIVE_COST,
            },
            f,
            indent=2,
        )
    results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    # A reference sample of the training distribution for Module 7 drift
    # monitoring (kept small so it can live in the repo).
    X_train.assign(isFraud=y_train.values).sample(
        n=min(5000, len(X_train)), random_state=42
    ).to_csv(ARTIFACT_DIR / "training_reference_sample.csv", index=False)

    print(f"      wrote artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
