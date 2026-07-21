"""
Module 4/5 - class balancing experiments for payment fraud detection.

This script keeps all balancing-specific work in ``data_balancing`` while
reading the cleaned dataset produced by the earlier modules.

Run from the repository root:
    conda run -n ba python data_balancing/balancing_experiments.py

Use the full dataset:
    conda run -n ba python data_balancing/balancing_experiments.py --max-rows 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "data_balancing" / "outputs" / ".mplconfig"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEANED_DATA = (
    ROOT
    / "feature_engineering"
    / "fraud-detection"
    / "data"
    / "processed"
    / "transactions_cleaned.parquet"
)
OUTPUT_DIR = ROOT / "data_balancing" / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"

SEED = 42
FALSE_POSITIVE_COST = 5.0
N_JOBS = 4
XGBOOST_DEVICE = "cuda"

TARGET = "isFraud"
DROP_COLUMNS = [
    TARGET,
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
    "home_device_id",
    "device_id",
    "browser_fingerprint",
]
CATEGORICAL_COLUMNS = ["type", "home_billing_country", "ip_country"]
CORRELATION_THRESHOLD = 0.98
ANALYSIS_SAMPLE_ROWS = 50_000


# ---------------------------------------------------------------------------
# Phase 0 - CLI and logging
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path) -> str:
    """Return a stable repo-relative path for logs and reports when possible."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def merge_metric_table(existing_path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if not existing_path.exists() or existing_path.stat().st_size == 0:
        return new_df
    existing = pd.read_csv(existing_path)
    if existing.empty:
        return new_df
    missing = [col for col in key_cols if col not in existing.columns or col not in new_df.columns]
    if missing:
        return new_df
    merged = pd.concat([existing, new_df], ignore_index=True)
    return merged.drop_duplicates(subset=key_cols, keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original, undersampling, SMOTE and class-weight balancing."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_CLEANED_DATA)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Stratified cap for speed. Use 0 for the full dataset.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--split",
        choices=["stratified", "time"],
        default="stratified",
        help="Holdout strategy. `stratified` preserves class rate; `time` evaluates on latest steps.",
    )
    parser.add_argument(
        "--fraud-review-size",
        type=int,
        default=1000,
        help="Normal rows added to the fraud-focused review set. Use 0 to skip.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="CPU worker count for supported models.",
    )
    parser.add_argument(
        "--xgboost-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost device. auto tries cuda first, then falls back to cpu.",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Also run k-fold AUC-PR validation curves. Slower.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified validation folds when --run-cv is enabled.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["original", "undersampling", "smote", "class_weights"],
        default=["original", "undersampling", "smote", "class_weights"],
        help="Balancing strategies to train in this run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting"],
        default=["logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting"],
        help="Model names to train in this run.",
    )
    parser.add_argument(
        "--allow-heavy-combos",
        action="store_true",
        help="Allow memory-heavy combinations such as full-data SMOTE + RandomForest.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 - Load the cleaned dataset
# ---------------------------------------------------------------------------
def load_data(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    """Read cleaned transactions and optionally take a stratified run sample."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {display_path(path)}. Run synthetic generation and cleaning first."
        )

    log(f"[1/8] Reading data: {display_path(path)}")
    df = pd.read_parquet(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    if max_rows and len(df) > max_rows:
        log(f"[2/8] Stratified sampling to {max_rows:,} rows for this run")
        frac = max_rows / len(df)
        parts = [
            group.sample(frac=frac, random_state=seed)
            for _, group in df.groupby(TARGET, sort=False)
        ]
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)

    log(
        f"      rows={len(df):,}, fraud={int(df[TARGET].sum()):,}, "
        f"fraud_rate={df[TARGET].mean():.4%}"
    )
    return df


# ---------------------------------------------------------------------------
# Phase 2 - Feature engineering
# ---------------------------------------------------------------------------
def add_balancing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create risk and account-history features used by balancing experiments."""
    out = df.copy()

    amount = out["amount"].astype(float)
    old_org = out["oldbalanceOrg"].astype(float)
    new_org = out["newbalanceOrig"].astype(float)
    old_dest = out["oldbalanceDest"].astype(float)
    new_dest = out["newbalanceDest"].astype(float)

    out["log_amount"] = np.log1p(amount.clip(lower=0))
    out["errorBalanceOrig"] = new_org + amount - old_org
    out["errorBalanceDest"] = old_dest + amount - new_dest
    out["orig_balance_delta"] = old_org - new_org
    out["dest_balance_delta"] = new_dest - old_dest
    out["amount_to_oldOrg_ratio"] = amount / (old_org + 1.0)
    out["amount_to_oldDest_ratio"] = amount / (old_dest + 1.0)
    out["zero_dest_balance"] = ((old_dest == 0) & (new_dest == 0)).astype(int)
    out["ip_country_mismatch"] = (
        out["ip_country"].astype(str).str.upper()
        != out["home_billing_country"].astype(str).str.upper()
    ).astype(int)

    if "hour_of_day" not in out.columns:
        out["hour_of_day"] = out["step"].astype(int) % 24
    if "is_night" not in out.columns:
        out["is_night"] = out["hour_of_day"].between(0, 5).astype(int)

    if {"nameOrig", "step"}.issubset(out.columns):
        ordered = out[["nameOrig", "step", "amount"]].copy()
        ordered["_row_id"] = np.arange(len(out))
        ordered = ordered.sort_values(["nameOrig", "step", "_row_id"])
        grouped = ordered.groupby("nameOrig", sort=False)
        ordered["tx_count_prev_orig"] = grouped.cumcount()
        ordered["time_since_prev_orig"] = grouped["step"].diff().fillna(9999).clip(lower=0)
        ordered["amount_mean_prev_orig"] = (
            grouped["amount"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
            .fillna(0)
        )
        ordered["amount_vs_prev_mean_orig"] = ordered["amount"] / (
            ordered["amount_mean_prev_orig"] + 1.0
        )
        ordered = ordered.sort_values("_row_id")
        out["tx_count_prev_orig"] = ordered["tx_count_prev_orig"].to_numpy()
        out["time_since_prev_orig"] = ordered["time_since_prev_orig"].to_numpy()
        out["amount_mean_prev_orig"] = ordered["amount_mean_prev_orig"].to_numpy()
        out["amount_vs_prev_mean_orig"] = ordered["amount_vs_prev_mean_orig"].to_numpy()

    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate feature matrix, target and amount values for cost evaluation."""
    log("[3/8] Engineering model features, including account velocity/time-gap fields")
    df = add_balancing_features(df)
    y = df[TARGET].astype(int)
    amounts = df["amount"].astype(float)
    drop = [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=drop)

    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().str.upper()

    return X, y, amounts


# ---------------------------------------------------------------------------
# Phase 3 - Train/test split design
# ---------------------------------------------------------------------------
def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    amounts: pd.Series,
    test_size: float,
    split: str,
    seed: int,
):
    """Create either a natural stratified holdout or a production-style time holdout."""
    if split == "stratified":
        return train_test_split(
            X,
            y,
            amounts,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )

    if split == "time":
        if "step" not in X.columns:
            raise ValueError("Time split requires the engineered `step` feature.")
        order = X["step"].sort_values().index
        test_n = max(1, int(round(len(order) * test_size)))
        test_idx = order[-test_n:]
        train_idx = order[:-test_n]
        if y.loc[test_idx].nunique() < 2:
            raise ValueError(
                "Time split test set contains only one class. Increase --test-size "
                "or use --split stratified."
            )
        return (
            X.loc[train_idx],
            X.loc[test_idx],
            y.loc[train_idx],
            y.loc[test_idx],
            amounts.loc[train_idx],
            amounts.loc[test_idx],
        )

    raise ValueError(f"Unknown split: {split}")


def make_fraud_review_set(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts_test: pd.Series,
    normal_rows: int,
    seed: int,
):
    """Build a fraud-focused auxiliary set without replacing the main holdout."""
    if normal_rows <= 0:
        return None
    fraud_idx = y_test[y_test == 1].index
    normal_idx = y_test[y_test == 0].index
    if len(fraud_idx) == 0 or len(normal_idx) == 0:
        return None

    sampled_normal = pd.Series(normal_idx).sample(
        n=min(normal_rows, len(normal_idx)),
        random_state=seed,
    )
    review_idx = fraud_idx.union(pd.Index(sampled_normal))
    return X_test.loc[review_idx], y_test.loc[review_idx], amounts_test.loc[review_idx]


# ---------------------------------------------------------------------------
# Phase 4 - Feature analysis and feature selection
# ---------------------------------------------------------------------------
def select_uncorrelated_features(
    X: pd.DataFrame, y: pd.Series, threshold: float = CORRELATION_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove highly correlated numeric features before preprocessing/modeling."""
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if X[col].nunique(dropna=False) > 1]
    if not numeric_cols:
        return X, pd.DataFrame(), pd.DataFrame()

    target_corr = (
        X[numeric_cols]
        .corrwith(y)
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    corr = X[numeric_cols].corr().abs().fillna(0)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = []
    decisions = []
    for left in upper.columns:
        highly_corr = upper.index[upper[left] > threshold].tolist()
        for right in highly_corr:
            if left in drop_cols or right in drop_cols:
                continue
            drop = left if target_corr[left] < target_corr[right] else right
            keep = right if drop == left else left
            drop_cols.append(drop)
            decisions.append(
                {
                    "dropped_feature": drop,
                    "kept_feature": keep,
                    "absolute_correlation": float(corr.loc[left, right]),
                    "dropped_target_corr": float(target_corr[drop]),
                    "kept_target_corr": float(target_corr[keep]),
                }
            )

    selected = X.drop(columns=drop_cols)
    return selected, corr, pd.DataFrame(decisions)


# ---------------------------------------------------------------------------
# Phase 5 - Preprocessing, balancing strategies and models
# ---------------------------------------------------------------------------
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Standardize numeric fields and one-hot encode low-cardinality categories."""
    categorical = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    numeric = [col for col in X.columns if col not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(
    model_name: str,
    weighted: bool,
    seed: int,
    scale_pos_weight: float = 1.0,
):
    """Return a fresh classifier, with class weighting only when requested."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced" if weighted else None,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=60,
            min_samples_leaf=2,
            n_jobs=N_JOBS,
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=60,
            learning_rate=0.08,
            random_state=seed,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight if weighted else 1.0,
            eval_metric="aucpr",
            tree_method="hist",
            device=XGBOOST_DEVICE,
            n_jobs=N_JOBS,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def xgboost_fit_with_fallback(model_name: str, model, X_train, y_train, fit_kwargs: dict):
    if model_name != "xgboost" or XGBOOST_DEVICE != "cuda":
        model.fit(X_train, y_train, **fit_kwargs)
        return model
    try:
        model.fit(X_train, y_train, **fit_kwargs)
        return model
    except Exception as exc:
        log(f"      XGBoost CUDA failed, falling back to CPU: {exc}")
        model.set_params(device="cpu")
        model.fit(X_train, y_train, **fit_kwargs)
        return model


def pipeline_fit_with_fallback(model_name: str, pipe, X_train, y_train, fit_kwargs: dict):
    if model_name != "xgboost" or XGBOOST_DEVICE != "cuda":
        pipe.fit(X_train, y_train, **fit_kwargs)
        return pipe
    try:
        pipe.fit(X_train, y_train, **fit_kwargs)
        return pipe
    except Exception as exc:
        log(f"      XGBoost CUDA failed, falling back to CPU: {exc}")
        pipe.set_params(model__device="cpu")
        pipe.fit(X_train, y_train, **fit_kwargs)
        return pipe


def make_pipeline(
    model_name: str,
    strategy: str,
    preprocessor,
    seed: int,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """Build the train-only balancing pipeline for one method/model pair."""
    weighted = strategy == "class_weights"
    steps = [("preprocess", preprocessor)]

    if strategy == "undersampling":
        steps.append(
            (
                "sampler",
                RandomUnderSampler(
                    sampling_strategy=0.35,
                    random_state=seed,
                ),
            )
        )
    elif strategy == "smote":
        steps.append(
            (
                "sampler",
                SMOTE(
                    sampling_strategy=0.25,
                    k_neighbors=5,
                    random_state=seed,
                ),
            )
        )

    steps.append(("model", make_model(model_name, weighted, seed, scale_pos_weight)))
    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# Phase 6 - Evaluation metrics and threshold selection
# ---------------------------------------------------------------------------
def choose_threshold(y_true, y_score, amounts) -> tuple[float, float]:
    """Choose the probability threshold that minimizes the business cost proxy."""
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=float)
    best_threshold = 0.5
    best_cost = np.inf

    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (y_score >= threshold).astype(int)
        missed_fraud = (y_true == 1) & (y_pred == 0)
        false_alarm = (y_true == 0) & (y_pred == 1)
        cost = amounts[missed_fraud].sum() + FALSE_POSITIVE_COST * false_alarm.sum()
        if cost < best_cost:
            best_threshold = float(threshold)
            best_cost = float(cost)

    return best_threshold, best_cost


def score_model(name, strategy, pipe, X_test, y_test, amounts_test, fit_seconds) -> dict:
    """Score one fitted model on the natural holdout distribution."""
    y_score = pipe.predict_proba(X_test)[:, 1]
    threshold, business_cost = choose_threshold(y_test, y_score, amounts_test)
    y_pred = (y_score >= threshold).astype(int)
    y_true = np.asarray(y_test)
    amounts = np.asarray(amounts_test, dtype=float)
    fraud_mask = y_true == 1
    normal_mask = y_true == 0
    predicted_fraud = y_pred == 1
    captured_fraud = fraud_mask & predicted_fraud
    missed_fraud = fraud_mask & (~predicted_fraud)
    false_alerts = normal_mask & predicted_fraud
    total_fraud_cases = int(fraud_mask.sum())
    captured_fraud_cases = int(captured_fraud.sum())
    missed_fraud_cases = int(missed_fraud.sum())
    review_queue_size = int(predicted_fraud.sum())
    fraud_amount_total = float(amounts[fraud_mask].sum())
    fraud_amount_captured = float(amounts[captured_fraud].sum())
    fraud_amount_missed = float(amounts[missed_fraud].sum())

    return {
        "strategy": strategy,
        "model": name,
        "auc_pr": average_precision_score(y_test, y_score),
        "roc_auc": roc_auc_score(y_test, y_score),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "test_fraud_cases": total_fraud_cases,
        "fraud_cases_captured": captured_fraud_cases,
        "fraud_cases_missed": missed_fraud_cases,
        "fraud_case_capture_rate": (
            captured_fraud_cases / total_fraud_cases if total_fraud_cases else 0.0
        ),
        "fraud_amount_total": fraud_amount_total,
        "fraud_amount_captured": fraud_amount_captured,
        "fraud_amount_missed": fraud_amount_missed,
        "fraud_value_capture_rate": (
            fraud_amount_captured / fraud_amount_total if fraud_amount_total else 0.0
        ),
        "review_queue_size": review_queue_size,
        "false_alerts": int(false_alerts.sum()),
        "review_fraud_hit_rate": (
            captured_fraud_cases / review_queue_size if review_queue_size else 0.0
        ),
        "operating_threshold": threshold,
        "business_cost": business_cost,
        "fit_seconds": fit_seconds,
    }


def score_review_set(
    name,
    strategy,
    pipe,
    X_review,
    y_review,
    amounts_review,
    threshold: float,
) -> dict:
    """Score the auxiliary fraud-focused review set."""
    y_score = pipe.predict_proba(X_review)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    fraud_mask = y_review == 1
    captured_value = float(amounts_review[(fraud_mask) & (y_pred == 1)].sum())
    total_fraud_value = float(amounts_review[fraud_mask].sum())
    return {
        "strategy": strategy,
        "model": name,
        "review_rows": int(len(y_review)),
        "review_fraud_rows": int(fraud_mask.sum()),
        "review_precision": precision_score(y_review, y_pred, zero_division=0),
        "review_recall": recall_score(y_review, y_pred, zero_division=0),
        "review_f1": f1_score(y_review, y_pred, zero_division=0),
        "review_fraud_value_capture_rate": (
            captured_value / total_fraud_value if total_fraud_value else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Phase 7 - Visualization helpers
# ---------------------------------------------------------------------------
def plot_class_distribution(y: pd.Series, path: Path) -> None:
    """Plot the raw class imbalance for the run dataset."""
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette=["#4C78A8", "#E45756"],
        legend=False,
    )
    ax.set_title("Original Class Distribution")
    ax.set_xlabel("isFraud")
    ax.set_ylabel("Rows")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """Plot important numeric feature distributions by fraud class."""
    candidate_cols = [
        "log_amount",
        "errorBalanceOrig",
        "errorBalanceDest",
        "account_age_days",
        "failed_payment_attempts",
        "ip_billing_distance_km",
        "time_since_prev_orig",
        "tx_count_prev_orig",
        "amount_vs_prev_mean_orig",
    ]
    cols = [col for col in candidate_cols if col in X.columns]
    if not cols:
        return

    if len(X) > ANALYSIS_SAMPLE_ROWS:
        sample_idx = (
            pd.DataFrame({TARGET: y})
            .groupby(TARGET, group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, ANALYSIS_SAMPLE_ROWS // 2)),
                    random_state=SEED,
                )
            )
            .index
        )
        sample = X.loc[sample_idx, cols].copy()
        sample[TARGET] = y.loc[sample_idx].values
    else:
        sample = X[cols].copy()
        sample[TARGET] = y.values
    sample = sample.reset_index(drop=True)

    long = sample.melt(id_vars=TARGET, value_vars=cols, var_name="feature")
    g = sns.FacetGrid(
        long,
        col="feature",
        hue=TARGET,
        col_wrap=3,
        sharex=False,
        sharey=False,
        height=3.0,
        palette=["#4C78A8", "#E45756"],
    )
    g.map_dataframe(sns.kdeplot, x="value", common_norm=False, warn_singular=False)
    g.add_legend(title="isFraud")
    g.fig.suptitle("Feature Distributions by Class", y=1.02)
    g.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(g.fig)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    """Plot the numeric-feature correlation matrix after feature creation."""
    if corr.empty:
        return
    mean_corr = corr.mean().sort_values(ascending=False)
    cols = mean_corr.index[: min(24, len(mean_corr))]
    plt.figure(figsize=(13, 10))
    ax = sns.heatmap(
        corr.loc[cols, cols],
        cmap="vlag",
        center=0,
        linewidths=0.2,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_feature_target_scores(X: pd.DataFrame, y: pd.Series, path: Path) -> pd.DataFrame:
    """Estimate and plot feature-target association with mutual information."""
    sample_X = X
    sample_y = y
    if len(X) > ANALYSIS_SAMPLE_ROWS:
        sample_idx = (
            pd.DataFrame({TARGET: y})
            .groupby(TARGET, group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, ANALYSIS_SAMPLE_ROWS // 2)),
                    random_state=SEED,
                )
            )
            .index
        )
        sample_X = X.loc[sample_idx]
        sample_y = y.loc[sample_idx]

    preprocessor = make_preprocessor(sample_X)
    Xt = preprocessor.fit_transform(sample_X)
    feature_names = preprocessor.get_feature_names_out()
    scores = mutual_info_classif(Xt, sample_y, discrete_features=False, random_state=SEED)
    out = (
        pd.DataFrame({"feature": feature_names, "mutual_info": scores})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )

    top = out.head(20).sort_values("mutual_info")
    plt.figure(figsize=(9, 7))
    ax = sns.barplot(data=top, x="mutual_info", y="feature", color="#4C78A8")
    ax.set_title("Top Feature-Target Association Scores")
    ax.set_xlabel("Mutual information with isFraud")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return out


def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Compute model-level feature importance on the untouched holdout set."""
    sample_X = X_test
    sample_y = y_test
    if len(X_test) > ANALYSIS_SAMPLE_ROWS:
        sample_idx = (
            pd.DataFrame({TARGET: y_test})
            .groupby(TARGET, group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, ANALYSIS_SAMPLE_ROWS // 2)),
                    random_state=seed,
                )
            )
            .index
        )
        sample_X = X_test.loc[sample_idx]
        sample_y = y_test.loc[sample_idx]

    result = permutation_importance(
        model,
        sample_X,
        sample_y,
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    return (
        pd.DataFrame(
            {
                "feature": sample_X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_model_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    """Plot permutation importance for the selected best model."""
    if importance.empty:
        return
    top = importance.head(20).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    ax = sns.barplot(data=top, x="importance_mean", y="feature", color="#E45756")
    ax.set_title("Best Model Permutation Importance")
    ax.set_xlabel("Mean AUC-PR drop after shuffling")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_metric_comparison(results: pd.DataFrame, path: Path) -> None:
    """Plot each model metric separately so balancing trade-offs are readable."""
    metrics = ["auc_pr", "precision", "recall", "f1"]
    long = results.melt(
        id_vars=["strategy", "model"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    g = sns.catplot(
        data=long,
        kind="bar",
        x="strategy",
        y="value",
        hue="model",
        col="metric",
        col_wrap=2,
        sharey=False,
        height=4,
        aspect=1.35,
    )
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
    g.fig.suptitle("Model Quality by Balancing Strategy", y=1.03)
    g.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(g.fig)


def plot_precision_recall_curves(curves: list[dict], path: Path) -> None:
    """Plot PR curves, the most informative curve family for rare fraud."""
    plt.figure(figsize=(9, 7))
    for curve in curves:
        precision, recall, _ = precision_recall_curve(curve["y_true"], curve["y_score"])
        label = f"{curve['model']} | {curve['strategy']} (AP={curve['auc_pr']:.3f})"
        plt.plot(recall, precision, linewidth=1.6, label=label)
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion_matrix(y_true, y_score, threshold, title: str, path: Path) -> None:
    """Plot the confusion matrix for the selected best operating threshold."""
    y_pred = (y_score >= threshold).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Legit", "Fraud"],
        cmap="Blues",
        values_format="d",
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_cv_scores(results: pd.DataFrame, path: Path) -> None:
    """Plot optional cross-validation AUC-PR stability by fold."""
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=results,
        x="fold",
        y="auc_pr",
        hue="strategy",
        style="model",
        markers=True,
        dashes=False,
    )
    ax.set_title("Cross-Validation AUC-PR by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC-PR")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Phase 8 - Optional cross-validation and reporting
# ---------------------------------------------------------------------------
def run_quick_cv(X, y, strategies, models, seed, cv_folds: int) -> pd.DataFrame:
    """Run optional k-fold AUC-PR validation for stability checks."""
    cv_rows = []
    cv_folds = max(2, int(cv_folds))
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        scale_pos_weight = float((y_train == 0).sum() / max(int((y_train == 1).sum()), 1))

        for strategy in strategies:
            for model_name in models:
                preprocessor = make_preprocessor(X_train)
                pipe = make_pipeline(
                    model_name,
                    strategy,
                    preprocessor,
                    seed + fold,
                    scale_pos_weight=scale_pos_weight,
                )
                fit_kwargs = {}
                if strategy == "class_weights" and model_name == "hist_gradient_boosting":
                    weights = compute_sample_weight("balanced", y_train)
                    fit_kwargs["model__sample_weight"] = weights
                pipeline_fit_with_fallback(model_name, pipe, X_train, y_train, fit_kwargs)
                y_score = pipe.predict_proba(X_valid)[:, 1]
                cv_rows.append(
                    {
                        "fold": fold,
                        "strategy": strategy,
                        "model": model_name,
                        "auc_pr": average_precision_score(y_valid, y_score),
                    }
                )

    return pd.DataFrame(cv_rows)


def write_report(
    results: pd.DataFrame,
    review_results: pd.DataFrame,
    metadata: dict,
    path: Path,
) -> None:
    """Write a Markdown report tying feature, balancing and model decisions together."""
    best = results.sort_values(["auc_pr", "f1"], ascending=False).iloc[0]
    table = results.sort_values(["auc_pr", "f1"], ascending=False).copy()
    table = table[
        [
            "strategy",
            "model",
            "auc_pr",
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
            "operating_threshold",
            "business_cost",
            "fit_seconds",
        ]
    ]
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = []
    for _, row in table.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    markdown_table = "\n".join([header, separator, *rows])
    review_section = []
    if not review_results.empty:
        review_table = review_results.sort_values(
            ["review_recall", "review_precision"], ascending=False
        ).copy()
        review_header = "| " + " | ".join(review_table.columns) + " |"
        review_separator = "| " + " | ".join(["---"] * len(review_table.columns)) + " |"
        review_rows = []
        for _, row in review_table.iterrows():
            cells = []
            for value in row:
                if isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            review_rows.append("| " + " | ".join(cells) + " |")
        review_section = [
            "",
            "## Fraud-Focused Review Set",
            "",
            "This is not the main scorecard. It contains all fraud rows from the holdout plus a sampled set of normal rows, so it is useful for inspecting fraud capture but not for estimating production precision.",
            "",
            "\n".join([review_header, review_separator, *review_rows]),
        ]
    figure_lines = [
        "- `figures/class_distribution.png`",
        "- `figures/feature_distributions_by_class.png`",
        "- `figures/numeric_feature_correlation.png`",
        "- `figures/feature_target_scores.png`",
        "- `figures/model_permutation_importance.png`",
        "- `figures/metric_comparison.png`",
        "- `figures/precision_recall_curves.png`",
        "- `figures/confusion_matrix_best.png`",
    ]
    if metadata["run_cv"]:
        figure_lines.insert(3, "- `figures/cv_aucpr_by_fold.png`")

    lines = [
        "# Data Balancing Report",
        "",
        "## Input",
        "",
        f"- Data file: `{metadata['data_path']}`",
        f"- Rows used: {metadata['rows_used']:,}",
        f"- Fraud rows: {metadata['fraud_rows']:,}",
        f"- Fraud rate: {metadata['fraud_rate']:.4%}",
        f"- Split mode: `{metadata['split']}`",
        f"- Test size: {metadata['test_size']:.0%}",
        f"- Train fraud rate: {metadata['train_fraud_rate']:.4%}",
        f"- Test fraud rate: {metadata['test_fraud_rate']:.4%}",
        f"- Features before correlation filter: {metadata['features_before_selection']}",
        f"- Features after correlation filter: {metadata['features_after_selection']}",
        f"- Highly correlated features dropped: {metadata['correlated_features_dropped']}",
        "",
        "## Feature Decisions",
        "",
        "- Dropped high-cardinality identifiers: `nameOrig`, `nameDest`, `home_device_id`, `device_id`, `browser_fingerprint`.",
        "- Dropped `isFlaggedFraud` because it is a rule flag, not a behavioral input.",
        "- Kept transaction, synthetic and derived risk signals: amount/balance checks, new device, failed attempts, country mismatch, account age, IP distance and time-of-day.",
        "- Derived account-history fields from PaySim `step`: `tx_count_prev_orig`, `time_since_prev_orig`, `amount_mean_prev_orig`, and `amount_vs_prev_mean_orig`.",
        f"- Removed numeric features with absolute pairwise correlation above {CORRELATION_THRESHOLD:.2f}; when two features were redundant, the one with weaker target correlation was dropped.",
        "- One-hot encoded categorical fields and standardized numeric fields before SMOTE/model training.",
        "",
        "## Balancing Methods",
        "",
        "- `original`: no resampling or class weights; baseline for raw class imbalance.",
        "- `undersampling`: randomly reduces the normal/majority class in the training fold only; the test set keeps the original fraud rate.",
        "- `smote`: synthesizes minority fraud-class training examples after preprocessing; no synthetic rows are added to validation/test data.",
        "- `class_weights`: keeps all rows but increases minority-class penalty during model fitting (`class_weight` for Logistic Regression/Random Forest, `sample_weight` for HistGradientBoosting).",
        "",
        "## Best Result",
        "",
        f"- Strategy: `{best['strategy']}`",
        f"- Model: `{best['model']}`",
        f"- AUC-PR: {best['auc_pr']:.4f}",
        f"- Fraud cases captured: {int(best['fraud_cases_captured']):,}/{int(best['test_fraud_cases']):,}",
        f"- Fraud cases missed: {int(best['fraud_cases_missed']):,}",
        f"- Fraud case capture rate: {best['fraud_case_capture_rate']:.4f}",
        f"- Fraud value captured: {best['fraud_amount_captured']:,.2f}",
        f"- Fraud value missed: {best['fraud_amount_missed']:,.2f}",
        f"- Fraud value capture rate: {best['fraud_value_capture_rate']:.4f}",
        f"- Review queue size: {int(best['review_queue_size']):,}",
        f"- False alerts: {int(best['false_alerts']):,}",
        f"- Cost-optimal threshold: {best['operating_threshold']:.2f}",
        f"- Business cost: {best['business_cost']:,.2f}",
        "",
        "## Full Results",
        "",
        markdown_table,
        *review_section,
        "",
        "## Figures",
        "",
        *figure_lines,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all phases end to end and write tables, figures and artifacts."""
    global N_JOBS, XGBOOST_DEVICE
    args = parse_args()
    N_JOBS = max(1, int(args.n_jobs))
    XGBOOST_DEVICE = "cuda" if args.xgboost_device == "auto" else args.xgboost_device
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.max_rows, args.seed)
    X, y, amounts = split_features(df)
    features_before_selection = X.shape[1]
    log("[4/8] Plotting feature distributions and selecting uncorrelated features")
    plot_class_distribution(y, FIGURE_DIR / "class_distribution.png")
    plot_feature_distributions(X, y, FIGURE_DIR / "feature_distributions_by_class.png")
    X, corr, dropped_corr = select_uncorrelated_features(X, y)
    corr.to_csv(OUTPUT_DIR / "feature_correlation_matrix.csv")
    dropped_corr.to_csv(OUTPUT_DIR / "correlated_features_dropped.csv", index=False)
    plot_correlation_heatmap(corr, FIGURE_DIR / "numeric_feature_correlation.png")
    feature_scores = plot_feature_target_scores(
        X, y, FIGURE_DIR / "feature_target_scores.png"
    )
    feature_scores.to_csv(OUTPUT_DIR / "feature_target_scores.csv", index=False)
    log(
        f"      selected {X.shape[1]} / {features_before_selection} features; "
        f"dropped {len(dropped_corr)} correlated numeric features"
    )

    log(f"[5/8] Creating {args.split} train/test split")
    X_train, X_test, y_train, y_test, amount_train, amount_test = make_holdout_split(
        X,
        y,
        amounts,
        args.test_size,
        args.split,
        args.seed,
    )
    log(
        f"      train fraud_rate={y_train.mean():.4%} ({int(y_train.sum()):,}/{len(y_train):,}); "
        f"test fraud_rate={y_test.mean():.4%} ({int(y_test.sum()):,}/{len(y_test):,})"
    )
    scale_pos_weight = float((y_train == 0).sum() / max(int((y_train == 1).sum()), 1))
    review_set = make_fraud_review_set(
        X_test, y_test, amount_test, args.fraud_review_size, args.seed
    )
    if review_set is not None:
        _, review_y, _ = review_set
        log(
            f"      fraud-review set={len(review_y):,} rows "
            f"({int(review_y.sum()):,} fraud + {int((review_y == 0).sum()):,} normal)"
        )

    strategies = list(args.strategies)
    models = list(args.models)

    results = []
    review_results = []
    curves = []
    smote_cache = None

    log("[6/8] Training balancing/model combinations")
    for strategy in strategies:
        for model_name in models:
            label = f"{strategy}__{model_name}"
            if (
                strategy == "smote"
                and model_name == "random_forest"
                and args.max_rows == 0
                and not args.allow_heavy_combos
            ):
                log(
                    f"      Skipping {label}: full-data SMOTE + RandomForest is memory-heavy. "
                    "Use --allow-heavy-combos to force it."
                )
                continue
            log(f"      Training {label} ...")
            fit_kwargs = {}

            if strategy == "smote":
                if smote_cache is None:
                    log("      Fitting preprocess + SMOTE once for all SMOTE models ...")
                    smote_started = time.perf_counter()
                    smote_preprocessor = make_preprocessor(X_train)
                    X_train_preprocessed = smote_preprocessor.fit_transform(X_train)
                    X_train_smote, y_train_smote = SMOTE(
                        sampling_strategy=0.25,
                        k_neighbors=5,
                        random_state=args.seed,
                    ).fit_resample(X_train_preprocessed, y_train)
                    smote_seconds = time.perf_counter() - smote_started
                    log(
                        f"      SMOTE cache rows={len(y_train_smote):,} "
                        f"fraud={int(pd.Series(y_train_smote).sum()):,}; "
                        f"seconds={smote_seconds:.1f}"
                    )
                    smote_cache = {
                        "preprocessor": smote_preprocessor,
                        "X_train": X_train_smote,
                        "y_train": y_train_smote,
                        "seconds": smote_seconds,
                    }

                model = make_model(
                    model_name,
                    weighted=False,
                    seed=args.seed,
                    scale_pos_weight=scale_pos_weight,
                )
                started = time.perf_counter()
                xgboost_fit_with_fallback(
                    model_name,
                    model,
                    smote_cache["X_train"],
                    smote_cache["y_train"],
                    {},
                )
                fit_seconds = time.perf_counter() - started
                pipe = Pipeline(
                    [
                        ("preprocess", smote_cache["preprocessor"]),
                        ("model", model),
                    ]
                )
                fit_seconds_total = fit_seconds
            else:
                preprocessor = make_preprocessor(X_train)
                pipe = make_pipeline(
                    model_name,
                    strategy,
                    preprocessor,
                    args.seed,
                    scale_pos_weight=scale_pos_weight,
                )
                if strategy == "class_weights" and model_name == "hist_gradient_boosting":
                    fit_kwargs["model__sample_weight"] = compute_sample_weight(
                        "balanced", y_train
                    )

                started = time.perf_counter()
                pipeline_fit_with_fallback(model_name, pipe, X_train, y_train, fit_kwargs)
                fit_seconds = time.perf_counter() - started
                fit_seconds_total = fit_seconds
            checkpoint_path = CHECKPOINT_DIR / f"{label}.joblib"
            joblib.dump(pipe, checkpoint_path)

            row = score_model(
                model_name,
                strategy,
                pipe,
                X_test,
                y_test,
                amount_test,
                fit_seconds_total,
            )
            if strategy == "smote":
                row["smote_shared_seconds"] = float(smote_cache["seconds"])
                row["model_fit_seconds"] = float(fit_seconds)
            else:
                row["smote_shared_seconds"] = 0.0
                row["model_fit_seconds"] = float(fit_seconds)
            row["checkpoint_path"] = str(checkpoint_path.relative_to(ROOT))
            results.append(row)
            y_score = pipe.predict_proba(X_test)[:, 1]
            if review_set is not None:
                review_results.append(
                    score_review_set(
                        model_name,
                        strategy,
                        pipe,
                        review_set[0],
                        review_set[1],
                        review_set[2],
                        row["operating_threshold"],
                    )
                )
            curves.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "y_true": y_test.copy(),
                    "y_score": y_score,
                    "auc_pr": row["auc_pr"],
                }
            )
            del pipe
            if strategy == "smote":
                del model
            gc.collect()
        if strategy == "smote" and smote_cache is not None:
            log("      Releasing SMOTE cache from memory")
            del smote_cache
            smote_cache = None
            gc.collect()

    log("[7/8] Writing evaluation tables and figures")
    results_df = pd.DataFrame(results)
    results_df = merge_metric_table(
        OUTPUT_DIR / "balancing_model_results.csv",
        results_df,
        ["strategy", "model"],
    ).sort_values(["auc_pr", "f1"], ascending=False)
    results_df.to_csv(OUTPUT_DIR / "balancing_model_results.csv", index=False)
    checkpoint_manifest = results_df[
        [
            "strategy",
            "model",
            "auc_pr",
            "precision",
            "recall",
            "f1",
            "test_fraud_cases",
            "fraud_cases_captured",
            "fraud_cases_missed",
            "fraud_case_capture_rate",
            "fraud_amount_captured",
            "fraud_amount_missed",
            "fraud_value_capture_rate",
            "review_queue_size",
            "false_alerts",
            "business_cost",
            "operating_threshold",
            "checkpoint_path",
        ]
    ].copy()
    checkpoint_manifest["label"] = checkpoint_manifest["strategy"] + "__" + checkpoint_manifest["model"]
    checkpoint_manifest = checkpoint_manifest[
        [
            "label",
            "strategy",
            "model",
            "auc_pr",
            "precision",
            "recall",
            "f1",
            "test_fraud_cases",
            "fraud_cases_captured",
            "fraud_cases_missed",
            "fraud_case_capture_rate",
            "fraud_amount_captured",
            "fraud_amount_missed",
            "fraud_value_capture_rate",
            "review_queue_size",
            "false_alerts",
            "business_cost",
            "operating_threshold",
            "checkpoint_path",
        ]
    ]
    checkpoint_manifest.to_csv(ARTIFACT_DIR / "model_checkpoints.csv", index=False)
    review_results_df = pd.DataFrame(review_results)
    if not review_results_df.empty:
        review_results_df = merge_metric_table(
            OUTPUT_DIR / "fraud_review_results.csv",
            review_results_df,
            ["strategy", "model"],
        )
        review_sort_cols = [
            col
            for col in ["review_fraud_hit_rate", "review_precision", "review_recall", "review_f1"]
            if col in review_results_df.columns
        ]
        if review_sort_cols:
            review_results_df = review_results_df.sort_values(review_sort_cols, ascending=False)
    if not review_results_df.empty:
        review_results_df.to_csv(OUTPUT_DIR / "fraud_review_results.csv", index=False)
    plot_metric_comparison(results_df, FIGURE_DIR / "metric_comparison.png")
    plot_precision_recall_curves(curves, FIGURE_DIR / "precision_recall_curves.png")

    current_labels = {f"{row['strategy']}__{row['model']}" for row in results}
    plot_candidates = results_df[
        (results_df["strategy"] + "__" + results_df["model"]).isin(current_labels)
    ]
    best_row = plot_candidates.iloc[0] if not plot_candidates.empty else pd.DataFrame(results).iloc[0]
    best_label = f"{best_row['strategy']}__{best_row['model']}"
    best_curve = next(
        curve
        for curve in curves
        if curve["strategy"] == best_row["strategy"] and curve["model"] == best_row["model"]
    )
    plot_confusion_matrix(
        y_test,
        best_curve["y_score"],
        best_row["operating_threshold"],
        f"Best Confusion Matrix: {best_label}",
        FIGURE_DIR / "confusion_matrix_best.png",
    )
    best_model = joblib.load(CHECKPOINT_DIR / f"{best_label}.joblib")
    importance_df = compute_permutation_importance(best_model, X_test, y_test, args.seed)
    importance_df.to_csv(OUTPUT_DIR / "model_permutation_importance.csv", index=False)
    plot_model_feature_importance(
        importance_df, FIGURE_DIR / "model_permutation_importance.png"
    )
    joblib.dump(best_model, ARTIFACT_DIR / "best_balancing_model.joblib")
    del best_model
    gc.collect()

    if args.run_cv:
        log(f"[8/8] Running optional {args.cv_folds}-fold CV curves")
        cv_df = run_quick_cv(X_train, y_train, strategies, models, args.seed, args.cv_folds)
        cv_df = merge_metric_table(
            OUTPUT_DIR / "cv_aucpr_by_fold.csv",
            cv_df,
            ["strategy", "model", "fold"],
        )
        cv_df.to_csv(OUTPUT_DIR / "cv_aucpr_by_fold.csv", index=False)
        plot_cv_scores(cv_df, FIGURE_DIR / "cv_aucpr_by_fold.png")

    metadata = {
        "data_path": display_path(args.data),
        "rows_used": int(len(df)),
        "fraud_rows": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "false_positive_cost": FALSE_POSITIVE_COST,
        "split": args.split,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "feature_columns": list(X.columns),
        "features_before_selection": int(features_before_selection),
        "features_after_selection": int(X.shape[1]),
        "correlated_features_dropped": int(len(dropped_corr)),
        "run_cv": bool(args.run_cv),
        "cv_folds": int(args.cv_folds),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(
        results_df,
        review_results_df,
        metadata,
        OUTPUT_DIR / "balancing_report.md",
    )

    log("\n=== Results sorted by AUC-PR/F1 ===")
    log(results_df.to_string(index=False))
    log(f"\nWrote outputs to: {display_path(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
