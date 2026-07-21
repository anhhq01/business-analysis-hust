"""
MODULE 4 - orchestration: engineer -> split -> fit-on-train -> save artifacts.

Produces everything Module 5 needs, with strict anti-leakage ordering:
  1. engineer causal features on the cleaned data
  2. STRATIFIED train/test split (preserves the ~0.3% fraud rate in both folds)
  3. fit the preprocessor on TRAIN ONLY
  4. demonstrate imbalance handling (class weights + SMOTE) on TRAIN ONLY
  5. persist: train/test parquet, fitted preprocessor, feature config, MI ranking

Run:  python src/build_features.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

from config import DATA_PROCESSED, ROOT, SEED
from features import (engineer_features, build_preprocessor, model_feature_columns,
                      transformed_feature_names, TARGET, NUMERIC_FEATURES)

MODELS = ROOT / "models"
SPLIT = DATA_PROCESSED / "split"


def main():
    MODELS.mkdir(exist_ok=True)
    SPLIT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PROCESSED / "transactions_clean.parquet")
    print(f"clean data: {len(df):,} rows")

    df = engineer_features(df)
    feat_cols = model_feature_columns()
    keep = feat_cols + [TARGET]                     # amount is already in feat_cols
    data = df[keep].copy()

    # ---- 2) stratified split BEFORE any resampling / fitting -----------------
    X = data[feat_cols]
    y = data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    print(f"train {len(X_tr):,} (fraud {y_tr.sum():,}, {y_tr.mean()*100:.3f}%) | "
          f"test {len(X_te):,} (fraud {y_te.sum():,}, {y_te.mean()*100:.3f}%)")

    # ---- 3) fit preprocessor on TRAIN ONLY ----------------------------------
    pre = build_preprocessor()
    pre.fit(X_tr)
    feat_names = transformed_feature_names(pre)
    joblib.dump(pre, MODELS / "preprocessor.joblib")

    # persist raw (pre-transform) engineered splits for Module 5
    # (amount is already among the feature columns, kept for the cost metric)
    train_out = X_tr.copy(); train_out[TARGET] = y_tr.values
    test_out = X_te.copy(); test_out[TARGET] = y_te.values
    train_out.to_parquet(SPLIT / "train.parquet", index=False)
    test_out.to_parquet(SPLIT / "test.parquet", index=False)

    # ---- 4) imbalance handling (documented) ---------------------------------
    # (a) class weight for models that support it
    pos = int(y_tr.sum()); neg = int((~y_tr.astype(bool)).sum())
    scale_pos_weight = neg / max(pos, 1)          # for XGBoost/LightGBM
    class_weight = {0: 1.0, 1: scale_pos_weight}  # for LogReg/RandomForest

    # (b) SMOTE demo on TRANSFORMED TRAIN ONLY (never on test)
    Xtr_t = pre.transform(X_tr)
    sm = SMOTE(sampling_strategy=0.1, random_state=SEED)  # bring minority to 10%
    Xtr_res, ytr_res = sm.fit_resample(Xtr_t, y_tr)
    print(f"SMOTE (train only): {len(y_tr):,} -> {len(ytr_res):,} rows; "
          f"fraud {y_tr.mean()*100:.3f}% -> {ytr_res.mean()*100:.3f}%")

    # ---- 5) feature selection: mutual information (on a train subsample) -----
    sub = X_tr.sample(min(50_000, len(X_tr)), random_state=SEED)
    sub_t = pre.transform(sub)
    mi = mutual_info_classif(sub_t, y_tr.loc[sub.index],
                             discrete_features=False, random_state=SEED)
    mi_rank = (pd.Series(mi, index=feat_names)
               .sort_values(ascending=False).round(5))

    config = {
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "numeric_features": NUMERIC_FEATURES,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "class_weight": {int(k): float(v) for k, v in class_weight.items()},
        "smote_sampling_strategy": 0.1,
        "mi_ranking_top15": mi_rank.head(15).to_dict(),
    }
    (MODELS / "feature_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nscale_pos_weight (neg/pos) = {scale_pos_weight:,.1f}")
    print("top features by mutual information:")
    print(mi_rank.head(12).to_string())
    print(f"\nsaved -> {MODELS/'preprocessor.joblib'}, {MODELS/'feature_config.json'}, "
          f"{SPLIT/'train.parquet'}, {SPLIT/'test.parquet'}")


if __name__ == "__main__":
    main()
