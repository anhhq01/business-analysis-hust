# Module 5 — Model Development Results

**Dataset:** `online_payment_fraud_detection.csv` (6,362,620 rows)
**Training sample:** 600,000 rows (stratified), fraud rate **0.129%**
**Split:** 75% train / 25% test (stratified), test set ≈ 150,000 rows
**Run date:** 2026-07-03

## Model comparison (test set, sorted by AUC-PR)

| Model | AUC-PR | ROC-AUC | Precision | Recall | F1 | Threshold | Business cost |
|-------|-------:|--------:|----------:|-------:|---:|----------:|--------------:|
| **Random Forest** ⭐ | **0.9947** | 0.9974 | 0.9796 | 0.9948 | **0.9871** | 0.09 | **23,312** |
| XGBoost | 0.9671 | 0.9962 | 0.8539 | 0.9689 | 0.9078 | 0.03 | 23,252,574 |
| Logistic Regression | 0.7696 | 0.9834 | 0.0362 | 0.9793 | 0.0698 | 0.43 | 23,219,548 |

⭐ = selected model (highest AUC-PR).

## How to read these numbers

- **AUC-PR** is the headline metric for this severely imbalanced problem (fraud ≈ 0.13%). ROC-AUC looks high for every model because true negatives dominate, so it is *not* a reliable discriminator here — AUC-PR is.
- **Precision vs. Recall:** all three models catch ~97–99% of fraud (recall), but they differ sharply in false alarms. Logistic Regression flags a huge number of legitimate orders (precision 3.6%), which would overwhelm the analyst queue. Random Forest keeps recall at 99.5% *and* precision at 98%.
- **Business cost** = `sum(amount of missed frauds) + $5 × (false alarms)`, evaluated at each model's own cost-optimal threshold. Random Forest's cost (~23K) is **1000× lower** than the others (~23M) because it misses almost no high-value fraud. This is the metric that matches the business question.

## Selected operating point

- **Model:** Random Forest (200 trees, `class_weight="balanced"`)
- **Operating threshold:** **0.09** (transactions scoring ≥ 0.09 → BLOCK / REVIEW)
- The threshold was chosen to minimise the business cost above, i.e. it reflects the trade-off between fraud loss and customer friction rather than a default 0.5 cut-off.

## Class-imbalance handling

- Logistic Regression / Random Forest: `class_weight="balanced"`.
- XGBoost: `scale_pos_weight = (#negatives / #positives)`.
- No resampling (SMOTE/undersampling) was needed — reweighting was sufficient and keeps the pipeline simple and fast.

## Features used (18)

`log_amount, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
errorBalanceOrig, errorBalanceDest, orig_balance_delta, dest_balance_delta,
amount_to_oldOrg_ratio, zero_dest_balance, hour_of_day`, plus one-hot
`type_{CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER}`.

The strongest signals are the balance-error checks (`errorBalanceOrig`/`errorBalanceDest`)
and `type` — consistent with the EDA finding that fraud occurs only in TRANSFER and
CASH_OUT transactions.

## Artifacts produced (`artifacts/`)

| File | Purpose |
|------|---------|
| `model.joblib` | Trained Random Forest, loaded by the API (Module 6) |
| `model_meta.json` | Model name, operating threshold, feature list, cost assumption |
| `model_comparison.csv` | Raw metrics table above |
| `training_reference_sample.csv` | 5,000-row training-distribution baseline for Module 7 drift monitoring |

## Notes / caveats

- Trained on a 600K stratified sample for speed. Re-run with `--max-rows 0` to use all
  6.3M rows if you want the final production model.
- Near-perfect scores are expected on this dataset: the balance-error features are highly
  predictive of the labelled fraud pattern. This is a known property of the PaySim-derived
  data, not overfitting — but it is worth stating explicitly in the report, and it is why
  the synthetic contextual data (Module 1) matters for realism.
