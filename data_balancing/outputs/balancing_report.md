# Data Balancing Report

## Input

- Data file: `feature_engineering/fraud-detection/data/processed/transactions_cleaned.parquet`
- Rows used: 10,000
- Fraud rows: 13
- Fraud rate: 0.1300%
- Split mode: `stratified`
- Test size: 25%
- Train fraud rate: 0.1333%
- Test fraud rate: 0.1200%
- Features before correlation filter: 34
- Features after correlation filter: 32
- Highly correlated features dropped: 2

## Feature Decisions

- Dropped high-cardinality identifiers: `nameOrig`, `nameDest`, `home_device_id`, `device_id`, `browser_fingerprint`.
- Dropped `isFlaggedFraud` because it is a rule flag, not a behavioral input.
- Kept transaction, synthetic and derived risk signals: amount/balance checks, new device, failed attempts, country mismatch, account age, IP distance and time-of-day.
- Derived account-history fields from PaySim `step`: `tx_count_prev_orig`, `time_since_prev_orig`, `amount_mean_prev_orig`, and `amount_vs_prev_mean_orig`.
- Removed numeric features with absolute pairwise correlation above 0.98; when two features were redundant, the one with weaker target correlation was dropped.
- One-hot encoded categorical fields and standardized numeric fields before SMOTE/model training.

## Balancing Methods

- `original`: no resampling or class weights; baseline for raw class imbalance.
- `undersampling`: randomly reduces the normal/majority class in the training fold only; the test set keeps the original fraud rate.
- `smote`: synthesizes minority fraud-class training examples after preprocessing; no synthetic rows are added to validation/test data.
- `class_weights`: keeps all rows but increases minority-class penalty during model fitting (`class_weight` for Logistic Regression/Random Forest, `sample_weight` for HistGradientBoosting).

## Best Result

- Strategy: `original`
- Model: `random_forest`
- AUC-PR: 0.9387
- Fraud cases captured: 16/16
- Fraud cases missed: 0
- Fraud case capture rate: 1.0000
- Fraud value captured: 28,091,103.45
- Fraud value missed: 0.00
- Fraud value capture rate: 1.0000
- Review queue size: 92
- False alerts: 76
- Cost-optimal threshold: 0.02
- Business cost: 380.00

## Full Results

| strategy | model | auc_pr | test_fraud_cases | fraud_cases_captured | fraud_cases_missed | fraud_case_capture_rate | fraud_amount_captured | fraud_amount_missed | fraud_value_capture_rate | review_queue_size | false_alerts | review_fraud_hit_rate | operating_threshold | business_cost | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | random_forest | 0.9387 | 16 | 16 | 0 | 1.0000 | 28091103.4500 | 0.0000 | 1.0000 | 92 | 76 | 0.1739 | 0.0200 | 380.0000 | 1.4778 |
| original | xgboost | 0.9167 | 3 | 3 | 0 | 1.0000 | 11754212.0700 | 0.0000 | 1.0000 | 4 | 1 | 0.7500 | 0.0200 | 5.0000 | 0.3027 |
| original | logistic_regression | 0.7381 | 3 | 2 | 1 | 0.6667 | 11576531.5300 | 177680.5400 | 0.9849 | 2 | 0 | 1.0000 | 0.1900 | 177680.5400 | 0.0179 |

## Fraud-Focused Review Set

This is not the main scorecard. It contains all fraud rows from the holdout plus a sampled set of normal rows, so it is useful for inspecting fraud capture but not for estimating production precision.

| strategy | model | review_rows | review_fraud_rows | review_precision | review_recall | review_f1 | review_fraud_value_capture_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | xgboost | 103 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| original | random_forest | 3016 | 16 | 0.4571 | 1.0000 | 0.6275 | 1.0000 |
| original | logistic_regression | 103 | 3 | 1.0000 | 0.6667 | 0.8000 | 0.9849 |

## Figures

- `figures/class_distribution.png`
- `figures/feature_distributions_by_class.png`
- `figures/numeric_feature_correlation.png`
- `figures/feature_target_scores.png`
- `figures/model_permutation_importance.png`
- `figures/metric_comparison.png`
- `figures/precision_recall_curves.png`
- `figures/confusion_matrix_best.png`