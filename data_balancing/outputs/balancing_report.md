# Data Balancing Report

## Input

- Data file: `D:\BA\business-analysis-hust\feature_engineering\fraud-detection\data\processed\transactions_cleaned.parquet`
- Rows used: 50,000
- Fraud rows: 65
- Fraud rate: 0.1300%
- Split mode: `stratified`
- Test size: 25%
- Train fraud rate: 0.1307%
- Test fraud rate: 0.1280%
- Features before correlation filter: 34
- Features after correlation filter: 33
- Highly correlated features dropped: 1

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
- Model: `logistic_regression`
- AUC-PR: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1: 1.0000
- Cost-optimal threshold: 0.30
- Business cost: 0.00

## Full Results

| strategy | model | auc_pr | roc_auc | precision | recall | f1 | operating_threshold | business_cost | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3000 | 0.0000 | 0.2091 |
| smote | logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9300 | 0.0000 | 0.5913 |
| smote | xgboost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0200 | 0.0000 | 1.2882 |
| class_weights | random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1900 | 0.0000 | 0.7767 |
| class_weights | xgboost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9600 | 0.0000 | 0.8751 |
| original | xgboost | 0.9963 | 1.0000 | 0.9412 | 1.0000 | 0.9697 | 0.0800 | 5.0000 | 5.4328 |
| smote | random_forest | 0.9963 | 1.0000 | 0.9412 | 1.0000 | 0.9697 | 0.2700 | 5.0000 | 1.1435 |
| class_weights | logistic_regression | 0.9875 | 1.0000 | 0.7273 | 1.0000 | 0.8421 | 0.9600 | 30.0000 | 0.5367 |
| undersampling | hist_gradient_boosting | 0.9770 | 1.0000 | 0.6667 | 1.0000 | 0.8000 | 0.9600 | 40.0000 | 0.2396 |
| undersampling | random_forest | 0.9622 | 0.9999 | 0.4571 | 1.0000 | 0.6275 | 0.5800 | 95.0000 | 0.3052 |
| smote | hist_gradient_boosting | 0.9575 | 0.9998 | 1.0000 | 0.9375 | 0.9677 | 0.0500 | 44577.3200 | 1.1288 |
| original | random_forest | 0.9387 | 0.9998 | 0.1739 | 1.0000 | 0.2963 | 0.0200 | 380.0000 | 0.7658 |
| undersampling | xgboost | 0.9148 | 0.9997 | 0.3200 | 1.0000 | 0.4848 | 0.7600 | 170.0000 | 0.1914 |
| class_weights | hist_gradient_boosting | 0.8710 | 0.9381 | 0.0013 | 1.0000 | 0.0026 | 0.0400 | 62275.0000 | 3.5471 |
| undersampling | logistic_regression | 0.8454 | 0.9998 | 0.8000 | 1.0000 | 0.8889 | 0.9600 | 20.0000 | 0.1887 |
| original | hist_gradient_boosting | 0.2422 | 0.8114 | 0.3333 | 0.6875 | 0.4490 | 0.4800 | 941466.4900 | 1.6837 |

## Fraud-Focused Review Set

This is not the main scorecard. It contains all fraud rows from the holdout plus a sampled set of normal rows, so it is useful for inspecting fraud capture but not for estimating production precision.

| strategy | model | review_rows | review_fraud_rows | review_precision | review_recall | review_f1 | review_fraud_value_capture_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | logistic_regression | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| original | xgboost | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| undersampling | hist_gradient_boosting | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| smote | logistic_regression | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| smote | xgboost | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| class_weights | random_forest | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| class_weights | xgboost | 1016 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| undersampling | logistic_regression | 1016 | 16 | 0.9412 | 1.0000 | 0.9697 | 1.0000 |
| undersampling | xgboost | 1016 | 16 | 0.9412 | 1.0000 | 0.9697 | 1.0000 |
| smote | random_forest | 1016 | 16 | 0.9412 | 1.0000 | 0.9697 | 1.0000 |
| class_weights | logistic_regression | 1016 | 16 | 0.9412 | 1.0000 | 0.9697 | 1.0000 |
| undersampling | random_forest | 1016 | 16 | 0.8889 | 1.0000 | 0.9412 | 1.0000 |
| original | random_forest | 1016 | 16 | 0.8000 | 1.0000 | 0.8889 | 1.0000 |
| class_weights | hist_gradient_boosting | 1016 | 16 | 0.0158 | 1.0000 | 0.0311 | 1.0000 |
| smote | hist_gradient_boosting | 1016 | 16 | 1.0000 | 0.9375 | 0.9677 | 0.9984 |
| original | hist_gradient_boosting | 1016 | 16 | 0.7333 | 0.6875 | 0.7097 | 0.9665 |

## Figures

- `figures/class_distribution.png`
- `figures/feature_distributions_by_class.png`
- `figures/numeric_feature_correlation.png`
- `figures/feature_target_scores.png`
- `figures/model_permutation_importance.png`
- `figures/metric_comparison.png`
- `figures/precision_recall_curves.png`
- `figures/confusion_matrix_best.png`