# Data Balancing Experiments

This module covers the class-imbalance part of Module 4/5 in the project brief.
It compares the original imbalanced data against undersampling, SMOTE and
class-weighted training, then evaluates each method with fraud-appropriate
metrics.

## What This Module Does

The workflow is intentionally split into large phases:

1. **Load cleaned data**
   Reads `transactions_cleaned.parquet` produced by the previous cleaning module.

2. **Engineer balancing/modeling features**
   Drops raw IDs and leakage-prone rule flags, keeps synthetic risk signals, and
   derives account-history features from `step`/`nameOrig`.

3. **Analyze and select features**
   Plots feature distributions, computes numeric correlations, removes highly
   correlated numeric features, and scores feature-target association.

4. **Create a valid holdout split**
   Uses either a stratified split that preserves the fraud rate, or a time split
   that evaluates on the latest transaction steps.

5. **Apply balancing only on training data**
   The test set is never undersampled, oversampled, or SMOTE-generated.

6. **Train and compare models**
   Models include Logistic Regression, Random Forest, XGBoost, and sklearn
   HistGradientBoosting.

7. **Evaluate with imbalance-aware metrics**
   Main metrics are AUC-PR, precision, recall, F1, and business cost. Accuracy is
   intentionally not used as a headline metric because fraud is rare.

8. **Explain and monitor**
   Computes model-level permutation importance and provides a separate drift
   monitoring script for realtime/batch incoming transactions.

## Input

The script reads:

```text
feature_engineering/fraud-detection/data/processed/transactions_cleaned.parquet
```

Before running this module, make sure the earlier pipeline has produced:

```text
feature_engineering/fraud-detection/data/processed/transactions_enriched.parquet
feature_engineering/fraud-detection/data/processed/transactions_cleaned.parquet
```

## Feature Decisions

Dropped fields:

- `nameOrig`, `nameDest`: high-cardinality account IDs, useful for traceability
  but risky as direct model features.
- `home_device_id`, `device_id`, `browser_fingerprint`: high-cardinality device
  identifiers.
- `isFlaggedFraud`: existing rule flag, not a raw behavior signal.

Kept or derived fields:

- Base transaction signals: `amount`, balances, transaction `type`, `step`.
- Synthetic/context signals: account age, new device, shipping/billing mismatch,
  failed payment attempts, IP country, billing distance.
- Derived balance signals: balance errors, balance deltas, amount ratios.
- Derived time/account-history signals:
  `hour_of_day`, `is_night`, `tx_count_prev_orig`,
  `time_since_prev_orig`, `amount_mean_prev_orig`,
  `amount_vs_prev_mean_orig`.

Highly correlated numeric features are removed using a configurable threshold
(`0.98` by default). When two features are redundant, the script keeps the one
with stronger absolute correlation to `isFraud`.

## Balancing Methods

- `original`: no balancing; this is the raw imbalanced baseline.
- `undersampling`: reduces the normal/majority class in the training set only.
- `smote`: creates synthetic fraud/minority examples in the training set only.
- `class_weights`: keeps all rows and changes the training loss/penalty. Logistic
  Regression and Random Forest use `class_weight`; XGBoost uses
  `scale_pos_weight`; HistGradientBoosting receives `sample_weight`.

## Splits

Default:

```text
--split stratified
```

This preserves the original fraud ratio in train and test. It is the safest
main scorecard because precision, false positives and business cost require both
normal and fraud transactions.

Production-style option:

```text
--split time
```

This sorts by `step`, trains on older transactions and tests on the latest
transaction steps.

The script also writes a fraud-focused review scorecard containing all fraud
rows from the holdout plus sampled normal rows. This is useful for recall/fraud
capture inspection, but it is not the main production metric.

## Run

From the repository root:

```powershell
cd D:\BA\business-analysis-hust
python data_balancing\balancing_experiments.py
```

Log to a file while still showing progress:

```powershell
python data_balancing\balancing_experiments.py 2>&1 | Tee-Object -FilePath data_balancing\outputs\sample_run.log
```

Run all 6.36M rows with the stratified scorecard:

```powershell
python data_balancing\balancing_experiments.py --max-rows 0 --split stratified 2>&1 | Tee-Object -FilePath data_balancing\outputs\full_run_stratified.log
```

Run all 6.36M rows with a time-based holdout:

```powershell
python data_balancing\balancing_experiments.py --max-rows 0 --split time 2>&1 | Tee-Object -FilePath data_balancing\outputs\full_run_time.log
```

Optional 3-fold AUC-PR validation curves:

```powershell
python data_balancing\balancing_experiments.py --run-cv
```

## Outputs

Outputs are written to `data_balancing/outputs/`.

Tables:

- `balancing_model_results.csv`: main holdout scorecard.
- `fraud_review_results.csv`: fraud-focused review scorecard.
- `feature_correlation_matrix.csv`: numeric correlation matrix.
- `correlated_features_dropped.csv`: redundant features removed.
- `feature_target_scores.csv`: feature-target mutual information scores.
- `model_permutation_importance.csv`: best-model permutation importance.
- `cv_aucpr_by_fold.csv`: created only with `--run-cv`.
- `run_metadata.json`: run settings and selected feature list.
- `balancing_report.md`: human-readable summary.

Figures:

- `figures/class_distribution.png`
- `figures/feature_distributions_by_class.png`
- `figures/numeric_feature_correlation.png`
- `figures/feature_target_scores.png`
- `figures/model_permutation_importance.png`
- `figures/metric_comparison.png`
- `figures/precision_recall_curves.png`
- `figures/confusion_matrix_best.png`
- `figures/cv_aucpr_by_fold.png` when `--run-cv` is enabled

Artifacts:

- `artifacts/best_balancing_model.joblib`

## Feature Importance

The training script writes two kinds of importance:

- `feature_target_scores.csv`: model-free mutual information between features
  and `isFraud`.
- `model_permutation_importance.csv`: model-level permutation importance for
  the selected best model, measured by AUC-PR drop on the holdout set.

Use both together:

- Mutual information helps decide whether a feature has standalone signal.
- Permutation importance shows whether the trained model actually depends on
  that feature after preprocessing, balancing and interaction with other fields.

## Monitoring and Data Drift

`monitor.py` is separate from training because monitoring should run after
deployment on incoming transaction windows.

Build or refresh a reference distribution:

```powershell
python data_balancing\monitor.py --build-reference --current feature_engineering\fraud-detection\data\processed\transactions_cleaned.parquet
```

Quick monitoring test on a small early sample:

```powershell
python data_balancing\monitor.py --build-reference --source-max-rows 5000 --max-reference-rows 5000
python data_balancing\monitor.py --source-max-rows 5000 --max-current-rows 5000
```

Compare a current/incoming window against the reference:

```powershell
python data_balancing\monitor.py --current feature_engineering\fraud-detection\data\processed\transactions_cleaned.parquet 2>&1 | Tee-Object -FilePath data_balancing\outputs\monitoring_run.log
```

Monitoring outputs:

- `monitoring_reference.parquet`: saved baseline distribution.
- `monitoring_drift_report.csv`: feature-level PSI, KS and categorical drift.
- `monitoring_summary.md`: human-readable drift summary and realtime monitoring recommendations.
- `figures/monitoring_drift_summary.png`: top drift signals.

Recommended realtime monitoring:

- Score distribution drift: rolling score percentiles, PSI and review queue size.
- Feature drift: amount ratios, balance errors, failed attempts, new device,
  IP distance, country mismatch and transaction type.
- Operational drift: approval/block/review volume by hour and transaction type.
- Delayed label metrics: precision, recall, fraud value captured and false
  positive rate once chargeback/fraud labels arrive.

Future work for online transaction data:

- Start with simple synthetic/balancing methods such as undersampling, SMOTE and
  KNN-style nearest-neighbor synthesis.
- Then test stronger generative methods when feasible, such as CTGAN/TVAE for
  tabular synthetic fraud patterns.
- Add graph features because fraud often links accounts, devices, IPs and
  destination accounts: shared-device count, shared-IP count, shared-destination
  count, connected component risk and account-device-IP bipartite graph scores.

## Streamlit Dashboard

`monitor_app.py` provides a visual dashboard over the generated outputs.

Install the local requirements if needed:

```powershell
pip install -r data_balancing\requirements.txt
```

Run the dashboard:

```powershell
streamlit run data_balancing\monitor_app.py
```

Dashboard tabs:

- **Overview:** best model and current drift status.
- **Realtime Attack:** simulated incoming transaction batches with attack drift timeline.
- **Drift:** PSI/KS/category drift table and top drift chart.
- **Models:** model/balancing comparison and fraud-review scorecard.
- **Features:** mutual information, permutation importance and dropped correlated features.
- **Reports:** generated Markdown reports and refresh commands.

Realtime attack scenarios:

- `Account takeover`: new devices, address mismatch, IP mismatch, failed attempts.
- `High-value cashout`: larger amounts and cash-out-like amount/balance behavior.
- `Bot burst`: short time gaps, many previous transactions and failed attempts.
- `Foreign IP wave`: sudden shift toward foreign/high-distance IP transactions.

Realtime monitoring components:

- Threat banner showing current status: `normal`, `warning`, `alert`, or `critical`.
- Threat gauge for max PSI in the latest incoming batch.
- Time-based alert timeline with warning/alert thresholds.
- Feature-by-batch heatmap to see which signal is drifting.
- Event log ordered by simulated time and threat level.
- Threat detail selector: choose a feature and inspect reference vs current batch distribution.
