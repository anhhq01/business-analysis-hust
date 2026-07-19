# EDA run report - e-commerce fraud detection

Generated: **2026-07-19 14:43:03**
Base file: `/home/raven/Workspaces/ba/business-analysis-hust/feature_engineering/fraud-detection/data/raw/online_fraud_detection.csv`
Enriched file: `/home/raven/Workspaces/ba/business-analysis-hust/feature_engineering/fraud-detection/data/processed/transactions_enriched.parquet` (present)

---

# Part A - Before synthetic enrichment (base PaySim)

## 1. Dataset overview (base PaySim)

Source file: `online_fraud_detection.csv`
Rows: **6,362,620**  |  Columns: **11**
Duplicate rows: **0**  |  Total missing cells: **0**

|                | dtype   |   n_missing |   n_unique |
|:---------------|:--------|------------:|-----------:|
| step           | int64   |           0 |        743 |
| type           | str     |           0 |          5 |
| amount         | float64 |           0 |    5316900 |
| nameOrig       | str     |           0 |    6353307 |
| oldbalanceOrg  | float64 |           0 |    1845844 |
| newbalanceOrig | float64 |           0 |    2682586 |
| nameDest       | str     |           0 |    2722362 |
| oldbalanceDest | float64 |           0 |    3614697 |
| newbalanceDest | float64 |           0 |    3555499 |
| isFraud        | int64   |           0 |          2 |
| isFlaggedFraud | int64   |           0 |          2 |

## 2. Class imbalance & value-at-risk

- Fraud transactions: **8,213** of 6,362,620
- Prevalence: **0.1291%**  ->  roughly **1 in 775**
- A "never fraud" classifier is 99.8709% accurate and useless,
  which is why we optimise **AUC-PR / precision-recall**, not accuracy.

**Value at risk** (drives `C_fn` in the Module-5 cost model):

- Fraud transaction value: **12,056,415,428**
- Total transaction value: **1,144,392,944,760**
- Fraud is **1.0535%** of value while only
  0.1291% of count -> fraudulent transactions are far larger than
  average, so a missed fraud costs on the order of its `amount`.

![Legit vs fraud counts (log scale).](reports/figures/01_class_imbalance.png)

*Legit vs fraud counts (log scale).*

## 3. Fraud by transaction type

Fraud occurs **only** in `TRANSFER`, `CASH_OUT` - every
other type has zero fraud. Restricting to those two types, prevalence rises
to **0.2965%** (vs 0.1291%
overall): the modelling universe is far less imbalanced than the headline.

| type     |              n |      fraud |   fraud_rate_% |
|:---------|---------------:|-----------:|---------------:|
| TRANSFER |   532,909.0000 | 4,097.0000 |         0.7688 |
| CASH_OUT | 2,237,500.0000 | 4,116.0000 |         0.1840 |
| CASH_IN  | 1,399,284.0000 |     0.0000 |         0.0000 |
| DEBIT    |    41,432.0000 |     0.0000 |         0.0000 |
| PAYMENT  | 2,151,495.0000 |     0.0000 |         0.0000 |

![Fraud rate by type (TRANSFER & CASH_OUT only).](reports/figures/02_fraud_by_type.png)

*Fraud rate by type (TRANSFER & CASH_OUT only).*

## 4. Transaction amount distribution

Amounts are heavily right-skewed, so we view them on a log scale. Fraudulent
transactions sit systematically higher, consistent with account-draining.

|       |        mean |       50% |         90% |          99% |          max |
|:------|------------:|----------:|------------:|-------------:|-------------:|
| legit |   178,197.0 |  74,684.7 |   364,373.4 |  1,586,064.2 | 92,445,516.6 |
| fraud | 1,467,967.3 | 441,423.4 | 4,521,723.5 | 10,000,000.0 | 10,000,000.0 |

![Log-amount density, legit vs fraud.](reports/figures/03_amount_log.png)

*Log-amount density, legit vs fraud.*

## 5. Balance mechanics (the drained-origin signature)

For fraud, the origin account is typically emptied
(`oldbalanceOrg approx amount`, `newbalanceOrig = 0`):
**8,053 of 8,213** fraud rows (98.1%)
end with a zero origin balance.

We compute two reconciliation-error terms:
`orig_err = oldbalanceOrg - amount - newbalanceOrig` and
`dest_err = oldbalanceDest + amount - newbalanceDest`. About
**80.6%** of rows have a non-zero `orig_err` (a known PaySim
accounting quirk, and a Module-3 item). These balance signals are highly
predictive but partly mechanical, so they must be handled carefully to avoid
leakage in later modules.

|       |   orig_err |   dest_err |
|:------|-----------:|-----------:|
| legit | -201,338.6 |   54,692.2 |
| fraud |  -10,692.3 |  732,509.3 |

## 6. Destination type

Merchant destinations (`M...`) are never defrauded; all fraud lands on
customer destinations (`C...`). A merchant destination is effectively a
strong negative signal.

| dest_type    |              n |      fraud |   fraud_rate_% |
|:-------------|---------------:|-----------:|---------------:|
| customer (C) | 4,211,125.0000 | 8,213.0000 |         0.1950 |
| merchant (M) | 2,151,495.0000 |     0.0000 |         0.0000 |

## 7. Time-of-day pattern

`hour_of_day = step % 24` is **derived**, so any pattern here reflects the
real PaySim clock rather than anything injected. Fraud is disproportionately
concentrated in the low-activity night hours.

![Fraud rate across the 24-hour cycle.](reports/figures/04_fraud_by_hour.png)

*Fraud rate across the 24-hour cycle.*

## 8. `isFlaggedFraud` (PaySim's built-in rule)

Fires **16** times in total and catches only **16** of
8,213 frauds (recall **0.19%**). It is far too conservative to
be useful and is dropped from modelling, as the data dictionary advises.

## 9. Numeric correlation (base features)

Correlations among the base numeric fields and the label. Balance/amount
fields carry most of the base signal; nothing else is strongly linear.

![Base numeric correlation matrix.](reports/figures/05_corr_base.png)

*Base numeric correlation matrix.*

# Part B - After synthetic enrichment

## 10. Schema diff & merge integrity

Enrichment adds **12** columns and changes **no** base values.

- Base rows: **6,362,620**  |  Enriched rows: **6,362,620**
  -> row count preserved.
- New columns introduced no nulls: **True**.

Account-level fields (`account_age_days`, `home_billing_country`, `home_device_id`) are
**stable per `nameOrig`**: True. (Note: PaySim origin accounts are
almost all unique, so "stable per account" is rarely exercised in practice -
a limitation of the base data, not the generator.)

New columns:

- account-level: `account_age_days`, `home_billing_country`, `home_device_id`
- transaction-level: `device_id`, `is_new_device`, `browser_fingerprint`, `shipping_billing_mismatch`, `failed_payment_attempts`, `ip_country`, `ip_billing_distance_km`
- derived: `hour_of_day`, `is_night`

## 11. Anti-leakage lift tables

The core validation: each risky signal must **raise** the fraud rate without
**separating** the classes. Below, fraud rate conditional on each binary flag
(=1 vs =0). Elevated-but-overlapping is correct; anything near 100%/0% would
indicate leakage.

| signal                    |   fraud_rate_% (=1) |   fraud_rate_% (=0) |   lift_x |
|:--------------------------|--------------------:|--------------------:|---------:|
| is_new_device             |               1.129 |               0.041 |   27.328 |
| shipping_billing_mismatch |               0.718 |               0.063 |   11.395 |
| is_night                  |               1.773 |               0.099 |   17.820 |

For the continuous signals, fraud rate by decile (monotone-ish increase
expected, never a clean step to 100%):

![Fraud rate across account_age_days deciles.](reports/figures/06_decile_account_age_days.png)

*Fraud rate across account_age_days deciles.*

![Fraud rate across failed_payment_attempts deciles.](reports/figures/06_decile_failed_payment_attempts.png)

*Fraud rate across failed_payment_attempts deciles.*

![Fraud rate across ip_billing_distance_km deciles.](reports/figures/06_decile_ip_billing_distance_km.png)

*Fraud rate across ip_billing_distance_km deciles.*

## 12. Correlation of synthetic features with `isFraud`

Every synthetic feature correlates with the label only **weakly** - none
approaches 1.0. This is the single-glance anti-leakage check: no field is a
deterministic (or near-deterministic) function of fraud.

![Weak per-feature correlation with the label.](reports/figures/07_synth_corr.png)

*Weak per-feature correlation with the label.*

|                           |   corr_with_isFraud |
|:--------------------------|--------------------:|
| account_age_days          |             -0.0251 |
| ip_billing_distance_km    |              0.0496 |
| shipping_billing_mismatch |              0.0549 |
| is_night                  |              0.0614 |
| is_new_device             |              0.0825 |
| failed_payment_attempts   |              0.1438 |

## 13. Multivariate lift (signals must be combined)

Stacking risky flags (`is_new_device`, `shipping_billing_mismatch`, `is_night`) raises the fraud
rate sharply - the model must **combine** signals rather than rely on any one.
This both validates the generation design and motivates Module-4 feature
engineering.

|   _nflags |             n |     fraud |   fraud_rate_% |
|----------:|--------------:|----------:|---------------:|
|         0 | 5,169,983.000 |   807.000 |          0.016 |
|         1 | 1,119,887.000 | 3,197.000 |          0.285 |
|         2 |    71,109.000 | 3,426.000 |          4.818 |
|         3 |     1,641.000 |   783.000 |         47.715 |

![Fraud rate rises as flags stack.](reports/figures/08_multivariate_lift.png)

*Fraud rate rises as flags stack.*

## 14. Data-quality / reproducibility notes (for Module 3)

- **`browser_fingerprint` is not reproducible across runs.** It is built with
  Python's built-in `hash()`, which is salted per process via
  `PYTHONHASHSEED` and is **not** governed by the numpy seed in `config.py`.
  Within a run it is stable per `device_id` (as documented), but the actual
  `fp_...` values differ between runs/machines. If reproducibility matters,
  seed the mapping with numpy or set `PYTHONHASHSEED=0`. It does not affect
  modelling (the raw id would be hashed or dropped anyway).
- **Balance-reconciliation quirks** (non-zero `orig_err`) are inherent to
  PaySim and are a cleaning item, not a bug.
- **Origin accounts are near-unique**, so origin-level velocity features will
  be degenerate; build velocity on `nameDest` (customer destinations recur).
