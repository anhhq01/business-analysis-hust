# Module 3 — Data Cleaning Decisions

## Before / after summary

|                |         before |          after | changed   |
|:---------------|---------------:|---------------:|:----------|
| rows           |    6.36262e+06 |    6.36262e+06 | False     |
| columns        |   23           |   23           | False     |
| missing_cells  |    0           |    0           | False     |
| duplicate_rows |    0           |    0           | False     |
| fraud_count    | 8213           | 8213           | False     |
| fraud_rate_%   |    0.1291      |    0.1291      | False     |

## Validity checks

| column                    |   min_allowed |   max_allowed |   n_below |   n_above |   observed_min |    observed_max |
|:--------------------------|--------------:|--------------:|----------:|----------:|---------------:|----------------:|
| step                      |             1 |           743 |         0 |         0 |              1 |   743           |
| hour_of_day               |             0 |            23 |         0 |         0 |              0 |    23           |
| account_age_days          |             1 |          3650 |         0 |         0 |              1 |  3360           |
| failed_payment_attempts   |             0 |            12 |         0 |         0 |              0 |    11           |
| ip_billing_distance_km    |             0 |         23000 |         0 |        49 |              0 | 36738.4         |
| amount                    |             0 |           inf |         0 |         0 |              0 |     9.24455e+07 |
| isFraud                   |             0 |             1 |         0 |         0 |              0 |     1           |
| isFlaggedFraud            |             0 |             1 |         0 |         0 |              0 |     1           |
| is_new_device             |             0 |             1 |         0 |         0 |              0 |     1           |
| shipping_billing_mismatch |             0 |             1 |         0 |         0 |              0 |     1           |
| is_night                  |             0 |             1 |         0 |         0 |              0 |     1           |

## Balance-reconciliation convention (documented, retained)

| metric                           |   pct_of_rows |
|:---------------------------------|--------------:|
| orig not reconciling             |          80.6 |
| dest not reconciling             |          68.6 |
| dest balances both zero (masked) |          36.4 |

## Decisions

| Issue                                                     | Decision                                                                        | Reason                                                                                         |
|:----------------------------------------------------------|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| Schema validation                                         | Assert all 23 required columns present; fail otherwise                          | Contract gate — a malformed upstream file must stop the run, not reach the model.              |
| Missing values                                            | None present; policy = drop unparseable numerics, never impute financial fields | Imputing a balance/amount fabricates money movement; dropping is safer for a risk model.       |
| Duplicate rows                                            | Remove exact full-row duplicates                                                | Duplicates bias training and EDA; per-txn ids make genuine dupes ~impossible here.             |
| Inconsistent categories                                   | Trim + upper-case type/countries, then validate against vocabulary              | Prevents case/whitespace splitting a category and catches out-of-vocab values before encoding. |
| Range / timestamp validity                                | Check both bounds for every dictionary range; binaries strict {0,1}             | Explicit contracts; also seeds Module-7 drift checks. Values are clipped by the generator.     |
| ip_billing_distance_km > ~23000                           | Keep (documented)                                                               | Legitimate exponential-tail draws, not corruption; the ceiling is approximate.                 |
| amount < 0                                                | Remove if present (none in PaySim)                                              | Negative payment amount is invalid; kept as a defensive guard.                                 |
| amount == 0                                               | Keep + note                                                                     | May be system/rule artefact or card-testing signal; deleting without evidence loses signal.    |
| Balance reconciliation gap                                | Document and RETAIN; do not 'correct'                                           | PaySim's non-reconciling ledger and masked zero-balances are conventions and are predictive.   |
| Derived error terms (orig_err/dest_err)                   | Documented here, engineered in Module 4                                         | Keeps cleaning and feature engineering separate and auditable.                                 |
| isFlaggedFraud                                            | Keep in cleaned data, exclude from default model features later                 | Existing near-useless rule flag (EDA: 0.19% recall); retained for traceability only.           |
| Raw ids (nameOrig/nameDest/device_id/browser_fingerprint) | Keep for traceability, not as raw features                                      | High-cardinality identifiers; used for joins/velocity in FE, not fed directly to the model.    |
