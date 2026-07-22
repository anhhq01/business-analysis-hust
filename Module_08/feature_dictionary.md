# Module 4 — Engineered Feature Dictionary

Final feature count: **46**. Modelling universe: ['TRANSFER', 'CASH_OUT']. Temporal split at step ≤ 354 (train) vs later (test).

| Feature | Type | Definition | Source |
|---|---|---|---|
| `log_amount` | numeric | log10(amount+1); tames right-skew | derived from amount |
| `orig_balance_err` | numeric | oldbalanceOrg-amount-newbalanceOrig; ledger gap (mechanical caveat) | derived |
| `dest_balance_err` | numeric | oldbalanceDest+amount-newbalanceDest; ledger gap | derived |
| `drained_origin` | flag | 1 if origin emptied (newbalanceOrig==0 & oldbalanceOrg>0) | derived |
| `amount_to_oldbalanceOrg` | numeric | amount / (oldbalanceOrg+1); share of balance moved | derived |
| `dest_balances_masked` | flag | 1 if both dest balances 0 (PaySim masking) | derived |
| `ip_country_mismatch` | flag | 1 if ip_country != home_billing_country | synthetic geo |
| `log_ip_distance` | numeric | log10(ip_billing_distance_km+1) | synthetic geo |
| `young_account` | flag | 1 if account_age_days<=90 | synthetic account |
| `high_failed_attempts` | flag | 1 if failed_payment_attempts>=3 | synthetic behavioural |
| `dest_prior_count` | numeric | causal count of prior txns to this nameDest | velocity (dest) |
| `dest_steps_since_last` | numeric | steps since dest last seen (-1=never) | recency (dest) |
| `dest_first_seen` | flag | 1 if first sighting of this destination | recency (dest) |
| `hour_sin/hour_cos` | numeric | cyclical encoding of hour_of_day | derived time |
| `is_transfer` | flag | 1 if TRANSFER else CASH_OUT (universe is binary) | encoded type |
| `home_* / ip_* one-hots` | flag | one-hot of home/ip country | encoded geo |

## Leakage & monitoring notes

- Balance-error / drained-origin features are partly **mechanical** in PaySim; retained but flagged for drift monitoring (Module 7).
- All velocity/recency features are **causal** (prior rows only) and built on `nameDest` because origin accounts are near-unique.
- `isFlaggedFraud` and raw identifiers are **excluded** from features.
- Scaler fit on **train only**; resampling applied to **train only**.