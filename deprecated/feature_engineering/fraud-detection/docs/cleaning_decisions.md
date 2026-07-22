# Module 3 — Data Cleaning Decisions

## Before / After Summary

| Check | Value |
|---|---:|
| `rows_before` | 6362620 |
| `columns_before` | 23 |
| `missing_values_before` | 0 |
| `duplicate_rows_before` | 0 |
| `fraud_count_before` | 8213 |
| `fraud_rate_before_pct` | 0.12908204481801522 |
| `invalid_binary_isFraud` | 0 |
| `invalid_binary_isFlaggedFraud` | 0 |
| `invalid_binary_is_new_device` | 0 |
| `invalid_binary_shipping_billing_mismatch` | 0 |
| `invalid_binary_is_night` | 0 |
| `negative_amount_rows_removed` | 0 |
| `invalid_ip_distance_rows` | 0 |
| `invalid_account_age_rows` | 0 |
| `rows_after` | 6362620 |
| `columns_after` | 28 |
| `missing_values_after` | 0 |
| `duplicate_rows_after` | 0 |
| `fraud_count_after` | 8213 |
| `fraud_rate_after_pct` | 0.12908204481801522 |

## Cleaning Decisions

| Issue | Decision | Reason |
|---|---|---|
| Exact duplicate rows | Removed if present | Duplicate rows can bias model training and EDA summaries. |
| `amount < 0` | Removed | Negative transaction amount is invalid for payment transactions. |
| `amount = 0` | Kept with `amount_zero_flag` | Zero-amount transactions may be system-generated or rule-related; they should not be deleted without evidence. |
| Extreme amount values | Kept | Large transactions may contain fraud signal; use `log_amount` later in feature engineering. |
| `isFlaggedFraud` | Kept in cleaned data but excluded from default model features later | It is an existing rule flag, not a raw behavioural signal. |
| Raw IDs: `nameOrig`, `nameDest`, `device_id`, `browser_fingerprint` | Kept for traceability | These are high-cardinality identifiers and should not be used directly as default model features. |