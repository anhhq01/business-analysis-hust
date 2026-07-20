# Data Dictionary

Documents every field produced by `src/generate_synthetic.py`. Base PaySim
columns are listed first for context; **synthetic** and **derived** fields
(the ones we author) are documented in full per the project requirement:
column name, data type, unit, valid range, and generation logic / business
assumption.

---

## Base fields (from Kaggle PaySim — not authored by us)

| Column | Type | Unit | Notes |
|---|---|---|---|
| `step` | int | hours | 1 hour per step, ~30 days total (1–743). |
| `type` | str | — | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER. Fraud only in TRANSFER/CASH_OUT. |
| `amount` | float | currency | Transaction amount. Highly right-skewed. |
| `nameOrig` | str | — | Origin account id (join key for synthetic profile). |
| `oldbalanceOrg` / `newbalanceOrig` | float | currency | Origin balance before/after. For fraud, origin is drained. |
| `nameDest` | str | — | Destination id. `M…` = merchant (never defrauded), `C…` = customer. |
| `oldbalanceDest` / `newbalanceDest` | float | currency | Destination balance before/after. |
| `isFraud` | int (0/1) | — | Ground-truth label. |
| `isFlaggedFraud` | int (0/1) | — | PaySim's naive rule flag; flags almost nothing — treat as near-useless / drop. |

---

## Synthetic fields (authored — account level, stable per `nameOrig`)

| Column | Type | Unit | Valid range | Generation logic / business assumption |
|---|---|---|---|---|
| `account_age_days` | int | days | 1 – 3650 | Days since account creation. Base draw `Gamma(k=2, θ=180)` clipped to [1, 3650]. Accounts ever involved in fraud are scaled younger by a factor `U(0.25, 0.75)`, so fraud skews toward newer accounts while retaining heavy overlap with legit. Stable per account. |
| `home_billing_country` | str | ISO-ish code | 10 countries | Account's usual billing country, drawn from a weighted list (US/GB/DE most common). Stable per account; baseline for geo-mismatch signals. |
| `home_device_id` | str | id | — | The account's habitual device (`dev_xxxxxxxx`). Stable per account; baseline for the new-device signal. |

## Synthetic fields (authored — transaction level, vary per row)

| Column | Type | Unit | Valid range | Generation logic / business assumption |
|---|---|---|---|---|
| `device_id` | str | id | — | Device used for **this** transaction. Equals `home_device_id` unless `is_new_device`, in which case a fresh id is drawn. |
| `is_new_device` | int (0/1) | flag | 0 / 1 | 1 = transaction from a device not seen for this account. Drawn from a Bernoulli with P=0.70 for fraud rows vs P=0.08 for legit. Fraud rings frequently operate from fresh devices. |
| `browser_fingerprint` | str | id | — | Browser fingerprint (`fp_xxxxxxxxxx`), derived deterministically from `device_id` (stable per device). |
| `shipping_billing_mismatch` | int (0/1) | flag | 0 / 1 | 1 = shipping address ≠ billing address. Bernoulli P=0.55 (fraud) vs P=0.10 (legit). Reshipping/mule addresses are a classic fraud signal. |
| `failed_payment_attempts` | int | count | 0 – 12 | Failed payment attempts immediately preceding this transaction. `Poisson(λ=2.5)` for fraud vs `Poisson(λ=0.3)` for legit, capped at 12. Card-testing behaviour. |
| `ip_country` | str | code | 10 countries | Country of the transaction IP. Differs from `home_billing_country` with P=0.60 (fraud) vs P=0.07 (legit). |
| `ip_billing_distance_km` | float | kilometres | 0 – ~23000 | Great-circle-style proxy distance between IP and billing location. If `ip_country == home_billing_country`: `Exp(mean=120)`. If different: `Exp(mean=2500)+300`. Large distances indicate account takeover / location spoofing. |

## Derived fields (computed from base — not injected)

| Column | Type | Unit | Valid range | Generation logic |
|---|---|---|---|---|
| `hour_of_day` | int | hour | 0 – 23 | `step % 24`. Time-of-day proxy. **Derived, not fraud-biased** — any correlation reflects the real data only. |
| `is_night` | int (0/1) | flag | 0 / 1 | 1 if `hour_of_day` in 0–5. Derived; not injected. |

---

### Anti-leakage note
Every risky signal is drawn from an overlapping distribution — the fraud-vs-legit
probabilities differ but neither class is perfectly separated by any single
feature (e.g. `is_new_device` fraud-rate ≈ 1.0% when 1 vs 0.05% when 0, not 100%/0%).
This forces the model to combine signals and preserves realistic false-positive
and missed-fraud behaviour. No synthetic field is a deterministic function of
`isFraud`.
