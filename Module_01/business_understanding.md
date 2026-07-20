# Module 1 — Business Understanding

## The business question
> Which incoming transactions are likely to be fraudulent, and how should the
> platform balance **blocking fraud** against **approving legitimate orders**
> without unnecessary friction?

We act as the risk-analytics function of an e-commerce platform. The model must
flag high-risk transactions **in real time** on a severely imbalanced stream
(fraud is ~0.13% of transactions in PaySim).

## Why this is a trade-off, not an accuracy problem
Because fraud is ~1 in 775 transactions, a model that predicts "never fraud"
is 99.87% accurate and completely useless. Every decision has two failure modes
with very different costs:

| Decision error | What happens | Cost driver |
|---|---|---|
| **False negative** (miss fraud) | Fraudulent order approved | Direct financial loss ≈ transaction `amount` + chargeback fees |
| **False positive** (block legit) | Real customer declined / sent to manual review | Review cost + lost sale + churn / trust damage |

The right model is the one that **minimises total business cost**, not the one
with the highest accuracy or even the highest AUC.

## KPIs we track
1. **Fraud rate (prevalence)** — fraud transactions ÷ total. Baseline context.
2. **Recall / fraud capture rate** — fraud caught ÷ all fraud. "How much fraud do we stop?"
3. **Precision** — true fraud ÷ all flagged. Drives the analyst review workload.
4. **False-positive rate** — legit flagged ÷ all legit. Proxy for customer friction.
5. **Financial loss avoided** — Σ`amount` of fraud correctly blocked.
6. **Net cost** (primary optimisation metric, see below).

## Cost model (used for threshold selection in Module 5)
For a chosen decision threshold, define:

```
net_cost =  C_fn * (missed fraud value)          # value of fraud we let through
          + C_fp * (number of false positives)   # review + friction cost per wrongly blocked order
```

- `C_fn` (cost of a missed fraud) ≈ the transaction `amount` (we lose the money),
  optionally + a fixed chargeback fee.
- `C_fp` (cost of a false alarm) ≈ a fixed per-order cost: manual-review labour +
  expected lost-sale/churn. Start with a defensible constant (e.g. $5–$25) and
  run sensitivity analysis.

The operating threshold is chosen to **minimise `net_cost`**, and we report the
trade-off curve rather than a single number. Evaluation metrics for the model
itself are **AUC-PR**, precision/recall, and F1 — **not** ROC-AUC or accuracy,
which are misleading under this level of imbalance.

## Data strategy
- **Base (Kaggle PaySim):** transaction records + fraud labels. Note two structural
  facts: fraud occurs only in `TRANSFER` and `CASH_OUT`, and the origin account is
  typically drained (`oldbalanceOrg ≈ amount`, `newbalanceOrig = 0`).
- **Synthetic extension (this module):** account age, device/browser fingerprint,
  shipping/billing mismatch, failed payment attempts, IP-to-billing distance, and
  time-of-day — generated in Python, keyed on `nameOrig`, documented in
  `data_dictionary.md`. These are correlated with fraud *realistically* (noisy, no
  single feature separates the classes) so the model must combine signals and the
  fraud-vs-friction trade-off is preserved.
