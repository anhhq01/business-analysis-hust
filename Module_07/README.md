# Module 7 - Model Monitoring

This module adds a lightweight monitoring scaffold around the deployed fraud
model in `models/`.

It does four things:

1. Rebuild the deployed behavioural feature set from
   `data/processed/transactions_enriched.parquet`.
2. Score the current slice with `models/best_model.joblib` and the deployed
   threshold in `models/decision_threshold.json`.
3. Track drift and rolling classification metrics.
4. Export a simple Evidently HTML dashboard plus trigger files.

## Outputs

Running the script writes files under `Module_07/outputs/`:

- `monitoring_dashboard.html` - Evidently dashboard
- `rolling_metrics.csv` - rolling precision/recall/F1/AP by step window
- `drift_summary.csv` - per-feature drift checks
- `trigger_summary.json` - retraining trigger decisions
- `monitor_summary.md` - readable summary

## Quick run

From the repo root:

```bash
python Module_07/monitoring.py
```

Quick smoke test with smaller dashboard samples:

```bash
python Module_07/monitoring.py --dashboard-max-rows 50000
```

## Monitoring design

- `reference data`: transactions with `step <= 354` to mirror the original
  training period documented in Module 4.
- `current data`: later transactions with `step > 354`.
- `rolling window`: by default 168 steps, which is 7 days because PaySim uses
  one hour per step.

## Default retraining triggers

- More than 30% of monitored features drift.
- Latest rolling precision falls below 70% of the deployment baseline.
- Latest rolling recall falls more than 15 percentage points below baseline.

These are intentionally simple thresholds for coursework. In production, tie
them to review capacity, fraud-loss tolerance, and label delay.