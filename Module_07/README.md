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

Module 6 integration log path:

- `Module_07/inputs/module6_scored_events.jsonl` - events written by
  `Module_06/api.py` on every `/score` or `/score/batch` request

## Quick run

From the repo root:

```bash
python Module_07/monitoring.py
```

Quick smoke test with smaller dashboard samples:

```bash
python Module_07/monitoring.py --dashboard-max-rows 50000

# explicitly point to the Module 6 event log
python Module_07/monitoring.py --events-log Module_07/inputs/module6_scored_events.jsonl
```

## Run every 5 minutes (Module 6 -> Module 7 trigger)

Use the scheduler to check the Module 6 event log every 5 minutes and trigger
Module 7 only when new events were appended.

```bash
python Module_07/scheduler.py --interval-seconds 300
```

One-shot health check (single cycle):

```bash
python Module_07/scheduler.py --once
```

Run once immediately at startup, then continue 5-minute cycles:

```bash
python Module_07/scheduler.py --interval-seconds 300 --run-on-start
```

On Windows, if you want this always-on after reboot, create a Task Scheduler
task that starts:

```bash
d:\2026\SoICT_DataScience\Business_Intelligent\business-analysis-hust\.venv\Scripts\python.exe Module_07/scheduler.py --interval-seconds 300
```

## Monitoring design

- `reference data`: transactions with `step <= 354` to mirror the original
  training period documented in Module 4.
- `current data`: later transactions with `step > 354`.
- If the Module 6 event log exists and is non-empty, it is used as
  `current data` automatically (this is the Module 6 -> Module 7 integration).
- `Module_07/scheduler.py` polls this log every 5 minutes and triggers
  `Module_07/monitoring.py` only if the log file has grown/changed.
- `rolling window`: by default 168 steps, which is 7 days because PaySim uses
  one hour per step.

For live deployment, labels are often delayed. Module 6 accepts an optional
`actual_label` field (0/1). When present in events, Module 7 computes rolling
precision/recall/F1; when absent, drift still runs and performance triggers are
left inactive until labels arrive.

If the current live batch has only one class label (e.g., all 0 or all 1),
the Evidently classification preset is skipped automatically and a drift-only
dashboard is still generated.

## Default retraining triggers

- More than 30% of monitored features drift.
- Latest rolling precision falls below 70% of the deployment baseline.
- Latest rolling recall falls more than 15 percentage points below baseline.

These are intentionally simple thresholds for coursework. In production, tie
them to review capacity, fraud-loss tolerance, and label delay.