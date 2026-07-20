# Reproducing Modules 1-2 (data + EDA)

Everything is driven by `src/config.py`, so there are no absolute paths to edit.
Run these from the **repo root**.

## 0. Requirements

```bash
pip install pandas numpy pyarrow matplotlib tabulate
```

(`pyarrow` is needed to write/read the enriched parquet; `tabulate` is needed
for the markdown tables in the EDA report.)

## 1. Get the data

```bash
bash setup_data.sh
```

Downloads the Kaggle PaySim CSV into `data/raw/online_fraud_detection.csv`
(~470 MB, 6,362,620 rows). See the bottom of `setup_data.sh` for manual /
Kaggle-CLI fallbacks if the direct link stops serving.

## 2. Generate the synthetic layer (Module 1)

```bash
python src/generate_synthetic.py
```

Writes `data/processed/transactions_enriched.parquet` (23 columns).

> **Memory note:** on the full 6.36M-row file this step peaks above ~4 GB RAM
> (origin accounts are almost all unique, so the account table is nearly as long
> as the full frame, and it builds two 6.3M-element id/fingerprint arrays). Run
> it on a machine with ~8 GB free. It is left exactly as authored.

## 3. Run the EDA (Module 2)

```bash
python src/eda.py
```

Writes `reports/eda_report.md` and PNGs into `figures/`. The report has two parts:

- **Part A - before enrichment:** imbalance + value-at-risk, fraud by type,
  amount/balance/destination/time patterns, `isFlaggedFraud` check, base
  correlations. Runs on the base CSV alone.
- **Part B - after enrichment:** schema diff + merge integrity, anti-leakage
  lift tables, per-feature correlation with `isFraud`, and multivariate lift.
  Only runs if the enriched parquet exists (step 2); otherwise it is skipped
  with a note.

### Useful flags

```bash
python src/eda.py --sample 500000          # quick pass on a random raw sample
python src/eda.py --raw <csv> --enriched <parquet> --outdir <dir> --figdir <dir>
```

## Verified full-data figures (Part A)

| Metric | Value |
|---|---|
| Rows | 6,362,620 |
| Fraud | 8,213 (0.1291%, ~1 in 775) |
| Fraud types | TRANSFER (0.77%), CASH_OUT (0.18%) only |
| Fraud share of $ value | 1.05% (vs 0.13% of count) |
| `isFlaggedFraud` recall | 16 / 8,213 = 0.19% (drop it) |
| Nulls / duplicate rows | 0 / 0 |

## Notes carried into Module 3

- `browser_fingerprint` uses Python's `hash()` -> not reproducible across runs
  (salted by `PYTHONHASHSEED`, not the numpy seed). Stable within a run.
- Non-zero balance-reconciliation terms are a known PaySim quirk (cleaning item).
- Origin accounts are near-unique -> build velocity features on `nameDest`.