# Real-Time Payment Fraud Detection

End-to-end analytics project on PaySim + a synthetic contextual layer.

## Setup
```bash
pip install -r requirements.txt
```
Put the Kaggle CSV at `data/raw/online_fraud_detection.csv`
(or set `RAW_DATA_PATH` in `src/config.py`).

## Run — Modules 1 & 2
```bash
# (optional) create a mock dataset to test without the 700MB file
PYTHONPATH=src python scripts/make_mock_paysim.py

# Module 1: generate synthetic contextual data + enriched parquet
PYTHONPATH=src python src/generate_synthetic.py

# Module 2: (re)build and run EDA
PYTHONPATH=src python scripts/build_eda_notebook.py
jupyter notebook notebooks/02_eda.ipynb   # or run all cells
```

## Layout
```
src/config.py            single source of paths/seed/constants
src/generate_synthetic.py  Module 1 synthetic generator (keyed on nameOrig)
docs/business_understanding.md  business problem, KPIs, cost model
docs/data_dictionary.md  every synthetic/derived field documented
notebooks/02_eda.ipynb   Module 2 EDA (executed)
scripts/make_mock_paysim.py  test fixture (PaySim-shaped)
figures/                 saved plots
```

## Status
- [x] Module 1 — Business understanding, synthetic data, data dictionary
- [x] Module 2 — EDA
- [x] Module 3 — Data cleaning
- [ ] Module 4 — Feature engineering
- [ ] ...
