# Real-Time Payment Fraud Detection

End-to-end analytics project on PaySim + a synthetic contextual layer.

## Setup
```bash
pip install -r requirements.txt
```
Put the Kaggle CSV at `data/raw/online_fraud_detection.csv`
(or set `RAW_DATA_PATH` in `src/config.py`).

## Run — Modules 1 to 4

Before running the pipeline, put the Kaggle PaySim CSV at:

```text
data/raw/online_fraud_detection.csv
```

or update `RAW_DATA_PATH` in:

```text
src/config.py
```

Recommended execution directory:

```powershell
cd feature_engineering/fraud-detection
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

For PowerShell, set `PYTHONPATH`:

```powershell
$env:PYTHONPATH="src"
```

For Git Bash / Linux / macOS:

```bash
export PYTHONPATH=src
```

---

### Optional: create mock data

Use this only when you want to test the pipeline without the full Kaggle dataset.

PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/make_mock_paysim.py
```

Git Bash / Linux / macOS:

```bash
PYTHONPATH=src python scripts/make_mock_paysim.py
```

---

## Module 1 — Synthetic Data Generation

Purpose:

- Load the original PaySim transaction dataset.
- Generate synthetic contextual features such as account age, device behavior, browser fingerprint, failed payment attempts, IP country, billing country, and time-of-day fields.
- Save the enriched dataset as Parquet.

Run:

```powershell
$env:PYTHONPATH="src"
python src/generate_synthetic.py
```

Expected output:

```text
data/processed/transactions_enriched.parquet
```

Quick check:

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/processed/transactions_enriched.parquet'); print(df.shape); print(df.head())"
```

---

## Module 2 — Exploratory Data Analysis

Purpose:

- Analyze class imbalance.
- Analyze fraud by transaction type.
- Analyze amount distribution and outliers.
- Analyze time-of-day fraud behavior.
- Analyze synthetic contextual fraud signals.
- Generate EDA notebook and figures.

Run:

```powershell
$env:PYTHONPATH="src"
python scripts/build_eda_notebook.py
```

Open notebook:

```powershell
jupyter notebook notebooks/02_eda.ipynb
```

Expected outputs:

```text
notebooks/02_eda.ipynb
figures/
```

---

## Module 3 — Data Cleaning

Purpose:

- Validate required columns.
- Check missing values and duplicate rows.
- Standardize categorical values.
- Validate binary flags.
- Validate synthetic feature ranges.
- Create cleaning documentation.
- Save cleaned transaction data.

Run:

```powershell
$env:PYTHONPATH="src"
python src/clean.py
```

Then build the cleaning notebook:

```powershell
$env:PYTHONPATH="src"
python scripts/build_cleaning_notebook.py
```

Open notebook:

```powershell
jupyter notebook notebooks/03_cleaning.ipynb
```

Expected outputs:

```text
data/processed/transactions_clean.parquet
docs/cleaning_decisions.md
notebooks/03_cleaning.ipynb
```

If your local `clean.py` writes `transactions_cleaned.parquet` instead of `transactions_clean.parquet`, follow the filename defined in `src/config.py`.

---

## Module 4 — Feature Engineering

Purpose:

- Build model-ready fraud detection features.
- Create amount-based features.
- Create balance-behavior features.
- Create transaction type and time features.
- Fit preprocessing artifacts on training data only.
- Save feature engineering artifacts for modeling.

Run:

```powershell
$env:PYTHONPATH="src"
python src/build_features.py
```

Then build the feature engineering notebook:

```powershell
$env:PYTHONPATH="src"
python scripts/build_features_notebook.py
```

Open notebook:

```powershell
jupyter notebook notebooks/04_features.ipynb
```

Expected outputs may include:

```text
models/preprocessor.joblib
models/feature_config.json
notebooks/04_features.ipynb
```

---

## Full Run Order

Run the modules in this order:

```powershell
cd feature_engineering/fraud-detection

pip install -r requirements.txt

$env:PYTHONPATH="src"

# Module 1
python src/generate_synthetic.py

# Module 2
python scripts/build_eda_notebook.py

# Module 3
python src/clean.py
python scripts/build_cleaning_notebook.py

# Module 4
python src/build_features.py
python scripts/build_features_notebook.py
```

After that, open the notebooks and run all cells if needed:

```powershell
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_cleaning.ipynb
jupyter notebook notebooks/04_features.ipynb
```

---

## Pipeline Outputs

| Module | Main Script | Main Output |
|---|---|---|
| Module 1 | `src/generate_synthetic.py` | `data/processed/transactions_enriched.parquet` |
| Module 2 | `scripts/build_eda_notebook.py` | `notebooks/02_eda.ipynb`, `figures/` |
| Module 3 | `src/clean.py` | `data/processed/transactions_clean.parquet`, `docs/cleaning_decisions.md` |
| Module 4 | `src/build_features.py` | `models/preprocessor.joblib`, `models/feature_config.json` |

Generated datasets such as `.csv` and `.parquet` should not be committed to Git.

---

## Layout

```text
src/config.py                    Single source of paths, seed, and constants
src/generate_synthetic.py         Module 1 synthetic generator
src/clean.py                      Module 3 reusable cleaning pipeline
src/features.py                   Module 4 feature definitions
src/build_features.py             Module 4 feature build and preprocessing artifacts

scripts/make_mock_paysim.py       Optional mock PaySim-shaped dataset
scripts/build_eda_notebook.py     Builds Module 2 EDA notebook
scripts/build_cleaning_notebook.py Builds Module 3 cleaning notebook
scripts/build_features_notebook.py Builds Module 4 feature notebook

docs/business_understanding.md    Business problem, KPIs, and cost model
docs/data_dictionary.md           Original, synthetic, and derived fields
docs/cleaning_decisions.md        Data cleaning rules and before/after summary

notebooks/02_eda.ipynb            Module 2 EDA notebook
notebooks/03_cleaning.ipynb       Module 3 cleaning notebook
notebooks/04_features.ipynb       Module 4 feature engineering notebook

data/raw/                         Local raw dataset location
data/processed/                   Local generated Parquet outputs
figures/                          Saved EDA plots
models/                           Fitted preprocessing artifacts
```

---

## Status

- [x] Module 1 — Business understanding, synthetic data, data dictionary
- [x] Module 2 — EDA
- [x] Module 3 — Data cleaning
- [x] Module 4 — Feature engineering
- [x] Module 5 — Model development
- [x] Module 6 — Deployment
- [x] Module 7 — Monitoring

## Status
- [x] Module 1 — Business understanding, synthetic data, data dictionary
- [x] Module 2 — EDA
- [x] Module 3 — Data cleaning
- [ ] Module 4 — Feature engineering
- [ ] ...
