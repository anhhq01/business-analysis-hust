"""
Central configuration for the fraud-detection project.

Every notebook/script imports paths and constants from here so there are
NO hardcoded absolute paths scattered around the repo. To run on your own
machine, set RAW_DATA_PATH once (or drop the CSV into data/raw/).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the repo root, so they work on any teammate's machine)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# The original Kaggle PaySim CSV. Point this at your file, e.g. on Windows:
#   RAW_DATA_PATH = Path(r"D:\study\online_fraud_detection.csv")
# or (recommended) copy it into data/raw/ and keep the default below.
RAW_DATA_PATH = DATA_RAW / "online_fraud_detection.csv"

# Output of Module 1 (base data + synthetic contextual fields, joined)
ENRICHED_DATA_PATH = DATA_PROCESSED / "transactions_enriched.parquet"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42          # global seed for all randomness
SALT = 20240101    # namespacing salt for per-account determinism

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
# Fraud in PaySim only ever occurs in these two transaction types.
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# Simplified country set for synthetic geo features (home country + IP country).
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "NG", "RU", "CN", "VN"]
# Weights make US/GB/DE common "home" countries; used for legit baseline.
COUNTRY_WEIGHTS = [0.30, 0.15, 0.12, 0.10, 0.10, 0.07, 0.06, 0.04, 0.03, 0.03]

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
