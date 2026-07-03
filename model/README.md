# Module 5 & 6 — Model Development and Deployment

Fraud-detection model training (Module 5) and real-time deployment
(Module 6) for the Business Analytics group project.

## Files

| File | Module | Purpose |
|------|--------|---------|
| `data_prep.py` | 4–5 | Shared feature engineering used by **both** training and serving (prevents train/serve skew) |
| `train.py` | 5 | Trains & compares Logistic Regression, Random Forest, XGBoost; imbalanced-data metrics + cost-based threshold; saves artifacts |
| `api.py` | 6 | FastAPI service scoring transactions in real time (`/score`, `/score_batch`) |
| `streamlit_app.py` | 6 | Fraud-analyst review-queue demo — **calls the API over HTTP** (does not load the model itself) |
| `requirements.txt` | 5–6 | Dependencies |

## Setup

```bash
cd model
python -m venv venv && source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Module 5 — train

```bash
python train.py --data "D:/study/online_fraud_detection.csv"
```

- Uses a stratified sample (`--max-rows`, default 600k) for speed; pass
  `--max-rows 0` to train on the full dataset.
- Handles class imbalance via `class_weight="balanced"` (LR/RF) and
  `scale_pos_weight` (XGBoost).
- Evaluation metrics: **AUC-PR** (headline for imbalanced data), ROC-AUC,
  precision, recall, F1.
- **Operating threshold** is chosen to minimise a business cost:
  `missed-fraud loss (transaction amount) + $5 per false alarm`.
  Adjust `FALSE_POSITIVE_COST` in `train.py` and justify it in the report.
- Writes to `artifacts/`: `model.joblib`, `model_meta.json`,
  `model_comparison.csv`, and `training_reference_sample.csv` (for the
  Module 7 drift baseline).

## Module 6 — deploy

The Streamlit demo is a **client of the API** — start the API first, then
the UI. Architecture: `Streamlit UI → HTTP → FastAPI /score_batch → model`.

**1. API** (real-time scoring service):

```bash
uvicorn api:app --reload
# open http://127.0.0.1:8000/docs   (interactive Swagger UI)
```

Endpoints: `GET /health`, `POST /score` (one transaction),
`POST /score_batch` (a list of transactions).

**2. Review-queue demo** (in a second terminal, from the `model/` folder):

```bash
streamlit run streamlit_app.py
```

Point it at a remote API by setting `API_URL`, e.g.
`API_URL=https://your-api.onrender.com streamlit run streamlit_app.py`,
or edit the "API URL" box in the sidebar.

### Deploy to a cloud free tier

- **API (Render / Hugging Face Spaces):** start command
  `uvicorn api:app --host 0.0.0.0 --port $PORT`. Include the `artifacts/`
  folder in the image (commit it or run `train.py` in the build step).
- **Streamlit (Community Cloud):** point it at `model/streamlit_app.py`.

## Notes

- `artifacts/` is git-ignored — regenerate it by running `train.py`.
- The feature list is defined once in `data_prep.FEATURE_COLUMNS`; the API
  and Streamlit app import the exact same transform.
