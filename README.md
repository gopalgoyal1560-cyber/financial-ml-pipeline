# Financial ML Pipeline

An end-to-end pipeline that ingests daily market and macroeconomic data, engineers technical features, validates incoming data against schema drift, trains a return-prediction model, and serves predictions through a FastAPI endpoint — automated daily via GitHub Actions.

> **Status:** Data ingestion, validation, and feature engineering are production-ready and running daily. Model development is experimental/in progress — see [Model Development](#model-development-model1_protoipynb) for an honest account of what has (and hasn't) worked so far.

---

## What it does

```
Alpha Vantage (Treasury yield, WTI crude, Gold)  ─┐
Yahoo Finance (IBM OHLCV)                         ─┼──▶ Ingest ──▶ Schema Validate ──▶ Feature Engineer ──▶ Merge ──▶ daily_latest.csv
                                                    │                                                                        │
                                                    ▼                                                                        ▼
                                          Raw JSON snapshots                                                    FastAPI model endpoint
                                          (data/raw_response/)                                                  (fast_api_connection.py)
```

Every weekday, a GitHub Actions workflow:
1. Pulls the latest IBM OHLCV bars (Yahoo Finance) and Treasury yield / WTI crude / Gold prices (Alpha Vantage).
2. Validates each source's schema against the previous day's data — stops the pipeline on breaking changes (removed fields or type changes) rather than silently ingesting bad data.
3. Computes ~30 technical indicators, merges all sources on date, forward-fills gaps, and appends the newest row to `data/processed/daily_latest.csv`.
4. A second workflow then posts that row to a deployed prediction API and logs the response.

## Repository structure

```
.
├── training_Data_ingestion.py   # Core library: fetch/parse/validate helpers + feature engineering (SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, returns, volatility, momentum)
├── daily_data_fetch.py          # Daily orchestration script — imports the above, adds schema-drift checks, appends the latest row
├── response.py                  # Reads the latest row and posts it to the deployed prediction API
├── fast_api_connection.py       # FastAPI app that loads pipeline_rasso.pkl and serves /Post_values predictions
├── model1_proto.ipynb           # Model research notebook (feature selection, PCA, Ridge/Lasso/RandomForest comparison)
├── pipeline_rasso.pkl           # Serialized preprocessing + Lasso regression pipeline used by the API
├── data/
│   ├── raw_response/            # Raw JSON snapshots per source per day (schema-validation baseline)
│   └── processed/                # Merged, feature-engineered datasets (training set + daily_latest.csv)
├── log/
│   ├── ingestion.log             # Logs from the full historical fetch (training_Data_ingestion.py)
│   └── daily_ingestion.log       # Logs from the daily incremental fetch (daily_data_fetch.py)
├── project_requirements/
│   ├── requirements.txt          # Full pipeline + modeling dependencies
│   └── requirements_api.txt      # Minimal dependencies for the FastAPI service
├── .github/workflows/
│   ├── daily_fetch.yml           # Runs daily_data_fetch.py on a weekday cron, commits new data
│   └── response.yml              # Triggers after a successful fetch, calls response.py, commits the log
└── .env.example                  # Expected environment variable(s)
```

## Data sources

| Source | Provider | Fields |
|---|---|---|
| IBM daily OHLCV | Yahoo Finance (`yfinance`) | open, high, low, close, adjusted close, volume |
| 30-year Treasury yield | Alpha Vantage | daily yield |
| WTI crude oil price | Alpha Vantage | daily price |
| Gold price | Alpha Vantage | daily price |

## Feature engineering

`training_Data_ingestion.py` derives the following from the merged OHLCV series:

- **Trend:** SMA (10/20/50), EMA (10/20/50)
- **Momentum/oscillators:** RSI (14), MACD + signal + histogram, momentum & rate-of-change (10/20)
- **Volatility:** Bollinger Bands (mid/upper/lower/%B), ATR (14), rolling volatility (10/20, annualized)
- **Volume:** On-Balance Volume (OBV)
- **Returns:** simple % return, log return

## Data quality & validation

Before any new data is merged in, `daily_data_fetch.py` compares it against the previous day's snapshot:
- **JSON endpoints** (Treasury/WTI/Gold): builds a schema with `genson` and diffs it with `deepdiff`. Removed fields or type changes are treated as **breaking** and halt the pipeline; new fields are logged as non-breaking.
- **OHLCV DataFrame**: compares columns and dtypes directly.
- If there's no prior snapshot to compare against, validation is skipped and logged as a warning rather than failing.

This is designed so a silent schema change upstream (e.g. Alpha Vantage renaming a field) surfaces as a stopped pipeline and a log entry, not a corrupted dataset.

## Automation (GitHub Actions)

- **`daily_fetch.yml`** — runs weekdays at 22:23 UTC, installs `requirements.txt`, runs `daily_data_fetch.py`, and commits/pushes any changed files under `data/` and `log/`.
- **`response.yml`** — triggers after `daily_fetch.yml` completes successfully, waits 5 minutes, pulls the latest data, runs `response.py` (which posts the newest row to the deployed model API), and commits the updated log.

Both require an Alpha Vantage API key available as the `KEY_alpha` repository secret.

## Model development (`model1_proto.ipynb`)

The notebook documents an iterative, transparently-logged modeling process on ~3,800 rows of daily data:

- Started with plain Linear Regression on imputed raw features — poor, highly negative R².
- Added scaling, then mutual-information-based feature selection, then PCA on correlated feature groups — each step tested against MAE/R² to isolate what was actually helping.
- Corrected methodology issues along the way (data leakage from computing MI scores on the full dataset instead of the training split, switching from k-fold to `TimeSeriesSplit`, fixing the return-shift/target alignment).
- Established a naive baseline (predict 0 / mean return / yesterday's return) — most model variants failed to beat it, which drove further investigation into feature quality rather than model choice.
- Compared Ridge, Lasso, and Random Forest on the full (collinear) feature set. **Lasso** was selected: it zeroed out all but one feature (`adjusted`), which was read as strong evidence that most engineered features carry weak/noisy signal for this target, and it had the best MAE/R² among the models tried.
- The selected pipeline (imputation → scaling → `LassoCV`) is serialized as `pipeline_rasso.pkl` and loaded by the FastAPI service.

**Read this as a research log, not a validated production model** — the notebook's own conclusion is that engineered technical features on a single equity show weak predictive signal relative to a naive baseline, and further work (more assets, longer history, different targets) is called out as needed.

## Prediction API

`fast_api_connection.py` loads `pipeline_rasso.pkl`, dynamically builds a Pydantic request model from the pipeline's expected feature names/dtypes, and exposes:

```
POST /Post_values
```

`response.py` shows the intended client usage: read the latest row from `data/processed/daily_latest.csv`, drop the date column, and POST the remaining fields as JSON to get back a prediction.

## Setup

```bash
git clone https://github.com/gopalgoyal1560-cyber/financial-ml-pipeline.git
cd financial-ml-pipeline
pip install -r project_requirements/requirements.txt
cp .env.example .env   # then fill in KEY_alpha with your Alpha Vantage API key
```

**Run a full historical fetch + feature build:**
```bash
python training_Data_ingestion.py
```

**Run the daily incremental fetch** (mirrors what the GitHub Action does):
```bash
python daily_data_fetch.py
```

**Serve the model locally:**
```bash
pip install -r project_requirements/requirements_api.txt
uvicorn fast_api_connection:api --reload
```

**Get a prediction for the latest row:**
```bash
python response.py
```

## Environment variables

| Variable | Description |
|---|---|
| `KEY_alpha` | Alpha Vantage API key, used for Treasury yield / WTI / Gold requests |

## Roadmap

- [ ] Expand beyond a single equity (IBM) to test whether weak signal is asset-specific
- [ ] Formal model evaluation framework / experiment tracking
- [ ] Production model selection criteria
- [ ] Dockerize the API service
- [ ] Monitoring for the deployed prediction endpoint and data pipeline
