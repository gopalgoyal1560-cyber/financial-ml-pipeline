# Financial ML Pipeline

> **An end-to-end financial machine learning system built from data ingestion and validation to feature engineering, model experimentation, automated execution, and API-based prediction.**

**Overview**

Financial ML Pipeline is a complete machine learning workflow for experimenting with financial return prediction while treating **data engineering, validation, reproducibility, and deployment** as first-class components.

The system:

* Collects financial and macroeconomic data from multiple sources
* Stores raw API responses for reproducibility and validation
* Detects upstream schema changes before they silently corrupt the dataset
* Engineers technical indicators and statistical features
* Builds a merged daily feature dataset
* Experiments with multiple regression models and feature-selection techniques
* Serializes the selected preprocessing/model pipeline
* Serves predictions through a FastAPI endpoint
* Runs daily data ingestion through GitHub Actions
* Automatically sends newly generated data to the deployed prediction API
* Maintains logs of ingestion and prediction activity

The project deliberately documents both what worked and what did not.

That distinction matters in financial machine learning: a sophisticated pipeline does not automatically imply that the underlying market signal is strong.

---

## Live Demo

**Deployed application:**
https://financial-ml-pipeline-a38mtptcqzhz2manbjgrdv.streamlit.app/

**Source repository:**
https://github.com/gopalgoyal1560-cyber/financial-ml-pipeline

---

# Architecture

```text
                    ┌──────────────────────────┐
                    │      External Sources    │
                    │                          │
                    │ Yahoo Finance             │
                    │ Alpha Vantage             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Data Ingestion       │
                    │                          │
                    │ OHLCV                     │
                    │ Treasury Yield            │
                    │ WTI Crude                 │
                    │ Gold                      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Raw Data Snapshots     │
                    │                          │
                    │ JSON / historical data   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Schema Validation      │
                    │                          │
                    │ Column changes           │
                    │ Type changes             │
                    │ Missing fields           │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Feature Engineering    │
                    │                          │
                    │ SMA / EMA                │
                    │ RSI / MACD               │
                    │ Bollinger Bands           │
                    │ ATR / OBV                │
                    │ Returns / Volatility     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Merged Dataset         │
                    │                          │
                    │ daily_latest.csv         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     ML Experimentation   │
                    │                          │
                    │ Linear Regression         │
                    │ Ridge                     │
                    │ Lasso                     │
                    │ Random Forest             │
                    │ PCA / Feature Selection   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Serialized ML Pipeline   │
                    │                          │
                    │ Imputation               │
                    │ Scaling                  │
                    │ LassoCV                  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       FastAPI             │
                    │     Prediction API        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Client / Demo       │
                    │                          │
                    │ Streamlit / response.py  │
                    └──────────────────────────┘
```

---

# Data Sources

The pipeline currently combines:

| Source                     | Data                   |
| -------------------------- | ---------------------- |
| Yahoo Finance / `yfinance` | IBM daily OHLCV data   |
| Alpha Vantage              | 30-year Treasury yield |
| Alpha Vantage              | WTI crude oil          |
| Alpha Vantage              | Gold price             |

The external data is combined by date to construct the modelling dataset.

---

# Feature Engineering

The ingestion pipeline derives approximately 30 technical/statistical features.

### Trend

* SMA 10
* SMA 20
* SMA 50
* EMA 10
* EMA 20
* EMA 50

### Momentum

* RSI 14
* MACD
* MACD signal
* MACD histogram
* Momentum 10
* Momentum 20
* Rate of change 10
* Rate of change 20

### Volatility

* Bollinger middle band
* Bollinger upper band
* Bollinger lower band
* Bollinger `%B`
* ATR 14
* Rolling volatility 10
* Rolling volatility 20
* Annualized volatility

### Volume

* On-Balance Volume (OBV)

### Returns

* Simple percentage return
* Log return

These features are generated from the merged market data before being passed into the modelling pipeline.

---

# Data Quality & Schema Drift Detection

One of the main engineering goals of this project is to prevent upstream data changes from silently corrupting the ML pipeline.

Before new data is merged, the system compares the current source structure with the previous snapshot.

For JSON-based sources:

```text
Current response
       │
       ▼
   Genson schema
       │
       ▼
Compare against previous schema
       │
       ├── Breaking change → Stop pipeline
       │
       └── New/non-breaking field → Log change
```

Breaking changes include:

* Removed fields
* Data type changes

New fields are logged without automatically breaking the pipeline.

For OHLCV data, columns and dtypes are compared directly.

If no previous snapshot exists, validation is skipped with a warning rather than failing the initial ingestion.

This makes the data pipeline more defensive against upstream API changes.

---

# Machine Learning

The modelling stage was treated as an experiment rather than assuming that a complex model would automatically produce useful predictions.

The development process included:

1. Linear Regression
2. Feature scaling
3. Mutual-information feature selection
4. PCA for correlated feature groups
5. Time-series-aware validation
6. Naive baselines
7. Ridge regression
8. Lasso regression
9. Random Forest

Several methodological problems were identified and corrected during development, including:

* Feature-selection leakage
* Incorrect validation methodology
* Return/target alignment issues

`TimeSeriesSplit` was ultimately used instead of ordinary k-fold validation because the data is time-dependent.

---

# Model Result

The final experimental pipeline uses:

```text
Imputation
     ↓
Scaling
     ↓
LassoCV
```

The selected pipeline is serialized in:

```text
pipeline_rasso.pkl
```

An important finding from the experiments was that the engineered features did **not** demonstrate strong predictive signal relative to simple baselines.

The Lasso model effectively reduced the feature set to a single dominant feature, `adjusted`.

This was treated as a useful research result rather than hidden:

> More features and a more complicated model do not necessarily produce more predictive information.

The repository therefore considers the current model **experimental**, not a validated production financial forecasting system.

---

# API

The prediction service is implemented using FastAPI.

The API loads:

```text
pipeline_rasso.pkl
```

and dynamically constructs its request model from the pipeline's expected feature names and dtypes.

### Endpoint

```http
POST /Post_values
```

The client workflow is:

```text
daily_latest.csv
        ↓
Remove date column
        ↓
Convert row to JSON
        ↓
POST /Post_values
        ↓
Model pipeline
        ↓
Prediction
```

The repository includes `response.py` as an example client implementation.

---

# Automated Daily Pipeline

The project uses GitHub Actions to automate the workflow.

## Workflow 1 — Data ingestion

```text
Scheduled GitHub Action
        ↓
Fetch latest market data
        ↓
Validate schemas
        ↓
Feature engineering
        ↓
Update daily_latest.csv
        ↓
Commit changes
```

The ingestion workflow runs on weekdays.

## Workflow 2 — Prediction

After successful ingestion:

```text
Successful data workflow
        ↓
Wait
        ↓
Pull latest dataset
        ↓
Run response.py
        ↓
POST latest row to prediction API
        ↓
Record response in log
        ↓
Commit updated log
```

The workflows require the `KEY_alpha` repository secret for Alpha Vantage access.

---

# Repository Structure

```text
financial-ml-pipeline/
│
├── .github/
│   └── workflows/
│       ├── daily_fetch.yml
│       └── response.yml
│
├── data/
│   ├── raw_response/
│   └── processed/
│
├── log/
│   ├── ingestion.log
│   └── daily_ingestion.log
│
├── project_requirements/
│   ├── requirements.txt
│   └── requirements_api.txt
│
├── .env.example
├── .gitignore
│
├── daily_data_fetch.py
├── training_Data_ingestion.py
├── fast_api_connection.py
├── response.py
│
├── model1_proto.ipynb
├── pipeline_rasso.pkl
│
└── README.md
```

---

# Key Components

### `training_Data_ingestion.py`

Core data-ingestion and feature-engineering library.

Responsible for:

* fetching historical data
* parsing external responses
* validation helpers
* technical indicators
* feature construction

### `daily_data_fetch.py`

Daily orchestration layer.

Responsible for:

* fetching current data
* schema-drift checks
* incremental dataset updates
* logging

### `model1_proto.ipynb`

Research and experimentation notebook containing:

* model comparisons
* feature-selection experiments
* PCA experiments
* baseline evaluation
* methodological corrections
* final model selection

### `fast_api_connection.py`

FastAPI prediction service.

Responsible for:

* loading the serialized pipeline
* defining the prediction endpoint
* validating incoming features
* generating predictions

### `response.py`

Example API client.

Reads the latest feature row and sends it to the deployed prediction endpoint.

---

# Local Setup

Clone the repository:

```bash
git clone https://github.com/gopalgoyal1560-cyber/financial-ml-pipeline.git
cd financial-ml-pipeline
```

Install dependencies:

```bash
pip install -r project_requirements/requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Add your Alpha Vantage API key:

```env
KEY_alpha=your_api_key
```

---

# Run Historical Data Ingestion

```bash
python training_Data_ingestion.py
```

This performs the full historical ingestion and feature-building process.

---

# Run Daily Ingestion

```bash
python daily_data_fetch.py
```

This mirrors the automated daily ingestion workflow.

---

# Run the API Locally

Install the API dependencies:

```bash
pip install -r project_requirements/requirements_api.txt
```

Start FastAPI:

```bash
uvicorn fast_api_connection:api --reload
```

---

# Request a Prediction

```bash
python response.py
```

This reads the latest processed row and sends it to the prediction API.

---

# Environment Variables

| Variable    | Purpose               |
| ----------- | --------------------- |
| `KEY_alpha` | Alpha Vantage API key |

Never commit real API credentials to the repository.

---

# Engineering Lessons

This project was not just about fitting a regression model.

The more important lessons were around building a system that survives contact with real data:

### 1. Data pipelines fail before models do

An upstream API changing a field name can invalidate everything downstream.

### 2. Time-series validation matters

Randomly shuffling financial observations can create misleading evaluation results.

### 3. Leakage can make bad models look good

Feature selection and preprocessing must respect the train/test boundary.

### 4. Baselines are mandatory

A model that cannot consistently beat a simple baseline has not demonstrated useful predictive value.

### 5. More features are not necessarily better

Technical indicators can be highly correlated and may contain little independent predictive information.

### 6. Deployment changes the problem

A notebook that produces predictions is not the same thing as a functioning ML system.

This project therefore treats ingestion, validation, automation, serving, and logging as part of the ML system rather than separate afterthoughts.

---

# Limitations

The current system has important limitations:

* The modelling experiment focuses on a single equity: IBM.
* The available historical sample is relatively small for financial modelling.
* The current engineered features show weak predictive signal.
* The deployed model should not be interpreted as a reliable financial forecasting or investment system.
* Production-grade model monitoring is not yet implemented.
* The API is not yet containerized.
* Experiment tracking is still limited.
* A broader multi-asset dataset would be required to determine whether the observed results generalize.

These limitations are intentionally documented rather than hidden.

---

# Future Work

Potential extensions include:

* Expand from IBM to multiple equities and asset classes
* Increase historical coverage
* Test alternative prediction targets
* Build a formal experiment-tracking system
* Establish explicit production model-selection criteria
* Containerize the API
* Add endpoint monitoring
* Add data-quality monitoring
* Improve model evaluation and statistical testing
* Investigate whether the weak signal is asset-specific

---

# Disclaimer

This project is an **educational and experimental machine learning system**.

It is **not financial advice**, and its predictions should not be interpreted as recommendations to buy, sell, or hold any financial instrument.

The primary objective of the project is to demonstrate an end-to-end ML engineering workflow and investigate whether engineered market features provide useful predictive information.

The current experiments do not establish that they do.

---

# Author

**Gopal Goyal**

Built as an end-to-end exploration of:

```text
Data Engineering
        +
Machine Learning
        +
Financial Data
        +
MLOps
        +
API Deployment
        +
Automation
```

---

## Project Status

**Completed as an end-to-end experimental pipeline.**

The data ingestion, validation, feature engineering, automation, model serving, and deployment components are functional. The modelling results are intentionally treated as experimental because the current evidence does not establish strong predictive signal.
