# Financial ML Pipeline

**End-to-end financial machine learning pipeline for market data ingestion, feature engineering, validation, modeling, and eventual deployment.**

> **Project status:** Active development — data ingestion and feature-generation layers are implemented; model experimentation has started; production deployment and monitoring are still under development.

---

## Overview

This project is being built as a complete **financial machine learning pipeline**, starting with reliable market-data ingestion and progressively extending toward model training, evaluation, deployment, and monitoring.

The current implementation focuses on creating a reproducible dataset from multiple financial data sources, generating technical and macroeconomic features, validating incoming data, and maintaining daily updates.

The long-term objective is to move from:

**Raw Financial Data → Validated Dataset → Feature Engineering → ML Model → Evaluation → Deployment → Monitoring**

without treating any individual notebook or model as the final system.

---

## Current Status

| Component                   | Status      |
| --------------------------- | ----------- |
| Financial API ingestion     | Done        |
| Yahoo Finance ingestion     | Done        |
| Raw response storage        | Done        |
| Retry & HTTP error handling | Done        |
| Structured logging          | Done        |
| Date normalization          | Done        |
| Dataset alignment           | Done        |
| Technical indicators        | Done        |
| Dataset merging             | Done        |
| Schema validation           | Done        |
| Missing-value handling      | Done        |
| Daily data update logic     | Done        |
| ML experimentation          | In progress |
| Model evaluation framework  | In progress |
| Production model selection  | Planned     |
| Model serving API           | Planned     |
| Dockerization               | Planned     |
| Monitoring                  | Planned     |

The repository currently contains a modeling prototype (`model1_proto.ipynb`) and a serialized pipeline (`pipeline_rasso.pkl`), but these should be considered **experimental/model-development artifacts**, not a production ML service.

---

## Architecture

```text
                         FINANCIAL DATA SOURCES
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              Alpha Vantage                Yahoo Finance
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         Data Ingestion Layer
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                Raw JSON Storage          Data Parsing
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                         Data Validation
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              Schema Validation          Data Quality
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         Data Transformation
                                  │
                                  ▼
                       Feature Engineering
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
        Technical             Market/Macro         Derived
        Indicators              Features           Features
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ▼
                         Dataset Construction
                                  │
                                  ▼
                       ML Experimentation
                                  │
                                  ▼
                       Model Evaluation
                                  │
                         ┌────────┴────────┐
                         │                 │
                    Model Artifact    Backtesting
                         │                 │
                         └────────┬────────┘
                                  ▼
                         Future Deployment
                                  │
                                  ▼
                           Monitoring
```

---

## Data Sources

The current pipeline works with:

* **Alpha Vantage** — financial and macroeconomic API data
* **Yahoo Finance** — market price and volume data through `yfinance`

The ingestion code also preserves raw API responses locally so that transformations can be inspected and reproduced instead of relying only on the final processed dataset.

---

## Data Pipeline

### 1. Ingestion

The ingestion layer retrieves data from external financial APIs.

Implemented functionality includes:

* API requests
* Connection-error handling
* Timeout handling
* HTTP error handling
* Automatic retries with exponential backoff
* JSON parsing
* Request logging
* Raw-response persistence
* Environment-variable based API-key loading

The retry mechanism uses Tenacity and retries connection/timeout failures with bounded exponential waiting.

---

### 2. Raw Data Storage

Raw API responses are stored under:

```text
data/
└── raw_response/
    ├── <date>_<endpoint>.json
    └── ...
```

This provides a basic raw-data layer that can be used for debugging, schema comparison, and reproducing downstream transformations.

---

### 3. Data Processing

The processing layer converts raw responses into structured pandas DataFrames.

Current processing includes:

* JSON parsing
* Date normalization
* Sorting
* Numeric conversion
* Dataset alignment
* DataFrame construction
* Multi-source dataset merging

---

### 4. Schema Validation

The daily pipeline includes schema validation before continuing with downstream processing.

For JSON-based sources, the project uses:

* **Genson** for schema generation
* **DeepDiff** for schema comparison

Breaking changes such as removed fields or type changes can cause validation to fail, while newly added fields are treated separately. DataFrame-level validation also checks missing columns and dtype changes.

This is important for financial pipelines because upstream APIs can change their response structures without warning.

---

### 5. Missing-Value Handling

The current daily pipeline applies forward-fill imputation to the merged dataset before saving the latest observation.

```python
df = df.ffill()
```

The pipeline also records missing-value counts before and after imputation in the ingestion logs.

---

## Feature Engineering

The current feature layer generates technical indicators and derived market features.

### Technical Indicators

* SMA — Simple Moving Average
* EMA — Exponential Moving Average
* RSI — Relative Strength Index
* MACD — Moving Average Convergence Divergence
* Bollinger Bands
* ATR — Average True Range
* OBV — On-Balance Volume
* Returns
* Volatility
* Momentum
* Rate of Change

These features form the initial feature set for subsequent modeling work.

---

## Daily Data Pipeline

`daily_data_fetch.py` is responsible for orchestrating the recurring data-update process.

The current workflow includes:

```text
Fetch market data
       ↓
Load/validate source data
       ↓
Build and compare schemas
       ↓
Parse & align datasets
       ↓
Generate indicators
       ↓
Merge datasets
       ↓
Impute missing values
       ↓
Save latest observation
       ↓
Write execution logs
```

The daily pipeline writes its ingestion log to:

```text
log/daily_ingestion.log
```

The underlying ingestion module writes to:

```text
log/ingestion.log
```

---

## Machine Learning

The project has now moved beyond pure ingestion into **model experimentation**.

The repository contains:

```text
model1_proto.ipynb
pipeline_rasso.pkl
```

The notebook represents the current experimental modeling work, while the serialized `.pkl` artifact represents a saved pipeline/model artifact.

### Important

The presence of a saved model artifact does **not** mean that the ML layer is production-ready.

The next ML-stage work should establish:

1. Explicit prediction target
2. Time-aware train/validation/test splitting
3. Leakage prevention
4. Baseline model
5. Model comparison
6. Appropriate financial evaluation metrics
7. Backtesting
8. Error analysis
9. Model versioning
10. Reproducible model training

---

## Repository Structure

```text
financial-ml-pipeline/
│
├── .github/
│   └── workflows/
│       └── ...                  # GitHub Actions workflows
│
├── data/
│   ├── processed/               # Generated processed datasets
│   └── raw_response/            # Raw API responses
│
├── log/
│   ├── ingestion.log
│   └── daily_ingestion.log
│
├── daily_data_fetch.py          # Daily pipeline orchestration
├── training_Data_ingestion.py   # Data ingestion & feature preparation
│
├── model1_proto.ipynb           # ML experimentation
├── pipeline_rasso.pkl           # Serialized experimental pipeline
│
├── .env.example                 # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

The current repository structure includes the ingestion scripts, data directories, logs, notebook, serialized pipeline, environment template, dependency file, and GitHub Actions configuration.

---

## Tech Stack

| Area                   | Technology       |
| ---------------------- | ---------------- |
| Language               | Python           |
| Data processing        | pandas           |
| Numerical computing    | NumPy            |
| Market data            | yfinance         |
| Financial API          | Alpha Vantage    |
| HTTP                   | requests         |
| Retry handling         | Tenacity         |
| Configuration          | python-dotenv    |
| Model serialization    | Joblib           |
| JSON schema generation | Genson           |
| Schema comparison      | DeepDiff         |
| Experimentation        | Jupyter Notebook |
| Automation             | GitHub Actions   |

The current dependency file contains these core packages.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/gopalgoyal1560-cyber/financial-ml-pipeline.git
cd financial-ml-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project currently maintains its Python dependencies in `requirements.txt`.

### 4. Configure environment variables

Create `.env` in the project root:

```env
KEY_alpha=your_alpha_vantage_api_key
```

Do not commit your real API key.

The ingestion module reads the Alpha Vantage key through the environment using `python-dotenv`.

---

## Running the Data Pipeline

The repository currently does **not** contain the `main.py` referenced by the older README.

The primary implemented scripts are:

```bash
python training_Data_ingestion.py
```

and:

```bash
python daily_data_fetch.py
```

Before treating either command as the canonical production entry point, the project should consolidate orchestration into a single explicit CLI entry point.

A future interface should look more like:

```bash
python -m pipeline ingest
python -m pipeline validate
python -m pipeline features
python -m pipeline train
python -m pipeline evaluate
```

This would make the project easier to automate and maintain.

---

## Data Outputs

Generated data is stored locally under:

```text
data/
├── raw_response/
└── processed/
```

The daily pipeline currently maintains a latest-row dataset:

```text
data/processed/daily_latest.csv
```

Raw responses and generated datasets are intended to remain outside version control.

---

## Logging

Pipeline execution is logged rather than relying entirely on console output.

Current log files include:

```text
log/ingestion.log
log/daily_ingestion.log
```

Logs capture information such as:

* API requests
* API failures
* parsing errors
* schema-validation results
* missing-value counts
* generated output paths
* daily pipeline execution details

---

## ML Development Roadmap

### Phase 1 — Data Foundation

* [x] API data ingestion
* [x] Raw response persistence
* [x] Retry handling
* [x] HTTP error handling
* [x] Logging
* [x] JSON parsing
* [x] Date normalization
* [x] Dataset alignment
* [x] Technical indicators
* [x] Dataset merging

### Phase 2 — Data Quality

* [x] JSON schema generation
* [x] Schema comparison
* [x] DataFrame schema validation
* [x] Missing-value handling
* [ ] Comprehensive data-quality tests
* [ ] Outlier detection
* [ ] Duplicate detection
* [ ] Data-quality reporting

### Phase 3 — ML Development

* [x] Initial modeling prototype
* [x] Serialized experimental pipeline
* [ ] Define prediction target
* [ ] Establish baseline
* [ ] Time-series train/validation/test strategy
* [ ] Leakage checks
* [ ] Feature selection
* [ ] Model comparison
* [ ] Hyperparameter tuning
* [ ] Walk-forward evaluation
* [ ] Backtesting
* [ ] Error analysis

### Phase 4 — Production ML

* [ ] Reproducible training pipeline
* [ ] Model registry/versioning
* [ ] Automated model evaluation
* [ ] Model promotion criteria
* [ ] Prediction service
* [ ] API layer
* [ ] Docker image
* [ ] CI/CD

### Phase 5 — Monitoring

* [ ] Data drift detection
* [ ] Feature drift detection
* [ ] Prediction monitoring
* [ ] Model performance monitoring
* [ ] Pipeline health checks
* [ ] Alerting
* [ ] Automated retraining strategy

---

## Financial ML Principles

Because this is a financial machine-learning project, conventional random train/test splitting should not be treated as sufficient.

Future modeling work should explicitly address:

### Temporal ordering

Training data must precede validation and test data chronologically.

### Data leakage

Features must only use information that would have been available at prediction time.

### Baselines

Every ML model should be compared against a simple baseline rather than evaluated in isolation.

### Backtesting

Model performance should be evaluated under a realistic historical simulation.

### Transaction assumptions

Future trading-oriented evaluation should account for realistic assumptions such as transaction costs and execution constraints.

### Reproducibility

Training datasets, feature definitions, parameters, model versions, and evaluation results should be traceable.

---

## Project Design Philosophy

The project is intentionally being developed incrementally.

The goal is not simply:

```text
Dataset → Train Model → Accuracy
```

The intended system is:

```text
Reliable Data
      ↓
Validated Data
      ↓
Reproducible Features
      ↓
Leakage-Controlled Training
      ↓
Time-Aware Evaluation
      ↓
Backtesting
      ↓
Deployable Model
      ↓
Continuous Monitoring
```

This distinction matters because a model can achieve strong historical metrics while still being unsuitable for real-world financial use.

---

## Known Limitations

The current implementation is still under development.

Known limitations include:

* No finalized production model
* No documented target-definition contract
* No complete model evaluation framework
* No production prediction API
* No Docker deployment
* No formal model registry
* No production monitoring system
* No documented backtesting framework
* No comprehensive automated data-quality test suite
* Data-source schemas can change and require defensive validation
* Some orchestration remains distributed across standalone scripts

These are development tasks, not claims of completed functionality.

---

## Security & Configuration

Never commit:

```text
.env
API keys
private credentials
generated secrets
```

Use `.env.example` as the configuration template and keep actual credentials local or in the CI/CD secret store.

---

## Contributing

The project is under active development.

When adding a new pipeline component:

1. Keep ingestion, transformation, modeling, and deployment logic separated.
2. Avoid hard-coded credentials.
3. Add logging for important pipeline operations.
4. Preserve reproducibility.
5. Add validation around external data.
6. Avoid introducing look-ahead bias.
7. Document new features and dependencies.
8. Update the roadmap when implementation status changes.

---

## Disclaimer

This project is for **educational, research, and software-engineering purposes**.

Financial markets are uncertain, and historical model performance does not guarantee future results. Nothing in this repository should be interpreted as financial advice or as a guarantee of investment performance.

---

## License

See the repository for the applicable license.

---

## Project Status

**Current focus:**

> Strengthen the data foundation → formalize the ML pipeline → establish rigorous time-aware evaluation → move toward reproducible deployment and monitoring.

The repository is intentionally being built in stages rather than presenting an unfinished prototype as a production financial prediction system.
