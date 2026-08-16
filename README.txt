
# Financial ML Pipeline

An end-to-end financial machine learning project being built incrementally, from data collection to deployment and monitoring.

## Current Stage

### Milestone 1 — Data Ingestion & Dataset Construction

The current implementation focuses on collecting financial data, preserving raw API responses, transforming the data, generating technical indicators, and producing a processed dataset for future machine learning work.

## Data Sources
- Alpha Vantage API
- Yahoo Finance

## Current Pipeline
```text
Alpha Vantage API ──┐
                    │
                    ├──> Data Parsing
                    │
Yahoo Finance ──────┘
                         ↓
                  Date Alignment
                         ↓
              Technical Indicators
                         ↓
                  Dataset Merge
                         ↓
              Processed CSV Dataset

##Implemented
API-based financial data collection
Retry mechanism for connection and timeout failures
HTTP error handling
Request logging
Raw API response storage
JSON response parsing
Date normalization and sorting
Numeric conversion
Dataset alignment by date
Technical indicator generation
Merging multiple financial data sources
Processed CSV dataset generation
Technical Indicators

##The current pipeline generates:

SMA
EMA
RSI
MACD
Bollinger Bands
ATR
OBV
Returns
Volatility
Momentum
Rate of Change

##Project Roadmap
[x] Data collection
[x] Raw data storage
[x] Basic data processing
[x] Feature generation
[x] Dataset construction
[ ] Data validation
[ ] Data cleaning
[ ] Exploratory Data Analysis
[ ] Feature engineering refinement
[ ] Target definition
[ ] Model training
[ ] Model evaluation
[ ] Model selection
[ ] Model serialization
[ ] API deployment
[ ] Dockerization
[ ] User interface
[ ] Monitoring

##Setup
Clone the repository and install the required Python packages:

pip install -r requirements.txt

Create a .env file in the project root and add your own Alpha Vantage API key:
KEY_alpha=your_alpha_vantage_api_key

The .env file is intentionally excluded from version control.

#Running the Pipeline
Run:
python main.py

The pipeline collects the required data, generates the features, and creates the processed dataset.

Note:
Raw API responses, generated datasets, logs, and environment files are excluded from version control. They are generated locally when the pipeline is executed.
