import requests as rq
import pandas as pd
import logging
from pathlib import Path
import sys

Path("log").mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("log/daily_ingestion.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False

df = pd.read_csv("data/processed/daily_latest.csv")
df = df.drop("Date", axis=1)
parameters = df.tail(1)
payload = parameters.to_dict(orient='records')[0]

url = "https://financial-ml-pipeline-up86.onrender.com/Post_values"

try:
    response = rq.post(url=url, json=payload, timeout=(5, 30))
    response.raise_for_status()
    prediction = response.json()
    logger.info(f"Prediction request SUCCESS | url : {url} | prediction : {prediction}")
    print(prediction)
except rq.exceptions.Timeout as t:
    logger.error(f"Prediction request FAILED (timeout) | url : {url} | error : {t}")
    sys.exit(1)
except rq.exceptions.ConnectionError as c:
    logger.error(f"Prediction request FAILED (connection error) | url : {url} | error : {c}")
    sys.exit(1)
except rq.exceptions.HTTPError as h:
    logger.error(f"Prediction request FAILED (HTTP error) | url : {url} | status : {response.status_code} | response : {response.text[:200]}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Prediction request FAILED (unexpected) | url : {url} | error : {e}")
    sys.exit(1)
