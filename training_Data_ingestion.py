import requests
import time
from datetime import datetime,timezone
from tenacity import (retry,stop_after_attempt,wait_exponential,retry_if_exception_type)
import json,csv,os,pandas as pd,logging
from pathlib import Path
from joblib import dump
from dotenv import load_dotenv
import numpy as np
import yfinance as yf

load_dotenv()
key = os.getenv("KEY_alpha")
Path("log").mkdir(parents=True,exist_ok=True)

logging.basicConfig(
    filename="log/ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1,min=2,max=20),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True)
def fetch_data(URL:str,query:dict,endpoint_name:str):
    # today = datetime.now().strftime("%d-%m-%y")
    try:
        response = requests.get(url=URL,params=query,timeout=(5,30))
        response.raise_for_status()
        query['apikey'] = "your_alpha_vantage_key"
    except requests.exceptions.ConnectionError as c:
        logger.info(f"ConnectionError | endpoint : {endpoint_name} | query : {query} | error : {c}")
        raise
    except requests.exceptions.HTTPError as h:
        logger.info(f"HTTPError | endpoint : {endpoint_name} | query : {query} | error : {h}")
        raise
    except requests.RequestException as e:
        logger.info(f"RequestException | endpoint : {endpoint_name} | query : {query} | error : {e}")
        raise

    logger.info(f"Request | endpoint : {endpoint_name} | query :{query}")
    try:
        data = response.json()
    except json.JSONDecodeError as j:
        logger.error(f"Json Prraing Error | endpoint : {endpoint_name} | query : {query} | error : {j} | portion of response : {response.text[:100]}")
        raise

    logger.info(f"Info | endpoint : {endpoint_name} | fetched {len(data)}  data | query : {query}")
    return data

def save_raw_response(data:dict,endpoint:str):
    today = datetime.now().strftime("%d-%m-%y")
    file_path = Path(f"data/raw_response")
    file_path.mkdir(parents=True,exist_ok=True)

    file_name = f"{file_path}/{today}_{endpoint}.json"
    try:
        with open(file_name,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
            logger.info(f"Saved Raw Response | file : {file_name}")
    except Exception as e:
        logger.error(f"Error in saving raw response | error : {e} | intended file : {file_name}")

def data_prasing(d2:dict,d3:dict,d4:dict):
    # full1 = []
    # for i,j in d1["Time Series (Daily)"].items():
    #     chunk = []
    #     row = [i,j["1. open"],j["2. high"],j["3. low"],j["4. close"],j["5. adjusted"],j["6. volume"],j["7. dividend amount"],j["8. split coefficient"]]
    #     chunk.append(row)
    #     full1.append(chunk)
    # df1 = pd.DataFrame(full1,columns=["Date","open","high","low","close","adjusted","volume","dividend_amount","split_coefficient"])

    full2 = []
    for i in d2["data"]:
        chunk = []
        chunk = [i["date"],i["value"]]
        full2.append(chunk)
    df2 = pd.DataFrame(full2,columns=["Date","Value"])

    full3 = []
    for i in d3["data"]:
        chunk = []
        chunk = [i["date"],i["value"]]
        full3.append(chunk)
    df3 = pd.DataFrame(full3,columns=["Date","wti_value"])

    full4 = []
    for i in d4["data"]:
        chunk = []
        chunk = [i["date"],i["price"]]
        full4.append(chunk)
    df4 = pd.DataFrame(full4,columns=["Date","gold_price"])
    return df2,df3,df4

def frame_prep(df:pd.DataFrame,date_col:str = "Date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df  = df.sort_values(date_col).set_index(date_col)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c],errors="coerce")
    return df

def add_sma(df, price_col="adjusted", windows=(10, 20, 50)):
    for w in windows:
        df[f"SMA_{w}"] = df[price_col].rolling(window=w).mean()
    return df

def add_ema(df, price_col="adjusted", windows=(10, 20, 50)):
    for w in windows:
        df[f"EMA_{w}"] = df[price_col].ewm(span=w, adjust=False).mean()
    return df

def add_rsi(df, price_col="adjusted", period=14):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, price_col="adjusted", fast=12, slow=26, signal=9):
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_bbands(df, price_col="adjusted", window=20, num_std=2):
    mid = df[price_col].rolling(window=window).mean()
    std = df[price_col].rolling(window=window).std()

    df[f"BB_mid_{window}"] = mid
    df[f"BB_upper_{window}"] = mid + num_std * std
    df[f"BB_lower_{window}"] = mid - num_std * std
    df[f"BB_pctB_{window}"] = (df[price_col] - df[f"BB_lower_{window}"]) / (
        df[f"BB_upper_{window}"] - df[f"BB_lower_{window}"]
    )
    return df


def add_atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return df


def add_obv(df, price_col="adjusted"):
    direction = np.sign(df[price_col].diff()).fillna(0)
    df["OBV"] = (direction * df["volume"]).cumsum()
    return df


def add_returns(df, price_col="adjusted"):
    df["return_pct"] = df[price_col].pct_change(fill_method = None)
    df["return_log"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_volatility(df, windows=(10, 20)):
    if "return_pct" not in df.columns:
        df = add_returns(df)
    for w in windows:
        df[f"volatility_{w}"] = df["return_pct"].rolling(window=w).std() * np.sqrt(252)
    return df


def add_momentum(df, price_col="adjusted", periods=(10, 20)):
    for p in periods:
        df[f"momentum_{p}"] = df[price_col] - df[price_col].shift(p)
        df[f"roc_{p}"] = df[price_col].pct_change(fill_method = None,periods=p) * 100
    return df


def add_all_indicators(df):
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bbands(df)
    df = add_atr(df)       # needs high/low/close — raw OHLC, not adjusted
    df = add_obv(df)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_momentum(df)
    return df

def merged_data(df1,df2,df3,df4):
    frames = {"ibm":df1,"treasury":df2,"wti":df3,"Gold_history":df4}
    start_least_old = {name:df.index.min() for name,df in frames.items()}

    cut_off = max(start_least_old.values())
    limiting_source = max(start_least_old,key = start_least_old.get)
    logger.info(f"Limiting dataset: {limiting_source} | cutoff date: {cut_off}")

    trimmed = {name:df[df.index >= cut_off] for name,df in frames.items()}
    merged = trimmed["ibm"].join([trimmed["treasury"],trimmed["wti"],trimmed["Gold_history"]],how="left")
    merged = merged.sort_index(ascending=True)
    return merged

def fetch_all_data():
    url = "https://www.alphavantage.co/query"

    y = yf.download("IBM",interval="1d",period="max",auto_adjust=False)
    if isinstance(y.columns,pd.MultiIndex):
        y.columns = y.columns.get_level_values(0)

    y = y.rename(columns={"Open": "open", "High": "high", "Low": "low","Close": "close", "Adj Close": "adjusted", "Volume": "volume",})
    y.index.name = "Date"
    y = y.reset_index()
    logger.info(f"Fetched IBM ohlcv data using downloaded api| {len(y)} rows and {len(y.columns)} columns")
    # query1 = {
    #     "function":"TIME_SERIES_DAILY_ADJUSTED",
    #     "symbol":"IBM",
    #     "outputsize":"full",
    #     "datatype":"json",
    #     "entitlement":"realtime",
    #     "apikey":key
    # }
    # TBM = fetch_data(url,query1,"TIME_SERIES_DAILY_ADJUSTED")
    # save_raw_response(TBM,"TIME_SERIES_DAILY_ADJUSTED")

    query2 = {
        "function":"TREASURY_YIELD",
        "interval":"daily",
        "maturity":"30year",
        "datatype":"json",
        "apikey":key
        }
    treasury = fetch_data(url,query2,"Tresury")
    save_raw_response(treasury,"Treasury")

    query3 = {
        "function":"WTI",
        "interval":"daily",
        "datatype":"json",
        "apikey":key
    }
    wti = fetch_data(url,query3,"WTI")
    save_raw_response(wti,"WTI")

    query4 = {
        "function":"GOLD_SILVER_HISTORY",
        "symbol":"GOLD",
        "interval":"daily",
        "apikey":key
    }
    gold_history = fetch_data(url,query4,"Gold_history")
    save_raw_response(gold_history,"Gold_history")
    df2,df3,df4 = data_prasing(treasury,wti,gold_history)
    y  = frame_prep(y,"Date")
    y = add_all_indicators(y)
    df2 = frame_prep(df2,"Date")
    df3 = frame_prep(df3,"Date")
    df4 = frame_prep(df4,"Date")
    final = merged_data(y,df2,df3,df4)
    folder = Path("data/processed")
    folder.mkdir(parents=True,exist_ok=True)
    final.to_csv(f"{folder}/merged_imb_dataset.csv",index = True)
    logger.info(f"Training Data saved successfully | file name : data/processed/merged_imb_dataset.csv | final_length : {len(final)} | missing_values : {final.isnull().sum().sum()}")


if __name__ == "__main__":
    fetch_all_data()
