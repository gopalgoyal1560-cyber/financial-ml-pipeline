
import json, os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pandas as pd
import yfinance as yf
from genson import SchemaBuilder
from deepdiff import DeepDiff
import time

from training_Data_ingestion import (
    key,
    fetch_data,
    save_raw_response,
    data_prasing,
    frame_prep,
    add_all_indicators,
    merged_data,
)

Path("log").mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("log/daily_ingestion.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False  # don't also send these up to the root logger / ingestion.log

# --------------------------------------------------------------------------
# 1) LOAD a previously saved raw json file (same naming convention as save_raw_response)
# --------------------------------------------------------------------------
def load_raw_response(endpoint: str, date: str = None):
    """
    Loads a raw json file saved by save_raw_response() in training_Data_ingestion.py.
    date must be 'dd-mm-yy' (same as the file naming). Defaults to today.
    Returns parsed json on success, or None if missing/corrupt - never raises.
    """
    date = date or datetime.now().strftime("%d-%m-%y")
    file_name = Path(f"data/raw_response/31-08-26_{endpoint}.json")

    if not file_name.exists():
        logger.warning(f"Raw response file not found | file : {file_name}")
        return None
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded Raw Response | file : {file_name}")
        return data
    except json.JSONDecodeError as j:
        logger.error(f"Corrupt raw response file | file : {file_name} | error : {j}")
        return None
    except Exception as e:
        logger.error(f"Error loading raw response | file : {file_name} | error : {e}")
        return None


# --------------------------------------------------------------------------
# 2) BUILD SCHEMA using genson (json data, or a df's column/dtype map)
# --------------------------------------------------------------------------
def build_schema(data):
    if data is None:
        return None
    try:
        builder = SchemaBuilder()
        if isinstance(data, list):
            for item in data:
                builder.add_object(item)
        else:
            builder.add_object(data)
        return builder.to_schema()
    except Exception as e:
        logger.error(f"Error building schema | error : {e}")
        return None


def get_df_schema(df: pd.DataFrame) -> dict:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


# --------------------------------------------------------------------------
# 3) COMPARE two schemas -> pass/fail (try/except + return only, no sys.exit)
# --------------------------------------------------------------------------
def compare_schema(old_schema: dict, new_schema: dict):
    diff = DeepDiff(old_schema, new_schema, ignore_order=True)
    breaking = {}
    non_breaking = {}

    if "dictionary_item_removed" in diff:
        breaking["fields_removed"] = list(diff["dictionary_item_removed"])

    if "type_changes" in diff:
        breaking["type_changed"] = {
            str(k): {"old": str(v["old_type"]), "new": str(v["new_type"])}
            for k, v in diff["type_changes"].items()
        }
    if "values_changed" in diff:
        type_value_changes = {
            str(k): {"old": v["old_value"], "new": v["new_value"]}
            for k, v in diff["values_changed"].items()
            if str(k).endswith("['type']")
        }
        if type_value_changes:
            breaking.setdefault("type_changed", {}).update(type_value_changes)

    if "dictionary_item_added" in diff:
        non_breaking["fields_added"] = list(diff["dictionary_item_added"])

    return breaking, non_breaking


def validate_schema(old_data, new_data, source_name: str) -> bool:
    """
    Gate for json endpoints (Treasury / WTI / Gold_history).
    old_data / new_data are raw json (dict/list) - schemas are built here.
    If old_data is None (no previous file to compare against), logs a warning and passes.
    Returns True/False - never raises/exits.
    """
    try:
        if old_data is None:
            logger.warning(f"No previous data to compare against for {source_name} - skipping check, treating as PASS.")
            return True

        old_schema = build_schema(old_data)
        new_schema = build_schema(new_data)
        if old_schema is None or new_schema is None:
            logger.error(f"Schema validation FAILED | source : {source_name} | reason : could not build schema")
            return False

        breaking, non_breaking = compare_schema(old_schema, new_schema)
        if breaking:
            logger.error(f"Schema validation FAILED (STOP) | source : {source_name} | breaking : {json.dumps(breaking)}")
            return False

        if non_breaking:
            logger.info(f"Schema validation PASSED (additions) | source : {source_name} | {json.dumps(non_breaking)}")
        else:
            logger.info(f"Schema validation PASSED (unchanged) | source : {source_name}")
        return True

    except Exception as e:
        logger.error(f"Schema validation FAILED (unexpected error) | source : {source_name} | error : {e}")
        return False


def validate_df_schema(old_df: pd.DataFrame, new_df: pd.DataFrame, name: str) -> bool:
    """
    Gate for a DataFrame - compares df.columns and dtypes between old_df and new_df, nothing saved.
    If old_df is None, logs a warning and passes.
    Returns True/False - never raises/exits.
    """
    try:
        if old_df is None:
            logger.warning(f"No previous data to compare against for {name} - skipping check, treating as PASS.")
            return True

        old_schema = get_df_schema(old_df)
        new_schema = get_df_schema(new_df)

        missing_cols = [c for c in old_schema if c not in new_schema]
        added_cols = [c for c in new_schema if c not in old_schema]
        dtype_changed = {
            c: {"old": old_schema[c], "new": new_schema[c]}
            for c in old_schema
            if c in new_schema and old_schema[c] != new_schema[c]
        }

        if missing_cols or dtype_changed:
            logger.error(
                f"DataFrame schema validation FAILED (STOP) | name : {name} | "
                f"missing_columns : {missing_cols} | dtype_changed : {dtype_changed}"
            )
            return False

        if added_cols:
            logger.info(f"DataFrame schema validation PASSED (new columns) | name : {name} | added : {added_cols}")
        else:
            logger.info(f"DataFrame schema validation PASSED (unchanged) | name : {name}")
        return True

    except Exception as e:
        logger.error(f"DataFrame schema validation FAILED (unexpected error) | name : {name} | error : {e}")
        return False


# --------------------------------------------------------------------------
# Impute + save last (today's) row
# --------------------------------------------------------------------------
def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    before_na = df.isnull().sum().sum()
    df = df.ffill()
    after_na = df.isnull().sum().sum()
    logger.info(f"Imputed merged dataset with ffill() | missing before : {before_na} | missing after : {after_na}")
    return df


def save_latest_row(df: pd.DataFrame):
    folder = Path("data/processed")
    folder.mkdir(parents=True, exist_ok=True)
    file_name = folder / "daily_latest.csv"

    last_row = df.tail(1)
    write_header = not file_name.exists()
    last_row.to_csv(file_name, mode="a", header=write_header, index=True)
    logger.info(f"Saved latest row | file : {file_name} | date : {last_row.index[-1]}")


# --------------------------------------------------------------------------
# Main daily pipeline - orchestration only, all heavy lifting imported
# --------------------------------------------------------------------------
def fetch_daily_data():
    url = "https://www.alphavantage.co/query"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%d-%m-%y")

    y = yf.download("IBM", interval="1d", period="1y", auto_adjust=False)
    if isinstance(y.columns, pd.MultiIndex):
        y.columns = y.columns.get_level_values(0)
    y = y.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adjusted", "Volume": "volume"})
    y.index.name = "Date"
    y = y.reset_index()
    logger.info(f"Fetched IBM ohlcv data | {len(y)} rows and {len(y.columns)} columns")
    time.sleep(1)
    # compare against yesterday's row already sitting in daily_latest.csv, if it exists
    prev_row_path = Path("data/processed/merged_imb_dataset2.csv")
    old_y = pd.read_csv(prev_row_path) if prev_row_path.exists() else None
    if not validate_df_schema(old_y[["adjusted","close","high","low","open","volume"]].astype({"volume": "float64"}), y.drop("Date",axis=1).astype("float64"), "ibm_ohlcv"):
        logger.critical("Stopping pipeline: IBM OHLCV schema check failed.")
        return None

    query2 = {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "30year", "datatype": "json", "apikey": key}
    treasury = fetch_data(url, query2, "Treasury")
    old_treasury = load_raw_response("Treasury", yesterday)
    if not validate_schema(old_treasury, treasury, "Treasury"):
        logger.critical("Stopping pipeline: Treasury schema check failed.")
        return None
    time.sleep(1)
    query3 = {"function": "WTI", "interval": "daily", "datatype": "json", "apikey": key}
    wti = fetch_data(url, query3, "WTI")
    old_wti = load_raw_response("WTI", yesterday)
    if not validate_schema(old_wti, wti, "WTI"):
        logger.critical("Stopping pipeline: WTI schema check failed.")
        return None
    time.sleep(1)
    query4 = {"function": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily", "apikey": key}
    gold_history = fetch_data(url, query4, "Gold_history")
    old_gold = load_raw_response("Gold_history", yesterday)
    if not validate_schema(old_gold, gold_history, "Gold_history"):
        logger.critical("Stopping pipeline: Gold_history schema check failed.")
        return None
    time.sleep(1)
    # all schema checks passed -> reuse the imported parsing/merge pipeline
    df2, df3, df4 = data_prasing(treasury, wti, gold_history)
    y = frame_prep(y, "Date")
    y = add_all_indicators(y)
    df2 = frame_prep(df2, "Date")
    df3 = frame_prep(df3, "Date")
    df4 = frame_prep(df4, "Date")

    final = merged_data(y, df2, df3, df4)
    final = impute_data(final)
    save_latest_row(final)

    logger.info(f"Daily pipeline completed successfully | final_length : {len(final)} | missing_values : {final.isnull().sum().sum()}")
    return final


if __name__ == "__main__":
    result = fetch_daily_data()
    if result is None:
        logger.error("Daily pipeline did not complete due to a schema validation failure - check log/daily_ingestion.log")
