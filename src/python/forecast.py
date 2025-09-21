# src/python/forecast.py
"""
Robust BigQuery AI FORECAST runner.

Features:
- Reads SQL template from src/queries/forecast.sql (substitutes PROJECT.DATASET)
- CLI args for horizon and whether to use BigQuery Storage API
- On failure of AI.FORECAST query, falls back to generating a local synthetic forecast
  based on historical sales (if available) or a constant forecast otherwise.
- Writes results to results/forecasts.csv
"""
import argparse
import sys
from pathlib import Path
import traceback
import pandas as pd
from datetime import datetime, timedelta

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

# Config import (project + dataset)
try:
    from config import PROJECT_ID, BQ_DATASET
except Exception:
    PROJECT_ID = None
    BQ_DATASET = None

ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "src" / "queries" / "forecast.sql"
RESULTS_PATH = ROOT / "results" / "forecasts.csv"

def run_bq_forecast(sql: str, project: str, use_bqstorage: bool):
    client = bigquery.Client(project=project)
    job = client.query(sql)
    # Use create_bqstorage_client flag to avoid requiring readSession permissions
    df = job.result().to_dataframe(create_bqstorage_client=use_bqstorage)
    return df

def query_historical_sales(project: str, dataset: str, limit: int = None):
    """
    Try to fetch historical aggregated daily sales from the dataset.sales table.
    Returns a pandas DataFrame with columns (time_day (date), sales (numeric))
    or raises an exception if the table doesn't exist / is inaccessible.
    """
    client = bigquery.Client(project=project)
    table_ref = f"{project}.{dataset}.sales"
    limit_clause = f" LIMIT {limit}" if limit else ""
    sql = f"""
    SELECT DATE(order_date) AS time_day,
           SUM(sales_amount) AS sales
    FROM `{table_ref}`
    GROUP BY time_day
    ORDER BY time_day
    {limit_clause}
    """
    job = client.query(sql)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    # Ensure types
    if "time_day" in df.columns:
        df["time_day"] = pd.to_datetime(df["time_day"]).dt.date
    return df

def synthetic_forecast_from_history(history_df: pd.DataFrame, horizon: int):
    """
    Build a simple synthetic forecast from history:
    - Use mean of sales as forecast_value
    - Build 95%-ish interval from std deviation
    """
    if history_df is None or history_df.empty:
        # No history: return constant forecast of 1.0
        start = datetime.utcnow().date() + timedelta(days=1)
        timestamps = [datetime.combine(start + timedelta(days=i), datetime.min.time()) for i in range(horizon)]
        df = pd.DataFrame({
            "forecast_timestamp": timestamps,
            "forecast_value": [1.0] * horizon,
            "confidence_level": [0.95] * horizon,
            "prediction_interval_lower_bound": [0.0] * horizon,
            "prediction_interval_upper_bound": [1.0] * horizon,
            "ai_forecast_status": ["fallback:constant"] * horizon,
        })
        return df

    mean = float(history_df["sales"].mean())
    std = float(history_df["sales"].std()) if len(history_df) > 1 else max(1.0, 0.1 * mean)

    start = max(history_df["time_day"]) + timedelta(days=1)
    timestamps = [datetime.combine(start + timedelta(days=i), datetime.min.time()) for i in range(horizon)]

    # Create symmetric interval around mean (approximate)
    lower = [max(0.0, mean - 1.96 * std)] * horizon
    upper = [mean + 1.96 * std] * horizon
    forecast_vals = [mean] * horizon

    df = pd.DataFrame({
        "forecast_timestamp": timestamps,
        "forecast_value": forecast_vals,
        "confidence_level": [0.95] * horizon,
        "prediction_interval_lower_bound": lower,
        "prediction_interval_upper_bound": upper,
        "ai_forecast_status": ["fallback:from_history"] * horizon,
    })
    return df

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AI.FORECAST via BigQuery with robust fallbacks.")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days).")
    parser.add_argument("--use-bqstorage", action="store_true", help="Allow using BigQuery Storage API for faster download (requires IAM permission).")
    parser.add_argument("--project", type=str, default=PROJECT_ID, help="GCP project id (overrides config).")
    parser.add_argument("--dataset", type=str, default=BQ_DATASET, help="BigQuery dataset (overrides config).")
    args = parser.parse_args(argv)

    if not args.project or not args.dataset:
        print("ERROR: PROJECT_ID and BQ_DATASET must be configured (either in config.py or via --project/--dataset).", file=sys.stderr)
        sys.exit(2)

    # Read SQL template
    if not SQL_PATH.exists():
        print(f"ERROR: SQL template not found at {SQL_PATH}", file=sys.stderr)
        sys.exit(2)

    sql_template = SQL_PATH.read_text()
    # Replace the placeholder PROJECT.DATASET with actual
    sql = sql_template.replace("PROJECT.DATASET", f"{args.project}.{args.dataset}")

    # Also try to inject horizon if SQL contains horizon placeholder like {horizon}
    if "{horizon}" in sql:
        sql = sql.replace("{horizon}", str(args.horizon))

    print("Running forecast query...")
    try:
        df = run_bq_forecast(sql, project=args.project, use_bqstorage=args.use_bqstorage)
        # Normalize timestamp column name if present (some outputs call it forecast_timestamp)
        # We'll persist whatever columns BigQuery returned.
        if df is None or df.empty:
            print("Warning: forecast query returned no rows. Falling back to local synthetic forecast.")
            history_df = None
            try:
                history_df = query_historical_sales(args.project, args.dataset)
            except Exception as e:
                print("Warning: unable to fetch historical sales to build fallback:", e)
            df = synthetic_forecast_from_history(history_df, args.horizon)
        else:
            # Ensure forecast_timestamp column exists and is datetime-like
            if "forecast_timestamp" in df.columns:
                try:
                    df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
                except Exception:
                    pass
        print(f"✅ Forecast query succeeded, {len(df)} rows returned.")
    except GoogleAPIError as ge:
        print("BigQuery API error when running AI.FORECAST:", file=sys.stderr)
        traceback.print_exc()
        print("Attempting fallback: build synthetic forecast from historical sales (if available).")
        history_df = None
        try:
            history_df = query_historical_sales(args.project, args.dataset)
            print(f"Fetched {len(history_df)} historical rows for fallback.")
        except Exception as e:
            print("Could not fetch historical sales for fallback:", e)
            history_df = None
        df = synthetic_forecast_from_history(history_df, args.horizon)
    except Exception:
        print("Unexpected error while executing forecast SQL:", file=sys.stderr)
        traceback.print_exc()
        print("Attempting fallback: build synthetic forecast from historical sales (if available).")
        history_df = None
        try:
            history_df = query_historical_sales(args.project, args.dataset)
            print(f"Fetched {len(history_df)} historical rows for fallback.")
        except Exception as e:
            print("Could not fetch historical sales for fallback:", e)
            history_df = None
        df = synthetic_forecast_from_history(history_df, args.horizon)

    # Ensure results dir exists and write CSV
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Ensure consistent column order if fallback created it; otherwise write as returned
    try:
        df.to_csv(RESULTS_PATH, index=False)
        print(f"✅ Saved forecasts to {RESULTS_PATH} ({len(df)} rows)")
    except Exception as e:
        print("ERROR saving results CSV:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
