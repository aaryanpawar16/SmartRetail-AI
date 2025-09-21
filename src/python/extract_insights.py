"""
Run AI.GENERATE_TABLE to extract structured insights from support calls,
save results to results/support_insights.csv.

If AI.GENERATE_TABLE fails (model syntax/permission), the script will:
 - fall back to a simple SQL-based heuristic extraction (keyword matching + short summary),
 - still save results so your demo can proceed.
"""
from google.cloud import bigquery
from google.oauth2 import service_account
from pathlib import Path
import pandas as pd
import traceback
import json
import sys
import logging
import numpy as np
import os

# Try to load from Streamlit secrets if available
try:
    import streamlit as st
    if "gcp_service_account" in st.secrets:
        sa_info = dict(st.secrets["gcp_service_account"])
        CREDENTIALS = service_account.Credentials.from_service_account_info(sa_info)
        PROJECT_ID = sa_info["project_id"]
    else:
        CREDENTIALS = None
        from config import PROJECT_ID, BQ_DATASET
except ImportError:
    # Not running inside Streamlit, fall back to config.py
    CREDENTIALS = None
    from config import PROJECT_ID, BQ_DATASET

from config import BQ_DATASET

ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "src" / "queries" / "extract_insights.sql"
OUT_PATH = ROOT / "results" / "support_insights.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def safe_print(s: str):
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or 'utf-8'
        print(s.encode('utf-8', errors='replace').decode(enc, errors='replace'))


def make_bq_client():
    if CREDENTIALS:
        return bigquery.Client(credentials=CREDENTIALS, project=PROJECT_ID)
    return bigquery.Client(project=PROJECT_ID)


def run_ai_generate_table(sql_text: str) -> pd.DataFrame:
    client = make_bq_client()
    logger.info("Submitting query to BigQuery (job will run asynchronously)...")
    job = client.query(sql_text)
    try:
        logger.info("Waiting for BigQuery job result...")
        df = job.result().to_dataframe(create_bqstorage_client=False)
        logger.info("BigQuery job completed. Rows: %s", len(df))
        return df
    except Exception as e:
        logger.error("BigQuery job failed: %s", repr(e))
        try:
            job_id = getattr(job, 'job_id', None)
            logger.error("Job ID: %s", job_id)
            logger.error("Job errors: %s", getattr(job, 'errors', None))
            logger.error("Job error_result: %s", getattr(job, 'error_result', None))
        except Exception:
            pass
        raise


def fallback_heuristic() -> pd.DataFrame:
    client = make_bq_client()
    q = f"SELECT call_id, customer_id, call_text, call_timestamp FROM `{PROJECT_ID}.{BQ_DATASET}.support_calls` ORDER BY call_timestamp"
    df = client.query(q).result().to_dataframe(create_bqstorage_client=False)

    def naive_sentiment(text: str) -> str:
        txt = (text or "").lower()
        neg = ["not", "unusable", "damage", "damaged", "refund", "charged twice", "reboot", "fail"]
        pos = ["great", "thanks", "thank you", "good", "excellent", "easy"]
        if any(k in txt for k in neg):
            return "negative"
        if any(k in txt for k in pos):
            return "positive"
        return "neutral"

    def naive_priority(text: str) -> str:
        txt = (text or "").lower()
        high = ["refund", "escalat", "unusable", "charge", "charged", "replacement"]
        if any(k in txt for k in high):
            return "high"
        return "medium"

    rows = []
    for _, r in df.iterrows():
        text = r.get("call_text") if pd.notna(r.get("call_text")) else ""
        summary = (text or "").strip()[:160]
        sentiment = naive_sentiment(text)
        priority = naive_priority(text)
        tags = []
        for kw in ["refund","replacement","shipping","update","warranty","escalate","subscription","charge","address","reboot"]:
            if kw in (text or "").lower():
                tags.append(kw)
        actions = []
        low = (text or "").lower()
        if "refund" in low or "charged" in low:
            actions.append("Investigate billing; process refund if valid")
        if "replacement" in low or "damaged" in low:
            actions.append("Initiate replacement/shipping")
        if "escalat" in low or "unusable" in low:
            actions.append("Escalate to engineering")
        if "address" in low:
            actions.append("Update shipping address")
        rows.append({
            "call_id": r["call_id"],
            "summary": summary,
            "sentiment": sentiment,
            "priority": priority,
            "tags": tags,
            "action_items": actions
        })
    return pd.DataFrame(rows)


def normalize_cell(v) -> str:
    if v is None:
        return "[]"
    try:
        if pd.isna(v):
            return "[]"
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return json.dumps(v)
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist())
    if isinstance(v, pd.Series):
        return json.dumps(v.tolist())
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v)
    except TypeError:
        return json.dumps(str(v))


def main():
    SQL = SQL_PATH.read_text().replace("PROJECT.DATASET", f"{PROJECT_ID}.{BQ_DATASET}")
    safe_print("Running AI.GENERATE_TABLE extraction (this may fail)...")

    debug_dir = ROOT / "tmp"
    debug_dir.mkdir(exist_ok=True)
    (debug_dir / "last_query.sql").write_text(SQL, encoding="utf-8")

    try:
        df = run_ai_generate_table(SQL)
        safe_print("✅ AI.GENERATE_TABLE succeeded.")
    except Exception:
        safe_print("⚠️ AI.GENERATE_TABLE failed. Falling back to heuristic extraction.")
        logger.exception("AI.GENERATE_TABLE failure details:")
        df = fallback_heuristic()
        safe_print("✅ Fallback extraction completed.")

    for col in ["tags", "action_items"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_cell)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    safe_print(f"✅ Saved insights to {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
