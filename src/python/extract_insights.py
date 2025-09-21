"""
Run AI.GENERATE_TABLE to extract structured insights from support calls,
save results to results/support_insights.csv.

If AI.GENERATE_TABLE fails (model syntax/permission), the script will:
 - fall back to a simple SQL-based heuristic extraction (keyword matching + short summary),
 - still save results so your demo can proceed.
"""
from google.cloud import bigquery
from pathlib import Path
import pandas as pd
import traceback
import json
import sys
import logging
import numpy as np
from config import PROJECT_ID, BQ_DATASET

ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "src" / "queries" / "extract_insights.sql"
OUT_PATH = ROOT / "results" / "support_insights.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Try to force utf-8 stdout on Windows so emoji/Unicode prints don't crash
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    # Older Pythons or environments may not support reconfigure; that's fine.
    pass


def safe_print(s: str):
    """Print but avoid crashing if the console encoding can't represent some chars."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or 'utf-8'
        print(s.encode('utf-8', errors='replace').decode(enc, errors='replace'))


def run_ai_generate_table(sql_text: str) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    logger.info("Submitting query to BigQuery (job will run asynchronously)...")
    logger.debug("SQL being submitted:\n%s", sql_text)

    job = client.query(sql_text)
    try:
        logger.info("Waiting for BigQuery job result (this can take a while)...")
        df = job.result().to_dataframe(create_bqstorage_client=False)
        logger.info("BigQuery job completed successfully. Rows: %s", len(df))
        return df
    except Exception as e:
        # Log rich job diagnostics if available and re-raise for the caller to handle
        logger.error("BigQuery job failed: %s", repr(e))
        try:
            job_id = getattr(job, 'job_id', None)
            logger.error("Job ID: %s", job_id)
            logger.error("Job errors: %s", getattr(job, 'errors', None))
            logger.error("Job error_result: %s", getattr(job, 'error_result', None))
            logger.error("Job statistics: %s", getattr(job, 'statistics', None))
        except Exception as ex2:
            logger.exception("Error while printing job diagnostics: %s", ex2)
        raise


def fallback_heuristic() -> pd.DataFrame:
    """
    A tiny fallback that summarizes text using simple heuristics:
    - summary: first 120 chars
    - sentiment: naive by keywords
    - priority: high if contains keywords 'refund', 'escalate', 'unusable', 'charged twice'
    - tags: keywords found
    - action_items: suggested actions
    """
    client = bigquery.Client(project=PROJECT_ID)
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
    out = pd.DataFrame(rows)
    return out


def normalize_cell(v) -> str:
    # None or NaN -> empty JSON array
    if v is None:
        return "[]"
    # pd.isna may raise/return array-like for complex objects; guard it
    try:
        if pd.isna(v):
            return "[]"
    except Exception:
        pass

    # list/tuple -> JSON string
    if isinstance(v, (list, tuple)):
        return json.dumps(v)

    # numpy arrays -> convert to list then JSON
    if isinstance(v, np.ndarray):
        try:
            return json.dumps(v.tolist())
        except Exception:
            return json.dumps([str(x) for x in v.tolist()])

    # pandas Series -> tolist
    try:
        if isinstance(v, pd.Series):
            return json.dumps(v.tolist())
    except Exception:
        pass

    # If already a string, return as-is
    if isinstance(v, str):
        return v

    # For dict or other JSON-serializable types
    try:
        return json.dumps(v)
    except TypeError:
        # last resort: convert to string
        return json.dumps(str(v))


def main():
    SQL = SQL_PATH.read_text().replace("PROJECT.DATASET", f"{PROJECT_ID}.{BQ_DATASET}")
    safe_print("Running AI.GENERATE_TABLE extraction (this may fail if model reference/permissions differ)...")

    # Save generated SQL for easier debugging (numbered lines)
    debug_dir = ROOT / "tmp"
    debug_dir.mkdir(exist_ok=True)
    debug_path = debug_dir / "last_query.sql"
    debug_path.write_text(SQL, encoding="utf-8")
    with open(debug_dir / "last_query_numbered.sql", "w", encoding="utf-8") as fh:
        for i, ln in enumerate(SQL.splitlines(), start=1):
            fh.write(f"{i:4d}: {ln}\n")
    logger.info("Saved generated SQL to %s and numbered view to %s", debug_path, debug_dir / "last_query_numbered.sql")
    logger.debug("Generated SQL:\n%s", SQL)

    try:
        df = run_ai_generate_table(SQL)
        safe_print("\u2705 AI.GENERATE_TABLE succeeded.")
    except Exception:
        # Use safe_print to avoid Unicode crashes in Windows consoles
        safe_print("WARNING: AI.GENERATE_TABLE failed. Falling back to heuristic extraction.")
        logger.exception("AI.GENERATE_TABLE failure details:")
        # If it's a syntax error from BigQuery, printing the SQL above (logger.debug) + job diagnostics
        # will help pinpoint the exact character/line causing the parse error.
        df = fallback_heuristic()
        safe_print("\u2705 Fallback extraction completed.")

    # Normalize arrays to JSON strings for CSV output (so CSV cells are readable)
    for col in ["tags", "action_items"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_cell)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    safe_print(f"\u2705 Saved insights to {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
