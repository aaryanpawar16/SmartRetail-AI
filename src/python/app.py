# src/python/app.py
"""
SmartRetail AI — Streamlit demo
Shows:
 - Forecast chart (with CI)
 - Personalized email samples
 - Support call insights (exec dashboard)
 - Simple product-similarity demo (local TF-IDF)
 - Buttons to run the existing forecast/personalize/insights scripts
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import subprocess
import shlex
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

st.set_page_config(page_title="SmartRetail AI – Demo", layout="wide")
st.title("SmartRetail AI — Demo")

# --- Helper to run scripts safely ---
def run_script_and_report(cmd: str):
    try:
        result = subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

# --- Section: Run jobs ---
st.sidebar.header("Actions")

if st.sidebar.button("Run Forecast (AI.FORECAST)"):
    with st.spinner("Running forecast.py..."):
        ok, out = run_script_and_report("python src/python/forecast.py")
        if ok:
            st.sidebar.success("Forecast completed")
        else:
            st.sidebar.error("Forecast script failed")
            st.sidebar.text(out)

if st.sidebar.button("Run Personalize (generate emails)"):
    with st.spinner("Running personalize.py..."):
        ok, out = run_script_and_report("python src/python/personalize.py")
        if ok:
            st.sidebar.success("Personalization completed")
        else:
            st.sidebar.error("Personalize script failed")
            st.sidebar.text(out)

if st.sidebar.button("Run Extract Insights (support calls)"):
    with st.spinner("Running extract_insights.py..."):
        ok, out = run_script_and_report("python src/python/extract_insights.py")
        if ok:
            st.sidebar.success("Support call insights completed")
        else:
            st.sidebar.error("Extract insights script failed")
            st.sidebar.text(out)

st.sidebar.markdown("---")
st.sidebar.info("This demo uses your existing scripts and the results CSVs in `results/`.")

# --- Forecasts ---
st.header("Forecasts")
forecast_file = RESULTS / "forecasts.csv"
if forecast_file.exists():
    try:
        df = pd.read_csv(forecast_file, parse_dates=["forecast_timestamp"])
        st.subheader("Raw forecast table")
        st.dataframe(df)

        # plot forecast with CI
        st.subheader("Forecast chart (with 95% CI)")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_sorted = df.sort_values("forecast_timestamp")
        ax.plot(df_sorted["forecast_timestamp"], df_sorted["forecast_value"], label="Forecast")
        if "prediction_interval_lower_bound" in df.columns and "prediction_interval_upper_bound" in df.columns:
            ax.fill_between(
                df_sorted["forecast_timestamp"],
                df_sorted["prediction_interval_lower_bound"],
                df_sorted["prediction_interval_upper_bound"],
                alpha=0.2,
                label="95% CI",
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Forecast value")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading forecast CSV: {e}")
else:
    st.info("No forecasts found. Click the sidebar button to run forecast.py (creates results/forecasts.csv)")

# --- Personalized emails ---
st.header("Personalized emails")
emails_file = RESULTS / "personalized_emails.csv"
if emails_file.exists():
    try:
        emails_df = pd.read_csv(emails_file)
        st.subheader("Sample personalized emails")
        st.dataframe(emails_df.head())

        # show as cards
        st.subheader("Email preview cards")
        for _, row in emails_df.head(10).iterrows():
            with st.container():
                st.markdown(f"**Customer ID:** {row.get('customer_id', '')}  ")
                st.markdown(f"**Email:** {row.get('email', '')}  ")
                st.markdown(f"**Draft:**  \n\n{row.get('marketing_email','')}")
                st.markdown("---")
    except Exception as e:
        st.error(f"Error loading personalized_emails.csv: {e}")
else:
    st.info("No personalized_emails.csv found. Click sidebar to run personalize.py.")

# --- Executive Insights (support calls) ---
st.header("Executive insights — Support calls")
insights_file = RESULTS / "support_insights.csv"
if insights_file.exists():
    try:
        ins_df = pd.read_csv(insights_file)
        st.subheader("Raw extracted insights")
        st.dataframe(ins_df)

        st.subheader("Top action items")
        def parse_actions(s):
            try:
                return json.loads(s) if isinstance(s, str) else []
            except Exception:
                return []
        all_actions = []
        for v in ins_df.get("action_items", []):
            all_actions.extend(parse_actions(v))
        counts = Counter(all_actions)
        for action, cnt in counts.most_common(10):
            st.write(f"- {action} ({cnt})")
    except Exception as e:
        st.error(f"Error loading support_insights.csv: {e}")
else:
    st.info("No support_insights.csv found. Click sidebar to run Extract Insights.")

# --- Product similarity demo (local) ---
st.header("Product similarity demo (local TF-IDF)")
st.info(
    "This is a local fallback demo of 'Vector Search' idea using TF-IDF. "
    "It doesn't require BigQuery vector features but shows how similarity search works."
)

product_names = []
try:
    cps = ROOT / "sample_customers_fixed.csv"
    if cps.exists():
        tmp = pd.read_csv(cps)
        product_names = tmp["product_name"].astype(str).tolist()
    elif (ROOT / "sample_customers.csv").exists():
        tmp = pd.read_csv(ROOT / "sample_customers.csv")
        product_names = tmp["product_name"].astype(str).tolist()
    elif emails_file.exists():
        product_names = ["Sneaker X", "Leather Wallet", "Compact Blender"]
except Exception:
    product_names = ["Sneaker X", "Leather Wallet", "Compact Blender"]

if not product_names:
    product_names = ["Sneaker X", "Leather Wallet", "Compact Blender"]

st.write("Product corpus (sample):", product_names[:10])

vectorizer = TfidfVectorizer().fit_transform(product_names)
cosine_sim = linear_kernel(vectorizer, vectorizer)

query = st.text_input("Enter a product name to find similar items:", value="Sneaker X")
if st.button("Find similar"):
    q_vec = TfidfVectorizer().fit(product_names + [query]).transform([query])
    sims = linear_kernel(q_vec, vectorizer).flatten()
    top_idx = sims.argsort()[::-1][:5]
    st.write("Top similar products:")
    for idx in top_idx:
        st.write(f"- {product_names[idx]} (score {sims[idx]:.3f})")

st.markdown("---")
st.caption("Demo app — replace TF-IDF section with BigQuery Vector Search for production.")
