# 🛍️ SmartRetail AI — BigQuery AI Hackathon Submission

## 📌 Project Overview
Retailers today face three common challenges:  
1. **Forecasting demand** to optimize inventory and reduce stockouts/overstock.  
2. **Personalizing marketing** at scale to improve customer engagement.  
3. **Extracting insights from unstructured support logs** (emails, chats, call transcripts).  

Most solutions require separate ML pipelines, manual effort, or exporting data from warehouses.  
**SmartRetail AI** solves this by building an **end-to-end AI application inside BigQuery** using its **Generative AI, Forecasting, and Vector Search** capabilities.  

The result:  
- Smarter demand forecasts 📈  
- Hyper-personalized marketing emails ✉️  
- Automated executive insights from messy support logs 🎧  

All orchestrated in a **single Streamlit demo app**, powered directly from BigQuery.  

⚙️ Tech Stack
BigQuery AI Functions

AI.FORECAST → demand forecasting

AI.GENERATE → personalized marketing text

AI.GENERATE_TABLE → extract structured insights from support calls

Python + BigFrames

Streamlit → interactive demo UI

Google Cloud Service Account for authentication

🚀 Features
✅ Forecasting: Generates product sales forecasts with 95% confidence intervals.
✅ Personalization: Produces unique marketing emails tailored to each customer.
✅ Executive Insights: Extracts action items, sentiment, and tags from raw support calls.
✅ Vector Search Demo: Shows product similarity search (TF-IDF fallback locally).
✅ Streamlit UI: Clean, interactive dashboard for forecasts, emails, and insights.

⚡ Quick Start
1. Clone Repo
git clone https://github.com/aaryanpawar16/smartretail-ai.git
cd smartretail-ai
2. Install Requirements
pip install -r requirements.txt
3. Authenticate with Google Cloud
Create a service account key:
gcloud iam service-accounts keys create key.json \
  --iam-account=bq-demo-sa@coral-hull-470715-m0.iam.gserviceaccount.com \
  --project=coral-hull-470715-m0
Then set:
export GOOGLE_APPLICATION_CREDENTIALS="key.json"
4. Run Demo
streamlit run src/python/app.py

🖥️ Demo Screens
Forecast chart with confidence intervals

Personalized email previews

Executive insights with action items

Product similarity search demo

🎥 Assets
Demo Video: https://www.youtube.com/watch?v=jO0NitgTyXE

Streamlit Live Link: https://smartretail-ai.streamlit.app/

💡 Feedback on BigQuery AI
Strengths:

Extremely easy to integrate forecasting and LLM-style text generation into SQL.

Removes the need for exporting data to external ML pipelines.

Challenges:

Error handling and debugging for AI.GENERATE_TABLE can be tricky (cryptic syntax errors).

Service account permissions sometimes block AI functions on free-tier projects.

📝 Survey
See user_survey.txt and feedback.txt in repo (attached in results section for submission).

📊 Impact
Forecasting → helps reduce stockouts/overstock → saves $$ in operations.

Personalization → improves CTR & conversion → boosts marketing ROI.

Support Insights → reduces analyst/manual effort → saves time and increases customer satisfaction.

