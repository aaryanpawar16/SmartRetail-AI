# Example showing how to call BigQuery using AI.GENERATE_TABLE for summarization
from google.cloud import bigquery
import pandas as pd
from config import PROJECT_ID, BQ_DATASET


client = bigquery.Client(project=PROJECT_ID)
SQL = open('../../src/queries/generate_table.sql').read()
SQL = SQL.replace('PROJECT.DATASET', f'{PROJECT_ID}.{BQ_DATASET}')


job = client.query(SQL)
rows = job.result()
# Convert result to DataFrame
df = rows.to_dataframe()


df.to_csv('../../results/insights.csv', index=False)
print('Saved insights to results/insights.csv')