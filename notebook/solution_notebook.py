# This is a notebook-like script (useable as a .py or convert to .ipynb)
# Steps:
# 1) Configure project & dataset
# 2) Run forecast.sql via BigQuery
# 3) Run generate_table.sql to produce insights
# 4) Run personalize.sql to produce marketing emails


from src.python.config import PROJECT_ID, BQ_DATASET
from src.python.forecast import *
# For a real notebook, split into cells and show outputs inline using pandas