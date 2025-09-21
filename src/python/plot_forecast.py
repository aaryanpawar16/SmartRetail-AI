# src/python/plot_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

p = Path('results/forecasts.csv')
df = pd.read_csv(p, parse_dates=['forecast_timestamp'])
df = df.sort_values('forecast_timestamp')

plt.figure(figsize=(10,5))
plt.plot(df['forecast_timestamp'], df['forecast_value'], label='Forecast', linewidth=2)
if 'prediction_interval_lower_bound' in df.columns and 'prediction_interval_upper_bound' in df.columns:
    plt.fill_between(df['forecast_timestamp'],
                     df['prediction_interval_lower_bound'],
                     df['prediction_interval_upper_bound'],
                     alpha=0.2, label='95% CI')
plt.xlabel('Date')
plt.ylabel('Forecast value')
plt.title('AI.FORECAST results')
plt.legend()
plt.tight_layout()
plt.show()
