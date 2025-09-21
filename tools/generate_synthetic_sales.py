# tools/generate_synthetic_sales.py
"""
Generate synthetic daily sales with seasonality + trend + noise.
Writes sample_synthetic_sales.csv in repo root.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

OUT = Path("sample_synthetic_sales.csv")

# Config: length in days
DAYS = 180  # change to 90 if you prefer shorter history
START = date.today() - timedelta(days=DAYS)  # history ending today- useful for forecasts
# or set a fixed start date, e.g. date(2025,1,1)

rng = np.random.default_rng(42)

dates = [START + timedelta(days=i) for i in range(DAYS)]

# Seasonality: weekly pattern (higher on weekends)
weekday = np.array([d.weekday() for d in dates])  # Mon=0..Sun=6
weekly_season = 1.0 + 0.2 * ((weekday >= 5).astype(float))   # +20% on Sat/Sun

# Annual/Monthly season: gentle sinusoid (not necessary but nice)
t = np.arange(DAYS)
annual = 1.0 + 0.15 * np.sin(2 * np.pi * t / 30.0)  # monthly-ish cycles

# Trend: small upward trend
trend = 1.0 + 0.0015 * t  # grows slowly

# Base demand + noise
base = 100.0
noise = rng.normal(scale=8.0, size=DAYS)  # Gaussian noise

sales = (base * weekly_season * annual * trend + noise).round().astype(int)
sales = (sales.clip(min=1))  # no negatives

df = pd.DataFrame({"order_date": [d.isoformat() for d in dates], "sales_amount": sales})
OUT.write_text(df.to_csv(index=False))
print(f"WROTE {OUT} ({len(df)} rows)")
