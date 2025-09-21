# src/python/clean_personalized.py
"""
Trim whitespace and tidy up the personalized_emails CSV.
Writes results/personalized_emails_clean.csv (and optionally overwrites original).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
in_path = ROOT / "results" / "personalized_emails.csv"
out_path = ROOT / "results" / "personalized_emails_clean.csv"

if not in_path.exists():
    raise SystemExit(f"Input file not found: {in_path}")

df = pd.read_csv(in_path)

# Trim whitespace on all string columns, collapse multiple spaces to one
for c in df.select_dtypes(include=["object"]).columns:
    # convert NaN to empty string safely, then strip and normalize spaces
    df[c] = df[c].fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

# Optional: remove any space before punctuation (like "word ." -> "word.")
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].str.replace(r"\s+([.,!?;:])", r"\1", regex=True)

# Save cleaned CSV
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"WROTE {out_path} ({len(df)} rows)")

# If you want to overwrite the original, uncomment the following line:
# df.to_csv(in_path, index=False); print(f"OVERWROTE {in_path}")
