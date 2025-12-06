# src/convert.py

import pandas as pd

from src.raw_data_utils import ensure_nhcs_stata

# Paths are handled by raw_data_utils
stata_file = ensure_nhcs_stata()
csv_file = "data/nhcs2021ip.csv"

print(f"📂 Loading NHCS Stata file from: {stata_file}")

df = pd.read_stata(stata_file)

print(f"✅ Loaded {len(df):,} rows from Stata file")

df.to_csv(csv_file, index=False)
print(f"💾 Saved {len(df):,} rows to {csv_file}")
