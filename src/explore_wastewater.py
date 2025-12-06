# src/explore_wastewater.py  (or convert_wastewater.py)

import sys
from pathlib import Path
import pandas as pd

from src.raw_data_utils import ensure_cdc_wastewater

# ---------- CONFIG ----------
RAW_CSV_PATH = ensure_cdc_wastewater()  # auto-download if needed
OUTPUT_CSV_PATH = Path("data/simplified_wastewater.csv")
# ----------------------------

print("\n📂 Loading wastewater data...")
print("Python executable:", sys.executable)
print(f"Using CDC CSV at: {RAW_CSV_PATH}")

# 1. Load the full CDC dataset
df = pd.read_csv(RAW_CSV_PATH, low_memory=False)

print("\n✅ Loaded CSV successfully!")
print(f"Rows: {len(df):,},  Columns: {df.shape[1]}")

# 2. Show first few rows and columns
print("\n📌 First 5 rows:")
print(df.head(5), "\n")

print("📌 Columns in dataset:")
for col in df.columns:
    print(f"   - {col}")
print()

# 3. Inspect PCR target values to see how SARS-CoV-2 is labeled
if "pcr_target" in df.columns:
    print("📌 Unique pcr_target values (first 20):")
    print(df["pcr_target"].dropna().unique()[:20])
    print()
else:
    print("⚠️ No 'pcr_target' column found. Check the CSV structure.")
    sys.exit(1)

# 4. Filter to rows that look like SARS-CoV-2
virus_mask = df["pcr_target"].str.contains("sars", case=False, na=False) | \
             df["pcr_target"].str.contains("cov-2", case=False, na=False)

virus_df = df[virus_mask].copy()

if virus_df.empty:
    print("⚠️ No rows found where pcr_target contains 'sars' or 'cov-2'.")
    sys.exit(1)

print(f"✅ Filtered to SARS-CoV-2-like targets. Rows remaining: {len(virus_df):,}")

# 5. Choose a viral-load column to use as our signal
signal_col_options = [
    "pcr_target_flowpop_lin",
    "pcr_target_avg_conc_lin",
    "pcr_target_avg_conc",
]

signal_col = None
for col in signal_col_options:
    if col in virus_df.columns:
        signal_col = col
        break

if signal_col is None:
    print("⚠️ Could not find any of the expected viral-load columns:")
    print("   ", signal_col_options)
    sys.exit(1)

print(f"✅ Using '{signal_col}' as the viral-load signal.\n")

# 6. Keep only rows where the signal is not null
virus_df = virus_df.dropna(subset=[signal_col])

# 7. Choose one sewershed
counts_by_sewershed = virus_df["sewershed_id"].value_counts()
top_sewershed = counts_by_sewershed.index[0]
print(f"📌 Using sewershed_id = {top_sewershed} (rows: {counts_by_sewershed.iloc[0]})\n")

simple = virus_df[virus_df["sewershed_id"] == top_sewershed].copy()

required_columns = ["sewershed_id", "population_served", "sample_collect_date", signal_col]
missing = [c for c in required_columns if c not in simple.columns]
if missing:
    print("⚠️ Missing expected columns in filtered data:", missing)
    sys.exit(1)

simple = simple[required_columns].rename(
    columns={signal_col: "synthetic_viral_load"}
)

simple = simple.sort_values("sample_collect_date").reset_index(drop=True)

print("📌 Preview of simplified dataset:")
print(simple.head(10), "\n")

OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
simple.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Saved simplified wastewater data to: {OUTPUT_CSV_PATH}")
print("   Columns:", list(simple.columns))
