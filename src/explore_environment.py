# src/explore_environment.py

import sys
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/Air_Quality_Measures_on_the_National_Environmental_Health_Tracking_Network.csv")

def main():
    print(f"Python executable: {sys.executable}")
    print(f"Current working dir: {Path.cwd()}")
    print(f"\n📂 Loading air-quality data from: {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Could not find {RAW_PATH}. Check the filename and path.")

    # low_memory=False so pandas doesn't guess types in chunks
    df = pd.read_csv(RAW_PATH, low_memory=False)

    print("\n✅ Loaded CSV successfully!")
    print(f"Rows: {len(df):,},  Columns: {df.shape[1]}")

    print("\n📌 First 5 rows:")
    print(df.head(), "\n")

    print("📌 Columns in dataset:")
    for col in df.columns:
        print(f"  - {col}")

    # OPTIONAL: show unique values for likely 'Measure' column
    guess_cols = [c for c in df.columns if "measure" in c.lower()]
    if guess_cols:
        col = guess_cols[0]
        print(f"\n🔍 Example values in '{col}' (first 20 unique):")
        print(df[col].dropna().unique()[:20])

    # OPTIONAL: show some state/county columns if present
    loc_cols = [c for c in df.columns if any(k in c.lower() for k in ["county", "state", "fips"])]
    if loc_cols:
        print("\n📌 Location-related columns found:")
        for c in loc_cols:
            print(f"  - {c}")
            print("    sample values:", df[c].dropna().unique()[:5])

if __name__ == "__main__":
    main()
