# src/explore_hospital.py

import pandas as pd
import os

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "CDC_hopstial_admissions_sample.csv")
SIMPLIFIED_FILE = os.path.join(DATA_DIR, "simplified_hospital.csv")

def main():
    print("📂 Loading hospital data...")
    df = pd.read_csv(RAW_FILE)

    print("\n✅ Loaded CSV successfully!")
    print(f"Rows: {len(df):,},  Columns: {df.shape[1]}\n")

    print("📌 First 5 rows:")
    print(df.head(), "\n")

    print("📌 Columns in dataset:")
    for col in df.columns:
        print(f"   - {col}")
    print()

    # ---- MODIFY THIS PART TO MATCH YOUR REAL COLUMN NAMES ----
    # I'll assume the RESP-NET export has columns named something like:
    #   "Site", "Week Ending Date", "Weekly Rate"
    # Adjust these names if your CSV uses slightly different ones.
    site_col = "Site"
    date_col = "Week Ending Date"
    rate_col = "Weekly Rate"

    # Filter to overall age group / sex / race if needed
    # (Only if those columns exist – if not, this will just be skipped.)
    for col, value in [
        ("Age group", "Overall"),
        ("Sex", "Overall"),
        ("Race/Ethnicity", "Overall"),
    ]:
        if col in df.columns:
            df = df[df[col] == value]

    # Keep only the columns we need and rename them
    simplified = df[[site_col, date_col, rate_col]].copy()
    simplified = simplified.rename(columns={
        site_col: "location_id",
        date_col: "sample_date",
        rate_col: "baseline_rate"
    })

    # Add a fake population_served column for now (you can adjust later)
    simplified["population_served"] = 100000  # fake placeholder

    # Sort by location and date
    simplified["sample_date"] = pd.to_datetime(simplified["sample_date"])
    simplified = simplified.sort_values(["location_id", "sample_date"])

    # Reorder columns
    simplified = simplified[["location_id", "population_served",
                             "sample_date", "baseline_rate"]]

    simplified.to_csv(SIMPLIFIED_FILE, index=False)
    print(f"\n💾 Saved simplified hospital data to: {SIMPLIFIED_FILE}")
    print(f"Rows: {len(simplified):,}")

if __name__ == "__main__":
    main()
