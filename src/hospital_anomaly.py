# src/hospital_anomaly.py

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"
SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_hospital.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "hospital_with_anomalies.csv")
PLOT_FILE = os.path.join(DATA_DIR, "hospital_anomalies.png")

ROLLING_WINDOW = 14  # days or 14 time points, depending on your granularity
Z_THRESHOLD = 3.0

def detect_anomalies(df, value_col="synthetic_admissions",
                     window=ROLLING_WINDOW, z_thresh=Z_THRESHOLD):
    df = df.sort_values("sample_date").copy()

    df["rolling_mean"] = df[value_col].rolling(window=window, min_periods=window).mean()
    df["rolling_std"] = df[value_col].rolling(window=window, min_periods=window).std()

    # avoid division by zero
    df["rolling_std"] = df["rolling_std"].replace(0, pd.NA)

    df["z_score"] = (df[value_col] - df["rolling_mean"]) / df["rolling_std"]
    df["z_score"] = df["z_score"].fillna(0)

    df["is_anomaly"] = df["z_score"] > z_thresh

    return df

def main():
    print("📂 Loading synthetic hospital data...")
    df = pd.read_csv(SYNTHETIC_FILE, parse_dates=["sample_date"])

    print(f"Rows: {len(df):,}")
    print(df.head(), "\n")

    # Apply anomaly detection per location
    results = []
    for loc, group in df.groupby("location_id"):
        annotated = detect_anomalies(group)
        results.append(annotated)

    full = pd.concat(results, ignore_index=True)
    full.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved hospital anomalies to: {OUTPUT_FILE}")

    # Simple plot for one location (first one) to visually inspect
    first_loc = full["location_id"].iloc[0]
    subset = full[full["location_id"] == first_loc]

    plt.figure(figsize=(10, 5))
    plt.plot(subset["sample_date"], subset["synthetic_admissions"], label="Admissions")
    plt.plot(subset["sample_date"], subset["rolling_mean"], label="Rolling mean", linestyle="--")

    anomalies = subset[subset["is_anomaly"]]
    plt.scatter(anomalies["sample_date"], anomalies["synthetic_admissions"],
                marker="o", s=50, label="Anomalies")

    plt.title(f"Hospital Admissions with Anomalies (location {first_loc})")
    plt.xlabel("Date")
    plt.ylabel("Synthetic admissions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"🖼️ Saved anomaly plot to: {PLOT_FILE}")

if __name__ == "__main__":
    main()
