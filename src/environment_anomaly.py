# src/environment_anomaly.py

"""
Run simple rolling z-score anomaly detection on synthetic_environment.csv.

Input:  data/synthetic_environment.csv
Output: data/environment_with_anomalies.csv
        data/environment_anomalies.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "data/synthetic_environment.csv"
OUTPUT_CSV = "data/environment_with_anomalies.csv"
OUTPUT_FIG = "data/environment_anomalies.png"


def rolling_zscore(series: pd.Series, window: int = 28):
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    z = (series - mean) / std
    return z


def main():
    root = Path(__file__).resolve().parents[1]
    input_path = root / INPUT_PATH
    output_csv = root / OUTPUT_CSV
    output_fig = root / OUTPUT_FIG

    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}\n")

    print(f"📂 Loading synthetic environment data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "synthetic_aq_index" not in df.columns:
        raise KeyError("Expected column 'synthetic_aq_index' in synthetic_environment.csv")

    # Compute z-score and anomaly flag
    df["zscore"] = rolling_zscore(df["synthetic_aq_index"], window=28)
    df["is_anomaly"] = df["zscore"] > 3.0

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved environment_with_anomalies.csv to: {output_csv}")
    print(f"Total rows: {len(df):,}")
    print(f"Number of anomalies flagged: {df['is_anomaly'].sum():,}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["synthetic_aq_index"], label="Synthetic AQ index")
    plt.scatter(
        df.loc[df["is_anomaly"], "date"],
        df.loc[df["is_anomaly"], "synthetic_aq_index"],
        marker="o",
        s=20,
        label="Anomalies",
    )
    plt.xlabel("Date")
    plt.ylabel("Synthetic air-quality index")
    plt.title("Environment anomalies (air quality)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()

    print(f"📈 Saved anomaly plot to: {output_fig}")


if __name__ == "__main__":
    main()
