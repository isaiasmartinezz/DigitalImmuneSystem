import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "data/simplified_wastewater.csv"
OUTPUT_FILE = "data/wastewater_with_anomalies.csv"
PLOT_FILE = "data/wastewater_anomalies.png"
ROLLING_WINDOW = 14
Z_THRESHOLD = 3


print("\n📂 Loading wastewater data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["sample_collect_date"])

df = df.sort_values("sample_collect_date").reset_index(drop=True)

print(f"Loaded {len(df)} rows.")


# -----------------------------
# Step 1: Compute rolling stats
# -----------------------------
df["rolling_mean"] = df["synthetic_viral_load"].rolling(ROLLING_WINDOW, min_periods=5).mean()
df["rolling_std"] = df["synthetic_viral_load"].rolling(ROLLING_WINDOW, min_periods=5).std()

# -----------------------------
# Step 2: Compute z-score
# -----------------------------
df["z_score"] = (df["synthetic_viral_load"] - df["rolling_mean"]) / df["rolling_std"]

# -----------------------------
# Step 3: Identify anomalies
# -----------------------------
df["is_anomaly"] = df["z_score"] > Z_THRESHOLD

num_anomalies = df["is_anomaly"].sum()
print(f"\n🚨 Detected {num_anomalies} anomalies.")


# -----------------------------
# Step 4: Save processed dataset
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"📁 Saved output to {OUTPUT_FILE}")


# -----------------------------
# Step 5: Generate plot
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df["sample_collect_date"], df["synthetic_viral_load"], label="Viral Load", color="blue")

# plot anomalies
anoms = df[df["is_anomaly"]]
plt.scatter(
    anoms["sample_collect_date"],
    anoms["synthetic_viral_load"],
    color="red",
    label="Anomaly",
    zorder=3
)

plt.title("Wastewater Viral Load With Anomaly Detection")
plt.xlabel("Date")
plt.ylabel("Synthetic Viral Load")
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=200)
print(f"🖼️ Saved plot to {PLOT_FILE}")

print("\nDone.\n")
