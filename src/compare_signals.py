import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/fused_signals.csv", parse_dates=["date"])

plt.figure(figsize=(12,8))

plt.plot(df["date"], df["ww_z"], label="Wastewater z-score")
plt.plot(df["date"], df["hosp_z"], label="Hospital z-score")
plt.plot(df["date"], df["env_z"], label="Environment z-score")
plt.plot(df["date"], df["fused_risk"], label="Fused Risk", linewidth=3)

plt.legend()
plt.title("Comparison of Individual Signals and Fused Risk")
plt.xlabel("Date")
plt.ylabel("Z-score / Risk")
plt.tight_layout()
plt.show()
