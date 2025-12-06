import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/fused_signals.csv", parse_dates=["date"])

plt.figure(figsize=(12,5))
plt.plot(df["date"], df["fused_risk"], label="Fused Risk")
plt.axhline(3, color="red", linestyle="--", label="Alert Threshold")

plt.title("Fused Risk Over Time")
plt.xlabel("Date")
plt.ylabel("Risk Score")
plt.legend()
plt.tight_layout()
plt.show()
