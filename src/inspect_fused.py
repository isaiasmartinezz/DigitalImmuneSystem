import pandas as pd

df = pd.read_csv("data/fused_signals.csv", parse_dates=["date"])

print("Rows:", len(df))
print("\nColumns:", df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nSummary of risk score:")
print(df["fused_risk"].describe())

print("\nNumber of alerts:", df["is_alert"].sum())
