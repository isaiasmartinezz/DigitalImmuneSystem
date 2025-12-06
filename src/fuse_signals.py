from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

WASTEWATER_FILE = DATA_DIR / "synthetic_wastewater.csv"
HOSPITAL_FILE = DATA_DIR / "synthetic_hospital.csv"
ENV_FILE = DATA_DIR / "synthetic_environment.csv"
OUTPUT_FILE = DATA_DIR / "fused_signals.csv"


# --------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------
def rolling_zscore(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute rolling z-score for a time series."""
    roll_mean = series.rolling(window=window, min_periods=window).mean()
    roll_std = series.rolling(window=window, min_periods=window).std()
    z = (series - roll_mean) / roll_std
    return z


# --------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------
def load_wastewater() -> pd.DataFrame:
    print(f"📂 Loading wastewater data from: {WASTEWATER_FILE}")
    df = pd.read_csv(WASTEWATER_FILE)

    # Detect date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif "sample_collect_date" in df.columns:
        df["date"] = pd.to_datetime(df["sample_collect_date"]).dt.date
    else:
        raise KeyError(
            "Wastewater CSV must have either 'date' or 'sample_collect_date' column."
        )

    # Ensure location_id
    if "location_id" not in df.columns:
        if "sewershed_id" in df.columns:
            df["location_id"] = df["sewershed_id"].astype(str)
        else:
            df["location_id"] = "SYNTH_REGION_1"

    # Ensure viral load column
    if "synthetic_viral_load" not in df.columns:
        candidates = [
            "viral_load",
            "pcr_target_flowpop_lin",
            "pcr_target_avg_conc",
        ]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise KeyError(
                "Could not find a wastewater viral load column. "
                "Expected 'synthetic_viral_load' or something similar."
            )
        df["synthetic_viral_load"] = df[found]

    df = df[["date", "location_id", "synthetic_viral_load"]].copy()
    df["location_id"] = df["location_id"].astype(str)
    df = df.sort_values(["location_id", "date"])

    # Add z-score
    df["ww_z"] = (
        df.groupby("location_id")["synthetic_viral_load"]
        .transform(lambda s: rolling_zscore(s, window=14))
    )

    print(f"✅ Wastewater rows: {len(df)}")
    return df


def load_hospital() -> pd.DataFrame:
    print(f"\n📂 Loading hospital data from: {HOSPITAL_FILE}")
    df = pd.read_csv(HOSPITAL_FILE)

    # Detect date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif "sample_date" in df.columns:
        df["date"] = pd.to_datetime(df["sample_date"]).dt.date
    else:
        raise KeyError(
            "Hospital CSV must have either 'date' or 'sample_date' column."
        )

    # Ensure location_id
    if "location_id" not in df.columns:
        df["location_id"] = "SYNTH_REGION_1"

    # Ensure admissions column
    if "synthetic_admissions" not in df.columns:
        candidates = ["admissions", "resp_visits", "visit_count"]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise KeyError(
                "Hospital CSV must have 'synthetic_admissions' or an equivalent admissions column."
            )
        df["synthetic_admissions"] = df[found]

    df = df[["date", "location_id", "synthetic_admissions"]].copy()
    df["location_id"] = df["location_id"].astype(str)
    df = df.sort_values(["location_id", "date"])

    df["hosp_z"] = (
        df.groupby("location_id")["synthetic_admissions"]
        .transform(lambda s: rolling_zscore(s, window=14))
    )

    print(f"✅ Hospital rows: {len(df)}")
    return df


def load_environment() -> pd.DataFrame:
    print(f"\n📂 Loading environment data from: {ENV_FILE}")
    df = pd.read_csv(ENV_FILE)

    # Detect date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif "sample_date" in df.columns:
        df["date"] = pd.to_datetime(df["sample_date"]).dt.date
    else:
        raise KeyError(
            "Environment CSV must have either 'date' or 'sample_date' column."
        )

    # Ensure location_id
    if "location_id" not in df.columns:
        df["location_id"] = "SYNTH_REGION_1"

    # Find environment value column
    candidates = [
        "synthetic_env",
        "synthetic_aq_index",
        "synthetic_pm25",
        "pm25",
        "pm_25",
        "Value",
        "env_value",
    ]
    env_col = None
    for c in candidates:
        if c in df.columns:
            env_col = c
            break

    if env_col is None:
        raise KeyError(
            "Could not find an environment value column (e.g., PM2.5). "
            f"Tried: {candidates}. Available columns: {list(df.columns)}"
        )

    df = df[["date", "location_id", env_col]].copy()
    df = df.rename(columns={env_col: "synthetic_env"})
    df["location_id"] = df["location_id"].astype(str)
    df = df.sort_values(["location_id", "date"])

    df["env_z"] = (
        df.groupby("location_id")["synthetic_env"]
        .transform(lambda s: rolling_zscore(s, window=14))
    )

    print(f"✅ Environment rows: {len(df)}")
    return df


# --------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------
def fuse_signals(
    ww: pd.DataFrame,
    hosp: pd.DataFrame,
    env: pd.DataFrame,
    use_location: bool = True,
) -> pd.DataFrame:
    """Fuse three signals into a single risk score."""
    if use_location:
        merge_keys = ["date", "location_id"]
        print(f"\n🔗 Fusing signals on {merge_keys}...")
    else:
        merge_keys = ["date"]
        print(f"\n🔗 Fusing signals on {merge_keys} (ignoring location_id)...")

    # If we're ignoring location, drop location_id before merge
    if not use_location:
        ww = ww.drop(columns=["location_id"]).copy()
        hosp = hosp.drop(columns=["location_id"]).copy()
        env = env.drop(columns=["location_id"]).copy()

    fused = ww.merge(hosp, on=merge_keys, how="inner", suffixes=("_ww", "_hosp"))
    fused = fused.merge(env, on=merge_keys, how="inner")

    if fused.empty:
        print("⚠️ Fusion produced 0 rows.")
        return fused

    # Compute fused risk score
    fused["fused_risk"] = (
        0.4 * fused["ww_z"] +
        0.4 * fused["hosp_z"] +
        0.2 * fused["env_z"]
    )

    # Alert rule
    fused["is_alert"] = fused["fused_risk"] > 2.5

    return fused


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    print(f"Python executable: {os.sys.executable}")
    print(f"Working directory: {os.getcwd()}\n")

    ww = load_wastewater()
    hosp = load_hospital()
    env = load_environment()

    # First try fusion using date + location_id
    fused = fuse_signals(ww, hosp, env, use_location=True)

    # If empty, try date-only fusion as fallback
    if fused.empty:
        print("\n⚠️ No rows after fusing on ['date', 'location_id']. "
              "Falling back to fusing on 'date' only...")
        fused = fuse_signals(ww, hosp, env, use_location=False)

    print(f"\n✅ Fused dataset rows: {len(fused)}")
    if not fused.empty:
        num_alerts = fused["is_alert"].sum()
        print(f"⚠️ Number of alert days (is_alert=True): {int(num_alerts)}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved fused signals to: {OUTPUT_FILE}")

    print("\n📌 Sample of fused data:")
    print(fused.head(10))


if __name__ == "__main__":
    main()
