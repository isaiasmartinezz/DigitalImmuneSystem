#!/usr/bin/env python3
"""
pipeline_cli.py

Command-line interface for your digital immune system prototype.

Usage (from project root):

    python src/pipeline_cli.py \
        --wastewater data/synthetic_wastewater.csv \
        --hospital data/synthetic_hospital.csv \
        --environment data/synthetic_environment.csv \
        --output-dir data \
        --window 28 \
        --risk-threshold 2.5

If you omit the paths, it will default to the synthetic CSVs above.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """
    Compute a rolling z-score for a 1D time series.

    We allow smaller min_periods at the beginning so you don't get all NaNs.
    """
    roll_mean = series.rolling(window=window, min_periods=max(3, window // 2)).mean()
    roll_std = series.rolling(window=window, min_periods=max(3, window // 2)).std()
    z = (series - roll_mean) / roll_std
    return z


def load_signal(
    path: Path,
    date_col: str,
    value_col: str,
    label: str,
    window: int,
) -> pd.DataFrame:
    """
    Generic loader for a single signal (wastewater, hospital, environment).

    - path: CSV path
    - date_col: name of the date column in that CSV
    - value_col: name of the numeric value column to use
    - label: base name for the signal (e.g. "synthetic_viral_load")
    - window: rolling window length for z-score
    """
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found at {path}")

    print(f"📂 Loading {label} from: {path}")
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise KeyError(
            f"{label}: expected date column '{date_col}' in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )
    if value_col not in df.columns:
        raise KeyError(
            f"{label}: expected value column '{value_col}' in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df = df.sort_values(date_col)

    z_col = f"{label}_z"

    df[z_col] = rolling_z(df[value_col], window=window)
    df.rename(columns={value_col: label, date_col: "date"}, inplace=True)

    print(f"   → rows: {len(df)}, columns: {list(df.columns)}")
    return df


def fuse_signals(
    wastewater: pd.DataFrame,
    hospital: pd.DataFrame,
    environment: pd.DataFrame,
    ww_weight: float,
    hosp_weight: float,
    env_weight: float,
    risk_threshold: float,
) -> pd.DataFrame:
    """
    Fuse three per-day signals into a single risk score and alert flag.

    Assumes each DataFrame has:
      - 'date'
      - '<label>' (raw value)
      - '<label>_z' (z-score)
    """

    print("\n🔗 Fusing signals on 'date' (inner join)...")
    fused = wastewater.merge(hospital, on="date", how="inner").merge(
        environment, on="date", how="inner"
    )

    if fused.empty:
        print("⚠️ No overlapping dates between the three signals.")
        return fused

    # standardize column names
    fused = fused.rename(
        columns={
            "synthetic_viral_load_z": "ww_z",
            "synthetic_admissions_z": "hosp_z",
            "synthetic_env_z": "env_z",
        }
    )

    # If any z-scores are missing for a day, treat them as 0 (neutral)
    for col in ["ww_z", "hosp_z", "env_z"]:
        if col not in fused.columns:
            raise KeyError(
                f"Expected z-score column '{col}' in fused DataFrame. "
                f"Have columns: {list(fused.columns)}"
            )
        fused[col] = fused[col].fillna(0.0)

    # Weighted risk score
    fused["fused_risk"] = (
        ww_weight * fused["ww_z"]
        + hosp_weight * fused["hosp_z"]
        + env_weight * fused["env_z"]
    )

    fused["is_alert"] = fused["fused_risk"] > risk_threshold

    print(f"\n✅ Fused dataset rows: {len(fused)}")
    print(f"⚠️ Number of alert days (is_alert=True): {fused['is_alert'].sum()}")

    return fused


def plot_fused(fused: pd.DataFrame, out_path: Path, risk_threshold: float) -> None:
    """Simple time-series plot of fused risk score with threshold line."""
    if fused.empty:
        print("⚠️ Skipping plot: fused dataset is empty.")
        return

    fused_sorted = fused.sort_values("date")

    fig, ax = plt.subplots()
    ax.plot(fused_sorted["date"], fused_sorted["fused_risk"], label="Fused risk")
    ax.axhline(risk_threshold, linestyle="--", label="Alert threshold")

    ax.set_xlabel("Date")
    ax.set_ylabel("Fused risk (z-score units)")
    ax.set_title("Digital immune system fused risk over time")
    ax.legend()
    fig.autofmt_xdate()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"📈 Saved fused risk plot to: {out_path}")


# ---------- CLI ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the digital immune system pipeline on three CSV files."
    )

    # default synthetic paths
    default_ww = DEFAULT_DATA_DIR / "synthetic_wastewater.csv"
    default_hosp = DEFAULT_DATA_DIR / "synthetic_hospital.csv"
    default_env = DEFAULT_DATA_DIR / "synthetic_environment.csv"

    parser.add_argument(
        "--wastewater",
        type=Path,
        default=default_ww,
        help=f"Path to wastewater CSV (default: {default_ww})",
    )
    parser.add_argument(
        "--hospital",
        type=Path,
        default=default_hosp,
        help=f"Path to hospital CSV (default: {default_hosp})",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=default_env,
        help=f"Path to environment CSV (default: {default_env})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory to write outputs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=28,
        help="Rolling window (days) for z-score calculation (default: 28)",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=2.5,
        help="Fused risk threshold for raising an alert (default: 2.5)",
    )
    parser.add_argument(
        "--ww-weight",
        type=float,
        default=0.4,
        help="Weight for wastewater z-score in fused risk (default: 0.4)",
    )
    parser.add_argument(
        "--hosp-weight",
        type=float,
        default=0.4,
        help="Weight for hospital z-score in fused risk (default: 0.4)",
    )
    parser.add_argument(
        "--env-weight",
        type=float,
        default=0.2,
        help="Weight for environment z-score in fused risk (default: 0.2)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Python executable: {Path.cwd()}")
    print(f"Working directory: {PROJECT_ROOT}\n")

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load each signal ----
    ww = load_signal(
        path=args.wastewater,
        date_col="date",
        value_col="synthetic_viral_load",
        label="synthetic_viral_load",
        window=args.window,
    )

    hosp = load_signal(
        path=args.hospital,
        date_col="date",
        value_col="synthetic_admissions",
        label="synthetic_admissions",
        window=args.window,
    )

    env = load_signal(
        path=args.environment,
        date_col="date",
        value_col="synthetic_env",
        label="synthetic_env",
        window=args.window,
    )

    # ---- fuse ----
    fused = fuse_signals(
        wastewater=ww,
        hospital=hosp,
        environment=env,
        ww_weight=args.ww_weight,
        hosp_weight=args.hosp_weight,
        env_weight=args.env_weight,
        risk_threshold=args.risk_threshold,
    )

    # ---- save + plot ----
    fused_csv = args.output_dir / "fused_signals_cli.csv"
    fused.to_csv(fused_csv, index=False)
    print(f"\n💾 Saved fused signals to: {fused_csv}")

    plot_path = args.output_dir / "fused_risk_cli.png"
    plot_fused(fused, plot_path, args.risk_threshold)

    # quick summary for the terminal
    if not fused.empty:
        print("\n📊 Fused risk summary:")
        print(fused["fused_risk"].describe())
        print("\nFirst 5 rows:")
        print(fused.head())


if __name__ == "__main__":
    main()
