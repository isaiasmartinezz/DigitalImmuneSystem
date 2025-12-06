#!/usr/bin/env python
"""
simulate_hospital.py

Generate fully synthetic hospital admissions data that is compatible with:
- synthetic_wastewater.csv
- synthetic_environment.csv
and ready to be consumed by fuse_signals.py.

Output: data/synthetic_hospital.csv with columns:
    - date
    - location_id
    - population_served
    - baseline_rate
    - synthetic_admissions
"""

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
N_DAYS = 180
START_DATE = date(2020, 1, 1)

LOCATION_ID = "SYNTH_REGION_1"
POPULATION_SERVED = 100_000

# Rough "seasonality" controls
BASE_RATE_PER_1000 = 0.2       # baseline daily admissions per 1000 people
SEASONAL_AMPLITUDE = 0.15      # magnitude of seasonal sinusoidal variation
NOISE_STD = 0.05               # random noise on the rate

# Outbreak window (align with wastewater/env if you like)
OUTBREAK_START_DAY = 60        # 0-indexed day into the series
OUTBREAK_LENGTH = 30           # days
OUTBREAK_MULTIPLIER = 3.0      # spike factor during outbreak


def make_date_range(start: date, n_days: int) -> list[date]:
    """Return a list of Python dates starting at `start` for `n_days`."""
    return [start + timedelta(days=i) for i in range(n_days)]


def generate_baseline_rates(n_days: int) -> np.ndarray:
    """
    Generate a seasonal baseline admission rate (per 1000 people per day)
    with some noise. This is a simple sinusoid plus Gaussian noise.
    """
    x = np.arange(n_days)

    # annual-ish sinusoidal pattern squashed into 180 days
    seasonal = SEASONAL_AMPLITUDE * np.sin(2 * math.pi * x / 60.0)

    # add base and noise, clip at >= 0
    rng = np.random.default_rng(seed=2024)
    noise = rng.normal(0.0, NOISE_STD, size=n_days)

    rate = BASE_RATE_PER_1000 + seasonal + noise
    rate = np.clip(rate, 0.01, None)  # avoid zero / negative

    return rate


def apply_outbreak(rate: np.ndarray) -> np.ndarray:
    """
    Apply a multiplicative outbreak spike over a specified window.
    """
    rate = rate.copy()
    start = OUTBREAK_START_DAY
    end = OUTBREAK_START_DAY + OUTBREAK_LENGTH
    end = min(end, len(rate))

    rate[start:end] *= OUTBREAK_MULTIPLIER
    return rate


def generate_admissions(rate_per_1000: np.ndarray, population: int) -> np.ndarray:
    """
    Convert a rate per 1000 people into integer admissions using a Poisson model.
    """
    lam = rate_per_1000 * (population / 1000.0)
    rng = np.random.default_rng(seed=2025)
    admissions = rng.poisson(lam)
    return admissions


def build_hospital_dataframe() -> pd.DataFrame:
    """
    Build the final hospital dataframe with all required columns.
    """
    dates = make_date_range(START_DATE, N_DAYS)
    baseline_rate = generate_baseline_rates(N_DAYS)
    rate_with_outbreak = apply_outbreak(baseline_rate)
    admissions = generate_admissions(rate_with_outbreak, POPULATION_SERVED)

    df = pd.DataFrame(
        {
            "date": dates,
            "location_id": LOCATION_ID,
            "population_served": POPULATION_SERVED,
            "baseline_rate": baseline_rate,
            "synthetic_admissions": admissions,
        }
    )

    return df


def main() -> None:
    # Resolve project root as the directory containing this file
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("Python executable:", Path(pd.__file__).resolve().parent.parent / "bin" / "python")
    print("Script location:", project_root)
    print()

    print("🧪 Generating synthetic hospital admissions...")
    df = build_hospital_dataframe()

    out_path = data_dir / "synthetic_hospital.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved synthetic hospital data to: {out_path}")
    print(f"Rows: {len(df)}")
    print("\n📌 First 5 rows:")
    print(df.head())
    print("\n📌 Summary of synthetic_admissions:")
    print(df["synthetic_admissions"].describe())


if __name__ == "__main__":
    main()
