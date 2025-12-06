# src/simulate_environment.py

"""
Generate synthetic environmental (air-quality-like) data
compatible with wastewater and hospital synthetic data.

Output:
    data/synthetic_environment.csv

Columns:
    - date           (YYYY-MM-DD, daily)
    - location_id    (string, same as other pipelines)
    - synthetic_env  (float, synthetic air-quality index)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Shared configuration
N_DAYS = 180
START_DATE_STR = "2020-01-01"
LOCATION_ID = "SYNTH_REGION_1"

# Outbreak / spike configuration (roughly aligned with others)
OUTBREAK_START_DAY = 60     # index (0-based)
OUTBREAK_DURATION = 14      # days
OUTBREAK_FACTOR = 2.5       # multiplier during spike

RANDOM_SEED = 42


def generate_environment_series(
    start_date: str = START_DATE_STR,
    n_days: int = N_DAYS,
    location_id: str = LOCATION_ID,
    outbreak_start: int = OUTBREAK_START_DAY,
    outbreak_duration: int = OUTBREAK_DURATION,
    outbreak_factor: float = OUTBREAK_FACTOR,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a daily synthetic environmental signal (e.g., air-quality index).

    We simulate:
    - A baseline with weekly seasonality + noise
    - An outbreak period where values are elevated
    """
    rng = np.random.default_rng(random_seed)

    # date index
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_index = np.arange(n_days)

    # baseline around 20
    baseline_level = 20.0

    # weekly seasonality
    weekly_seasonality = 3.0 * np.sin(2 * np.pi * day_index / 7.0)

    # noise
    noise = rng.normal(loc=0.0, scale=2.0, size=n_days)

    env_values = baseline_level + weekly_seasonality + noise
    env_values = np.clip(env_values, a_min=0.0, a_max=None)

    # outbreak spike
    spike_start = outbreak_start
    spike_end = min(outbreak_start + outbreak_duration, n_days)
    env_values[spike_start:spike_end] *= outbreak_factor

    df = pd.DataFrame(
        {
            "date": dates.date,
            "location_id": location_id,
            "synthetic_env": env_values,
        }
    )

    return df


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "synthetic_environment.csv"

    print("Python executable:", sys.executable)
    print("Working directory:", root)

    print("\n🧪 Generating synthetic environmental (air-quality) data...")
    df = generate_environment_series()

    print(f"✅ Generated {len(df)} daily rows from {df['date'].min()} to {df['date'].max()}")

    print("\n📌 Preview of synthetic_environment:")
    print(df.head())

    print("\n📊 Summary of 'synthetic_env':")
    print(df["synthetic_env"].describe())

    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved synthetic environment data to: {output_path}")


if __name__ == "__main__":
    main()
