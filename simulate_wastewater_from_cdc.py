# src/simulate_wastewater.py

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys

# Global settings shared across all simulators
N_DAYS = 180
START_DATE = datetime(2020, 1, 1)
LOCATION_ID = "SYNTH_REGION_1"
POPULATION_SERVED = 100_000


def generate_wastewater():
    dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]

    # Baseline viral load (log-normal-ish, mostly low-level background)
    np.random.seed(42)
    baseline = np.random.lognormal(mean=5.0, sigma=0.4, size=N_DAYS)  # around 150-ish

    # Inject an outbreak between day 60 and 90
    spike_start, spike_end = 60, 90
    spike_factor = 10.0
    viral_load = baseline.copy()
    viral_load[spike_start:spike_end] *= spike_factor

    df = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "location_id": LOCATION_ID,
            "population_served": POPULATION_SERVED,
            "synthetic_viral_load": viral_load,
        }
    )
    return df


def main():
    # Project root = parent of src/
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "synthetic_wastewater.csv"

    print("Python executable:", sys.executable)
    print("Working directory:", root)

    df = generate_wastewater()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✅ Saved wastewater to: {out_path}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

