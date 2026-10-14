
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "energy_consumption.csv"

START = "2024-01-01"
END = "2025-04-01"
SEED = 42


def generate_site(
    site_id: str,
    site_type: str,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate plausible hourly electricity consumption for one site."""
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    # Different sites have different consumption scales.
    base_load = {
        "office": 30.0,
        "retail": 45.0,
        "industrial": 100.0,
    }[site_type]

    # A daily operating pattern: daytime activity, lower overnight usage.
    daytime = np.where((hour >= 7) & (hour <= 19), 1.0, 0.35)

    # Offices and retail locations are quieter on weekends.
    weekend_factor = np.where(
        day_of_week >= 5,
        0.55 if site_type == "office" else 0.75,
        1.0,
    )

    # A simple annual pattern to imitate seasonal demand.
    seasonal_factor = 1.0 + 0.12 * np.cos(
        2 * np.pi * (day_of_year - 15) / 365.25
    )

    noise = rng.normal(loc=0.0, scale=base_load * 0.06, size=len(timestamps))

    consumption = (
        base_load * daytime * weekend_factor * seasonal_factor + noise
    )
    consumption = np.maximum(consumption, 0.1)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": site_id,
            "consumption_kwh": consumption.round(3),
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    timestamps = pd.date_range(
        start=START,
        end=END,
        freq="h",
        inclusive="left",
    )

    sites = [
        ("office_001", "office"),
        ("retail_002", "retail"),
        ("industrial_003", "industrial"),
        ("office_004", "office"),
    ]

    data = pd.concat(
        [
            generate_site(site_id, site_type, timestamps, rng)
            for site_id, site_type in sites
        ],
        ignore_index=True,
    )

    # Inject realistic data-quality problems for the exercise.
    # 1. Some measurements are missing.
    missing_indices = rng.choice(data.index, size=100, replace=False)
    data = data.drop(index=missing_indices)

    # 2. Some readings are extreme. They may be faults or genuine events.
    spike_indices = rng.choice(data.index, size=30, replace=False)
    data.loc[spike_indices, "consumption_kwh"] *= rng.uniform(
        4, 8, size=len(spike_indices)
    )

    # 3. Some readings are zero. Do not assume these are all errors.
    zero_indices = rng.choice(data.index, size=18, replace=False)
    data.loc[zero_indices, "consumption_kwh"] = 0.0

    # 4. A meter change shifts the scale for one site.
    meter_change = (
        (data["site_id"] == "office_004")
        & (data["timestamp"] >= "2024-11-01")
    )
    data.loc[meter_change, "consumption_kwh"] *= 1.22

    # 5. Duplicate some rows.
    duplicates = data.sample(n=22, random_state=SEED)
    data = pd.concat([data, duplicates], ignore_index=True)

    # Shuffle rows to ensure downstream code cannot assume sorted input.
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)

    print(f"Saved {len(data):,} rows to {OUTPUT}")
    print(f"Sites: {data['site_id'].nunique()}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")


if __name__ == "__main__":
    main()
