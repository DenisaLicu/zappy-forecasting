from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "energy_consumption.csv"
MODEL_PATH = ROOT / "artifacts" / "energy_forecaster.joblib"
OUTPUT_PATH = ROOT / "outputs" / "forecasts.csv"

SITE_COL = "site_id"
TIME_COL = "timestamp"
TARGET_COL = "consumption_kwh"

HORIZON_HOURS = 24

FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "site_id",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create exactly the calendar features used during training."""
    timestamps = pd.to_datetime(df[TIME_COL])

    features = pd.DataFrame(index=df.index)

    hour = timestamps.dt.hour
    day_of_week = timestamps.dt.dayofweek
    day_of_year = timestamps.dt.dayofyear

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    features["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    features["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    features["is_weekend"] = (
        day_of_week >= 5
    ).astype(int)

    features["site_id"] = df[SITE_COL].astype("category")

    return features[FEATURES]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run src/train.py first."
        )

    # Load the trained model; do not retrain during inference.
    model = joblib.load(MODEL_PATH)

    # Load and minimally clean the history used to identify
    # each site's latest observed timestamp.
    history = pd.read_csv(DATA_PATH)

    required_columns = {SITE_COL, TIME_COL, TARGET_COL}
    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            f"Input data is missing columns: {sorted(missing_columns)}"
        )

    history[TIME_COL] = pd.to_datetime(
        history[TIME_COL], errors="coerce"
    )
    history[TARGET_COL] = pd.to_numeric(
        history[TARGET_COL], errors="coerce"
    )

    history = history.dropna(
        subset=[SITE_COL, TIME_COL, TARGET_COL]
    )
    history = history[
        np.isfinite(history[TARGET_COL])
        & (history[TARGET_COL] >= 0)
    ]

    # Match the training script's duplicate policy.
    history = history.drop_duplicates(
        subset=[SITE_COL, TIME_COL], keep="first"
    )

    if history.empty:
        raise ValueError("No valid historical observations remain.")

    # The model was trained with site identity as a categorical
    # feature. Do not silently extrapolate to unseen sites.
    known_sites = set(history[SITE_COL].astype(str).unique())
    training_sites = set(model.classes_) if hasattr(model, "classes_") else None

    if training_sites is not None:
        unknown_sites = known_sites - training_sites
        if unknown_sites:
            raise ValueError(
                f"Unknown sites cannot be forecast: {sorted(unknown_sites)}"
            )

    # This exercise uses the sites present in the training data.
    # For a production system, store the training-site list as
    # explicit model metadata rather than infer it from current data.
    training_data = pd.read_csv(DATA_PATH, usecols=[SITE_COL])
    site_categories = sorted(
        training_data[SITE_COL].dropna().astype(str).unique()
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    forecast_frames = []

    for site_id, site_history in history.groupby(SITE_COL):
        site_history = site_history.sort_values(TIME_COL)
        last_observed = site_history[TIME_COL].max()

        # Forecast the 24 hours immediately after the last
        # timestamp observed for this site.
        future_timestamps = pd.date_range(
            start=last_observed + pd.Timedelta(hours=1),
            periods=HORIZON_HOURS,
            freq="h",
        )

        future = pd.DataFrame({
            SITE_COL: [str(site_id)] * HORIZON_HOURS,
            TIME_COL: future_timestamps,
        })

        X = make_features(future)

        # Use the same categorical vocabulary as the training data.
        X["site_id"] = pd.Categorical(
            X["site_id"],
            categories=site_categories,
        )

        if X["site_id"].isna().any():
            raise ValueError(
                f"Site {site_id!r} is not in the training categories."
            )

        predictions = np.asarray(model.predict(X), dtype=float)

        if len(predictions) != HORIZON_HOURS:
            raise ValueError(
                f"Expected {HORIZON_HOURS} predictions for {site_id}, "
                f"got {len(predictions)}."
            )

        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Non-finite predictions produced for site {site_id}."
            )

        # Consumption cannot be negative. Record how often this
        # guard is needed; frequent clipping deserves investigation.
        clipped_count = int((predictions < 0).sum())
        predictions = np.maximum(predictions, 0.0)

        forecast_frames.append(pd.DataFrame({
            SITE_COL: str(site_id),
            "forecast_for": future_timestamps,
            "prediction_kwh": predictions,
            "source_last_observation": last_observed,
            "generated_at": generated_at,
            "model_version": MODEL_PATH.name,
            "negative_predictions_clipped": clipped_count,
        }))

    forecasts = pd.concat(forecast_frames, ignore_index=True)

    # Validate the final output contract.
    if forecasts.duplicated([SITE_COL, "forecast_for"]).any():
        raise ValueError("Duplicate site/timestamp forecast rows found.")

    counts = forecasts.groupby(SITE_COL).size()
    if not (counts == HORIZON_HOURS).all():
        raise ValueError(
            f"Unexpected forecast counts by site: {counts.to_dict()}"
        )

    if not np.isfinite(forecasts["prediction_kwh"]).all():
        raise ValueError("Forecast output contains non-finite values.")

    if (forecasts["prediction_kwh"] < 0).any():
        raise ValueError("Forecast output contains negative consumption.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(OUTPUT_PATH, index=False)

    print(f"Forecast rows: {len(forecasts)}")
    print(f"Sites forecast: {forecasts[SITE_COL].nunique()}")
    print(f"Forecast horizon: {HORIZON_HOURS} hours per site")
    print(f"Output saved to: {OUTPUT_PATH}")
    print()
    print(forecasts.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
