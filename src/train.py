
from pathlib import Path
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models import infer_signature


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "energy_consumption.csv"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"

TARGET = "consumption_kwh"
KEYS = ["site_id", "timestamp"]

NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
]

FEATURES = NUMERIC_FEATURES + ["site_id"]


# ---------------------------------------------------------------------
# Data loading and basic quality checks
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run: uv run python src/generate_data.py"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    required = set(KEYS + [TARGET])
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print("=== Data audit ===")
    print(f"Rows loaded: {len(df):,}")
    print(f"Sites: {df['site_id'].nunique()}")
    print(f"Duplicate site/timestamp pairs: {df.duplicated(KEYS).sum()}")
    print(f"Missing values:\n{df[KEYS + [TARGET]].isna().sum()}")
    print(f"Timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # Exercise policy only: keep the first duplicate.
    # Production policy should investigate conflicting duplicate readings.
    df = df.drop_duplicates(KEYS, keep="first").copy()

    df = df.dropna(subset=KEYS + [TARGET])
    df = df[np.isfinite(df[TARGET])]
    df = df[df[TARGET] >= 0]

    df = df.sort_values(KEYS).reset_index(drop=True)

    print(f"Rows after basic cleaning: {len(df):,}")
    print()

    return df


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    hour = result["timestamp"].dt.hour
    day_of_week = result["timestamp"].dt.dayofweek
    day_of_year = result["timestamp"].dt.dayofyear

    # Cyclical encoding preserves the circular nature of time.
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    result["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    result["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    result["is_weekend"] = (day_of_week >= 5).astype(int)

    # HistGradientBoostingRegressor supports categorical pandas columns.
    result["site_id"] = result["site_id"].astype("category")

    return result


# ---------------------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------------------

def split_chronologically(df: pd.DataFrame):
    unique_timestamps = np.sort(df["timestamp"].unique())

    if len(unique_timestamps) < 2:
        raise ValueError("Not enough distinct timestamps to split the data.")

    cutoff_index = int(len(unique_timestamps) * 0.8)
    cutoff_index = min(max(cutoff_index, 1), len(unique_timestamps) - 1)
    cutoff = pd.Timestamp(unique_timestamps[cutoff_index])

    train = df[df["timestamp"] < cutoff].copy()
    test = df[df["timestamp"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

    print("=== Chronological split ===")
    print(f"Cutoff: {cutoff}")
    print(f"Train: {train['timestamp'].min()} -> {train['timestamp'].max()}")
    print(f"Test:  {test['timestamp'].min()} -> {test['timestamp'].max()}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print()

    return train, test, cutoff


# ---------------------------------------------------------------------
# Baseline forecasting
# ---------------------------------------------------------------------

def seasonal_naive(
    history: pd.DataFrame,
    requested: pd.DataFrame,
) -> pd.Series:
    """
    Predict consumption from the same site and hour 24 hours earlier.

    `history` may include observations from earlier in the test period.
    This is appropriate for rolling evaluation only when each lagged
    observation would have been available at the time of prediction.
    """
    lagged = history[KEYS + [TARGET]].copy()

    # A value observed at t becomes the prediction for t + 24 hours.
    lagged["timestamp"] += pd.Timedelta(hours=24)
    lagged = lagged.rename(columns={TARGET: "baseline_prediction"})

    joined = requested[KEYS].merge(
        lagged,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Merge order can be implementation-dependent; restore the requested
    # row order by joining on a temporary ID.
    requested_keys = requested[KEYS].copy()
    requested_keys["_row_id"] = np.arange(len(requested_keys))

    lagged = history[KEYS + [TARGET]].copy()
    lagged["timestamp"] += pd.Timedelta(hours=24)
    lagged = lagged.rename(columns={TARGET: "baseline_prediction"})

    aligned = requested_keys.merge(
        lagged,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row_id")

    return aligned["baseline_prediction"].reset_index(drop=True)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred) -> dict:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    coverage = float(valid.mean()) if len(valid) else 0.0

    if not valid.any():
        return {
            "mae": None,
            "rmse": None,
            "coverage": coverage,
            "n_valid": 0,
            "n_total": int(len(actual)),
        }

    mae = mean_absolute_error(actual[valid], predicted[valid])
    rmse = np.sqrt(mean_squared_error(actual[valid], predicted[valid]))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "coverage": coverage,
        "n_valid": int(valid.sum()),
        "n_total": int(len(actual)),
    }


def print_metrics(label: str, result: dict) -> None:
    if result["mae"] is None:
        print(f"{label}: no valid predictions")
        return

    print(
        f"{label}: "
        f"MAE={result['mae']:.3f} kWh, "
        f"RMSE={result['rmse']:.3f} kWh, "
        f"coverage={result['coverage']:.1%} "
        f"({result['n_valid']:,}/{result['n_total']:,})"
    )


# ---------------------------------------------------------------------
# Training and reporting
# ---------------------------------------------------------------------

def _run_training() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    data = add_features(raw)

    train, test, cutoff = split_chronologically(data)

    # Baseline can use historical observations from earlier test timestamps,
    # which simulates rolling forecasts with newly observed data.
    baseline_predictions = seasonal_naive(data, test).to_numpy()

    # Train the candidate model.
    # site_id is the final feature, so its categorical feature index is
    # derived from the feature list rather than hard-coded.
    categorical_indices = [FEATURES.index("site_id")]

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        categorical_features=categorical_indices,
        random_state=42,
    )

    mlflow.log_params({
        "model_type": type(model).__name__,
        "features": ",".join(FEATURES),
        "target": TARGET,
        "raw_rows": len(raw),
        "cleaned_rows": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "n_sites": data["site_id"].nunique(),
        "cutoff": cutoff.isoformat(),
        **{
            f"model_{key}": value
            for key, value in model.get_params().items()
            if key != "categorical_features"
        },
    })

    model.fit(train[FEATURES], train[TARGET])
    candidate_predictions = model.predict(test[FEATURES])

    print("=== Overall evaluation ===")
    baseline_metrics = calculate_metrics(test[TARGET], baseline_predictions)
    candidate_metrics = calculate_metrics(test[TARGET], candidate_predictions)

    print_metrics("Previous-day same-hour baseline", baseline_metrics)
    print_metrics("Gradient boosting candidate", candidate_metrics)

    # Fair comparison on rows where BOTH approaches produced predictions.
    actual = test[TARGET].to_numpy()
    comparable = (
        np.isfinite(actual)
        & np.isfinite(baseline_predictions)
        & np.isfinite(candidate_predictions)
    )

    if comparable.any():
        print("\n=== Head-to-head comparison ===")
        print(
            f"Comparable rows: {comparable.sum():,}/{len(test):,} "
            f"({comparable.mean():.1%})"
        )
        baseline_comparable_metrics = calculate_metrics(
            actual[comparable], baseline_predictions[comparable]
        )
        candidate_comparable_metrics = calculate_metrics(
            actual[comparable], candidate_predictions[comparable]
        )
        print_metrics(
            "Baseline on comparable rows",
            baseline_comparable_metrics,
        )
        print_metrics(
            "Candidate on comparable rows",
            candidate_comparable_metrics,
        )
    else:
        baseline_comparable_metrics = {}
        candidate_comparable_metrics = {}

    # Per-site metrics, using positional alignment within each site.
    test_results = test[KEYS + [TARGET]].copy().reset_index(drop=True)
    test_results["baseline_prediction"] = baseline_predictions
    test_results["candidate_prediction"] = candidate_predictions

    per_site_metrics = []

    print("\n=== Per-site evaluation ===")
    for site_id, group in test_results.groupby("site_id", sort=True):
        baseline_site = calculate_metrics(
            group[TARGET],
            group["baseline_prediction"],
        )
        candidate_site = calculate_metrics(
            group[TARGET],
            group["candidate_prediction"],
        )

        print(
            f"{site_id}: "
            f"baseline MAE={baseline_site['mae']}, "
            f"candidate MAE={candidate_site['mae']}"
        )

        per_site_metrics.append({
            "site_id": str(site_id),
            "n_test_rows": int(len(group)),
            "baseline": baseline_site,
            "candidate": candidate_site,
        })

    # Save the trained model.
    model_path = ARTIFACT_DIR / "energy_forecaster.joblib"
    joblib.dump(model, model_path)

    # Save evaluation and provenance information.
    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "sklearn_version": sklearn.__version__,
        "feature_columns": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": ["site_id"],
        "target": TARGET,
        "training_cutoff_exclusive": cutoff.isoformat(),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sites_seen": sorted(data["site_id"].astype(str).unique().tolist()),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "per_site_metrics": per_site_metrics,
        "model_path": str(model_path.relative_to(ROOT)),
        "limitations": [
            "Single chronological holdout; rolling-origin evaluation is recommended.",
            "Calendar features only; recent-consumption lag features are not included.",
            "Synthetic data does not establish real-world forecasting performance.",
            "Duplicate handling keeps the first row and requires review for production.",
        ],
    }

    report_path = OUTPUT_DIR / "training_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    # Save row-level predictions for later error analysis.
    prediction_path = OUTPUT_DIR / "test_predictions.csv"
    test_results.to_csv(prediction_path, index=False)

    metrics_to_log = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "baseline_comparable": baseline_comparable_metrics,
        "candidate_comparable": candidate_comparable_metrics,
    }
    mlflow.log_metrics({
        f"{prefix}_{name}": value
        for prefix, metrics in metrics_to_log.items()
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and value is not None
    })

    mlflow.log_artifact(str(model_path), artifact_path="files")
    mlflow.log_artifact(str(report_path), artifact_path="files")
    mlflow.log_artifact(str(prediction_path), artifact_path="files")
    mlflow.log_artifact(str(DATA_PATH), artifact_path="input")

    signature = infer_signature(test[FEATURES], candidate_predictions)
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="zappy_forecasting_model",
        signature=signature,
        input_example=test[FEATURES].head(5),
        skops_trusted_types=[
            "functools.partial",
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            "sklearn.utils.validation.check_array",
        ],
    )

    print("\n=== Artifacts saved ===")
    print(f"Model:       {model_path}")
    print(f"Report:      {report_path}")
    print(f"Predictions: {prediction_path}")


def main() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "energy-forecasting")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "project": "zappy-forecasting",
            "training_script": "src/train.py",
            "sklearn_version": sklearn.__version__,
        })
        _run_training()
        print(f"MLflow run: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
