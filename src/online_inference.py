"""FastAPI service for single-timestamp energy consumption predictions."""

from contextlib import asynccontextmanager
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

from batch_inference import MODEL_PATH, make_features


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "training_report.json"

HTTP_REQUESTS = Counter(
    "api_http_requests_total",
    "Total HTTP requests handled by the inference API.",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "api_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)


class PredictionRequest(BaseModel):
    site_id: Annotated[str, Field(min_length=1, examples=["office_001"])]
    timestamp: datetime = Field(
        description="Timestamp to predict, including timezone if relevant.",
        examples=["2025-04-01T12:00:00"],
    )


class PredictionResponse(BaseModel):
    site_id: str
    timestamp: datetime
    prediction_kwh: float = Field(ge=0)
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained model not found at {MODEL_PATH}. Run src/train.py first."
        )
    if not REPORT_PATH.exists():
        raise RuntimeError(
            f"Training report not found at {REPORT_PATH}; it contains the "
            "site categories used by the trained model."
        )

    with REPORT_PATH.open(encoding="utf-8") as report_file:
        training_report = json.load(report_file)

    app.state.model = joblib.load(MODEL_PATH)
    app.state.site_categories = training_report.get("sites_seen", [])
    if not app.state.site_categories:
        raise RuntimeError(f"No training sites were recorded in {REPORT_PATH}.")

    yield


app = FastAPI(
    title="Zappy Energy Forecasting API",
    description="Online predictions from the trained energy forecasting model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    started_at = time.perf_counter()
    status_code = "500"
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        labels = {"method": request.method, "path": request.url.path}
        HTTP_REQUESTS.labels(**labels, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(**labels).observe(
            time.perf_counter() - started_at
        )


app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(request.app.state, "model"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    site_id = payload.site_id.strip()
    if site_id not in request.app.state.site_categories:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown site_id {site_id!r}. Supported sites: "
                f"{request.app.state.site_categories}"
            ),
        )

    row = pd.DataFrame([{
        "site_id": site_id,
        "timestamp": pd.Timestamp(payload.timestamp),
    }])
    features = make_features(row)
    features["site_id"] = pd.Categorical(
        features["site_id"],
        categories=request.app.state.site_categories,
    )

    prediction = float(np.asarray(request.app.state.model.predict(features))[0])
    if not np.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail="The model returned a non-finite prediction.",
        )

    return PredictionResponse(
        site_id=site_id,
        timestamp=payload.timestamp,
        prediction_kwh=max(prediction, 0.0),
        model_version=MODEL_PATH.name,
    )
