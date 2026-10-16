# Zappy forecasting

## Training with MLflow

Install the project dependencies, then start the tracking server from the
project root:

```bash
uv sync
uv run mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 127.0.0.1 \
  --port 5000
```

In another terminal, run training:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python src/train.py
```

The default experiment is `energy-forecasting`. Set
`MLFLOW_EXPERIMENT_NAME` to use another experiment. Run metadata is stored in
`mlflow.db`, while logged models and other artifacts are stored in
`mlartifacts/`. Keep both paths in place between restarts; the paths above are
absolute so launching the server from another working directory will not
silently create a fresh store. Each training run also registers a new model
version under `zappy_recasting_model` in the MLflow Model Registry.

## Online inference API

Train a model first, then start the complete local stack from the project
root:

```bash
docker compose up --build -d
```

The API loads `artifacts/energy_forecaster.joblib` once at startup and uses the
training report's site categories to validate requests. The Compose stack also
starts MLflow, Prometheus, Loki, Grafana Alloy, and Grafana. Alloy ships Docker
container logs into Loki, and Prometheus scrapes the API's `/metrics` endpoint.

| Service | Local URL | Purpose |
| --- | --- | --- |
| Inference API | http://127.0.0.1:8000/docs | API and OpenAPI docs |
| MLflow | http://127.0.0.1:5000 | Experiments and model registry |
| Prometheus | http://127.0.0.1:9090 | Scraped API metrics |
| Loki | http://127.0.0.1:3100 | Log query endpoint |
| Grafana | http://127.0.0.1:3000 | Metrics and logs dashboards |

Grafana defaults to `admin` / `admin` for this local stack. Set
`GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in the environment before
starting Compose to change those credentials. The provisioned Energy
Forecasting dashboard includes request rate, prediction latency, and API logs.
Prometheus, Loki, Grafana, and Alloy state use named Docker volumes. MLflow
continues to use the workspace's `mlflow.db` and `mlartifacts/` paths. Alloy
reads Docker container logs through the Docker socket, so this Compose setup is
intended for local development.

Stop the stack without removing persistent data:

```bash
docker compose down
```

Do not add `--volumes` if you want to keep Grafana, Prometheus, and Loki data.

Send a prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"site_id":"office_001","timestamp":"2025-04-01T12:00:00"}'
```
