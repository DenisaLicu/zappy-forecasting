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
