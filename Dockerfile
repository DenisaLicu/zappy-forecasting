FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8000 5000

CMD ["uv", "run", "--no-sync", "uvicorn", "online_inference:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
