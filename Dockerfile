# MLflow 3 requires Python >= 3.10, and this matches the version used in CI.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# curl is needed by the container HEALTHCHECK below.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first so dependency layers stay cached across code changes.
COPY requirements.txt .
# Generous timeout/retries: the ML wheels are large and slow mirrors break builds.
RUN pip install --no-cache-dir --retries 10 --timeout 180 -r requirements.txt

COPY . .

# Default to the local model artefact; docker-compose can point this elsewhere.
ENV MODEL_PATH=/app/models/best_model.pkl \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    MLFLOW_ALLOW_FILE_STORE=true

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
