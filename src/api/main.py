"""FastAPI service exposing credit risk probabilities for new customers."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    BatchPredictionRequest,
    BatchPredictionResult,
    CustomerFeatures,
    PredictionResult,
)
from src.data_processing import CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS
from src.predict import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than on every request."""
    try:
        ml_models["model"] = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.warning("Model unavailable at startup: %s", exc)
    yield
    ml_models.clear()


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts the probability that a customer belongs to the high-risk segment.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model():
    model = ml_models.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model with `python -m src.train` first.",
        )
    return model


def _to_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    """Build a feature frame in the exact column order the model was trained on."""
    rows = [c.model_dump(exclude={"customer_id"}) for c in customers]
    return pd.DataFrame(rows)[FEATURE_COLS]


def _score(customer_id: str, proba: float) -> PredictionResult:
    """Map a probability onto a credit score, risk band, and lending recommendation."""
    score = int(round(MAX_CREDIT_SCORE - proba * (MAX_CREDIT_SCORE - MIN_CREDIT_SCORE)))
    score = max(MIN_CREDIT_SCORE, min(MAX_CREDIT_SCORE, score))

    if proba > HIGH_RISK_THRESHOLD:
        category = "High"
    elif proba > MEDIUM_RISK_THRESHOLD:
        category = "Medium"
    else:
        category = "Low"

    if score >= 750:
        limit, term = 10000.0, 60
    elif score >= 650:
        limit, term = 5000.0, 36
    elif score >= 550:
        limit, term = 2000.0, 24
    else:
        limit, term = 500.0, 12

    return PredictionResult(
        customer_id=customer_id,
        risk_probability=proba,
        risk_category=category,
        credit_score=score,
        recommended_limit=limit,
        recommended_term=term,
    )


@app.get("/")
async def health_check():
    """Liveness probe reporting whether the model is ready to serve."""
    return {"status": "healthy", "model_loaded": "model" in ml_models}


@app.post("/predict", response_model=PredictionResult)
async def predict(features: CustomerFeatures):
    """Score a single customer and return their risk probability."""
    model = _get_model()
    try:
        proba = float(model.predict_proba(_to_frame([features]))[0, 1])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return _score(features.customer_id, proba)


@app.post("/batch_predict", response_model=BatchPredictionResult)
async def batch_predict(request: BatchPredictionRequest):
    """Score many customers in a single call."""
    model = _get_model()
    try:
        probabilities = model.predict_proba(_to_frame(request.customers))[:, 1]
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    predictions = [
        _score(customer.customer_id, float(proba))
        for customer, proba in zip(request.customers, probabilities)
    ]
    return BatchPredictionResult(predictions=predictions, count=len(predictions))
