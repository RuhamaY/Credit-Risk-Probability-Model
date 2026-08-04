"""Inference utilities for the credit risk model."""

from __future__ import annotations

import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "CreditRiskClassifier"
DEFAULT_MODEL_PATH = os.path.join("models", "best_model.pkl")


def load_model(model_path: str | None = DEFAULT_MODEL_PATH):
    """Load the trained pipeline from disk, falling back to the MLflow registry.

    The local artefact is preferred because the container may run without a
    reachable tracking server.
    """
    if model_path and os.path.exists(model_path):
        logger.info("Loading model from %s", model_path)
        return joblib.load(model_path)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    logger.info("Loading model '%s' from the MLflow registry", REGISTERED_MODEL_NAME)
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")


def predict_risk(features: pd.DataFrame, model=None) -> np.ndarray:
    """Return high-risk probabilities for the provided feature rows."""
    if model is None:
        model = load_model()
    return model.predict_proba(features)[:, 1]
