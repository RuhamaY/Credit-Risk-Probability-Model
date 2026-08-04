"""Tests for the FastAPI scoring service."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


class DummyModel:
    """Stand-in estimator so API tests do not depend on a trained artefact."""

    def __init__(self, proba=0.9):
        self.proba = proba
        self.seen = None

    def predict_proba(self, X):
        self.seen = X
        return np.column_stack([np.full(len(X), 1 - self.proba), np.full(len(X), self.proba)])


@pytest.fixture
def customer_payload():
    return {
        "customer_id": "CustomerId_1",
        "Recency": 83,
        "Frequency": 1,
        "MonetarySum": 10000,
        "MonetaryMean": 10000,
        "MonetaryStd": 0,
        "TransactionHour": 16,
        "TransactionDay": 21,
        "TransactionMonth": 11,
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_2",
        "PricingStrategy": "4",
    }


@pytest.fixture
def client(monkeypatch):
    """Client whose lifespan loads a deterministic dummy model."""
    model = DummyModel()
    monkeypatch.setattr(api_main, "load_model", lambda *_args, **_kwargs: model)
    with TestClient(api_main.app) as test_client:
        test_client.model = model
        yield test_client


def test_health_check_reports_loaded_model(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_returns_scored_response(client, customer_payload):
    response = client.post("/predict", json=customer_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["customer_id"] == "CustomerId_1"
    assert body["risk_probability"] == pytest.approx(0.9)
    assert body["risk_category"] == "High"
    assert 300 <= body["credit_score"] <= 850


def test_predict_sends_features_in_training_order(client, customer_payload):
    """The frame handed to the model must match the trained column order exactly."""
    client.post("/predict", json=customer_payload)
    assert list(client.model.seen.columns) == api_main.FEATURE_COLS
    assert "customer_id" not in client.model.seen.columns


def test_low_probability_maps_to_low_risk(monkeypatch, customer_payload):
    monkeypatch.setattr(api_main, "load_model", lambda *_a, **_k: DummyModel(proba=0.01))
    with TestClient(api_main.app) as client:
        body = client.post("/predict", json=customer_payload).json()

    assert body["risk_category"] == "Low"
    assert body["credit_score"] > 800


def test_predict_rejects_invalid_payload(client, customer_payload):
    invalid = dict(customer_payload, TransactionMonth=99)
    assert client.post("/predict", json=invalid).status_code == 422
    assert client.post("/predict", json={"customer_id": "x"}).status_code == 422


def test_batch_predict_scores_every_customer(client, customer_payload):
    second = dict(customer_payload, customer_id="CustomerId_2")
    response = client.post("/batch_predict", json={"customers": [customer_payload, second]})
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 2
    assert [p["customer_id"] for p in body["predictions"]] == ["CustomerId_1", "CustomerId_2"]


def test_predict_returns_503_without_model(monkeypatch, customer_payload):
    """A missing model must surface as 'unavailable', not a 500."""
    def _fail(*_args, **_kwargs):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(api_main, "load_model", _fail)
    with TestClient(api_main.app) as client:
        assert client.get("/").json()["model_loaded"] is False
        assert client.post("/predict", json=customer_payload).status_code == 503
