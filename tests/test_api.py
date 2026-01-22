import pytest
from fastapi.testclient import TestClient
from sns_mlops.api import app

# Initialize test client
client = TestClient(app)

def test_read_main():
    # Verify welcome message at root path
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health_check():
    # Verify health check endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data

def test_predict_sentiment():
    # Verify prediction endpoint functionality
    payload = {"text": "Stock market is looking bullish today."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "probabilities" in data
    # Verify probability distribution validity
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5

def test_predict_invalid_input():
    # Verify error handling for invalid input
    response = client.post("/predict", json={"wrong_field": "data"})
    assert response.status_code == 422