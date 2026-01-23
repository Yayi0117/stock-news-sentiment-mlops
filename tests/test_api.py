from __future__ import annotations

import torch
from fastapi.testclient import TestClient

import sns_mlops.api as api


class DummyOutputs:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class DummyModel:
    def __init__(self):
        self.name_or_path = "dummy"

        class Config:
            id2label = {0: "negative", 1: "neutral", 2: "positive"}

        self.config = Config()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, **inputs):
        logits = torch.tensor([[0.0, 1.0, 2.0]])
        return DummyOutputs(logits=logits)


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * batch_size),
            "attention_mask": torch.tensor([[1, 1, 1]] * batch_size),
        }


def test_root_is_available():
    client = TestClient(api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_health_is_ok_without_model_loaded():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] in {True, False}


def test_predict_uses_dummy_model(monkeypatch):
    def fake_loader(model_name_or_path: str):
        return DummyTokenizer(), DummyModel(), "cpu"

    # CI runs with Hugging Face offline flags enabled. Make the API select a non-default
    # model source so `ensure_model_loaded()` does not fail before calling our fake loader.
    monkeypatch.setenv("SNS_MLOPS_MODEL_DIR", "dummy")

    monkeypatch.setattr(api, "_load_model_components", fake_loader)
    api.ml_models.clear()

    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "Stocks rallied today."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= payload["score"] <= 1.0
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6


def test_predict_invalid_input_returns_422():
    client = TestClient(api.app)
    response = client.post("/predict", json={"wrong_field": "data"})
    assert response.status_code == 422
