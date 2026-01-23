"""FastAPI application entrypoint.

This module provides a minimal inference API for sentiment classification with FinBERT.

Design goals:
- Keep CI stable: importing the app must not trigger Hugging Face downloads.
- Keep runtime simple: load a local exported model when available, otherwise fall back to
  the Hugging Face Hub when online.

Environment variables:
- `SNS_MLOPS_MODEL_DIR`: Optional explicit model directory to load (e.g. `models/finbert/dev/model`).
- `SNS_MLOPS_LOAD_MODEL_ON_STARTUP`: If set to `1`, load the model at app startup.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sns_mlops.model import DEFAULT_MODEL_NAME, build_tokenizer_and_model, tokenize_batch

logger = logging.getLogger(__name__)

MODEL_SEARCH_PATHS = [
    Path("models/finbert/full/model"),
    Path("models/finbert/dev/model"),
    Path("models/finbert/small/model"),
]

ml_models: dict[str, Any] = {}


class PredictRequest(BaseModel):
    """Prediction request schema."""

    text: str


class PredictResponse(BaseModel):
    """Prediction response schema."""

    text: str
    label: str
    score: float
    probabilities: dict[str, float]


def _select_model_source() -> str:
    explicit = os.environ.get("SNS_MLOPS_MODEL_DIR")
    if explicit:
        return explicit

    for path in MODEL_SEARCH_PATHS:
        if path.exists():
            return str(path)

    return DEFAULT_MODEL_NAME


def _load_model_components(model_name_or_path: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, str]:
    tokenizer, model = build_tokenizer_and_model(model_name=model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    return tokenizer, model, device


def ensure_model_loaded() -> None:
    if ml_models.get("model") and ml_models.get("tokenizer"):
        return

    model_name_or_path = _select_model_source()
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if offline and model_name_or_path == DEFAULT_MODEL_NAME:
        raise RuntimeError("Model is not available locally and Hugging Face downloads are disabled (offline mode).")

    logger.info("Loading inference model from %s", model_name_or_path)
    tokenizer, model, device = _load_model_components(model_name_or_path)

    ml_models["tokenizer"] = tokenizer
    ml_models["model"] = model
    ml_models["device"] = device
    ml_models["model_name_or_path"] = model_name_or_path
    ml_models["id2label"] = dict(model.config.id2label)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_on_startup = os.environ.get("SNS_MLOPS_LOAD_MODEL_ON_STARTUP", "0") == "1"
    if load_on_startup:
        try:
            ensure_model_loaded()
        except Exception as exc:
            logger.exception("Failed to load model on startup: %s", exc)
            raise

    yield

    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Stock News Sentiment API",
    description="Inference API for FinBERT financial sentiment classification",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint for basic connectivity check."""
    return {
        "message": "Welcome to the Stock News Sentiment API",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health")
async def health():
    """Liveness endpoint.

    This endpoint returns 200 even when the model is not loaded yet. Use `/ready`
    for a strict readiness check.
    """
    return {
        "status": "ok",
        "model_loaded": bool(ml_models.get("model") and ml_models.get("tokenizer")),
        "device": ml_models.get("device", "unknown"),
        "model_name_or_path": ml_models.get("model_name_or_path", "none"),
    }


@app.get("/ready")
async def ready():
    """Readiness endpoint (requires a loaded model)."""
    if not ml_models.get("model") or not ml_models.get("tokenizer"):
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not ready")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment for a single input text."""
    try:
        ensure_model_loaded()
        tokenizer: PreTrainedTokenizerBase = ml_models["tokenizer"]
        model: PreTrainedModel = ml_models["model"]
        device: str = ml_models["device"]
        id2label: dict[int, str] = ml_models["id2label"]

        inputs = tokenize_batch(tokenizer, [request.text], max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        top_score, top_label_id = torch.max(probs, dim=-1)
        top_label_id = int(top_label_id.item())
        top_score = float(top_score.item())

        prob_dict = {id2label[i]: float(probs[i].item()) for i in range(len(id2label))}

        return PredictResponse(
            text=request.text,
            label=id2label.get(top_label_id, "unknown"),
            score=top_score,
            probabilities=prob_dict,
        )

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)) from e
