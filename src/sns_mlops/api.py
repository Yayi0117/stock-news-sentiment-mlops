"""FastAPI application entrypoint.

This module provides a REST API for sentiment classification using the FinBERT model.
It integrates with the project's model management utilities to load artifacts from
local disk with a strict fallback strategy (Full -> Dev -> Small).
"""

from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sns_mlops.model import build_tokenizer_and_model, tokenize_batch

# 定义模型搜索路径优先级：Full -> Dev (中等) -> Small
MODEL_SEARCH_PATHS = [
    Path("models/finbert/full/model"),
    Path("models/finbert/dev/model"),
    Path("models/finbert/small/model"),
]

# 全局变量用于存储加载后的模型组件
ml_models: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    """Prediction request schema."""
    text: str


class PredictResponse(BaseModel):
    """Prediction response schema."""
    text: str
    label: str
    score: float
    probabilities: Dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    Handles loading the model on startup and cleaning up on shutdown.
    """
    model_name_or_path: Optional[str] = None
    
    # 1. 按照优先级 Full -> Dev -> Small 查找本地模型
    print("INFO: Searching for local models...")
    for path in MODEL_SEARCH_PATHS:
        if path.exists():
            model_name_or_path = str(path)
            print(f"INFO: Found local model at priority path: {model_name_or_path}")
            break
        else:
            print(f"DEBUG: Path not found: {path}")

    # 2. 如果所有路径都未找到，直接报错（不再回退到 Hugging Face）
    if model_name_or_path is None:
        error_msg = (
            f"CRITICAL: No local model found in any of the search paths: "
            f"{[str(p) for p in MODEL_SEARCH_PATHS]}. "
            "Please ensure you have run 'dvc repro train' or 'python -m sns_mlops.train' "
            "to generate model artifacts."
        )
        print(error_msg)
        raise RuntimeError(error_msg)

    try:
        print(f"INFO: Loading model artifacts from {model_name_or_path}...")
        # 3. 复用 src/sns_mlops/model.py 中的构建函数
        tokenizer, model = build_tokenizer_and_model(model_name=model_name_or_path)
        
        # 4. 配置推理环境 (GPU/CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()  # 切换到评估模式
        model.to(device)

        # 5. 保存到全局字典
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        ml_models["device"] = device
        # 保存 id2label 映射 (例如 {0: 'negative', ...}) 以便将预测 ID 转为文本
        ml_models["id2label"] = model.config.id2label
        
        print(f"INFO: Model loaded successfully on {device}")
    
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_name_or_path}. Details: {e}")
        # 如果模型加载失败，应用启动应报错
        raise RuntimeError("Could not load model artifacts") from e

    yield

    # Shutdown: 清理资源
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
        "health_url": "/health"
    }


@app.get("/health")
async def health():
    """Health check endpoint for Kubernetes/Docker probes."""
    # 检查模型是否已加载
    if not ml_models.get("model") or not ml_models.get("tokenizer"):
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not ready")
    
    return {
        "status": "ok", 
        "device": ml_models.get("device", "unknown"),
        "loaded_model": ml_models.get("model").name_or_path if ml_models.get("model") else "none"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Inference endpoint.
    Takes a text string and returns the sentiment label and probabilities.
    """
    # 获取模型组件
    tokenizer: PreTrainedTokenizerBase = ml_models.get("tokenizer")
    model: PreTrainedModel = ml_models.get("model")
    device: str = ml_models.get("device")
    id2label: Dict[int, str] = ml_models.get("id2label")

    if not tokenizer or not model:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        # 1. 预处理：调用 src/sns_mlops/model.py 中的工具
        # 注意：tokenize_batch 期望 list[str]，所以我们要把 request.text 包起来
        inputs = tokenize_batch(tokenizer, [request.text], max_length=128)
        
        # 将 tensor 移动到正确的设备 (CPU 或 GPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 2. 推理：不计算梯度
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 使用 Softmax 获取概率分布
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # 3. 后处理：获取最高分的标签
        top_score, top_label_id = torch.max(probs, dim=-1)
        top_label_id = int(top_label_id.item())
        top_score = float(top_score.item())

        # 构建完整的概率字典
        prob_dict = {
            id2label[i]: float(probs[i].item())
            for i in range(len(id2label))
        }

        return PredictResponse(
            text=request.text,
            label=id2label.get(top_label_id, "unknown"),
            score=top_score,
            probabilities=prob_dict
        )

    except Exception as e:
        # 捕获推理过程中的任何异常 (如 OOM, 输入过长等)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    # 允许直接运行脚本进行调试
    uvicorn.run("sns_mlops.api:app", host="0.0.0.0", port=8000, reload=True)