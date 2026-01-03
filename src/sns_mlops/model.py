from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MODEL_NAME: Final[str] = "ProsusAI/finbert"

DEFAULT_LABEL2ID: Final[dict[str, int]] = {"negative": 0, "neutral": 1, "positive": 2}
DEFAULT_ID2LABEL: Final[dict[int, str]] = {v: k for k, v in DEFAULT_LABEL2ID.items()}


def get_label_mappings(*, label2id: dict[str, int] | None = None) -> tuple[dict[str, int], dict[int, str]]:
    """Return copies of label mappings used for sentiment classification."""
    effective_label2id = DEFAULT_LABEL2ID if label2id is None else label2id
    return dict(effective_label2id), {v: k for k, v in effective_label2id.items()}


def build_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    """Build a tokenizer for the given Hugging Face model."""
    return AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        use_fast=use_fast,
    )


def build_model_config(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    label2id: dict[str, int] | None = None,
) -> PretrainedConfig:
    """Build a model config with an explicit label mapping."""
    final_label2id, final_id2label = get_label_mappings(label2id=label2id)
    return AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        num_labels=len(final_label2id),
        label2id=final_label2id,
        id2label=final_id2label,
    )


def build_pretrained_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    config: PretrainedConfig | None = None,
) -> PreTrainedModel:
    """Build a pretrained sequence classification model."""
    final_config = config or build_model_config(model_name, revision=revision, cache_dir=cache_dir)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        config=final_config,
    )


def build_tokenizer_and_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Convenience function to build tokenizer and pretrained model consistently."""
    config = build_model_config(model_name, revision=revision, cache_dir=cache_dir)
    tokenizer = build_tokenizer(model_name, revision=revision, cache_dir=cache_dir)
    model = build_pretrained_model(model_name, revision=revision, cache_dir=cache_dir, config=config)
    return tokenizer, model


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_length: int = 128,
) -> dict[str, Any]:
    """Tokenize a list of texts into a batch suitable for Transformer models."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
