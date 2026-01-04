from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from datasets import Dataset

import sns_mlops.train as t


def _write_processed_split(root: Path, *, name: str, texts: list[str], labels: list[int]) -> None:
    ds = Dataset.from_dict({"text": texts, "label": labels, "label_text": ["neutral"] * len(texts)})
    ds.to_parquet(str(root / f"{name}.parquet"))


def _clear_root_logging_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


class DummyTokenizer:
    def __call__(self, texts, *, truncation: bool = True, max_length: int = 128, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        return {
            "input_ids": [[1, 2, 3]] * batch_size,
            "attention_mask": [[1, 1, 1]] * batch_size,
        }

    def save_pretrained(self, output_dir: str, **kwargs) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyTrainOutput:
    def __init__(self, metrics: dict[str, float]):
        self.metrics = metrics


class DummyPredictOutput:
    def __init__(self, metrics: dict[str, float]):
        self.metrics = metrics


class DummyTrainer:
    def __init__(self, *positional_args, model=None, args=None, tokenizer=None, **kwargs):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer

    def train(self) -> DummyTrainOutput:
        return DummyTrainOutput({"train_loss": 1.0})

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.5, "eval_accuracy": 0.75}

    def predict(self, *args, **kwargs) -> DummyPredictOutput:
        return DummyPredictOutput({"test_loss": 0.4})

    def save_model(self, output_dir: str, **kwargs) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_text("dummy", encoding="utf-8")
        (out / "config.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize("save_model", [True, False])
def test_train_writes_artifacts_and_respects_save_flags(tmp_path, monkeypatch, save_model):
    processed_root = tmp_path / "processed"
    tier_dir = processed_root / "small"
    tier_dir.mkdir(parents=True, exist_ok=True)

    _write_processed_split(tier_dir, name="train", texts=["a", "b", "c", "d"], labels=[0, 1, 2, 1])
    _write_processed_split(tier_dir, name="val", texts=["e", "f"], labels=[0, 2])
    _write_processed_split(tier_dir, name="test", texts=["g", "h"], labels=[1, 2])

    (tier_dir / "metadata.json").write_text(
        json.dumps({"dataset_name": "dummy", "revision": None, "tier": "small"}, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "models"

    def fake_build_tokenizer_and_model(*args, **kwargs):
        return DummyTokenizer(), object()

    monkeypatch.setattr(t, "build_tokenizer_and_model", fake_build_tokenizer_and_model)
    monkeypatch.setattr(t, "Trainer", DummyTrainer)

    _clear_root_logging_handlers()
    t.train(
        tier="small",
        processed_root=processed_root,
        output_dir=output_dir,
        model_name="dummy/model",
        model_revision=None,
        seed=123,
        max_length=16,
        num_train_epochs=1.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_train_samples=None,
        max_eval_samples=None,
        max_test_samples=None,
        save_total_limit=1,
        overwrite_output_dir=True,
        save_model=save_model,
        save_checkpoints=False,
    )

    run_dir = output_dir / "small"
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "train.log").exists()

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["save_checkpoints"] is False
    assert run_config["training_args"]["save_strategy"] == "no"

    model_dir = run_dir / t.DEFAULT_MODEL_SUBDIR
    if save_model:
        assert (model_dir / "model.safetensors").exists()
        assert (model_dir / "tokenizer.json").exists()
    else:
        assert not model_dir.exists()
