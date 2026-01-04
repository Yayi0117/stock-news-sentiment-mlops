from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments, set_seed

from sns_mlops.model import DEFAULT_MODEL_NAME, build_tokenizer_and_model

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_ROOT: Final[Path] = Path("data/processed")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("models/finbert")
DEFAULT_MODEL_SUBDIR: Final[str] = "model"
TEXT_COLUMN: Final[str] = "text"
LABEL_COLUMN: Final[str] = "label"
LABEL_TEXT_COLUMN: Final[str] = "label_text"
TRAIN_SPLIT: Final[str] = "train"
VAL_SPLIT: Final[str] = "val"
TEST_SPLIT: Final[str] = "test"


@dataclass(frozen=True)
class RunConfig:
    """Serializable configuration snapshot for a training run."""

    created_at_utc: str
    model_name: str
    model_revision: str | None
    data_tier: str
    processed_root: str
    output_dir: str
    seed: int
    max_length: int
    save_model: bool
    save_checkpoints: bool
    training_args: dict[str, Any]
    data_metadata: dict[str, Any] | None
    versions: dict[str, str]
    git_commit: str | None


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _try_get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _load_processed_dataset(processed_root: Path, tier: str) -> DatasetDict:
    tier_dir = processed_root / tier
    data_files = {
        TRAIN_SPLIT: str(tier_dir / f"{TRAIN_SPLIT}.parquet"),
        VAL_SPLIT: str(tier_dir / f"{VAL_SPLIT}.parquet"),
        TEST_SPLIT: str(tier_dir / f"{TEST_SPLIT}.parquet"),
    }
    ds = load_dataset("parquet", data_files=data_files)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict from parquet loader, got {type(ds)!r}")
    return ds


def _maybe_subsample(dataset: Dataset, n: int | None, *, seed: int) -> Dataset:
    if n is None:
        return dataset
    if n <= 0:
        raise ValueError("`max_samples` must be > 0 when provided.")
    if n >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(list(range(n)))


def _prepare_tokenized_splits(
    splits: DatasetDict,
    *,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    if LABEL_COLUMN in splits[TRAIN_SPLIT].column_names:
        splits = splits.rename_column(LABEL_COLUMN, "labels")

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=max_length)

    remove_columns = [c for c in [TEXT_COLUMN, LABEL_TEXT_COLUMN] if c in splits[TRAIN_SPLIT].column_names]
    return splits.map(_tokenize, batched=True, remove_columns=remove_columns, desc="Tokenizing text")


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def train(
    *,
    tier: str,
    processed_root: Path,
    output_dir: Path,
    model_name: str,
    model_revision: str | None,
    seed: int,
    max_length: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    max_test_samples: int | None,
    save_total_limit: int,
    overwrite_output_dir: bool,
    save_model: bool,
    save_checkpoints: bool,
) -> None:
    """Fine-tune a FinBERT model using the Hugging Face Trainer API."""
    run_output_dir = output_dir / tier
    if overwrite_output_dir and run_output_dir.exists():
        shutil.rmtree(run_output_dir)
    _setup_logging(run_output_dir)

    logger.info("Starting training run")
    logger.info("tier=%s processed_root=%s output_dir=%s", tier, processed_root, run_output_dir)
    logger.info("model=%s revision=%s", model_name, model_revision)
    logger.info("save_model=%s save_checkpoints=%s overwrite_output_dir=%s", save_model, save_checkpoints, overwrite_output_dir)

    set_seed(seed)

    data_metadata_path = processed_root / tier / "metadata.json"
    data_metadata = _read_json(data_metadata_path) if data_metadata_path.exists() else None

    raw_splits = _load_processed_dataset(processed_root, tier)
    raw_splits[TRAIN_SPLIT] = _maybe_subsample(raw_splits[TRAIN_SPLIT], max_train_samples, seed=seed)
    raw_splits[VAL_SPLIT] = _maybe_subsample(raw_splits[VAL_SPLIT], max_eval_samples, seed=seed)
    raw_splits[TEST_SPLIT] = _maybe_subsample(raw_splits[TEST_SPLIT], max_test_samples, seed=seed)

    tokenizer, model = build_tokenizer_and_model(model_name, revision=model_revision)
    tokenized_splits = _prepare_tokenized_splits(raw_splits, tokenizer=tokenizer, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    save_strategy = "epoch" if save_checkpoints else "no"
    load_best_model_at_end = save_checkpoints

    args = TrainingArguments(
        output_dir=str(run_output_dir),
        overwrite_output_dir=overwrite_output_dir,
        seed=seed,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
    )

    run_config = RunConfig(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        model_revision=model_revision,
        data_tier=tier,
        processed_root=str(processed_root),
        output_dir=str(run_output_dir),
        seed=seed,
        max_length=max_length,
        save_model=save_model,
        save_checkpoints=save_checkpoints,
        training_args=args.to_dict(),
        data_metadata=data_metadata,
        versions={
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "transformers": __import__("transformers").__version__,
            "torch": __import__("torch").__version__,
        },
        git_commit=_try_get_git_commit(),
    )
    _write_json(run_output_dir / "run_config.json", asdict(run_config))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_splits[TRAIN_SPLIT],
        eval_dataset=tokenized_splits[VAL_SPLIT],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    train_result = trainer.train()
    if save_model:
        model_dir = run_output_dir / DEFAULT_MODEL_SUBDIR
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    eval_metrics = trainer.evaluate()
    test_output = trainer.predict(tokenized_splits[TEST_SPLIT])
    test_metrics = dict(test_output.metrics)

    metrics = {
        "train": dict(train_result.metrics),
        "val": dict(eval_metrics),
        "test": dict(test_metrics),
    }
    _write_json(run_output_dir / "metrics.json", metrics)

    logger.info("Finished training run. Metrics written to %s", run_output_dir / "metrics.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune ProsusAI/finbert using Hugging Face Trainer.")
    parser.add_argument("--tier", default="small", choices=["small", "dev", "full"])
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--overwrite-output-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    train(
        tier=args.tier,
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_revision=args.model_revision,
        seed=args.seed,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
        save_total_limit=args.save_total_limit,
        overwrite_output_dir=args.overwrite_output_dir,
        save_model=args.save_model,
        save_checkpoints=args.save_checkpoints,
    )


if __name__ == "__main__":
    main()
