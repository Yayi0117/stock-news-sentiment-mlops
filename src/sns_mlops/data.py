"""Data ingestion and preprocessing pipeline.

This module builds reproducible dataset tiers (`small`, `dev`, `full`) for a
sentiment classification task using the Hugging Face dataset
`NOSIBLE/financial-sentiment`.

Key features:
- Schema validation of the raw dataset (required/allowed columns + label types).
- Minimal cleaning (drop irrelevant columns, filter invalid rows, encode labels).
- Deterministic splits into train/val/test using a fixed seed.
- Materialization to Parquet with a `metadata.json` sidecar for traceability.
- Raw caching and reuse: a raw parquet artifact is stored under `data/raw/` and
  reused if `dataset_name` and `revision` match.

Outputs:
- `data/raw/train.parquet` + `data/raw/metadata.json`
- `data/processed/<tier>/{train,val,test}.parquet` + `data/processed/<tier>/metadata.json`
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import datasets as hf_datasets
import pyarrow
import typer
from datasets import ClassLabel, DatasetDict, load_dataset
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME: Final[str] = "NOSIBLE/financial-sentiment"
DEFAULT_RAW_ROOT: Final[Path] = Path("data/raw")
DEFAULT_TEXT_COLUMN: Final[str] = "text"
DEFAULT_LABEL_COLUMN: Final[str] = "label"
DEFAULT_DROP_COLUMNS: Final[tuple[str, ...]] = ("netloc", "url")
DEFAULT_LABEL2ID: Final[dict[str, int]] = {"negative": 0, "neutral": 1, "positive": 2}
DEFAULT_ID2LABEL: Final[dict[int, str]] = {v: k for k, v in DEFAULT_LABEL2ID.items()}
PYARROW_VERSION: Final[str] = pyarrow.__version__
RAW_TRAIN_FILENAME: Final[str] = "train.parquet"
RAW_METADATA_FILENAME: Final[str] = "metadata.json"


@dataclass(frozen=True)
class RawSchema:
    """The expected schema of the raw Hugging Face dataset."""

    text_column: str = DEFAULT_TEXT_COLUMN
    label_column: str = DEFAULT_LABEL_COLUMN
    drop_columns: tuple[str, ...] = DEFAULT_DROP_COLUMNS


@dataclass(frozen=True)
class ProcessedSchema:
    """The schema produced by this project's preprocessing step."""

    text_column: str = DEFAULT_TEXT_COLUMN
    label_id_column: str = "label"
    label_text_column: str = "label_text"
    label2id: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LABEL2ID))


def load_raw_hf_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> DatasetDict:
    """Load the raw dataset from the Hugging Face Hub.

    Args:
        dataset_name: Hugging Face dataset name.
        revision: Optional dataset revision (commit hash or tag). If `None`, the
            latest revision will be used.
        cache_dir: Optional local cache directory for datasets.

    Returns:
        A Hugging Face `DatasetDict` containing the dataset splits.
    """
    ds = load_dataset(dataset_name, revision=revision, cache_dir=str(cache_dir) if cache_dir else None)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected `load_dataset` to return a DatasetDict, got {type(ds)!r}")
    return ds


def validate_raw_schema(dataset: HFDataset, schema: RawSchema) -> None:
    """Validate raw columns and basic types.

    The pipeline expects at minimum a text column and a label column. Additional
    columns are only allowed if they are explicitly listed in `drop_columns`.

    Args:
        dataset: A Hugging Face Dataset representing the raw split.
        schema: The expected raw schema (column names and droppable columns).

    Raises:
        ValueError: If required columns are missing, unexpected columns are
            present, the dataset is empty, or types do not match expectations.
    """
    required_columns = {schema.text_column, schema.label_column}
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}. Available columns: {dataset.column_names}")

    allowed_columns = required_columns | set(schema.drop_columns)
    unexpected_columns = [col for col in dataset.column_names if col not in allowed_columns]
    if unexpected_columns:
        raise ValueError(f"Unexpected raw columns: {unexpected_columns}. Allowed columns: {sorted(allowed_columns)}")

    if len(dataset) == 0:
        raise ValueError("Raw dataset is empty.")

    text_feature = dataset.features[schema.text_column]
    text_dtype = getattr(text_feature, "dtype", None)
    if text_dtype not in {"string", "large_string"}:
        raise ValueError(
            f"Expected `{schema.text_column}` to be a string column, got {dataset.features[schema.text_column]!r}"
        )

    label_feature = dataset.features[schema.label_column]
    if isinstance(label_feature, ClassLabel):
        return

    label_dtype = getattr(label_feature, "dtype", None)
    is_int_label = isinstance(label_dtype, str) and (label_dtype.startswith("int") or label_dtype.startswith("uint"))
    if label_dtype not in {"string", "large_string"} and not is_int_label:
        raise ValueError(
            f"Expected `{schema.label_column}` to be a ClassLabel, an integer column, or a string column, "
            f"got {label_feature!r}"
        )


def clean_and_encode_labels(
    dataset: HFDataset,
    *,
    raw_schema: RawSchema,
    processed_schema: ProcessedSchema,
) -> HFDataset:
    """Drop irrelevant columns, normalize text/labels, and encode labels to integers.

    Cleaning rules are intentionally simple and deterministic:
    - Filter empty/whitespace-only texts.
    - Drop explicitly droppable raw columns.
    - Map labels to integer ids and store both `label` and `label_text`.

    Args:
        dataset: Raw dataset split (e.g., the `train` split).
        raw_schema: Schema describing raw column names.
        processed_schema: Schema describing processed column names and label mapping.

    Returns:
        A cleaned dataset containing only the processed columns.

    Raises:
        ValueError: If a label is outside the supported label set.
    """
    to_remove = [c for c in raw_schema.drop_columns if c in dataset.column_names]
    if to_remove:
        dataset = dataset.remove_columns(to_remove)

    source_label_column = raw_schema.label_column
    if raw_schema.label_column == processed_schema.label_id_column:
        source_label_column = "__raw_label__"
        dataset = dataset.rename_column(raw_schema.label_column, source_label_column)

    label2id = processed_schema.label2id
    id2label = {v: k for k, v in label2id.items()}
    label_feature = dataset.features[source_label_column]
    class_label_names = label_feature.names if isinstance(label_feature, ClassLabel) else None

    def _normalize(example: dict[str, Any]) -> dict[str, Any]:
        text = example.get(raw_schema.text_column)
        label_value = example.get(source_label_column)

        text_norm = "" if text is None else str(text).strip()
        if label_value is None:
            label_norm = ""
        elif isinstance(label_value, str):
            label_norm = label_value.strip().lower()
        else:
            try:
                label_idx = int(label_value)
            except (TypeError, ValueError):
                label_norm = str(label_value).strip().lower()
            else:
                if class_label_names and 0 <= label_idx < len(class_label_names):
                    label_norm = str(class_label_names[label_idx]).strip().lower()
                else:
                    label_norm = id2label.get(label_idx, "")
        label_id = label2id.get(label_norm, -1)

        return {
            raw_schema.text_column: text_norm,
            processed_schema.label_text_column: label_norm,
            processed_schema.label_id_column: label_id,
        }

    dataset = dataset.map(_normalize, remove_columns=[source_label_column], desc="Normalizing text and labels")
    dataset = dataset.filter(
        lambda ex: ex[raw_schema.text_column] != "" and ex[processed_schema.label_id_column] != -1,
        desc="Filtering invalid rows",
    )

    unique_ids = set(dataset.unique(processed_schema.label_id_column))
    expected_ids = set(processed_schema.label2id.values())
    if not unique_ids.issubset(expected_ids):
        raise ValueError(f"Unexpected label ids found after preprocessing: {sorted(unique_ids)}")
    return dataset


def make_train_val_test_splits(
    dataset: HFDataset,
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> DatasetDict:
    """Create deterministic train/val/test splits from a single dataset.

    Args:
        dataset: The dataset to split.
        seed: Random seed for deterministic splitting.
        test_size: Fraction reserved for the test split.
        val_size: Fraction reserved for the validation split (applied after test split).

    Returns:
        A `DatasetDict` with keys `train`, `val`, and `test`.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("`test_size` must be between 0 and 1.")
    if not 0.0 < val_size < 1.0:
        raise ValueError("`val_size` must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("`test_size + val_size` must be < 1.")

    first_split = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    train = first_split["train"]
    test = first_split["test"]

    val_fraction_of_train = val_size / (1.0 - test_size)
    second_split = train.train_test_split(test_size=val_fraction_of_train, seed=seed, shuffle=True)
    return DatasetDict(train=second_split["train"], val=second_split["test"], test=test)


def _select_n(dataset: HFDataset, n: int, *, seed: int) -> HFDataset:
    if n <= 0:
        raise ValueError("Subset size must be > 0.")
    if n >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(list(range(n)))


def write_processed_splits(
    splits: DatasetDict,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Write the processed splits to parquet with a metadata sidecar.

    Args:
        splits: A `DatasetDict` containing `train`, `val`, and `test`.
        output_dir: Output directory for the tier.
        metadata: JSON-serializable metadata payload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise ValueError(f"Missing split `{split_name}`. Available: {list(splits.keys())}")

    for split_name, split_ds in splits.items():
        out_path = output_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_fingerprint(dataset: HFDataset) -> str | None:
    fingerprint = getattr(dataset, "fingerprint", None)
    if fingerprint is not None:
        return str(fingerprint)
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None:
        return str(fingerprint)
    return None


def _raw_metadata_matches(meta: dict[str, Any], *, dataset_name: str, revision: str | None) -> bool:
    if meta.get("dataset_name") != dataset_name:
        return False
    if meta.get("revision") != revision:
        return False
    return True


def _load_local_raw_train(raw_root: Path) -> tuple[HFDataset, dict[str, Any]]:
    meta_path = raw_root / RAW_METADATA_FILENAME
    train_path = raw_root / RAW_TRAIN_FILENAME

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    ds = load_dataset("parquet", data_files={"train": str(train_path)})
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict from parquet loader, got {type(ds)!r}")
    return ds["train"], metadata


def _write_local_raw_train(
    raw_train: HFDataset,
    *,
    raw_root: Path,
    dataset_name: str,
    revision: str | None,
    raw_schema: RawSchema,
) -> dict[str, Any]:
    raw_root.mkdir(parents=True, exist_ok=True)
    train_path = raw_root / RAW_TRAIN_FILENAME
    meta_path = raw_root / RAW_METADATA_FILENAME

    raw_train.to_parquet(str(train_path))

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "revision": revision,
        "num_rows": len(raw_train),
        "columns": list(raw_train.column_names),
        "raw_schema": asdict(raw_schema),
        "pyarrow_version": PYARROW_VERSION,
        "datasets_version": hf_datasets.__version__,
        "fingerprint": _dataset_fingerprint(raw_train),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def load_or_prepare_raw_train(
    dataset_name: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    raw_root: Path,
    raw_schema: RawSchema,
    refresh_raw: bool,
) -> tuple[HFDataset, dict[str, Any], bool]:
    """Load raw train data from `data/raw/` if it matches, otherwise download and persist it.

    Raw reuse is only allowed when the local metadata matches the requested
    `dataset_name` and `revision` (including `None`).

    Args:
        dataset_name: Hugging Face dataset name.
        revision: Optional dataset revision (commit hash or tag).
        cache_dir: Optional Hugging Face cache directory.
        raw_root: Directory for raw artifacts.
        raw_schema: Expected schema for the raw dataset.
        refresh_raw: If True, forces a re-download and overwrites local raw artifacts.

    Returns:
        A tuple of `(raw_train_dataset, raw_metadata, reused)` where `reused` is
        True when local raw artifacts were reused.
    """
    meta_path = raw_root / RAW_METADATA_FILENAME
    train_path = raw_root / RAW_TRAIN_FILENAME

    if not refresh_raw and meta_path.exists() and train_path.exists():
        try:
            raw_train, metadata = _load_local_raw_train(raw_root)
        except Exception as exc:
            logger.info("Found local raw artifacts but failed to load them: %s. Re-downloading raw.", exc)
        else:
            if _raw_metadata_matches(metadata, dataset_name=dataset_name, revision=revision):
                logger.info("Reusing raw artifacts from %s", raw_root)
                return raw_train, metadata, True
            logger.info("Found local raw artifacts but metadata does not match. Re-downloading raw.")

    logger.info("Downloading raw dataset from Hugging Face: %s (revision=%s)", dataset_name, revision)
    ds = load_raw_hf_dataset(dataset_name, revision=revision, cache_dir=cache_dir)
    if "train" not in ds:
        raise ValueError(f"Expected a `train` split in the raw dataset. Available splits: {list(ds.keys())}")

    raw_train = ds["train"]
    validate_raw_schema(raw_train, raw_schema)

    logger.info("Writing raw artifacts to %s", raw_root)
    metadata = _write_local_raw_train(
        raw_train,
        raw_root=raw_root,
        dataset_name=dataset_name,
        revision=revision,
        raw_schema=raw_schema,
    )
    return raw_train, metadata, False


def build_processed_data(
    *,
    dataset_name: str = typer.Option(DEFAULT_DATASET_NAME, help="Hugging Face dataset name."),
    revision: str | None = typer.Option(None, help="Optional dataset revision (commit hash or tag)."),
    cache_dir: Path | None = typer.Option(None, help="Optional Hugging Face cache directory."),
    raw_root: Path = typer.Option(DEFAULT_RAW_ROOT, help="Directory for raw cached artifacts (parquet + metadata)."),
    refresh_raw: bool = typer.Option(False, "--refresh-raw", help="Force re-download and overwrite raw artifacts."),
    output_root: Path = typer.Option(Path("data/processed"), help="Where to write processed datasets."),
    seed: int = typer.Option(42, help="Random seed used for deterministic sampling/splitting."),
    test_size: float = typer.Option(0.1, help="Fraction used for the test split."),
    val_size: float = typer.Option(0.1, help="Fraction used for the validation split."),
    small_size: int = typer.Option(2_000, help="Number of examples for the `small` tier."),
    dev_size: int = typer.Option(10_000, help="Number of examples for the `dev` tier."),
) -> None:
    """Download, validate, preprocess, and persist `small`/`dev`/`full` datasets.

    This is the main CLI entrypoint (`python src/sns_mlops/data.py ...`).

    Writes:
    - `data/raw/train.parquet` + `data/raw/metadata.json` (reused when possible)
    - `data/processed/<tier>/{train,val,test}.parquet` + `metadata.json`
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raw_schema = RawSchema()
    processed_schema = ProcessedSchema(label2id=dict(DEFAULT_LABEL2ID))

    raw_train, raw_metadata, raw_reused = load_or_prepare_raw_train(
        dataset_name,
        revision=revision,
        cache_dir=cache_dir,
        raw_root=raw_root,
        raw_schema=raw_schema,
        refresh_raw=refresh_raw,
    )
    dropped_raw_columns = [c for c in raw_schema.drop_columns if c in raw_train.column_names]
    cleaned = clean_and_encode_labels(raw_train, raw_schema=raw_schema, processed_schema=processed_schema)

    tiers: list[tuple[str, HFDataset, int | None]] = [
        ("small", _select_n(cleaned, small_size, seed=seed), small_size),
        ("dev", _select_n(cleaned, dev_size, seed=seed), dev_size),
        ("full", cleaned, None),
    ]

    created_at = datetime.now(timezone.utc).isoformat()
    for tier_name, tier_ds, tier_n in tiers:
        tier_out = output_root / tier_name
        if tier_out.exists():
            shutil.rmtree(tier_out)

        splits = make_train_val_test_splits(tier_ds, seed=seed, test_size=test_size, val_size=val_size)
        metadata = {
            "created_at_utc": created_at,
            "dataset_name": dataset_name,
            "revision": revision,
            "seed": seed,
            "tier": tier_name,
            "requested_size": tier_n,
            "num_rows": {k: len(v) for k, v in splits.items()},
            "raw_schema": asdict(raw_schema),
            "processed_schema": asdict(processed_schema),
            "dropped_raw_columns": dropped_raw_columns,
            "label2id": dict(DEFAULT_LABEL2ID),
            "id2label": {str(k): v for k, v in DEFAULT_ID2LABEL.items()},
            "pyarrow_version": PYARROW_VERSION,
            "raw_artifact": {
                "root": str(raw_root),
                "train_parquet": str(raw_root / RAW_TRAIN_FILENAME),
                "metadata_json": str(raw_root / RAW_METADATA_FILENAME),
                "reused": raw_reused,
                "metadata": raw_metadata,
            },
            "fingerprints": {
                "raw_train": _dataset_fingerprint(raw_train),
                "cleaned": _dataset_fingerprint(cleaned),
                "tier_dataset": _dataset_fingerprint(tier_ds),
                "splits": {k: _dataset_fingerprint(v) for k, v in splits.items()},
            },
        }
        write_processed_splits(splits, output_dir=tier_out, metadata=metadata)
        logger.info("Wrote processed tier `%s` to %s", tier_name, tier_out)


def main() -> None:
    typer.run(build_processed_data)


if __name__ == "__main__":
    main()
