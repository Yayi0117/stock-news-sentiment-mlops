import json

import pytest
from datasets import ClassLabel, Features, Value
from datasets import Dataset as HFDataset

from sns_mlops.data import (
    ProcessedSchema,
    RawSchema,
    clean_and_encode_labels,
    load_or_prepare_raw_train,
    make_train_val_test_splits,
    validate_raw_schema,
    write_processed_splits,
)


def test_validate_raw_schema_ok():
    ds = HFDataset.from_dict({"text": ["hello"], "label": ["positive"], "url": ["https://example.com"]})
    validate_raw_schema(ds, RawSchema())


def test_validate_raw_schema_accepts_integer_labels():
    ds = HFDataset.from_dict({"text": ["hello"], "label": [2], "url": ["https://example.com"]})
    validate_raw_schema(ds, RawSchema())

    cleaned = clean_and_encode_labels(ds, raw_schema=RawSchema(), processed_schema=ProcessedSchema())
    assert cleaned[0]["label_text"] == "positive"
    assert cleaned[0]["label"] == 2


def test_validate_raw_schema_accepts_classlabel_labels():
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["negative", "neutral", "positive"]),
            "url": Value("string"),
        }
    )
    ds = HFDataset.from_dict({"text": ["hello"], "label": [0], "url": ["https://example.com"]}, features=features)
    validate_raw_schema(ds, RawSchema())

    cleaned = clean_and_encode_labels(ds, raw_schema=RawSchema(), processed_schema=ProcessedSchema())
    assert cleaned[0]["label_text"] == "negative"
    assert cleaned[0]["label"] == 0


def test_validate_raw_schema_missing_columns_raises():
    ds = HFDataset.from_dict({"text": ["hello"]})
    with pytest.raises(ValueError, match="Missing required raw columns"):
        validate_raw_schema(ds, RawSchema())


def test_clean_and_encode_labels_filters_invalid_rows_and_encodes():
    raw = HFDataset.from_dict(
        {
            "text": [" ok ", "", None, "fine"],
            "label": ["positive", "neutral", "negative", "unknown"],
            "netloc": ["a", "b", "c", "d"],
            "url": ["u1", "u2", "u3", "u4"],
        }
    )
    validate_raw_schema(raw, RawSchema())

    cleaned = clean_and_encode_labels(raw, raw_schema=RawSchema(), processed_schema=ProcessedSchema())
    assert set(cleaned.column_names) == {"text", "label_text", "label"}
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "ok"
    assert cleaned[0]["label_text"] == "positive"
    assert cleaned[0]["label"] == 2


def test_make_train_val_test_splits_partition_all_rows():
    raw = HFDataset.from_dict({"text": [f"t{i}" for i in range(100)], "label": ["positive"] * 100})
    validate_raw_schema(raw, RawSchema())
    cleaned = clean_and_encode_labels(raw, raw_schema=RawSchema(), processed_schema=ProcessedSchema())

    splits = make_train_val_test_splits(cleaned, seed=123, test_size=0.2, val_size=0.1)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["test"]) == 20
    assert len(splits["val"]) == 10
    assert len(splits["train"]) == 70
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(cleaned)


def test_write_processed_splits_writes_parquet_and_metadata(tmp_path):
    raw = HFDataset.from_dict({"text": [f"t{i}" for i in range(50)], "label": ["neutral"] * 50})
    validate_raw_schema(raw, RawSchema())
    cleaned = clean_and_encode_labels(raw, raw_schema=RawSchema(), processed_schema=ProcessedSchema())
    splits = make_train_val_test_splits(cleaned, seed=0, test_size=0.2, val_size=0.1)

    metadata = {"hello": "world"}
    write_processed_splits(splits, output_dir=tmp_path, metadata=metadata)

    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()

    stored = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert stored == metadata


def test_load_or_prepare_raw_train_reuses_matching_raw(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    raw = HFDataset.from_dict(
        {
            "text": ["hello", "world"],
            "label": ["positive", "negative"],
            "netloc": ["example.com", "example.com"],
            "url": ["https://example.com/1", "https://example.com/2"],
        }
    )
    raw.to_parquet(str(raw_root / "train.parquet"))
    (raw_root / "metadata.json").write_text(
        json.dumps({"dataset_name": "dummy", "revision": None, "num_rows": 2}, indent=2),
        encoding="utf-8",
    )

    loaded, meta, reused = load_or_prepare_raw_train(
        "dummy",
        revision=None,
        cache_dir=None,
        raw_root=raw_root,
        raw_schema=RawSchema(),
        refresh_raw=False,
    )
    assert reused is True
    assert meta["dataset_name"] == "dummy"
    assert len(loaded) == 2
