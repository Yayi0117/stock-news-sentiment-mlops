from __future__ import annotations

from pathlib import Path

import pytest
from transformers import BertTokenizerFast

import sns_mlops.model as m


def test_get_label_mappings_returns_copies():
    label2id, id2label = m.get_label_mappings()
    assert label2id == {"negative": 0, "neutral": 1, "positive": 2}
    assert id2label == {0: "negative", 1: "neutral", 2: "positive"}

    label2id["negative"] = 999
    assert m.DEFAULT_LABEL2ID["negative"] == 0


def test_build_model_config_passes_label_mapping(monkeypatch):
    captured = {}

    def fake_from_pretrained(
        model_name,
        *,
        revision=None,
        cache_dir=None,
        num_labels=None,
        label2id=None,
        id2label=None,
        **kwargs,
    ):
        captured.update(
            {
                "model_name": model_name,
                "revision": revision,
                "cache_dir": cache_dir,
                "num_labels": num_labels,
                "label2id": label2id,
                "id2label": id2label,
            }
        )

        class DummyConfig:
            pass

        cfg = DummyConfig()
        cfg.num_labels = num_labels
        cfg.label2id = label2id
        cfg.id2label = id2label
        return cfg

    monkeypatch.setattr(m.AutoConfig, "from_pretrained", fake_from_pretrained)

    cfg = m.build_model_config("dummy/model", revision="abc123", cache_dir=Path("hf_cache"))
    assert cfg.num_labels == 3
    assert captured["model_name"] == "dummy/model"
    assert captured["revision"] == "abc123"
    assert captured["cache_dir"] == "hf_cache"
    assert captured["label2id"] == m.DEFAULT_LABEL2ID
    assert captured["id2label"] == m.DEFAULT_ID2LABEL


def test_build_pretrained_model_uses_config(monkeypatch):
    captured = {}

    class DummyConfig:
        pass

    dummy_config = DummyConfig()

    def fake_from_pretrained(model_name, *args, revision=None, cache_dir=None, config=None, **kwargs):
        captured.update({"model_name": model_name, "revision": revision, "cache_dir": cache_dir, "config": config})
        return object()

    monkeypatch.setattr(m.AutoModelForSequenceClassification, "from_pretrained", fake_from_pretrained)

    _ = m.build_pretrained_model("dummy/model", revision="rev", cache_dir=Path("cache"), config=dummy_config)
    assert captured["model_name"] == "dummy/model"
    assert captured["revision"] == "rev"
    assert captured["cache_dir"] == "cache"
    assert captured["config"] is dummy_config


@pytest.mark.parametrize("max_length", [4, 8])
def test_tokenize_batch_local_tokenizer(tmp_path, max_length):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "finbert"]),
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    batch = m.tokenize_batch(tokenizer, ["hello world", "finbert"], max_length=max_length)

    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == 2
