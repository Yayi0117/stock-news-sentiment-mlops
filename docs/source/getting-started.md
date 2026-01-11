# Getting Started

This page describes the minimal setup to reproduce data and run a short (CPU) training smoke run.

## Prerequisites

- Conda (recommended for local development) or Python 3.11+ with pip
- Git

## Environment setup (Conda, recommended)

```bash
conda env create -f environment.yml
conda activate py312
pip install -e .
```

## Environment setup (pip only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .
```

## Build processed data

The repository uses DVC to reproduce the data pipeline:

```bash
dvc repro data
```

Expected outputs:
- `data/processed/small/{train,val,test}.parquet` + `metadata.json`
- `data/processed/dev/{train,val,test}.parquet` + `metadata.json`
- `data/processed/full/{train,val,test}.parquet` + `metadata.json`

Raw caching:
- The first run downloads the dataset from the Hugging Face Hub and writes a reusable raw artifact under `data/raw/`.
- Subsequent runs reuse `data/raw/` when `dataset_name` and `revision` match.

## Training smoke run (CPU)

Run a short training run on the `small` tier with sample caps:

```bash
python -m sns_mlops.train \
  --tier small \
  --processed-root data/processed \
  --output-dir models/finbert \
  --seed 42 \
  --num-train-epochs 1 \
  --max-length 64 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 8 \
  --max-train-samples 64 \
  --max-eval-samples 64 \
  --max-test-samples 64 \
  --no-save-checkpoints
```

Expected outputs:
- `models/finbert/small/run_config.json`
- `models/finbert/small/metrics.json`
- `models/finbert/small/train.log`

Optional output:
- `models/finbert/small/model/` (final model + tokenizer; not tracked by DVC and ignored by Git)

## Run tests and quality checks

```bash
pytest -q
ruff check .
ruff format --check .
pre-commit run --all-files
```

## Build documentation

```bash
mkdocs build --config-file docs/mkdocs.yaml
```
