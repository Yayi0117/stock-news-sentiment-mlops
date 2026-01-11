# Pipeline

This project is organized around two reproducible stages: `data` and `train`.

## Data stage

Entry point:
- CLI: `python src/sns_mlops/data.py --help`
- DVC stage: `dvc repro data`

Inputs:
- Hugging Face dataset: `NOSIBLE/financial-sentiment`

Processing steps:
- Validate raw schema (required columns, allowed columns, label type compatibility).
- Normalize text and labels.
- Drop explicitly droppable raw columns (e.g. `netloc`, `url`).
- Create deterministic `train`/`val`/`test` splits using a fixed seed.
- Materialize Parquet splits and a `metadata.json` sidecar.

Raw caching:
- A reproducible raw training artifact is written to `data/raw/train.parquet` plus `data/raw/metadata.json`.
- The pipeline reuses the raw artifact when `dataset_name` and `revision` match.

Outputs:
- `data/processed/<tier>/{train,val,test}.parquet`
- `data/processed/<tier>/metadata.json`

Tiers:
- `small`: intended for local development and CI smoke runs
- `dev`: intended for faster iteration with more data
- `full`: the complete cleaned dataset

## Train stage

Entry point:
- CLI: `python -m sns_mlops.train --help`
- DVC stage: `dvc repro train` (configured for a short `small` tier run)

Processing steps:
- Load processed Parquet splits using the `datasets` parquet loader.
- Tokenize with the FinBERT tokenizer.
- Fine-tune using the Hugging Face Transformers `Trainer`.
- Compute metrics with scikit-learn: `accuracy` and `macro-F1`.

Outputs (always written):
- `models/finbert/<tier>/run_config.json`: configuration snapshot, package versions, and git commit (best effort).
- `models/finbert/<tier>/metrics.json`: train/val/test metrics.
- `models/finbert/<tier>/train.log`: logs for the run.

Optional output:
- `models/finbert/<tier>/model/`: final model weights and tokenizer files (saved locally but not tracked by DVC).

## DVC tracking policy

The `train` stage intentionally tracks only small, high-signal artifacts as DVC outputs:
- `run_config.json`
- `metrics.json`
- `train.log`

Model weights are saved locally (for inference/deployment later) but are not tracked by DVC to avoid large artifacts.
