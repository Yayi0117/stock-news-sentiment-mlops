# sns_mlops

DTU MLOps course project focusing on engineering quality: reproducible pipelines,
data/model versioning, automated tests + CI, containerization, and documentation.

## Project overview

- Task: sentiment classification of financial news text.
- Dataset: Hugging Face `NOSIBLE/financial-sentiment`.
- Model: `ProsusAI/finbert` fine-tuned with the Transformers `Trainer` API.

## Quick start (Conda, recommended)

```bash
conda env create -f environment.yml
conda activate py312
pip install -e .
```

Windows PowerShell (equivalent):

```powershell
conda env create -f environment.yml
conda activate py312
pip install -e .
```

### Build processed data (DVC)

```bash
dvc repro data
```

Windows PowerShell (equivalent):

```powershell
dvc repro data
```

Expected outputs:
- `data/processed/small/{train,val,test}.parquet` + `metadata.json`
- `data/processed/dev/{train,val,test}.parquet` + `metadata.json`
- `data/processed/full/{train,val,test}.parquet` + `metadata.json`

Raw caching:
- The first run writes `data/raw/train.parquet` and `data/raw/metadata.json`.
- Subsequent runs reuse `data/raw/` if `dataset_name` and `revision` match.

### Training smoke run (CPU)

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

Windows PowerShell (equivalent):

```powershell
python -m sns_mlops.train `
  --tier small `
  --processed-root data/processed `
  --output-dir models/finbert `
  --seed 42 `
  --num-train-epochs 1 `
  --max-length 64 `
  --per-device-train-batch-size 4 `
  --per-device-eval-batch-size 8 `
  --max-train-samples 64 `
  --max-eval-samples 64 `
  --max-test-samples 64 `
  --no-save-checkpoints
```

Expected outputs:
- `models/finbert/small/run_config.json`
- `models/finbert/small/metrics.json`
- `models/finbert/small/train.log`

Optional output:
- `models/finbert/small/model/` (final model + tokenizer; ignored by Git and not tracked by DVC)

## DVC pipeline

Stages are defined in `dvc.yaml`:
- `data`: materializes processed tiers under `data/processed/`
- `train`: runs a short `small` tier training run and writes run artifacts

Reproduce everything:

```bash
dvc repro
```

Tracking policy:
- `train` stage tracks `run_config.json`, `metrics.json`, and `train.log` as DVC outputs.
- Model weights are saved locally but not tracked by DVC to avoid large artifacts.

## Tests, linting, and coverage

```bash
pytest -q
ruff check .
ruff format --check .
coverage run --source=sns_mlops -m pytest -q
coverage report -m
```

Pre-commit (recommended):

```bash
pre-commit install
pre-commit run --all-files
```

## Continuous integration (GitHub Actions)

Workflows live under `.github/workflows/`:
- `tests.yaml`: multi-OS (Ubuntu/Windows/macOS) and multi-Python (3.11/3.12) tests + coverage artifact (`coverage.xml`)
- `linting.yaml`: `ruff check .` and `ruff format --check .`

To prevent accidental Hugging Face downloads during CI, the tests job sets:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

## Docker (training image)

Build:

```bash
docker build -f dockerfiles/train.dockerfile -t sns-mlops-train .
```

Run (requires processed data on the host):

```bash
docker run --rm \
  -v ./data/processed:/app/data/processed:ro \
  -v ./models:/app/models \
  sns-mlops-train \
  --tier small --num-train-epochs 1 \
  --max-train-samples 16 --max-eval-samples 16 --max-test-samples 16 \
  --per-device-train-batch-size 4 --per-device-eval-batch-size 4 \
  --no-save-checkpoints --no-save-model
```

Windows PowerShell (equivalent):

```powershell
docker run --rm `
  -v "${PWD}\data\processed:/app/data/processed:ro" `
  -v "${PWD}\models:/app/models" `
  sns-mlops-train `
  --tier small --num-train-epochs 1 `
  --max-train-samples 16 --max-eval-samples 16 --max-test-samples 16 `
  --per-device-train-batch-size 4 --per-device-eval-batch-size 4 `
  --no-save-checkpoints --no-save-model
```

Optional: mount Hugging Face cache to reuse downloaded weights:

```bash
docker run --rm \
  -v ./data/processed:/app/data/processed:ro \
  -v ./models:/app/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  sns-mlops-train ...
```

## Documentation (MkDocs)

The documentation sources live under `docs/source/` and are built with
`docs/mkdocs.yaml`.

```bash
mkdocs serve --config-file docs/mkdocs.yaml
mkdocs build --config-file docs/mkdocs.yaml
```
