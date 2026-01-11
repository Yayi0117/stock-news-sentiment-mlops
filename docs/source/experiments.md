# Experiments

This project uses lightweight command-line arguments and DVC stages to run reproducible experiments.

## Smoke runs (fast)

Smoke runs are intended to validate the end-to-end training loop quickly on CPU.

```bash
python -m sns_mlops.train \
  --tier small \
  --num-train-epochs 1 \
  --max-train-samples 64 \
  --max-eval-samples 64 \
  --max-test-samples 64 \
  --no-save-checkpoints
```

Expected outputs:
- `models/finbert/small/run_config.json`
- `models/finbert/small/metrics.json`
- `models/finbert/small/train.log`

## Full runs (slower)

Full runs use the complete `full` tier and do not cap samples:

```bash
python -m sns_mlops.train \
  --tier full \
  --num-train-epochs 1 \
  --no-save-checkpoints
```

Notes:
- The first run downloads the model weights from the Hugging Face Hub.
- Increase `--num-train-epochs` and tune `--learning-rate` only after the pipeline is stable.

## Reproducing runs via DVC

The repository defines two stages in `dvc.yaml`:
- `data`: builds processed tiers under `data/processed/`
- `train`: runs a short `small` tier training and writes run artifacts

Reproduce both:

```bash
dvc repro
```

Reproduce only training (will rebuild data if needed):

```bash
dvc repro train
```

## Reproducibility measures

- Pinned dependencies: `requirements.txt`, `requirements_dev.txt`, `environment.yml`.
- Deterministic dataset splits: fixed seed and explicit split fractions.
- Dataset provenance: `metadata.json` in each tier records dataset source, schema, and fingerprints.
- Run provenance: `run_config.json` records training arguments and package versions.
- Pipeline provenance: `dvc.lock` records commands, code deps, and output hashes.
