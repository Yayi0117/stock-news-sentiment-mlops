"""Evaluation utilities (placeholder).

In this project, evaluation metrics are currently produced during training via the
Hugging Face Trainer `compute_metrics` callback and stored in `metrics.json` under
`models/finbert/<tier>/`.

If the project later needs a separate evaluation script, it should:
- Load processed data from `data/processed/<tier>/`.
- Load the saved model from `models/finbert/<tier>/model/` (not tracked by DVC).
- Write evaluation results to a dedicated metrics artifact (tracked by DVC).
"""
