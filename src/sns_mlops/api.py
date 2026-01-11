"""FastAPI application entrypoint.

This module is intentionally minimal in the current project stage. The repository
focuses on data ingestion, reproducible training, and engineering hygiene first.

Planned next step:
- Add a small inference API (e.g., `/predict`) that loads the locally saved model
  from `models/finbert/<tier>/model/` when available, otherwise downloads the base
  model from the Hugging Face Hub at runtime.
"""
