FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first to maximize Docker layer caching.
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN python -m pip install -U pip setuptools wheel && \
    pip wheel --wheel-dir /wheels \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt

# Copy source code last so changes do not invalidate dependency layers.
COPY src/ src/
RUN pip wheel --wheel-dir /wheels . --no-deps


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY src/ src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

ENTRYPOINT ["python", "-u", "-m", "sns_mlops.train"]
