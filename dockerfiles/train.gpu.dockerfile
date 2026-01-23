FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc libgomp1 python3.12 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy dependency definition first to maximize Docker layer caching.
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN python -m pip install -U pip setuptools wheel && \
    pip wheel --wheel-dir /wheels -r requirements.txt

# Copy source code last so changes do not invalidate dependency layers.
COPY src/ src/

RUN pip wheel --wheel-dir /wheels . --no-deps


FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y libgomp1 python3.12 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY src/ src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

ENTRYPOINT ["python", "-u", "-m", "sns_mlops.train"]
