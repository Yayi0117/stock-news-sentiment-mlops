FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc python3.12 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN python -m pip install -U pip setuptools wheel && \
    pip wheel --wheel-dir /wheels -r requirements.txt
COPY src/ src/
RUN pip wheel --wheel-dir /wheels . --no-deps


FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY src/ src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

ENTRYPOINT ["uvicorn", "src.sns_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
