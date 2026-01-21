FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml


RUN python -m pip install -U pip setuptools wheel && \
    pip wheel --wheel-dir /wheels \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY src/ src/
RUN pip wheel --wheel-dir /wheels . --no-deps


FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


RUN apt-get update && \
    apt-get install --no-install-recommends -y libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY src/ src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml


ENTRYPOINT ["uvicorn", "sns_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]