FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc python3.12 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["uvicorn", "src.sns_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
