FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app:/app/celery

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY common/requirements.txt common/requirements.txt
COPY celery/requirements.txt celery/requirements.txt

FROM base AS cpu
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r celery/requirements.txt
COPY common common
COPY celery/worker celery/worker
CMD ["python", "-m", "worker"]

FROM base AS amd
ARG ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.4
RUN pip install --no-cache-dir torch==2.8.0 --index-url ${ROCM_INDEX_URL} \
    && pip install --no-cache-dir -r celery/requirements.txt
COPY common common
COPY celery/worker celery/worker
CMD ["python", "-m", "worker"]
