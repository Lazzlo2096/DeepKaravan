FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app:/app/celery \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock .python-version ./

FROM base AS cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra cpu
COPY common common
COPY celery/worker celery/worker
CMD ["uv", "run", "--no-sync", "python", "-m", "worker"]

FROM base AS amd
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra amd
COPY common common
COPY celery/worker celery/worker
CMD ["uv", "run", "--no-sync", "python", "-m", "worker"]
