FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock .python-version ./
EXPOSE 8000
CMD ["python", "-m", "api"]

FROM base AS cpu
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra api --extra cpu
COPY common common
COPY api api
COPY gpu_worker gpu_worker
COPY scripts/check_amd.py scripts/check_amd.py

FROM base AS amd
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra api --extra amd
COPY common common
COPY api api
COPY gpu_worker gpu_worker
COPY scripts/check_amd.py scripts/check_amd.py

FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191 AS amd-wsl
USER root
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv
ADD --checksum=sha256:28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541 https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb /tmp/rocdxg.deb
RUN dpkg -i /tmp/rocdxg.deb && ldconfig && rm /tmp/rocdxg.deb
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=/app HSA_ENABLE_DXG_DETECTION=1 \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm/lib
WORKDIR /app
COPY requirements/amd-wsl.lock requirements/amd-wsl.lock
# Torch and Triton are supplied by the pinned AMD image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python python --no-deps -r requirements/amd-wsl.lock
COPY common common
COPY api api
COPY gpu_worker gpu_worker
COPY scripts/check_amd.py scripts/check_amd.py
EXPOSE 8000
CMD ["python", "-m", "api"]

FROM cpu AS default
