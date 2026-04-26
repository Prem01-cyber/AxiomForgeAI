
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# AxiomForgeAI OpenEnv server image
# Target: A100 PCIE 80 GB  |  Driver CUDA ≤ 13.2  |  PyTorch CUDA 12.4
#
# The server exposes the math RL environment over HTTP/WebSocket and supports
# optional GPU-accelerated PRM scoring when AXIOMFORGE_PRM_PATH is set.
#
# Build:
#   docker build -f server/Dockerfile -t axiomforgeai-server:latest .
#
# Run (CPU-only / validation):
#   docker run -p 8000:8000 axiomforgeai-server:latest
#
# Run (GPU + grounded data + PRM):
#   docker run --gpus all \
#     -e AXIOMFORGE_DATA_PATH=/data/gsm8k_sft.jsonl \
#     -e AXIOMFORGE_PRM_PATH=Qwen/Qwen2.5-Math-PRM-7B \
#     -v /host/data:/data \
#     -p 8000:8000 \
#     axiomforgeai-server:latest

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is required for VCS-based dependency installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=AxiomForgeAI

COPY . /app/env
WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install openenv-core + server deps (pyproject.toml)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Install the full ML training stack so the server can do GPU-based PRM scoring.
# PyTorch 2.5.1 + CUDA 12.4 wheels — compatible with the A100 PCIE driver
# (max CUDA 13.2) and with flash-attention-2 for BF16 attention.
RUN --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/pip install --no-deps --upgrade \
        torch==2.5.1 \
        torchvision \
        --extra-index-url https://download.pytorch.org/whl/cu124

RUN --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/pip install --no-cache-dir \
        transformers>=4.47.0 \
        accelerate>=1.2.1 \
        bitsandbytes>=0.43.0 \
        sympy>=1.12 \
        numpy>=1.26 \
        tqdm

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Optional: point at pre-downloaded HF model cache to avoid download on start
ENV HF_HOME="/app/hf_cache"
ENV TRANSFORMERS_CACHE="/app/hf_cache"

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
