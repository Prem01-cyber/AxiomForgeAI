
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# AxiomForgeAI OpenEnv server image
# ─────────────────────────────────────────────────────────────────────────────
# Hardware target  : A100 PCIE 80 GB  |  AMD EPYC 7V13
# CUDA driver      : >= 13.0  (enforced at container start)
# CUDA toolkit     : 12.4.1   (backward-compatible with driver 13.x)
# PyTorch          : 2.5.1+cu124  (pinned in /requirements.txt)
#
# The server exposes the math RL environment over HTTP/WebSocket and supports
# optional GPU-accelerated PRM scoring when AXIOMFORGE_PRM_PATH is set.
#
# ── Build ────────────────────────────────────────────────────────────────────
#   docker build -f server/Dockerfile -t axiomforgeai-server:latest .
#
# ── Run (CPU-only / validation) ───────────────────────────────────────────────
#   docker run -p 8000:8000 axiomforgeai-server:latest
#
# ── Run (GPU + grounded data + PRM) ──────────────────────────────────────────
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

# Install openenv-core + server deps (pyproject.toml / server/requirements.txt)
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

# ── ML stack for optional GPU-based PRM scoring ───────────────────────────────
# All versions are taken from the root requirements.txt so they stay in sync
# with the training image.  The cu124 extra-index is needed to resolve the
# correct CUDA-linked torch wheel.
COPY requirements.txt /tmp/ml-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r /tmp/ml-requirements.txt \
    || true   # non-fatal: server remains fully functional without the ML stack

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env      /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# HuggingFace model cache — mount a host path here to avoid re-downloading:
#   -v /host/hf_cache:/app/hf_cache
ENV HF_HOME="/app/hf_cache"
ENV TRANSFORMERS_CACHE="/app/hf_cache"

# A100 CUDA tuning (only effective when --gpus is passed)
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ── Runtime CUDA driver check (>= 13.0) ──────────────────────────────────────
RUN printf '%s\n' \
    '#!/bin/sh' \
    'if command -v nvidia-smi >/dev/null 2>&1; then' \
    '  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" || echo "0.0")' \
    '  MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)' \
    '  echo "[AxiomForgeAI-server] CUDA driver reports toolkit: $CUDA_VER"' \
    '  if [ "${MAJOR:-0}" -lt 13 ] 2>/dev/null; then' \
    '    echo "[ERROR] CUDA driver >= 13.0 required; detected $CUDA_VER. Upgrade your NVIDIA driver."' \
    '    exit 1' \
    '  fi' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
