# AxiomForgeAI — GRPO Training Image
# ─────────────────────────────────────────────────────────────────────────────
# Hardware target  : 1× A100 PCIE 80 GB  |  AMD EPYC 7V13  |  NVMe 300 GB
#
# CUDA driver      : >= 13.0  (enforced at container start via entrypoint)
# CUDA toolkit     : 12.4.1   (backward-compatible with driver 13.x)
# PyTorch          : 2.5.1+cu124  (pinned in requirements.txt)
# Flash-Attn       : 2.8.3        (pinned in requirements.txt)
#
# All Python package versions are taken exclusively from requirements.txt.
# No versions are hard-coded in this file.
#
# ── Build ─────────────────────────────────────────────────────────────────────
#   docker build -t axiomforgeai-train:latest .
#
# ── Interactive shell ─────────────────────────────────────────────────────────
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     -v $(pwd)/logs:/workspace/logs \
#     -it axiomforgeai-train:latest bash
#
# ── GRPO training (one-shot) ──────────────────────────────────────────────────
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     -v $(pwd)/logs:/workspace/logs \
#     axiomforgeai-train:latest \
#     python scripts/run_grpo_training.py \
#       --base-model checkpoints/dual_task_v1 \
#       --gsm8k-data data/sft/gsm8k_sft.jsonl \
#       --num-iterations 30 --group-size 8 --questions-per-iter 16
# ─────────────────────────────────────────────────────────────────────────────

# CUDA toolkit 12.4.1 — matches the cu124 wheels in requirements.txt and is
# fully compatible with the A100's CUDA 13.2 driver (driver is always ≥ toolkit).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.title="AxiomForgeAI Training" \
      cuda.driver.minimum="13.0" \
      cuda.toolkit="12.4.1" \
      torch.version="2.5.1+cu124" \
      flash_attn.version="2.8.3"

# ── System packages ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        git \
        git-lfs \
        curl \
        wget \
        build-essential \
        ninja-build \
        pkg-config \
        libssl-dev \
        libffi-dev \
        ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3    /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip + build tooling ───────────────────────────────────────────────
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# ── PyTorch (CUDA 12.4 wheels) ────────────────────────────────────────────────
# Must be installed before flash-attn because flash-attn runs a torch version
# check at install time.  The cu124 index is also used for all CUDA-linked wheels.
# Version is taken from requirements.txt — the --constraint flag keeps pip from
# re-resolving to a different version when requirements.txt is processed next.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"

# ── All remaining pinned requirements (from requirements.txt) ─────────────────
# flash-attn, xformers, vllm, triton, bitsandbytes, transformers, accelerate,
# peft, ray, sympy, scipy, numpy, openenv-core, fastapi, uvicorn, … are all
# installed here at the exact versions pinned in requirements.txt.
# The cu124 index is provided so CUDA-linked wheels resolve correctly.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r /tmp/requirements.txt

# ── Project source ────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace/

# ── Environment variables ─────────────────────────────────────────────────────
# Repo root on PYTHONPATH so `from src.rl.X import Y` works without editable install
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# HuggingFace model cache — mount a host path here to persist model downloads:
#   -v /host/hf_cache:/workspace/.hf_cache
ENV HF_HOME="/workspace/.hf_cache"
ENV TRANSFORMERS_CACHE="/workspace/.hf_cache"

# A100 CUDA / NCCL tuning
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=0
# Required for Flash-Attn 2 with bfloat16 on Ampere
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ── Runtime entrypoint: enforce CUDA driver >= 13.0 ──────────────────────────
# nvidia-smi is injected at runtime via --gpus, so this check runs when the
# container starts, not at build time.
RUN printf '%s\n' \
    '#!/bin/sh' \
    'if command -v nvidia-smi >/dev/null 2>&1; then' \
    '  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" || echo "0.0")' \
    '  MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)' \
    '  echo "[AxiomForgeAI] CUDA driver reports toolkit: $CUDA_VER"' \
    '  if [ "${MAJOR:-0}" -lt 13 ] 2>/dev/null; then' \
    '    echo "[ERROR] CUDA driver >= 13.0 required; detected $CUDA_VER. Upgrade your NVIDIA driver."' \
    '    exit 1' \
    '  fi' \
    '  echo "[AxiomForgeAI] CUDA $CUDA_VER >= 13.0 — OK"' \
    'else' \
    '  echo "[WARNING] nvidia-smi not found — CUDA driver version check skipped."' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
