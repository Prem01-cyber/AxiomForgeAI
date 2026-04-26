# AxiomForgeAI — GRPO Training Image
# Target hardware: 1× A100 PCIE 80 GB | AMD EPYC 7V13 | CUDA driver ≤ 13.2
#
# Uses PyTorch 2.5.1 + CUDA 12.4 which is within the driver's CUDA 13.2 ceiling.
# Flash Attention 2 is built for Ampere (A100 = sm_80) and provides O(T) attention
# memory instead of O(T²), critical for long math solution sequences.
#
# Build:
#   docker build -t axiomforgeai-train:latest .
#
# Run (interactive training):
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     -v $(pwd)/logs:/workspace/logs \
#     -it axiomforgeai-train:latest bash
#
# Run GRPO training directly:
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     -v $(pwd)/logs:/workspace/logs \
#     axiomforgeai-train:latest \
#     python scripts/run_grpo_training.py \
#       --base-model checkpoints/dual_task_v1 \
#       --gsm8k-data data/sft/gsm8k_sft.jsonl \
#       --num-iterations 30 --group-size 4 --questions-per-iter 16

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS base

# ── System packages ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies — split into layers for cache reuse ───────────────

# Layer 1: core ML stack (changes rarely)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Layer 2: HuggingFace stack + bitsandbytes for 4-bit quantisation
RUN pip install --no-cache-dir \
    transformers>=4.47.0 \
    accelerate>=1.2.1 \
    peft>=0.14.0 \
    datasets>=3.0.0 \
    tokenizers>=0.21.0 \
    safetensors>=0.4.5 \
    bitsandbytes>=0.43.0 \
    huggingface-hub>=0.26.0

# Layer 3: Flash Attention 2 (Ampere / A100 = sm_80)
# Pre-built wheel matching torch 2.5.1 + CUDA 12.4; avoids 20-min compilation.
RUN pip install --no-cache-dir \
    flash-attn==2.7.2.post1 \
    --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.7.2.post1 \
    || pip install --no-cache-dir flash-attn \
         --no-build-isolation \
         --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/

# Layer 4: math / science / training utilities
RUN pip install --no-cache-dir \
    sympy>=1.12 \
    numpy>=1.26 \
    scipy>=1.12 \
    tqdm>=4.66 \
    einops>=0.7.0 \
    packaging>=24.0

# Layer 5: experiment tracking + server deps
RUN pip install --no-cache-dir \
    openenv-core[core]>=0.2.2 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0 \
    cloudpickle>=3.0.0 \
    ray[default]>=2.38.0 \
    wandb>=0.18.0 \
    matplotlib>=3.9.0 \
    pandas>=2.2.0

# Layer 6: remaining requirements from the repo's pinned requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    || true  # non-critical: handles any version conflicts gracefully

# ── Copy project source ────────────────────────────────────────────────────
COPY . /workspace/

# Ensure the repo root is on PYTHONPATH so `from src.rl.X import Y` works.
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# HuggingFace model cache — mount a host directory here to avoid re-downloading
# large models on every container start.
ENV HF_HOME="/workspace/.hf_cache"
ENV TRANSFORMERS_CACHE="/workspace/.hf_cache"

# CUDA tuning for A100
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=0

# ── Verify GPU is visible at build time (will be skipped in --no-gpu builds) ──
# RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found'"

CMD ["bash"]
