set -euo pipefail

# ── Flash-Attention 2 install (if missing) ────────────────────────────────────
# flash-attn requires (torch version, CUDA version, Python version) alignment.
# MAX_JOBS caps parallel compilation; prebuilt wheel installs in <30 s.
# In the prior run (grpo_20260425_151304), flash-attn was absent → SDPA fallback
# → iter times of 262-330 s once question-gen started (vs ~150 s with Flash).
if ! python -c "import flash_attn; assert int(flash_attn.__version__.split('.')[0]) >= 2" 2>/dev/null; then
    echo "[launch] flash-attn not found or < v2 — installing now …"
    MAX_JOBS=4 pip install flash-attn --no-build-isolation -q
    echo "[launch] flash-attn installed."
else
    FLASH_VER=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    echo "[launch] flash-attn ${FLASH_VER} already installed — skipping install."
fi

# ── GPU / allocator ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# expandable_segments: recovers 2-4 GB fragmented VRAM during long Flash+HF runs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── CPU / threading ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# ── Triton / Flash-Attn compilation cache ─────────────────────────────────────
# Persists JIT kernels across runs — avoids ~30 s recompile each launch.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export FLASH_ATTENTION_SKIP_CUDA_BUILD=${FLASH_ATTENTION_SKIP_CUDA_BUILD:-FALSE}

# ── HuggingFace hub robustness ────────────────────────────────────────────────
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ── Pre-flight: GPU info ───────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "─── nvidia-smi ───────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader || true
    echo "──────────────────────────────────────────────────────────────────"
fi

# ── Confirm attention backend ─────────────────────────────────────────────────
python - <<'PYEOF'
import sys; sys.path.insert(0, '.')
from src.utils.attn_backend import select_attn_implementation
impl = select_attn_implementation()
tag = {
    "flash_attention_2": "FAST   — Flash-Attn 2 active (O(T) memory, ~1.5-2× faster)",
    "sdpa":              "OK     — SDPA active (install flash-attn for ~2× speedup)",
    "eager":             "SLOW   — Eager fallback (install flash-attn for best speed)",
}.get(impl, impl)
print(f"[launch] attn_backend = {tag}")
PYEOF

# ── Log tee ───────────────────────────────────────────────────────────────────
RUN_NAME="grpo_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/grpo"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

echo "[launch] run_name       = $RUN_NAME"
echo "[launch] base_model     = checkpoints/dual_task_v1"
echo "[launch] train_data     = data/sft/gsm8k_sft.jsonl + data/math/math_numeric.jsonl"
echo "[launch] eval_data      = data/sft/gsm8k_test.jsonl"
echo "[launch] log_file       = $LOG_FILE"
echo "[launch] architecture   = Two-phase self-play (K_q=2, K=10, N=20)"
echo "[launch] fixes_applied  = min-warmup↑12, selfplay-gt-thresh↑0.65, kl-coef↑0.06,"
echo "[launch]                  math-ramp-start↑18, group-size↑10, num-iters↑60"
echo "[launch] wall-time      ≈ 3.3 h (Flash active) / 4.5 h (SDPA fallback)"

# ── Train ─────────────────────────────────────────────────────────────────────
python -u scripts/run_grpo_training.py \
    --base-model            checkpoints/dual_task_v1 \
    --output-dir            checkpoints/grpo \
    --gsm8k-data            data/sft/gsm8k_sft.jsonl \
    --eval-data-path        data/sft/gsm8k_test.jsonl \
    \
    --num-iterations        60 \
    --group-size            10 \
    --q-group-size          2 \
    --questions-per-iter    20 \
    \
    --learning-rate         5e-6 \
    --max-new-tokens        1000 \
    --temperature           0.8 \
    --max-grad-norm         0.5 \
    --clip-eps              0.2 \
    --kl-coef               0.06 \
    --warmup-iters          8 \
    --min-lr-ratio          0.1 \
    \
    --difficulty-alpha      3.5 \
    --self-play-ratio       0.70 \
    \
    --math-mix-ratio        0.30 \
    --math-mix-ratio-late   0.50 \
    --math-ramp-start       18 \
    --math-max-difficulty   3 \
    \
    --overlong-filter \
    --min-warmup            12 \
    --selfplay-gt-thresh    0.65 \
    --selfplay-grounded-thresh 0.65 \
    --selfplay-step-thresh  0.68 \
    --selfplay-ramp-iters   28 \
    --grounded-floor        0.55 \
    \
    --extractor-model       Qwen/Qwen2.5-0.5B-Instruct \
    --extraction-cache      data/extraction_cache.json \
    \
    --eval-every            5 \
    --eval-max-samples      150 \
    --eval-max-new-tokens   1000 \
    --eval-pass-at-k        0 \
    --save-every            5 \
    --keep-last             4 \
    \
    --use-prm \
    --prm-model             Qwen/Qwen2.5-Math-PRM-7B \
    --run-name              "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
