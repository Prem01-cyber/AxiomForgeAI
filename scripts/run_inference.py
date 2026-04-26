#!/usr/bin/env python3
"""
Inference pipeline: Base Qwen2.5-Math-1.5B-Instruct vs RL fine-tuned checkpoint.

For each sampled GSM8K question, both models generate a step-by-step solution.
Results are saved to reports/<run_name>/ as JSON files for the Gradio demo.

Usage
-----
  # Full run (50 questions, both models):
  python scripts/run_inference.py \\
      --checkpoint checkpoints/grpo_run_v1 \\
      --num-questions 50 \\
      --run-name comparison_v1

  # Quick smoke test (10 questions, no RL model):
  python scripts/run_inference.py \\
      --num-questions 10 \\
      --base-only \\
      --run-name smoke

  # Custom data source:
  python scripts/run_inference.py \\
      --checkpoint checkpoints/grpo_run_v1 \\
      --data data/sft/gsm8k_test.jsonl \\
      --num-questions 30
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.prompts import create_solver_messages
from src.sft.solution_format import extract_final_answer_numeric_str
from src.utils.attn_backend import select_attn_implementation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
REPORTS_DIR   = Path("reports")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gsm8k_questions(
    data_path: Optional[str],
    num_questions: int,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Load GSM8K questions from a local JSONL file or fall back to HuggingFace.

    Each returned record has keys: ``question``, ``gold_final``, ``answer``.
    """
    # ── Try local JSONL first ────────────────────────────────────────────────
    candidates = [data_path] if data_path else []
    candidates += [
        "data/sft/gsm8k_test.jsonl",
        "data/sft/gsm8k_sft.jsonl",
    ]

    for path in candidates:
        if path and Path(path).exists():
            logger.info("Loading GSM8K from local file: %s", path)
            rows: List[Dict] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            rng = random.Random(seed)
            sample = rng.sample(rows, min(num_questions, len(rows)))
            logger.info("Sampled %d / %d questions.", len(sample), len(rows))
            return sample

    # ── Fall back to HuggingFace datasets ────────────────────────────────────
    logger.info("No local file found — downloading GSM8K from HuggingFace…")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        raise RuntimeError(
            "Could not load GSM8K. Provide --data or install datasets: pip install datasets"
        ) from e

    rows = []
    for item in ds:
        q = item["question"].strip()
        a = item["answer"].strip()
        # GSM8K answers end with "#### <number>"
        gold = a.split("####")[-1].strip() if "####" in a else ""
        rows.append({"question": q, "gold_final": gold, "answer": a})

    rng = random.Random(seed)
    sample = rng.sample(rows, min(num_questions, len(rows)))
    logger.info("Sampled %d questions from HF GSM8K test split.", len(sample))
    return sample


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(
    device: torch.device,
    attn_impl: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading base model: %s", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    logger.info("Base model loaded.")
    return model, tokenizer


def load_rl_model(
    checkpoint: str,
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    device: torch.device,
    attn_impl: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the RL fine-tuned checkpoint for comparison against the raw base model.

    Two checkpoint formats are supported:

    PEFT / LoRA adapter (has adapter_config.json)
        The already-loaded base model weights are deep-copied in CPU memory,
        the adapter is applied on top, then merged and unloaded.
        This avoids downloading the 1.5B base weights from HuggingFace a
        second time — the base model is downloaded only once per run.

    Full saved model (has config.json, no adapter_config.json)
        Loaded directly from disk with from_pretrained.
    """
    import copy

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    is_peft = (ckpt_path / "adapter_config.json").exists()

    if is_peft:
        logger.info(
            "Loading PEFT adapter from %s  (reusing base weights — no second HF download)",
            checkpoint,
        )
        from peft import PeftModel

        # Deep-copy the already-loaded base model so the base remains untouched
        # for side-by-side comparison.  For a 1.5B bfloat16 model this takes
        # ~1-2 s and avoids re-downloading ~3 GB from HuggingFace.
        base_copy = copy.deepcopy(base_model)
        model = PeftModel.from_pretrained(base_copy, checkpoint)
        model = model.merge_and_unload()
        model = model.to(device)
    else:
        logger.info("Loading full model checkpoint from %s", checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

    # Patch chat_template from base tokenizer if missing
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint if (ckpt_path / "tokenizer_config.json").exists() else BASE_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.chat_template is None and base_tokenizer.chat_template:
        tokenizer.chat_template = base_tokenizer.chat_template

    model.eval()
    logger.info("RL model loaded.")
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_solution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> Tuple[str, float]:
    """
    Generate a step-by-step solution for ``question``.

    Returns ``(solution_text, elapsed_seconds)``.
    Low temperature (0.1) for deterministic, greedy-like output during eval.
    """
    messages = create_solver_messages(question)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end not in stop_ids:
        stop_ids.append(im_end)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.05,
            temperature=temperature if temperature > 0.05 else None,
            top_p=0.95 if temperature > 0.05 else None,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    elapsed = time.time() - t0

    response_ids = output[0][prompt_len:]
    solution = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return solution, elapsed


def score_answer(solution: str, gold_final: str) -> Dict[str, Any]:
    """
    Extract the predicted final answer and compare with gold.
    Returns a dict with ``predicted``, ``gold``, ``correct``, ``match_type``.
    """
    predicted_raw = extract_final_answer_numeric_str(solution)

    if predicted_raw is None:
        return {
            "predicted": None,
            "gold": gold_final,
            "correct": False,
            "match_type": "no_answer_found",
        }

    # Normalise: strip whitespace, remove commas (e.g. "1,200" → "1200")
    def _norm(s: str) -> str:
        return s.strip().replace(",", "").rstrip(".").lower()

    pred_n = _norm(predicted_raw)
    gold_n = _norm(gold_final)

    # Direct string match
    if pred_n == gold_n:
        return {
            "predicted": predicted_raw,
            "gold": gold_final,
            "correct": True,
            "match_type": "exact",
        }

    # Numeric match (handles float/int equivalence)
    try:
        pred_f = float(pred_n)
        gold_f = float(gold_n)
        if abs(pred_f - gold_f) < 1e-6:
            return {
                "predicted": predicted_raw,
                "gold": gold_final,
                "correct": True,
                "match_type": "numeric",
            }
    except (ValueError, TypeError):
        pass

    return {
        "predicted": predicted_raw,
        "gold": gold_final,
        "correct": False,
        "match_type": "wrong",
    }


# ── Report serialisation ──────────────────────────────────────────────────────

def save_question_report(
    report_dir: Path,
    idx: int,
    question: str,
    gold_final: str,
    base_result: Dict[str, Any],
    rl_result: Optional[Dict[str, Any]],
) -> Path:
    record = {
        "idx": idx,
        "question": question,
        "gold_final": gold_final,
        "base_model": base_result,
        "rl_model": rl_result,
    }
    out = report_dir / f"q_{idx:04d}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def save_summary(
    report_dir: Path,
    run_name: str,
    checkpoint: Optional[str],
    base_correct: int,
    rl_correct: Optional[int],
    total: int,
    total_time_s: float,
    args_dict: Dict,
) -> None:
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "base_model": BASE_MODEL_ID,
        "rl_checkpoint": checkpoint,
        "num_questions": total,
        "base_accuracy": round(base_correct / total, 4) if total else 0,
        "rl_accuracy": round(rl_correct / total, 4) if (rl_correct is not None and total) else None,
        "base_correct": base_correct,
        "rl_correct": rl_correct,
        "total_time_s": round(total_time_s, 1),
        "args": args_dict,
    }
    out = report_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference: base vs RL model on GSM8K")
    p.add_argument("--checkpoint",      type=str,  default=None,
                   help="Path to RL fine-tuned model or PEFT adapter. "
                        "If omitted, only the base model is run.")
    p.add_argument("--data",            type=str,  default=None,
                   help="Path to local GSM8K JSONL file. "
                        "Defaults to data/sft/gsm8k_test.jsonl or HuggingFace.")
    p.add_argument("--num-questions",   type=int,  default=50)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--max-new-tokens",  type=int,  default=512)
    p.add_argument("--temperature",     type=float, default=0.1)
    p.add_argument("--run-name",        type=str,  default=None,
                   help="Report sub-folder name. Defaults to timestamp.")
    p.add_argument("--base-only",       action="store_true",
                   help="Skip RL model; only run the base model.")
    p.add_argument("--reports-dir",     type=str,  default="reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_name   = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    report_dir = Path(args.reports_dir) / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reports → %s", report_dir)

    # ── Device ────────────────────────────────────────────────────────────────
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attn_impl = select_attn_implementation()
    logger.info("Device: %s | attn: %s", device, attn_impl)
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | %.1f GB", g.name, g.total_memory / 1e9)

    # ── Data ──────────────────────────────────────────────────────────────────
    questions = load_gsm8k_questions(args.data, args.num_questions, args.seed)

    # ── Models ────────────────────────────────────────────────────────────────
    base_model, base_tokenizer = load_base_model(device, attn_impl)

    rl_model, rl_tokenizer = None, None
    if not args.base_only and args.checkpoint:
        rl_model, rl_tokenizer = load_rl_model(
            args.checkpoint, base_model, base_tokenizer, device, attn_impl
        )
    elif not args.base_only and not args.checkpoint:
        logger.warning("No --checkpoint provided. Running base model only.")

    # ── Inference loop ────────────────────────────────────────────────────────
    base_correct = 0
    rl_correct   = 0 if rl_model else None
    t_total_start = time.time()

    for idx, row in enumerate(tqdm(questions, desc="Inference")):
        question   = row["question"]
        gold_final = row.get("gold_final", "").strip()

        # Base model
        base_solution, base_time = generate_solution(
            base_model, base_tokenizer, question, device,
            args.max_new_tokens, args.temperature,
        )
        base_score = score_answer(base_solution, gold_final)
        if base_score["correct"]:
            base_correct += 1

        base_result = {
            "solution":  base_solution,
            "predicted": base_score["predicted"],
            "correct":   base_score["correct"],
            "match_type": base_score["match_type"],
            "time_s":    round(base_time, 2),
            "num_tokens": len(base_tokenizer.encode(base_solution)),
        }

        # RL model
        rl_result = None
        if rl_model is not None:
            rl_solution, rl_time = generate_solution(
                rl_model, rl_tokenizer, question, device,
                args.max_new_tokens, args.temperature,
            )
            rl_score = score_answer(rl_solution, gold_final)
            if rl_score["correct"]:
                rl_correct += 1

            rl_result = {
                "solution":  rl_solution,
                "predicted": rl_score["predicted"],
                "correct":   rl_score["correct"],
                "match_type": rl_score["match_type"],
                "time_s":    round(rl_time, 2),
                "num_tokens": len(rl_tokenizer.encode(rl_solution)),
            }

        save_question_report(report_dir, idx, question, gold_final, base_result, rl_result)

        # Live progress log every 10 questions
        if (idx + 1) % 10 == 0 or idx == len(questions) - 1:
            done = idx + 1
            b_acc = base_correct / done
            log_str = f"[{done}/{len(questions)}] Base acc: {b_acc:.1%}"
            if rl_correct is not None:
                log_str += f"  |  RL acc: {rl_correct / done:.1%}"
            logger.info(log_str)

    total_time = time.time() - t_total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    save_summary(
        report_dir=report_dir,
        run_name=run_name,
        checkpoint=args.checkpoint,
        base_correct=base_correct,
        rl_correct=rl_correct,
        total=len(questions),
        total_time_s=total_time,
        args_dict=vars(args),
    )

    logger.info("=" * 60)
    logger.info("Run complete: %s", run_name)
    logger.info("Base accuracy : %d / %d = %.1f%%",
                base_correct, len(questions), 100 * base_correct / len(questions))
    if rl_correct is not None:
        logger.info("RL accuracy   : %d / %d = %.1f%%",
                    rl_correct, len(questions), 100 * rl_correct / len(questions))
        delta = rl_correct - base_correct
        sign  = "+" if delta >= 0 else ""
        logger.info("Delta         : %s%d questions (%s%.1f%%)",
                    sign, delta, sign, 100 * delta / len(questions))
    logger.info("Reports       : %s", report_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
