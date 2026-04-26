"""
AxiomForgeAI — Inference Demo
==============================
Gradio UI that loads pre-saved inference reports and presents them as a
live side-by-side comparison between the base Qwen2.5-Math-1.5B model and
the RL fine-tuned model.

The typewriter streaming effect replays the saved solutions token-by-token
so the demo feels live without needing a GPU at presentation time.

Run
---
  python demo.py                          # auto-picks most recent report run
  python demo.py --run reports/run_v1    # pick a specific run folder
  python demo.py --port 7860             # custom port
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

REPORTS_DIR = Path("reports")
STREAM_DELAY_S = 0.012   # seconds between characters in typewriter effect
WORD_MODE      = True    # stream word-by-word (smoother than char-by-char)


# ── Report loading ────────────────────────────────────────────────────────────

def list_runs(reports_dir: Path = REPORTS_DIR) -> List[str]:
    if not reports_dir.exists():
        return []
    runs = sorted(
        [d.name for d in reports_dir.iterdir() if d.is_dir() and (d / "summary.json").exists()],
        reverse=True,
    )
    return runs


def load_summary(run: str, reports_dir: Path = REPORTS_DIR) -> Dict:
    path = reports_dir / run / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(run: str, reports_dir: Path = REPORTS_DIR) -> List[Dict]:
    run_dir = reports_dir / run
    files = sorted(run_dir.glob("q_*.json"))
    questions = []
    for f in files:
        try:
            questions.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return questions


# ── Text helpers ──────────────────────────────────────────────────────────────

def _colour_steps(solution: str) -> str:
    """
    Convert Step N: / Final Answer: lines to Markdown with light formatting.
    """
    lines = solution.split("\n")
    out = []
    for line in lines:
        if re.match(r"^Step\s+\d+\s*:", line, re.IGNORECASE):
            out.append(f"**{line}**")
        elif re.match(r"^Final\s+Answer\s*:", line, re.IGNORECASE):
            out.append(f"\n> 🎯 **{line}**")
        else:
            out.append(line)
    return "\n".join(out)


def _stream_text(text: str) -> Generator[str, None, None]:
    """
    Yield progressively longer prefixes of text, word-by-word,
    to create a typewriter / streaming effect.
    """
    if WORD_MODE:
        words = text.split(" ")
        buffer = ""
        for word in words:
            buffer += ("" if buffer == "" else " ") + word
            yield buffer
            time.sleep(STREAM_DELAY_S)
    else:
        for i in range(1, len(text) + 1):
            yield text[:i]
            time.sleep(STREAM_DELAY_S / 3)


def _correctness_badge(correct: Optional[bool]) -> str:
    if correct is True:
        return "✅ **CORRECT**"
    if correct is False:
        return "❌ **WRONG**"
    return "⚠️ **N/A**"


def _answer_line(result: Optional[Dict]) -> str:
    if result is None:
        return "_Not run_"
    pred = result.get("predicted") or "_not found_"
    gold = result.get("gold", "?")
    badge = _correctness_badge(result.get("correct"))
    return f"{badge}  |  Predicted: `{pred}`  |  Gold: `{gold}`"


# ── Summary markdown ──────────────────────────────────────────────────────────

def build_summary_md(summary: Dict, questions: List[Dict]) -> str:
    if not summary:
        return "_No summary available._"

    total = summary.get("num_questions", len(questions))
    b_acc = summary.get("base_accuracy", 0)
    r_acc = summary.get("rl_accuracy")
    base  = summary.get("base_model", "Qwen2.5-Math-1.5B-Instruct")
    ckpt  = summary.get("rl_checkpoint") or "—"
    ts    = summary.get("timestamp", "")[:19]

    lines = [
        f"## 📊 Run: `{summary.get('run_name', '—')}`",
        f"**Timestamp:** {ts}  ",
        "",
        "| | Base Model | RL Fine-Tuned |",
        "|---|---|---|",
        f"| **Model** | `{Path(base).name}` | `{Path(ckpt).name if ckpt != '—' else '—'}` |",
        f"| **Accuracy** | **{b_acc:.1%}** | "
        + (f"**{r_acc:.1%}**" if r_acc is not None else "_not run_") + " |",
        f"| **Correct** | {summary.get('base_correct', '?')} / {total} | "
        + (f"{summary.get('rl_correct', '?')} / {total}" if r_acc is not None else "—") + " |",
    ]

    if r_acc is not None:
        delta = r_acc - b_acc
        sign  = "+" if delta >= 0 else ""
        emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
        lines.append(f"| **Δ Accuracy** | | {emoji} {sign}{delta:.1%} |")

    lines += [
        "",
        f"**Questions sampled:** {total}  |  "
        f"**Total inference time:** {summary.get('total_time_s', 0):.0f}s",
    ]
    return "\n".join(lines)


# ── Gradio app ────────────────────────────────────────────────────────────────

def create_app(default_run: Optional[str] = None) -> gr.Blocks:
    runs        = list_runs()
    init_run    = default_run or (runs[0] if runs else None)
    init_qs     = load_questions(init_run) if init_run else []
    init_summ   = load_summary(init_run)   if init_run else {}
    q_labels    = [f"Q{i+1}: {q['question'][:80]}…" for i, q in enumerate(init_qs)]

    css = """
    .correct-box  { border-left: 4px solid #22c55e !important; padding-left: 8px; }
    .wrong-box    { border-left: 4px solid #ef4444 !important; padding-left: 8px; }
    .model-label  { font-size: 0.85rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase; }
    .question-box { background: #1e293b; border-radius: 8px; padding: 12px 16px; font-size: 1.05rem; line-height: 1.6; }
    #header-row   { text-align: center; margin-bottom: 4px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="AxiomForgeAI — Inference Demo",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="pink",
            neutral_hue="slate",
        ),
        css=css,
    ) as demo:

        # ── State ──────────────────────────────────────────────────────────
        state_questions = gr.State(init_qs)
        state_run       = gr.State(init_run)

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding: 24px 0 8px 0;">
          <h1 style="font-size:2rem; font-weight:700; margin:0; background: linear-gradient(90deg,#6366f1,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AxiomForgeAI
          </h1>
          <p style="color:#94a3b8; margin:4px 0 0 0; font-size:1rem; font-style:italic;">
            It doesn't just solve problems — it writes them, answers them, and grades every step of its own reasoning.
          </p>
        </div>
        """)

        # ── Run selector + summary ─────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                run_dropdown = gr.Dropdown(
                    choices=runs,
                    value=init_run,
                    label="📂 Select inference run",
                    interactive=True,
                )
                refresh_btn = gr.Button("🔄 Refresh runs", variant="secondary", size="sm")

            with gr.Column(scale=3):
                summary_md = gr.Markdown(
                    value=build_summary_md(init_summ, init_qs),
                    label="Run Summary",
                )

        gr.Divider()

        # ── Question selector ──────────────────────────────────────────────
        with gr.Row():
            q_dropdown = gr.Dropdown(
                choices=q_labels,
                value=q_labels[0] if q_labels else None,
                label="🎯 Question",
                scale=5,
                interactive=True,
            )
            random_btn = gr.Button("🎲 Random", variant="primary", scale=1)
            play_btn   = gr.Button("▶  Stream", variant="primary", scale=1)

        # ── Question text ──────────────────────────────────────────────────
        question_box = gr.Markdown(
            value=f"> **{init_qs[0]['question']}**" if init_qs else "",
            label="Question",
        )

        gr.HTML("<div style='height:6px'></div>")

        # ── Gold answer strip ──────────────────────────────────────────────
        gold_strip = gr.Markdown(
            value=f"**Gold answer:** `{init_qs[0]['gold_final']}`" if init_qs else "",
        )

        gr.Divider()

        # ── Side-by-side model outputs ─────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='model-label'>🟦 Base Model — Qwen2.5-Math-1.5B-Instruct</div>")
                base_answer_md  = gr.Markdown(value="", label="Answer")
                base_solution   = gr.Markdown(value="", label="Solution", height=420)
                base_meta       = gr.Markdown(value="", label="Metadata")

            with gr.Column():
                gr.HTML("<div class='model-label'>🟣 RL Fine-Tuned Model</div>")
                rl_answer_md    = gr.Markdown(value="", label="Answer")
                rl_solution     = gr.Markdown(value="", label="Solution", height=420)
                rl_meta         = gr.Markdown(value="", label="Metadata")

        # ── Accuracy bar (updates after each question is displayed) ────────
        accuracy_bar = gr.Markdown(value="", label="Session accuracy")

        # ── Session counters (hidden state) ────────────────────────────────
        session_total   = gr.State(0)
        session_b_ok    = gr.State(0)
        session_rl_ok   = gr.State(0)

        # ── Callbacks ──────────────────────────────────────────────────────

        def on_run_change(run_name: str):
            if not run_name:
                return gr.update(), gr.update(), gr.update(), [], None
            qs    = load_questions(run_name)
            summ  = load_summary(run_name)
            labels = [f"Q{i+1}: {q['question'][:80]}…" for i, q in enumerate(qs)]
            first_q = qs[0]["question"] if qs else ""
            first_g = qs[0].get("gold_final","") if qs else ""
            return (
                gr.update(choices=labels, value=labels[0] if labels else None),
                build_summary_md(summ, qs),
                f"> **{first_q}**",
                f"**Gold answer:** `{first_g}`",
                qs,
                run_name,
            )

        run_dropdown.change(
            on_run_change,
            inputs=[run_dropdown],
            outputs=[q_dropdown, summary_md, question_box, gold_strip,
                     state_questions, state_run],
        )

        def on_refresh():
            new_runs = list_runs()
            return gr.update(choices=new_runs, value=new_runs[0] if new_runs else None)

        refresh_btn.click(on_refresh, outputs=[run_dropdown])

        def on_q_change(label: str, questions: List[Dict]):
            if not label or not questions:
                return "", "", "", "", "", "", "", ""
            idx = int(re.match(r"Q(\d+):", label).group(1)) - 1
            idx = max(0, min(idx, len(questions) - 1))
            q   = questions[idx]
            return (
                f"> **{q['question']}**",
                f"**Gold answer:** `{q.get('gold_final','')}`",
                # clear outputs — user must press Stream to populate
                "", "", "", "", "", "",
            )

        q_dropdown.change(
            on_q_change,
            inputs=[q_dropdown, state_questions],
            outputs=[question_box, gold_strip,
                     base_answer_md, base_solution, base_meta,
                     rl_answer_md, rl_solution, rl_meta],
        )

        def on_random(questions: List[Dict]):
            if not questions:
                return gr.update()
            idx    = __import__("random").randint(0, len(questions) - 1)
            labels = [f"Q{i+1}: {q['question'][:80]}…" for i, q in enumerate(questions)]
            return gr.update(value=labels[idx])

        random_btn.click(on_random, inputs=[state_questions], outputs=[q_dropdown])

        def on_play(
            label: str,
            questions: List[Dict],
            s_total: int,
            s_b_ok: int,
            s_rl_ok: int,
        ):
            """
            Generator: streams base solution then RL solution word-by-word.
            Yields updates to all output components on every tick.
            """
            if not label or not questions:
                yield ("", "", "", "", "", "", "", "", 0, 0, 0)
                return

            idx = int(re.match(r"Q(\d+):", label).group(1)) - 1
            idx = max(0, min(idx, len(questions) - 1))
            q   = questions[idx]

            b   = q.get("base_model") or {}
            rl  = q.get("rl_model")
            g   = q.get("gold_final", "")

            b_sol  = _colour_steps(b.get("solution", "_Not available_"))
            rl_sol = _colour_steps(rl.get("solution", "_Not run_")) if rl else "_Not run_"

            b_ans_md  = _answer_line({**b, "gold": g})
            rl_ans_md = _answer_line({**rl, "gold": g}) if rl else "_Not run_"

            b_meta  = (f"⏱ `{b.get('time_s','?')}s`  |  "
                       f"📝 `{b.get('num_tokens','?')} tokens`")
            rl_meta = (f"⏱ `{rl.get('time_s','?')}s`  |  "
                       f"📝 `{rl.get('num_tokens','?')} tokens`"
                       if rl else "—")

            # ── Stream base solution ──────────────────────────────────────
            for partial in _stream_text(b_sol):
                yield (
                    b_ans_md, partial, b_meta,
                    "⏳ _generating…_", "", "",
                    "", s_total, s_b_ok, s_rl_ok,
                )

            # ── Stream RL solution ────────────────────────────────────────
            for partial in _stream_text(rl_sol):
                yield (
                    b_ans_md, b_sol, b_meta,
                    rl_ans_md, partial, rl_meta,
                    "", s_total, s_b_ok, s_rl_ok,
                )

            # ── Final frame: update session counters + accuracy bar ───────
            s_total += 1
            if b.get("correct"):  s_b_ok  += 1
            if rl and rl.get("correct"): s_rl_ok += 1

            bar_parts = [
                f"**Session accuracy** ({s_total} shown)  —  "
                f"Base: **{s_b_ok}/{s_total} = {s_b_ok/s_total:.0%}**"
            ]
            if rl:
                bar_parts.append(
                    f"  |  RL: **{s_rl_ok}/{s_total} = {s_rl_ok/s_total:.0%}**"
                )
                delta = s_rl_ok - s_b_ok
                sign  = "+" if delta >= 0 else ""
                emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
                bar_parts.append(f"  {emoji} Δ {sign}{delta}")
            acc_bar_val = "".join(bar_parts)

            yield (
                b_ans_md, b_sol, b_meta,
                rl_ans_md, rl_sol, rl_meta,
                acc_bar_val, s_total, s_b_ok, s_rl_ok,
            )

        play_btn.click(
            on_play,
            inputs=[q_dropdown, state_questions,
                    session_total, session_b_ok, session_rl_ok],
            outputs=[
                base_answer_md, base_solution, base_meta,
                rl_answer_md,   rl_solution,   rl_meta,
                accuracy_bar,
                session_total, session_b_ok, session_rl_ok,
            ],
        )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0 8px 0; color:#475569; font-size:0.85rem;">
          AxiomForgeAI &nbsp;·&nbsp; GRPO-trained Qwen2.5-Math-1.5B
          &nbsp;·&nbsp; Step-level PRM scoring &nbsp;·&nbsp; Adaptive curriculum
        </div>
        """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AxiomForgeAI Gradio inference demo")
    p.add_argument("--run",         type=str, default=None,
                   help="Path to a specific reports/<run> folder to preload.")
    p.add_argument("--reports-dir", type=str, default="reports",
                   help="Root reports directory (default: reports/)")
    p.add_argument("--port",        type=int, default=7860)
    p.add_argument("--share",       action="store_true",
                   help="Create a public Gradio share link.")
    p.add_argument("--stream-delay", type=float, default=0.012,
                   help="Seconds between words in typewriter effect (default: 0.012)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    global REPORTS_DIR, STREAM_DELAY_S
    REPORTS_DIR    = Path(args.reports_dir)
    STREAM_DELAY_S = args.stream_delay

    default_run = None
    if args.run:
        run_path = Path(args.run)
        default_run = run_path.name if run_path.is_dir() else args.run
        REPORTS_DIR = run_path.parent if run_path.is_dir() else REPORTS_DIR

    app = create_app(default_run=default_run)
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",   # bind to all interfaces so SSH port-forward works
    )
