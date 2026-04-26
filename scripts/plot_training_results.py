#!/usr/bin/env python3
"""
AxiomForgeAI — Training Results Plots
======================================
Reads the metrics CSV from a GRPO training run and generates five focused plots
that tell the story of what improved, how self-play was earned, and why step-level
reasoning quality matters as much as final-answer accuracy.

All plots are saved to images/ as high-resolution PNGs.

Usage
-----
  python scripts/plot_training_results.py
  python scripts/plot_training_results.py --metrics logs/grpo/grpo_20260426_032827/metrics.csv
  python scripts/plot_training_results.py --out images/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "indigo":    "#6366f1",
    "pink":      "#ec4899",
    "cyan":      "#06b6d4",
    "amber":     "#f59e0b",
    "emerald":   "#10b981",
    "slate":     "#94a3b8",
    "red":       "#ef4444",
    "violet":    "#8b5cf6",
    "white":     "#f8fafc",
    "bg":        "#0f172a",
    "bg2":       "#1e293b",
    "gridline":  "#1e293b",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["slate"],
    "axes.labelcolor":   PALETTE["white"],
    "axes.titlecolor":   PALETTE["white"],
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.grid":         True,
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.8,
    "xtick.color":       PALETTE["slate"],
    "ytick.color":       PALETTE["slate"],
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  "#1e293b",
    "legend.edgecolor":  PALETTE["slate"],
    "legend.labelcolor": PALETTE["white"],
    "legend.fontsize":   9,
    "text.color":        PALETTE["white"],
    "font.family":       "sans-serif",
    "lines.linewidth":   2.0,
})

PHASE_COLORS = {
    "GROUNDED_ONLY":  ("#6366f120", "#6366f1"),
    "SELFPLAY_RAMP":  ("#10b98120", "#10b981"),
}

DPI = 160
IMAGES_DIR = Path("images")

DEFAULT_METRICS = (
    "logs/grpo/grpo_20260426_032827/metrics.csv"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: v for k, v in r.items()})
    return rows


def f(row: Dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    try:
        return float(v) if v != "" else default
    except (ValueError, TypeError):
        return default


def moving_avg(values: List[float], w: int = 3) -> List[float]:
    result = []
    for i in range(len(values)):
        lo = max(0, i - w + 1)
        chunk = [v for v in values[lo : i + 1] if not np.isnan(v)]
        result.append(float(np.mean(chunk)) if chunk else float("nan"))
    return result


def shade_phases(ax, iters, phases):
    """Draw translucent background rectangles for each training phase."""
    prev_phase, start = None, iters[0]
    for it, ph in zip(iters, phases):
        if ph != prev_phase:
            if prev_phase is not None:
                bg, _ = PHASE_COLORS.get(prev_phase, ("#ffffff10", "#ffffff"))
                ax.axvspan(start - 0.5, it - 0.5, facecolor=bg, linewidth=0, zorder=0)
            prev_phase, start = ph, it
    if prev_phase is not None:
        bg, _ = PHASE_COLORS.get(prev_phase, ("#ffffff10", "#ffffff"))
        ax.axvspan(start - 0.5, iters[-1] + 0.5, facecolor=bg, linewidth=0, zorder=0)


def phase_legend_patches(phases):
    seen = []
    patches = []
    for ph in phases:
        if ph not in seen:
            seen.append(ph)
            _, edge = PHASE_COLORS.get(ph, ("#ffffff10", "#ffffff"))
            label = ph.replace("_", " ").title()
            patches.append(mpatches.Patch(facecolor=edge + "40", edgecolor=edge,
                                          linewidth=1.2, label=label))
    return patches


def annotate_transition(ax, x_iter, label, ypos=0.97, color="#94a3b8"):
    ax.axvline(x=x_iter - 0.5, color=color, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(x_iter, ypos, label, transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=color, ha="left", va="top",
            bbox=dict(facecolor=PALETTE["bg2"], edgecolor="none", pad=2))


def save(fig: plt.Figure, name: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓  {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Hero: Reasoning quality at evaluation checkpoints
# Shows four signals together: GSM8K accuracy, combined score, step accuracy,
# and LCCP.  The message: the model doesn't just get more answers right —
# every step of the reasoning chain gets better.
# ══════════════════════════════════════════════════════════════════════════════

def plot_eval_quality(rows: List[Dict], out: Path):
    eval_rows = [r for r in rows if r.get("eval_combined", "") != ""]
    iters     = [int(r["iteration"]) for r in eval_rows]

    gsm8k_acc  = [f(r, "eval_correct_rt") * 100 for r in eval_rows]
    combined   = [f(r, "eval_combined") * 100    for r in eval_rows]
    step_acc   = [f(r, "eval_step_acc") * 100    for r in eval_rows]
    lccp       = [f(r, "eval_lccp") * 100        for r in eval_rows]
    prm        = [f(r, "eval_prm") * 100         for r in eval_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Evaluation Quality Over Training — AxiomForgeAI",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=1.01)

    # --- lines
    ax.plot(iters, gsm8k_acc, "o-",  color=PALETTE["pink"],    label="GSM8K Accuracy (final answer)", ms=7, zorder=5)
    ax.plot(iters, combined,  "s-",  color=PALETTE["indigo"],  label="Combined Score",                 ms=6, zorder=5)
    ax.plot(iters, step_acc,  "^-",  color=PALETTE["cyan"],    label="Step Accuracy (reasoning chain)", ms=6, zorder=5)
    ax.plot(iters, lccp,      "D-",  color=PALETTE["emerald"], label="LCCP (chain integrity)",          ms=6, zorder=5)
    ax.plot(iters, prm,       "v--", color=PALETTE["amber"],   label="PRM Mean Score",                  ms=5, alpha=0.8, zorder=4)

    # annotate best GSM8K
    best_gsm = max(gsm8k_acc)
    bi = gsm8k_acc.index(best_gsm)
    ax.annotate(f"  {best_gsm:.1f}%",
                xy=(iters[bi], best_gsm), fontsize=9, color=PALETTE["pink"],
                va="bottom", ha="left")

    # annotate best combined
    best_c = max(combined)
    bci = combined.index(best_c)
    ax.annotate(f"  {best_c:.1f}",
                xy=(iters[bci], best_c), fontsize=9, color=PALETTE["indigo"],
                va="top", ha="left")

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(iters)
    ax.set_ylim(78, 96)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="lower right", framealpha=0.8)
    ax.set_title(
        "Four angles on quality — answer correctness, holistic score, per-step reasoning, and chain integrity",
        fontsize=9, color=PALETTE["slate"], pad=6,
    )

    fig.tight_layout()
    save(fig, "plot1_eval_quality.png", out)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Training Journey: full 30-iteration timeline with phase shading
# Shows mean reward, GT match rate, and step accuracy over every iteration.
# Phase backgrounds show when self-play unlocked and the curriculum ramped.
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_journey(rows: List[Dict], out: Path):
    iters      = [int(r["iteration"]) for r in rows]
    phases     = [r["training_phase"] for r in rows]
    mean_r     = [f(r, "mean_reward") * 100      for r in rows]
    gt_match   = [f(r, "gt_match_rate") * 100    for r in rows]
    step_acc   = [f(r, "step_accuracy") * 100    for r in rows]
    batch_acc  = [f(r, "batch_accuracy") * 100   for r in rows]

    ma_reward = moving_avg(mean_r,   w=4)
    ma_gt     = moving_avg(gt_match, w=4)
    ma_step   = moving_avg(step_acc, w=4)

    fig, ax = plt.subplots(figsize=(11, 5))
    shade_phases(ax, iters, phases)

    # raw (faint)
    ax.plot(iters, mean_r,   alpha=0.25, color=PALETTE["indigo"],  linewidth=1)
    ax.plot(iters, gt_match, alpha=0.25, color=PALETTE["pink"],    linewidth=1)
    ax.plot(iters, step_acc, alpha=0.25, color=PALETTE["cyan"],    linewidth=1)

    # smoothed (bold)
    ax.plot(iters, ma_reward, color=PALETTE["indigo"],  linewidth=2.5, label="Mean Reward (smooth)")
    ax.plot(iters, ma_gt,     color=PALETTE["pink"],    linewidth=2.5, label="GT Match Rate (smooth)")
    ax.plot(iters, ma_step,   color=PALETTE["cyan"],    linewidth=2.5, label="Step Accuracy (smooth)")

    # self-play transition annotation
    sp_start = next(i for i, p in enumerate(phases) if p == "SELFPLAY_RAMP")
    annotate_transition(ax, iters[sp_start], "Self-play\nunlocked", ypos=0.98,
                        color=PALETTE["emerald"])

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(55, 105)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax.set_xticks(range(1, max(iters) + 1, 2))
    ax.set_title("30-Iteration GRPO Training Timeline  |  Faint = raw  ·  Bold = 4-iter moving average",
                 fontsize=9, color=PALETTE["slate"], pad=6)
    fig.suptitle("Training Journey — Reward, GT Match & Step Accuracy",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=1.01)

    legend_patches = phase_legend_patches(phases)
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + legend_patches, loc="lower right", framealpha=0.8, ncol=2)

    fig.tight_layout()
    save(fig, "plot2_training_journey.png", out)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Self-Play Success: the curriculum earning its right to generate
# Shows the self-play ratio ramping up while question quality stays high.
# The headline: by iteration 30 more than 60% of training is model-generated,
# and those questions are 95-100% solvable and genuinely novel.
# ══════════════════════════════════════════════════════════════════════════════

def plot_selfplay_success(rows: List[Dict], out: Path):
    sp_rows = [r for r in rows if f(r, "q_reward") > 0]
    iters   = [int(r["iteration"]) for r in sp_rows]
    sp_rat  = [f(r, "sp_ratio") * 100      for r in sp_rows]
    q_sol   = [f(r, "q_solvability") * 100 for r in sp_rows]
    q_nov   = [f(r, "q_novelty") * 100     for r in sp_rows]
    q_rew   = [f(r, "q_reward") * 100      for r in sp_rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax2.tick_params(axis="y", labelcolor=PALETTE["slate"])
    ax2.spines["right"].set_color(PALETTE["slate"])

    # self-play ramp (left axis)
    ax1.fill_between(iters, sp_rat, alpha=0.18, color=PALETTE["emerald"])
    ax1.plot(iters, sp_rat, "o-", color=PALETTE["emerald"], ms=6,
             label="Self-play ratio", linewidth=2.5)
    ax1.set_ylabel("Self-play share of training (%)", color=PALETTE["emerald"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["emerald"])
    ax1.set_ylim(0, 80)

    # question quality (right axis)
    ax2.plot(iters, q_sol, "s--", color=PALETTE["cyan"],    ms=5, label="Solvability",   linewidth=1.8)
    ax2.plot(iters, q_nov, "^--", color=PALETTE["amber"],   ms=5, label="Novelty",        linewidth=1.8)
    ax2.plot(iters, q_rew, "D--", color=PALETTE["pink"],    ms=5, label="Q-Reward",       linewidth=1.8)
    ax2.set_ylabel("Question quality score (%)", color=PALETTE["slate"])
    ax2.set_ylim(0, 115)

    # merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.8)

    ax1.set_xlabel("Training Iteration")
    ax1.set_xticks(iters)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))

    # annotate final sp ratio
    ax1.annotate(f"  {sp_rat[-1]:.0f}% self-play\n  by iter {iters[-1]}",
                 xy=(iters[-1], sp_rat[-1]), fontsize=9, color=PALETTE["emerald"],
                 va="center", ha="left")

    fig.suptitle("Self-Play Curriculum — The Model Earns Its Own Training Data",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=1.01)
    ax1.set_title(
        "Self-play ratio ramps from 0 → 61%  ·  Generated questions stay 93-100% solvable throughout",
        fontsize=9, color=PALETTE["slate"], pad=6,
    )
    fig.tight_layout()
    save(fig, "plot3_selfplay_success.png", out)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Reward Signal Tightening: mean ± std over 30 iterations
# As the policy learns what "good" looks like, the spread between the best
# and worst solutions in a group narrows.  Lower variance = more consistent
# reasoning, not lucky guessing.
# ══════════════════════════════════════════════════════════════════════════════

def plot_reward_confidence(rows: List[Dict], out: Path):
    iters   = [int(r["iteration"]) for r in rows]
    phases  = [r["training_phase"]  for r in rows]
    mean_r  = np.array([f(r, "mean_reward")  for r in rows])
    std_r   = np.array([f(r, "std_reward")   for r in rows])
    skipped = np.array([f(r, "skipped_groups", 0) for r in rows])
    n_grps  = np.array([f(r, "n_groups", 1)        for r in rows])
    skip_rt = skipped / np.maximum(n_grps, 1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.2]})
    fig.suptitle("Reward Confidence — Mean ± Std  &  Skipped Groups Over 30 Iterations",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=1.01)

    shade_phases(ax1, iters, phases)

    ax1.fill_between(iters, (mean_r - std_r) * 100, (mean_r + std_r) * 100,
                     alpha=0.20, color=PALETTE["indigo"])
    ax1.plot(iters, mean_r * 100, color=PALETTE["indigo"], linewidth=2.5, label="Mean reward")
    ax1.plot(iters, (mean_r - std_r) * 100, "--", color=PALETTE["slate"], linewidth=1,
             alpha=0.6, label="±1 std")
    ax1.plot(iters, (mean_r + std_r) * 100, "--", color=PALETTE["slate"], linewidth=1,
             alpha=0.6)

    # highlight the two tight-cluster peaks
    for special_iter, label in [(11, "iter 11\nstd=0.098"), (22, "iter 22\nstd=0.124")]:
        si = iters.index(special_iter)
        ax1.annotate(label,
                     xy=(special_iter, (mean_r[si] + std_r[si]) * 100),
                     xytext=(special_iter + 1, (mean_r[si] + std_r[si]) * 100 + 2),
                     fontsize=8, color=PALETTE["amber"],
                     arrowprops=dict(arrowstyle="->", color=PALETTE["amber"], lw=1.2))

    ax1.set_ylabel("Reward (%)")
    ax1.set_ylim(55, 115)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + phase_legend_patches(phases), framealpha=0.8, ncol=3)

    # skip-rate bar chart (bottom panel)
    shade_phases(ax2, iters, phases)
    ax2.bar(iters, skip_rt, color=PALETTE["red"], alpha=0.7, width=0.7, label="Skipped groups %")
    ax2.set_ylabel("Skipped\ngroups (%)")
    ax2.set_xlabel("Training Iteration")
    ax2.set_ylim(0, 75)
    ax2.set_xticks(range(1, max(iters) + 1, 2))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()
    save(fig, "plot4_reward_confidence.png", out)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Step-Level Reasoning Quality: train vs eval
# Breaks down the two signals that measure HOW the model thinks (not just
# whether it gets the final answer right): step accuracy and LCCP.
# Train lines are noisy; eval lines show clean upward trends.
# ══════════════════════════════════════════════════════════════════════════════

def plot_reasoning_quality(rows: List[Dict], out: Path):
    iters   = [int(r["iteration"]) for r in rows]
    phases  = [r["training_phase"] for r in rows]

    # training
    t_step  = [f(r, "step_accuracy") * 100 for r in rows]
    t_lccp  = [f(r, "lccp") * 100          for r in rows]
    t_gt    = [f(r, "gt_match_rate") * 100 for r in rows]

    # eval (only at checkpoint iters)
    eval_rows  = [r for r in rows if r.get("eval_combined", "") != ""]
    e_iters    = [int(r["iteration"])        for r in eval_rows]
    e_step     = [f(r, "eval_step_acc") * 100 for r in eval_rows]
    e_lccp     = [f(r, "eval_lccp") * 100     for r in eval_rows]

    # moving averages
    ma_step = moving_avg(t_step, w=4)
    ma_lccp = moving_avg(t_lccp, w=4)
    ma_gt   = moving_avg(t_gt,   w=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Step-Level Reasoning Quality — Training vs Held-Out Evaluation",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=1.01)

    # ── LEFT: step accuracy ──
    shade_phases(ax1, iters, phases)
    ax1.plot(iters, t_step,  alpha=0.2,  color=PALETTE["cyan"],  linewidth=1)
    ax1.plot(iters, ma_step, color=PALETTE["cyan"],  linewidth=2.5, label="Train step acc (smooth)")
    ax1.plot(iters, t_gt,    alpha=0.15, color=PALETTE["pink"],  linewidth=1)
    ax1.plot(iters, ma_gt,   color=PALETTE["pink"],  linewidth=2.5, label="Train GT match (smooth)")
    ax1.plot(e_iters, e_step, "o-", color=PALETTE["white"], ms=8, linewidth=2,
             label="Eval step accuracy", zorder=6)

    # annotate eval start/end
    ax1.annotate(f"{e_step[0]:.1f}%",  xy=(e_iters[0], e_step[0]),
                 xytext=(e_iters[0] - 0.3, e_step[0] - 1.2), fontsize=8.5,
                 color=PALETTE["white"], ha="right")
    ax1.annotate(f"{e_step[-1]:.1f}%", xy=(e_iters[-1], e_step[-1]),
                 xytext=(e_iters[-1] + 0.3, e_step[-1] + 0.5), fontsize=8.5,
                 color=PALETTE["white"])
    ax1.annotate("", xy=(e_iters[-1], e_step[-1]),
                 xytext=(e_iters[0], e_step[0]),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["cyan"], lw=1.5,
                                 connectionstyle="arc3,rad=-0.3"))

    ax1.set_title("Step Accuracy  —  Did each reasoning step hold up?",
                  fontsize=9.5, color=PALETTE["slate"], pad=5)
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(55, 105)
    ax1.set_xticks(range(1, max(iters) + 1, 3))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + phase_legend_patches(phases),
               framealpha=0.8, ncol=1, loc="lower right")

    # ── RIGHT: LCCP ──
    shade_phases(ax2, iters, phases)
    ax2.plot(iters, t_lccp,  alpha=0.2,  color=PALETTE["emerald"], linewidth=1)
    ax2.plot(iters, ma_lccp, color=PALETTE["emerald"], linewidth=2.5, label="Train LCCP (smooth)")
    ax2.plot(e_iters, e_lccp, "o-", color=PALETTE["white"], ms=8, linewidth=2,
             label="Eval LCCP", zorder=6)

    ax2.annotate(f"{e_lccp[0]:.1f}%",  xy=(e_iters[0], e_lccp[0]),
                 xytext=(e_iters[0] - 0.3, e_lccp[0] - 1.5), fontsize=8.5,
                 color=PALETTE["white"], ha="right")
    ax2.annotate(f"{e_lccp[-1]:.1f}%", xy=(e_iters[-1], e_lccp[-1]),
                 xytext=(e_iters[-1] + 0.3, e_lccp[-1] + 0.5), fontsize=8.5,
                 color=PALETTE["white"])

    # show LCCP delta
    delta = e_lccp[-1] - e_lccp[0]
    ax2.text(0.97, 0.06,
             f"Eval LCCP  Δ = +{delta:.2f}pp\n(iter {e_iters[0]} → {e_iters[-1]})",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.5, color=PALETTE["emerald"],
             bbox=dict(facecolor=PALETTE["bg2"], edgecolor=PALETTE["emerald"],
                       linewidth=0.8, pad=5))

    ax2.set_title("LCCP  —  Did the chain of reasoning stay correct until the first error?",
                  fontsize=9.5, color=PALETTE["slate"], pad=5)
    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("LCCP (%)")
    ax2.set_ylim(55, 100)
    ax2.set_xticks(range(1, max(iters) + 1, 3))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(handles=ax2.get_legend_handles_labels()[0] + phase_legend_patches(phases),
               framealpha=0.8, ncol=1, loc="lower right")

    fig.tight_layout()
    save(fig, "plot5_reasoning_quality.png", out)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate AxiomForgeAI training plots")
    p.add_argument("--metrics", default=DEFAULT_METRICS,
                   help=f"Path to metrics.csv  (default: {DEFAULT_METRICS})")
    p.add_argument("--out", default="images",
                   help="Output directory for PNGs  (default: images/)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)

    print(f"Loading metrics from  : {args.metrics}")
    print(f"Saving plots to       : {out}/")
    print()

    rows = load_csv(args.metrics)
    print(f"Loaded {len(rows)} iterations.\n")

    print("Generating plots …")
    plot_eval_quality(rows, out)
    plot_training_journey(rows, out)
    plot_selfplay_success(rows, out)
    plot_reward_confidence(rows, out)
    plot_reasoning_quality(rows, out)

    print(f"\n✅  All 5 plots saved to {out}/")
    print("\nFiles:")
    for p in sorted(out.glob("plot*.png")):
        print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
