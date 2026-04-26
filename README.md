---
title: AxiomForgeAI Environment Server
emoji: 🌌
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AxiomForgeAI

[![OpenEnv](https://img.shields.io/badge/Powered%20by-OpenEnv-blue)](https://github.com/meta-pytorch/OpenEnv)

*A self-improving math environment where a model practices on verified problems, generates new challenges when ready, and learns from solution attempts whose reasoning steps and final answers agree.*

## The Problem

Math reasoning models can fail in two different ways. Sometimes the setup, arithmetic, and algebraic steps look reasonable, but the final answer is wrong. Sometimes the final answer is right, but the reasoning that produced it is incomplete, inconsistent, or hard to trust.

For a math user, both failures matter. Checking only the final answer misses where the solution went off track. Checking only the steps misses whether the work actually reaches the right result. The useful signal is the agreement between the reasoning path and the final answer.

This project builds a practice loop around that signal. The model first works on problems with known answers, gets feedback on both the chain of reasoning and the final result, and only then starts generating new challenges for itself. The constraint is intentionally small: a 1.5B math model.

## The Environment

The agent trains on two kinds of tasks: grounded math problems with known answers, and self-generated challenges from a curriculum.

In the grounded lane, the environment samples a dataset problem from sources such as GSM8K or MATH and keeps the gold final answer available for verification. In the self-play lane, the curriculum selects a target skill and difficulty, then the model generates a new question before attempting to solve it.

Both lanes then follow the same learning shape: sample multiple solution attempts, grade them with independent signals, compare attempts inside the group, and update the policy with GRPO. Training starts grounded-only, gradually adds self-play groups by ratio, and falls back to grounded signal when generated-question quality or answer correctness drops.

Diagram source: [`docs/environment-overview.puml`](docs/environment-overview.puml)

## How Self-Improvement Works

AxiomForgeAI treats reasoning as practice, not a one-shot answer. Each problem produces a group of candidate solutions. The reward separates answer correctness, step quality, chain consistency, and parseable final-answer format.

GRPO compares attempts within the same problem group. Stronger attempts receive positive relative signal, weaker attempts receive less, and groups with no useful reward difference can be skipped. Wrong final answers can still provide limited learning signal when the reasoning chain is partially correct, but correct and consistent solutions remain the target.

The loop is intentionally simple:

```text
practice -> generate multiple attempts -> verify -> compare -> reinforce -> adapt difficulty -> repeat
```

## Reward System

The reward is built from separate checks so the model is not trained on a single fragile signal.

- **Answer check:** when a gold answer exists, the final numeric answer is compared against the expected result.
- **Process reward:** a PRM scores the quality of intermediate reasoning steps.
- **Symbolic verification:** arithmetic and parseable expressions are normalized and checked with SymPy where possible.
- **Formatting:** solutions are rewarded for producing clear, parseable final answers.
- **Question quality:** self-generated problems are scored for topic fit, clarity, target difficulty, novelty, and solvability.

Diagram source: [`docs/reward-system.puml`](docs/reward-system.puml)

## Training Phases

The system is designed to start from stable feedback before asking the model to rely heavily on its own generated tasks. Grounded warmup gives the policy a reliable base signal, self-play ramps in curriculum-generated questions, and quality checks can push training back toward grounded data if generated questions become unclear or unsolvable.

Diagram source: [`docs/training-phases.puml`](docs/training-phases.puml)

## Results

This repository includes the training and demo paths for reporting results, but no completed `metrics.jsonl`, plots, or before/after demo output are currently committed. To avoid unsupported claims, this README does not report accuracy or reward improvements yet.

The GRPO trainer writes run logs and metrics for reward, grounded accuracy, curriculum behavior, and evaluation checkpoints. The demo script can read a trained checkpoint plus `metrics.jsonl` and produce a judge-friendly before/after comparison with example solutions.

```bash
python scripts/run_grpo_training.py \
  --base-model checkpoints/dual_task_v1 \
  --gsm8k-data data/sft/gsm8k_test.jsonl \
  --num-iterations 3 \
  --group-size 4 \
  --questions-per-iter 8 \
  --no-prm \
  --skip-initial-eval \
  --run-name smoke_grpo
```

```bash
python scripts/demo_before_after.py \
  --baseline-model checkpoints/dual_task_v1 \
  --trained-model checkpoints/grpo/<run>/best_policy \
  --metrics-jsonl checkpoints/grpo/<run>/metrics.jsonl \
  --problems data/sft/gsm8k_test.jsonl \
  --max-samples 100 \
  --records-out results/demo.json
```

## Why It Matters

Low-cost self-improvement for compact math models matters because many useful deployments cannot assume large hosted models, expensive training loops, or public data movement. A 1.5B parameter model is small enough to make local experimentation realistic, but still capable enough to expose the hard part: reasoning does not improve unless the system can turn mistakes into structured practice.

AxiomForgeAI is also a reusable pattern. Math is the first domain because verification is unusually clear, but the same environment idea applies to other tasks where attempts can be checked, compared, and turned into reward: code, logic, structured data transformations, and scientific problem solving.

## Quick Start

AxiomForgeAI is built on the OpenEnv standard and exposes a Gym-style `reset` / `step` interface.

```bash
docker build -t AxiomForgeAI-env:latest -f server/Dockerfile .
```

```bash
openenv push --namespace your-hf-username
```

The deployed Space provides a `/ws` WebSocket endpoint for RL training loops.

---
*Engineered for the OpenEnv Hackathon India 2026*
