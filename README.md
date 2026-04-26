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

The environment is a practice loop for math reasoning. Each training group starts with one problem, asks the model for multiple solution attempts, scores those attempts from several angles, and uses GRPO to reinforce the attempts that are stronger than the rest of the group.

![AxiomForgeAI environment overview](images/environment_overview.svg)

The environment has two task sources:

- **Grounded source:** A dataset problem from GSM8K / MATH comes with a known final answer. This gives the environment a reliable anchor for checking whether the model actually reached the right result.
- **Self-play source:** The curriculum selects a target skill and difficulty. The model writes a new question, then samples multiple solutions to that question. This adds practice beyond static datasets, but only after the grounded signal is stable enough.

Both sources feed the same scoring and update loop. For every selected problem, the model samples `K` candidate solutions. The environment checks final-answer correctness when a gold answer exists, scores reasoning quality with a PRM, checks chain consistency and symbolic arithmetic where possible, checks answer formatting, and scores self-generated questions for clarity, novelty, difficulty fit, and solvability.

GRPO then compares the `K` attempts against each other. The model is not rewarded for a solution in isolation; the strongest attempt in the group becomes the direction for learning. Training starts grounded-only, gradually mixes in self-play groups, and falls back to grounded practice if generated-question quality or answer correctness drops.

## How Self-Improvement Works

Self-improvement comes from turning each problem into a small comparison. The model does not produce one solution and move on; the environment samples several attempts, scores each attempt, and asks which reasoning path was strongest.

GRPO uses that within-group comparison as the learning signal. Attempts with correct answers, stronger reasoning chains, and cleaner final-answer format are reinforced. Attempts with broken chains or unsupported answers become weaker examples.

```text
practice -> sample attempts -> verify steps and answer -> compare -> reinforce -> adjust difficulty
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
