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

The reward is designed to avoid a common math-training failure: optimizing for either the final answer or the reasoning trace alone. A good solution should reach the right answer, explain the path clearly, and keep the final result consistent with the steps that produced it.

| Signal | What it checks | Why it matters |
| --- | --- | --- |
| Final answer | Matches the gold answer when one exists | Keeps grounded problems tied to objective correctness |
| Process score | PRM score over the reasoning steps | Rewards clear mathematical progress, not just the last line |
| Chain consistency | Correct-prefix and step-answer consistency signals | Gives partial learning signal when a solution goes wrong midway |
| Format | Parseable final answer and clean response structure | Makes automatic grading reliable |
| Question quality | Topic fit, difficulty fit, clarity, novelty, and solvability | Keeps self-play from generating vague or useless practice tasks |

Grounded problems use the gold answer as the anchor. Self-play problems add a question-quality score before the solution reward is trusted. Both paths produce one combined score for each sampled attempt, and GRPO uses those scores only in comparison with the other attempts from the same problem.

```text
grounded: answer correctness + process score + chain consistency + format
self-play: question quality + solution quality
both -> one combined score per attempt -> GRPO compares attempts within the group
```

## Training Phases

Training follows a simple three-phase schedule. It starts with grounded-only practice so the model learns to keep answers and reasoning stable on problems with known solutions. Self-play is then introduced gradually, while grounded questions remain as an anchor. Once both are stable, training continues with a mixed task source and falls back to grounded-only batches if answer quality drops.

![Training phases overview](images/training_phases.svg)

## Training Script

The GRPO training loop is available in two forms:

- [`scripts/launch_grpo.sh`](scripts/launch_grpo.sh) — the primary launch script; sets CUDA/threading env vars, verifies Flash-Attention, and calls `run_grpo_training.py` with the full parameter set.
- [`train_grpo.ipynb`](train_grpo.ipynb) — notebook version with the same parameters, structured around `env.reset / env.step / env.state / env.close` for interactive inspection.

```bash
bash scripts/launch_grpo.sh
```

## Results

These plots are from a single GPU training run. Each one measures something most models are never asked to prove.

---

### Evaluation Quality Over Training

![Evaluation quality over training](images/plot1_eval_quality.png)

Most models get evaluated on one thing: did the final answer match. This environment evaluates four things at once — final correctness, overall solution quality, per-step validity, and how far the reasoning held before it broke. All four trend upward together. That does not happen by accident. It happens because the reward was designed to refuse partial credit: a correct answer built on broken reasoning is penalised, not rewarded.

---

### Training Journey

![Training journey across all 30 iterations](images/plot2_training_journey.png)

The two background colours mark two different modes of learning. The first phase is grounded practice on real problems with known answers. The second phase introduces model-generated problems — but only after the model demonstrated it could hold its own on verified material. The transition was not scheduled. It was conditional. The model had to earn the right to train on problems it wrote itself. That distinction separates a principled curriculum from a training loop that just gets harder on a timer.

---

### Self-Play Curriculum

![Self-play curriculum ramp and question quality](images/plot3_selfplay_success.png)

By the end of training, the majority of the practice material came from the model itself. What makes this worth showing is not the quantity — it is the quality. The problems the model generated were consistently solvable and consistently novel. The model was not recycling what it had seen. It was constructing new problems, attempting them, and learning from the attempt. A model that can teach itself is fundamentally different from one that needs to be taught.

---

### Reward Confidence

![Reward confidence and skipped groups](images/plot4_reward_confidence.png)

The band in this chart shows the spread between the model's best and worst attempts on each problem. A wide spread is actually useful — it means there is something to learn from comparing strong and weak solutions. A narrow spread that appears briefly means the model has converged hard on that problem class: it knows what to do. Two such moments of near-total confidence appear during the run. They are not the end goal, but they are evidence that the reward signal is teaching something real. The bar chart below it shows the proportion of problems where the model's attempts were so similar that no useful comparison could be made — those groups are skipped. That rate falls as the curriculum introduces harder material, which is exactly the intended behaviour.

---

### Step-Level Reasoning Quality

![Step accuracy and LCCP across training](images/plot5_reasoning_quality.png)

This is the plot that justifies the entire environment design. Step accuracy asks whether each line of reasoning in a solution is valid. Chain integrity asks whether the valid steps form an unbroken sequence from the first line to the answer. A model that gets individual steps right but loses the thread partway through is not reasoning — it is pattern-matching. Both signals improve together across held-out evaluation, which means the model is not just producing better-looking outputs. It is building solutions that hold together from start to finish.

---

## Why It Matters

Reliable math reasoning needs more than fluent explanations or lucky final answers. A system that can separate correct reasoning from unsupported answers gives the model a better training target: not just "get the number," but build a chain of logic that reaches the number.

AxiomForgeAI matters because it turns that target into an environment. The same pattern can extend beyond math to other verifiable domains where attempts can be checked, compared, and improved: code, logic, structured data transformations, and scientific problem solving.

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
