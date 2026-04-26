# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AxiomForgeAI Math RL Environment.

Wraps CurriculumMathEnvironment from src/rl/math_environment_curriculum.py
to expose an OpenEnv-compatible interface (reset / step / state).

Episode semantics
-----------------
* reset()  — Samples a new question from the adaptive curriculum (or a
             grounded QA pair when a dataset is configured).  Returns the
             question in the observation; reward is 0.0.
* step(action) — Scores the agent's submitted solution with the full reward
             pipeline (PRM + SymPy + format) and returns reward + feedback.
             done=True always: one question per episode.

Environment variables
---------------------
AXIOMFORGE_DATA_PATH   Path to a JSONL file with {"question", "gold_final"}
                       records (e.g. data/sft/gsm8k_sft.jsonl).  When set,
                       the environment uses grounded QA pairs for questions
                       and ground-truth answer verification.

AXIOMFORGE_PRM_PATH    HuggingFace model ID or local path for the Process
                       Reward Model (default: Qwen/Qwen2.5-Math-PRM-7B).
                       Set to "" to disable PRM scoring (uses SymPy only).

AXIOMFORGE_CURRICULUM_DIR
                       Directory where the CurriculumManager persists its
                       state between runs.  Defaults to
                       "checkpoints/curriculum".
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AxiomforgeaiAction, AxiomforgeaiObservation

except ImportError:
    from models import AxiomforgeaiAction, AxiomforgeaiObservation

# ── Heavy RL imports — fail gracefully so openenv validate passes even when
#    the ML stack is not installed (e.g. lightweight CI / schema validation).
try:
    from src.rl.math_environment_curriculum import CurriculumMathEnvironment
    from src.rl.prm_scorer import ProcessRewardScorer
    from src.sft.solution_format import extract_final_answer_numeric_str

    _RL_AVAILABLE = True
except Exception as _rl_import_err:  # pragma: no cover
    _RL_AVAILABLE = False
    CurriculumMathEnvironment = None  # type: ignore[assignment,misc]
    ProcessRewardScorer = None  # type: ignore[assignment,misc]
    extract_final_answer_numeric_str = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

# Fallback question used during validation / when no dataset is configured.
_VALIDATION_QUESTION = (
    "A store sells apples for $2 each and oranges for $3 each. "
    "If Sarah buys 4 apples and 3 oranges, how much does she spend in total?"
)
_VALIDATION_GOLD = "17"
_VALIDATION_TOPIC = "basic_arithmetic"
_VALIDATION_DIFFICULTY = 0.1


def _load_qa_pairs(data_path: str) -> List[Dict[str, str]]:
    """Load {"question", "gold_final"} records from a JSONL file."""
    pairs: List[Dict[str, str]] = []
    p = Path(data_path)
    if not p.exists():
        logger.warning("AXIOMFORGE_DATA_PATH not found: %s", data_path)
        return pairs
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = rec.get("question", "").strip()
            g = rec.get("gold_final", "").strip()
            if q and g:
                pairs.append({"question": q, "gold_final": g})
    logger.info("Loaded %d QA pairs from %s", len(pairs), data_path)
    return pairs


class AxiomforgeaiEnvironment(Environment):
    """
    AxiomForgeAI math RL environment for OpenEnv.

    Uses CurriculumMathEnvironment from src/rl/ for adaptive question
    selection and reward computation.  When the ML stack is unavailable
    (e.g. during schema validation), falls back to a lightweight mode
    that uses only the installed openenv-core dependencies.

    Supports concurrent WebSocket sessions — each client gets its own
    instance with independent episode state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Per-episode state
        self._current_question: str = ""
        self._gold_final: str = ""
        self._current_topic: str = ""
        self._current_difficulty: float = 0.5

        self._math_env: Optional[Any] = None  # CurriculumMathEnvironment or None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not _RL_AVAILABLE:
            logger.warning(
                "RL stack (torch/transformers/sympy) not available — "
                "running in schema-validation mode with fixed fallback responses."
            )
            return

        # ── Load grounded QA pairs (optional) ─────────────────────────────
        grounded_qa_pairs: List[Dict[str, str]] = []
        data_path = os.environ.get("AXIOMFORGE_DATA_PATH", "")
        if data_path:
            grounded_qa_pairs = _load_qa_pairs(data_path)

        # ── Load PRM scorer (optional) ────────────────────────────────────
        prm: Optional[Any] = None  # ProcessRewardScorer or None
        prm_path = os.environ.get("AXIOMFORGE_PRM_PATH", "")
        if prm_path:
            try:
                prm = ProcessRewardScorer(
                    model_name=prm_path,
                    device=device,
                    load_in_4bit=True,
                )
                logger.info("PRM loaded: %s", prm_path)
            except Exception as exc:
                logger.warning("PRM load failed (%s) — scoring uses SymPy only.", exc)

        # ── Create CurriculumMathEnvironment in scoring-only mode ─────────
        # policy_model=None + tokenizer=None is safe when only reward-computation
        # methods are called (compute_grounded_reward, sample_instruction).
        # Generation methods (generate_with_logging, format_solution_prompt)
        # are NOT called from the server step path — the agent supplies solutions.
        curriculum_dir = os.environ.get(
            "AXIOMFORGE_CURRICULUM_DIR", "checkpoints/curriculum"
        )
        try:
            self._math_env = CurriculumMathEnvironment(
                policy_model=None,
                value_model=None,
                tokenizer=None,
                reference_questions=[qa["question"] for qa in grounded_qa_pairs],
                grounded_qa_pairs=grounded_qa_pairs,
                prm_scorer=prm,
                curriculum_checkpoint_dir=curriculum_dir,
                device=device,
            )
            logger.info(
                "CurriculumMathEnvironment ready (scoring-only, %d QA pairs, PRM=%s)",
                len(grounded_qa_pairs),
                "yes" if prm else "no",
            )
        except Exception as exc:
            logger.warning(
                "CurriculumMathEnvironment init failed (%s) — "
                "falling back to validation mode.",
                exc,
            )
            self._math_env = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        qa: Optional[Dict[str, str]] = None,
    ) -> AxiomforgeaiObservation:
        """
        Reset the environment and begin a new episode.

        Args:
            qa: Optional ``{"question": str, "gold_final": str}`` dict.
                When supplied the environment is seeded with this specific
                question and gold answer — used by the training loop for
                difficulty-sampled grounded episodes.  When omitted the
                environment draws from its internal grounded QA pool (if
                configured) or falls back to the curriculum instruction.

        Returns:
            AxiomforgeaiObservation with the question populated; reward=0.0.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if qa is not None:
            # Caller-supplied episode — honour it exactly.
            self._current_question = qa.get("question", "").strip()
            self._gold_final = qa.get("gold_final", "").strip()
            self._current_topic = qa.get("topic", "grounded")
            self._current_difficulty = float(qa.get("difficulty", 0.5))
        elif self._math_env is not None:
            try:
                instruction, topic, difficulty = self._math_env.sample_instruction()
                self._current_topic = topic
                self._current_difficulty = float(difficulty)
                if self._math_env.grounded_qa_pairs:
                    _qa = random.choice(self._math_env.grounded_qa_pairs)
                    self._current_question = _qa["question"]
                    self._gold_final = _qa["gold_final"]
                else:
                    self._current_question = instruction
                    self._gold_final = ""
            except Exception as exc:
                logger.warning("sample_instruction failed, using fallback: %s", exc)
                self._current_question = _VALIDATION_QUESTION
                self._gold_final = _VALIDATION_GOLD
                self._current_topic = _VALIDATION_TOPIC
                self._current_difficulty = _VALIDATION_DIFFICULTY
        else:
            self._current_question = _VALIDATION_QUESTION
            self._gold_final = _VALIDATION_GOLD
            self._current_topic = _VALIDATION_TOPIC
            self._current_difficulty = _VALIDATION_DIFFICULTY

        return AxiomforgeaiObservation(
            question=self._current_question,
            topic=self._current_topic,
            difficulty=self._current_difficulty,
            feedback="",
            done=False,
            reward=0.0,
        )

    def step(self, action: AxiomforgeaiAction) -> AxiomforgeaiObservation:  # type: ignore[override]
        """
        Score the agent's submitted solution.

        Uses compute_grounded_reward from CurriculumMathEnvironment when
        available (PRM + SymPy + format scoring).  Falls back to numeric
        answer extraction when the full RL stack is not loaded.

        Args:
            action: AxiomforgeaiAction containing the solution text.

        Returns:
            AxiomforgeaiObservation with reward, feedback, and metadata.
            done=True — one question per episode.
        """
        self._state.step_count += 1
        solution = action.solution

        reward: float = 0.0
        feedback: str = ""
        metadata: Dict[str, Any] = {}

        if self._math_env is not None and self._current_question:
            try:
                reward_result = self._math_env.compute_grounded_reward(
                    question=self._current_question,
                    solution=solution,
                    gold_final=self._gold_final,
                )
                reward = float(reward_result.get("combined_score", 0.0))
                gt = reward_result.get("gt_match", False)
                step_acc = reward_result.get("step_accuracy", 0.0)
                lccp = reward_result.get("lccp", 0.0)
                pred = reward_result.get("pred_final", "")
                feedback = (
                    f"gt_match={gt} pred={pred!r} gold={self._gold_final!r} "
                    f"step_acc={step_acc:.2f} lccp={lccp:.2f}"
                )
                # Serialise reward breakdown into metadata; skip non-serialisable lists.
                metadata = {
                    k: v
                    for k, v in reward_result.items()
                    if not isinstance(v, list)
                }
            except Exception as exc:
                logger.warning("compute_grounded_reward failed: %s", exc)
                reward, feedback, metadata = self._fallback_score(solution)
        else:
            reward, feedback, metadata = self._fallback_score(solution)

        return AxiomforgeaiObservation(
            question=self._current_question,
            topic=self._current_topic,
            difficulty=self._current_difficulty,
            feedback=feedback,
            done=True,
            reward=reward,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_score(
        self, solution: str
    ) -> tuple[float, str, Dict[str, Any]]:
        """Lightweight scoring used when the full RL stack is unavailable."""
        pred: str = ""
        if extract_final_answer_numeric_str is not None:
            pred = extract_final_answer_numeric_str(solution) or ""
        reward = 1.0 if pred and pred == self._gold_final else 0.0
        feedback = f"pred={pred!r} gold={self._gold_final!r}"
        return reward, feedback, {"pred_final": pred, "gold_final": self._gold_final}

    def close(self) -> None:
        """
        Persist curriculum state and release resources.

        Call once at the end of a training run so the CurriculumManager's
        per-topic statistics are saved to disk and can be resumed on the
        next run.  Safe to call multiple times.
        """
        if self._math_env is not None:
            try:
                self._math_env.curriculum_manager.save_state(
                    iteration=self._math_env.curriculum_manager.current_iteration,
                    rollout=None,
                )
                logger.info(
                    "Curriculum state saved (iteration %d).",
                    self._math_env.curriculum_manager.current_iteration,
                )
            except Exception as exc:
                logger.warning("close(): curriculum save failed — %s", exc)

    @property
    def state(self) -> State:
        """Return the current episode state (episode_id + step_count)."""
        return self._state
