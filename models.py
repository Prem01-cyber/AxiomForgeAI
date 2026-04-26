# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AxiomForgeAI math RL environment.

The AxiomForgeAI environment presents math questions drawn from an adaptive
curriculum; external agents submit step-by-step solutions and receive scored
observations.  The environment integrates with the GRPO training pipeline
defined in scripts/run_grpo_training.py.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AxiomforgeaiAction(Action):
    """Action for the AxiomForgeAI math environment.

    The agent submits a step-by-step solution to the current question.
    Solutions should follow the format::

        Step 1: <reasoning>
        Step 2: <reasoning>
        ...
        Final Answer: <numeric value>
    """

    solution: str = Field(
        default="",
        description=(
            "Step-by-step solution to the current math question. "
            "Use 'Step N: ...' lines and end with 'Final Answer: <value>'."
        ),
    )


class AxiomforgeaiObservation(Observation):
    """Observation from the AxiomForgeAI math environment.

    On reset the question is populated and reward/feedback are empty.
    After a step the reward and feedback reflect the quality of the submitted
    solution; done=True signals the end of the single-step episode.
    """

    question: str = Field(
        default="",
        description="Math question the agent must solve.",
    )
    topic: str = Field(
        default="",
        description="Mathematical topic of the question (e.g. 'algebra', 'geometry').",
    )
    difficulty: float = Field(
        default=0.5,
        description="Estimated difficulty of the question in [0, 1].",
    )
    feedback: str = Field(
        default="",
        description=(
            "Human-readable feedback on the submitted solution "
            "(empty on reset, populated after step)."
        ),
    )
