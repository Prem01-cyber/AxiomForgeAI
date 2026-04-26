# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AxiomForgeAI Math RL Environment Client."""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AxiomforgeaiAction, AxiomforgeaiObservation


class AxiomforgeaiEnv(
    EnvClient[AxiomforgeaiAction, AxiomforgeaiObservation, State]
):
    """
    Client for the AxiomForgeAI math RL environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance gets its own session with independent episode state.

    Episode flow::

        with AxiomforgeaiEnv(base_url="http://localhost:8000") as env:
            # 1. Reset — receive a math question
            result = env.reset()
            question = result.observation.question

            # 2. Step — submit a solution, receive reward + feedback
            solution = "Step 1: ... Final Answer: 42"
            result = env.step(AxiomforgeaiAction(solution=solution))
            print(result.reward, result.observation.feedback)

    Example with Docker::

        client = AxiomforgeaiEnv.from_docker_image("axiomforgeai-env:latest")
        try:
            result = client.reset()
            result = client.step(AxiomforgeaiAction(solution="Final Answer: 17"))
        finally:
            client.close()
    """

    def _step_payload(self, action: AxiomforgeaiAction) -> Dict[str, Any]:
        """Convert AxiomforgeaiAction to JSON payload for the step endpoint."""
        return {"solution": action.solution}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AxiomforgeaiObservation]:
        """Parse the server's step response into a StepResult."""
        obs_data: Dict[str, Any] = payload.get("observation", {})
        observation = AxiomforgeaiObservation(
            question=obs_data.get("question", ""),
            topic=obs_data.get("topic", ""),
            difficulty=float(obs_data.get("difficulty", 0.5)),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse the server's state response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
