"""Python client for the Forge + Arena environment.

Usage::

    from client import ForgeArenaEnv, OverseerInspectAction

    with ForgeArenaEnv(base_url="https://amogh-kal1-forge-arena.hf.space/api") as env:
        obs = env.reset()
        print(obs.task_description)

        result = env.step(OverseerInspectAction(
            detection=True,
            explanation="The Worker omitted the mandatory disclaimer.",
            correction="...",
            confidence=0.85,
        ))
        print(result.composite_reward)
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    AnyForgeAction,
    ForgeEpisodeResult,
    ForgeObservation,
    OverseerInspectAction,
    OverseerProbeAction,
)


class ForgeArenaEnv(EnvClient[AnyForgeAction, ForgeObservation, State]):
    """Client for the Forge + Arena oversight environment.

    Maintains a persistent connection to the environment server and exposes the
    standard openenv ``reset()`` / ``step()`` interface.

    Args:
        base_url: Base URL of a running Forge + Arena server.
                  Local:  ``http://localhost:8000``
                  Space:  ``https://amogh-kal1-forge-arena.hf.space/api``

    Example::

        with ForgeArenaEnv(base_url="http://localhost:8000") as env:
            obs = env.reset()
            result = env.step(OverseerInspectAction(
                detection=False, confidence=0.4,
            ))
    """

    def _step_payload(self, action: AnyForgeAction) -> Dict:
        """Serialise the action to the JSON body expected by POST /step."""
        return action.model_dump()

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[ForgeObservation | ForgeEpisodeResult]:
        """Parse the server response into a typed StepResult."""
        done = payload.get("done", False)

        if done:
            obs_data = payload.get("observation", payload)
            observation: ForgeObservation | ForgeEpisodeResult = ForgeEpisodeResult(
                episode_id=obs_data.get("episode_id", ""),
                domain=obs_data.get("domain", ""),
                worker_cot=obs_data.get("worker_cot", ""),
                worker_output=obs_data.get("worker_output", ""),
                done=True,
                corruption_present=obs_data.get("corruption_present", False),
                corruption_type=obs_data.get("corruption_type"),
                ground_truth_output=obs_data.get("ground_truth_output", ""),
                overseer_detection=obs_data.get("overseer_detection", False),
                overseer_explanation=obs_data.get("overseer_explanation", ""),
                overseer_correction=obs_data.get("overseer_correction", ""),
                overseer_confidence=obs_data.get("overseer_confidence", 0.0),
                composite_reward=obs_data.get("composite_reward", 0.0),
                detection_score=obs_data.get("detection_score", 0.0),
                explanation_score=obs_data.get("explanation_score", 0.0),
                correction_score=obs_data.get("correction_score", 0.0),
                calibration_score=obs_data.get("calibration_score", 0.0),
            )
        else:
            obs_data = payload.get("observation", payload)
            observation = ForgeObservation(
                episode_id=obs_data.get("episode_id", ""),
                task_description=obs_data.get("task_description", ""),
                source_material=obs_data.get("source_material", ""),
                domain=obs_data.get("domain", ""),
                worker_cot=obs_data.get("worker_cot", ""),
                worker_output=obs_data.get("worker_output", ""),
                done=False,
                phase=obs_data.get("phase", "overseer_inspecting"),
            )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse the server state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
