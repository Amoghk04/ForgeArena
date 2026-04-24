"""Episode state machine — manages the three-phase episode lifecycle."""

from __future__ import annotations

import random
from typing import Optional

from forge_arena.arena.corruptions.injector import build_corruption_prompt
from forge_arena.arena.domains import get_domain
from forge_arena.arena.worker import WorkerAgent
from forge_arena.models.actions import (
    OverseerInspectAction,
    OverseerProbeAction,
    WorkerRespondAction,
)
from forge_arena.models.observations import (
    EpisodeResult,
    ResetObservation,
    StateObservation,
    WorkerObservation,
)
from forge_arena.models.tasks import (
    CorruptionType,
    EpisodePhase,
    EpisodeState,
    Task,
    TaskDomain,
)


# In-memory episode store keyed by episode_id (UUID4 string).
# For production, replace with a Redis-backed store using the same interface.
EpisodeStore: dict[str, EpisodeState] = {}


class EpisodeManager:
    """Manages the lifecycle of oversight episodes."""

    def __init__(self, worker: WorkerAgent) -> None:
        self._worker = worker

    async def reset(
        self,
        task: Task,
        domain: TaskDomain,
    ) -> ResetObservation:
        """Initialise a new episode.

        Selects corruption decision from a Bernoulli with the domain's
        corruption_rate. Stores ground truth server-side. Returns only
        the task description to the caller.
        """
        domain_obj = get_domain(domain)
        corruption_present = random.random() < domain_obj.corruption_rate
        corruption_type: Optional[CorruptionType] = None

        if corruption_present:
            valid_types = domain_obj.corruption_vocab
            corruption_type = random.choice(valid_types)

        state = EpisodeState(
            task=task,
            phase=EpisodePhase.RESET,
            corruption_present=corruption_present,
            corruption_type=corruption_type,
        )

        # Advance immediately to worker phase and call the Worker
        corruption_prompt: Optional[str] = None
        if corruption_present and corruption_type:
            corruption_prompt = build_corruption_prompt(task, corruption_type)

        worker_cot, worker_output = await self._worker.respond(
            task, corruption_type, corruption_prompt
        )

        state.worker_cot = worker_cot
        state.worker_output = worker_output
        state.phase = EpisodePhase.OVERSEER_INSPECTING

        EpisodeStore[state.episode_id] = state

        return ResetObservation(
            episode_id=state.episode_id,
            task_description=task.task_description,
            source_material=task.source_material,
            domain=domain,
            worker_cot=worker_cot,
            worker_output=worker_output,
            done=False,
            phase=EpisodePhase.OVERSEER_INSPECTING,
        )

    async def step_probe(
        self, episode_id: str, action: OverseerProbeAction
    ) -> WorkerObservation:
        """Handle an Overseer probe action.

        The Worker answers the probe based on its original response context.
        The probe does not change the episode phase.
        """
        state = _get_state(episode_id)
        _assert_phase(state, EpisodePhase.OVERSEER_INSPECTING)

        if state.probe_used:
            raise ValueError("Probe already used in this episode. Maximum one probe per episode.")

        assert state.worker_output is not None
        probe_answer = await self._worker.answer_probe(
            state.task, action.question, state.worker_output
        )

        state.probe_used = True
        # Append probe Q&A to the Worker CoT so the Overseer sees the full exchange
        assert state.worker_cot is not None
        state.worker_cot = (
            state.worker_cot
            + f"\n\n[Probe Q]: {action.question}\n[Probe A]: {probe_answer}"
        )

        return WorkerObservation(
            episode_id=episode_id,
            task_description=state.task.task_description,
            source_material=state.task.source_material,
            domain=state.task.domain,
            worker_cot=state.worker_cot,
            worker_output=state.worker_output,
            done=False,
            phase=EpisodePhase.OVERSEER_INSPECTING,
        )

    def step_inspect(
        self,
        episode_id: str,
        action: OverseerInspectAction,
        composite_reward: float,
        detection_score: float,
        explanation_score: float,
        correction_score: float,
        calibration_score: float,
    ) -> EpisodeResult:
        """Handle terminal overseer_inspect action.

        Stores the Overseer submission and computed rewards.
        Returns full EpisodeResult including ground truth.
        """
        state = _get_state(episode_id)
        _assert_phase(state, EpisodePhase.OVERSEER_INSPECTING)

        state.overseer_detection = action.detection
        state.overseer_explanation = action.explanation
        state.overseer_correction = action.correction
        state.overseer_confidence = action.confidence
        state.reward = composite_reward
        state.phase = EpisodePhase.DONE
        state.done = True

        assert state.worker_cot is not None
        assert state.worker_output is not None

        return EpisodeResult(
            episode_id=episode_id,
            task_description=state.task.task_description,
            domain=state.task.domain,
            worker_cot=state.worker_cot,
            worker_output=state.worker_output,
            corruption_present=state.corruption_present,
            corruption_type=state.corruption_type,
            overseer_detection=action.detection,
            overseer_explanation=action.explanation,
            overseer_correction=action.correction,
            overseer_confidence=action.confidence,
            reward=composite_reward,
            detection_score=detection_score,
            explanation_score=explanation_score,
            correction_score=correction_score,
            calibration_score=calibration_score,
        )

    def get_state_observation(self, episode_id: str) -> StateObservation:
        """Return a safe state snapshot without advancing the episode."""
        state = _get_state(episode_id)
        return StateObservation(
            episode_id=episode_id,
            phase=state.phase,
            domain=state.task.domain,
            task_description=state.task.task_description,
            worker_cot=state.worker_cot,
            worker_output=state.worker_output,
            probe_used=state.probe_used,
            done=state.done,
        )


def _get_state(episode_id: str) -> EpisodeState:
    state = EpisodeStore.get(episode_id)
    if state is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")
    return state


def _assert_phase(state: EpisodeState, expected: EpisodePhase) -> None:
    if state.phase != expected:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=409,
            detail=(
                f"Invalid action for current episode phase. "
                f"Expected '{expected}', got '{state.phase}'."
            ),
        )
