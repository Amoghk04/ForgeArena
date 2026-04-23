"""Tests for the episode lifecycle — state machine, phase transitions, ground truth security."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from forge_arena.models.tasks import (
    CorruptionType,
    EpisodePhase,
    ObfuscationDepth,
    Task,
    TaskDomain,
)
from forge_arena.models.actions import OverseerInspectAction


def _make_task(task_id: str = "task-cs-test") -> Task:
    return Task(
        id=task_id,
        domain=TaskDomain.CUSTOMER_SUPPORT,
        task_description="Test task description.",
        source_material="Test context.",
        clean_worker_output="Correct output.",
        corruption_sophistication=0.4,
        obfuscation_depth=ObfuscationDepth.LOW,
        is_generated=False,
    )


def _build_manager():
    """Build a minimal EpisodeManager with mocked Worker."""
    from forge_arena.arena.domains import init_domain_registry
    from forge_arena.arena.episode import EpisodeManager

    # Domain registry must be initialised before EpisodeManager.reset() can work.
    # Pass empty list — domain class attributes (corruption_rate, etc.) are enough for tests.
    init_domain_registry([])

    worker = AsyncMock()
    worker.respond.return_value = ("<think>thinking step</think>", "Worker output.")

    # EpisodeManager only takes a worker — all other dependencies come from domain registry
    manager = EpisodeManager(worker=worker)
    return manager, worker


class TestEpisodeStateTransitions:
    """Unit tests for episode phase transitions with a mocked Worker."""

    @pytest.mark.asyncio
    async def test_reset_returns_reset_observation(self):
        manager, _ = _build_manager()
        obs = await manager.reset(_make_task(), TaskDomain.CUSTOMER_SUPPORT)
        assert obs.episode_id is not None
        assert obs.task_description == "Test task description."

    @pytest.mark.asyncio
    async def test_reset_does_not_leak_ground_truth(self):
        manager, _ = _build_manager()
        obs = await manager.reset(_make_task(), TaskDomain.CUSTOMER_SUPPORT)
        obs_dict = obs.model_dump()
        # These keys must never appear in the reset observation
        assert "corruption_present" not in obs_dict
        assert "corruption_type" not in obs_dict

    @pytest.mark.asyncio
    async def test_state_observation_does_not_leak_ground_truth(self):
        manager, _ = _build_manager()
        obs = await manager.reset(_make_task(), TaskDomain.CUSTOMER_SUPPORT)
        state = manager.get_state_observation(obs.episode_id)
        assert state is not None
        state_dict = state.model_dump()
        assert "corruption_present" not in state_dict
        assert "corruption_type" not in state_dict

    @pytest.mark.asyncio
    async def test_inspect_returns_episode_result_with_ground_truth(self):
        manager, _ = _build_manager()
        reset_obs = await manager.reset(_make_task(), TaskDomain.CUSTOMER_SUPPORT)

        action = OverseerInspectAction(
            detection=True,
            confidence=0.8,
            explanation="I detected a factual omission.",
            correction="Corrected output.",
        )
        result = manager.step_inspect(
            episode_id=reset_obs.episode_id,
            action=action,
            composite_reward=0.75,
            detection_score=1.0,
            explanation_score=0.6,
            correction_score=0.7,
            calibration_score=0.8,
        )
        # After terminal action, ground truth IS revealed
        assert hasattr(result, "corruption_present")
        assert result.done is True

    @pytest.mark.asyncio
    async def test_unknown_episode_id_raises(self):
        from fastapi import HTTPException

        manager, _ = _build_manager()
        action = OverseerInspectAction(
            detection=False,
            confidence=0.5,
            explanation="",
            correction="",
        )
        with pytest.raises(HTTPException) as exc_info:
            manager.step_inspect(
                episode_id=str(uuid.uuid4()),
                action=action,
                composite_reward=0.0,
                detection_score=0.0,
                explanation_score=0.0,
                correction_score=0.0,
                calibration_score=0.0,
            )
        assert exc_info.value.status_code == 404


class TestAdversarialScaling:
    """Tests for the Worker's adversarial temperature scaling."""

    def test_initial_obfuscation_level_is_zero(self):
        from forge_arena.arena.worker import WorkerAgent
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.worker_model.repo_id = "test"
        settings.worker_model.max_new_tokens = 100
        settings.worker_model.temperature = 0.7
        settings.hf_api_key = None
        worker = WorkerAgent(settings)
        assert worker.obfuscation_level == 0
        assert worker.current_top_p == 0.90

    def test_escalates_after_accuracy_crosses_threshold(self):
        from forge_arena.arena.worker import WorkerAgent
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.worker_model.repo_id = "test"
        settings.worker_model.max_new_tokens = 100
        settings.worker_model.temperature = 0.7
        settings.hf_api_key = None
        worker = WorkerAgent(settings)
        # Feed 10 correct detections (accuracy = 1.0 > 0.65 threshold)
        for _ in range(10):
            worker.record_detection_outcome(True)
        assert worker.obfuscation_level >= 1
        assert worker.current_top_p > 0.90

    def test_top_p_capped_at_097(self):
        from forge_arena.arena.worker import WorkerAgent
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.worker_model.repo_id = "test"
        settings.worker_model.max_new_tokens = 100
        settings.worker_model.temperature = 0.7
        settings.hf_api_key = None
        worker = WorkerAgent(settings)
        # Feed many correct detections to max out top_p
        for _ in range(200):
            worker.record_detection_outcome(True)
        assert worker.current_top_p <= 0.97

    def test_no_escalation_below_threshold(self):
        from forge_arena.arena.worker import WorkerAgent
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.worker_model.repo_id = "test"
        settings.worker_model.max_new_tokens = 100
        settings.worker_model.temperature = 0.7
        settings.hf_api_key = None
        worker = WorkerAgent(settings)
        # Feed all incorrect detections (accuracy = 0.0)
        for _ in range(20):
            worker.record_detection_outcome(False)
        assert worker.obfuscation_level == 0
        assert worker.current_top_p == 0.90

