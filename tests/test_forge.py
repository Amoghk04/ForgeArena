"""Tests for the Forge module — pass@k formula, difficulty classification, scheduler."""
from __future__ import annotations

import pytest

from forge_arena.forge.estimator import pass_at_k
from forge_arena.forge.generator import TaskGenerator
from forge_arena.models.tasks import CorruptionType, DifficultyTier, Task, TaskDomain


# ─────────────────────────────────────────────────────────────────────────────
# pass@k unbiased estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestPassAtK:
    def test_zero_correct(self):
        """0 correct answers → pass@k should be 0.0."""
        assert pass_at_k(n=10, c=0, k=8) == pytest.approx(0.0)

    def test_all_correct(self):
        """All correct → pass@k should be 1.0."""
        assert pass_at_k(n=10, c=10, k=8) == pytest.approx(1.0)

    def test_half_correct_k1(self):
        """pass@1 with half correct = 0.5."""
        result = pass_at_k(n=10, c=5, k=1)
        assert pytest.approx(0.5, abs=0.01) == result

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(n=4, c=2, k=8)

    def test_c_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(n=5, c=6, k=4)

    def test_monotone_in_c(self):
        """Increasing correct answers should increase pass@k."""
        results = [pass_at_k(n=10, c=c, k=8) for c in range(0, 11)]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_output_in_unit_interval(self):
        for n in [8, 10, 16]:
            for c in range(0, n + 1, 2):
                val = pass_at_k(n=n, c=c, k=min(8, n))
                assert 0.0 <= val <= 1.0, f"Out of range for n={n}, c={c}"


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty classification
# ─────────────────────────────────────────────────────────────────────────────

class TestDifficultyClassification:
    def setup_method(self):
        from unittest.mock import MagicMock
        from forge_arena.forge.estimator import DifficultyEstimator

        self.config = MagicMock()
        self.config.difficulty_thresholds.too_easy = 0.85
        self.config.difficulty_thresholds.too_hard = 0.20
        self.config.estimation_k = 8
        self.config.estimation_n = 10
        # DifficultyEstimator requires a shared mutable episode_counter list
        self.estimator = DifficultyEstimator(self.config, [])

    def test_too_easy(self):
        from forge_arena.forge.estimator import classify_difficulty
        tier = classify_difficulty(0.90, self.config)
        assert tier == DifficultyTier.TOO_EASY

    def test_too_hard(self):
        from forge_arena.forge.estimator import classify_difficulty
        tier = classify_difficulty(0.10, self.config)
        assert tier == DifficultyTier.TOO_HARD

    def test_learnable_midpoint(self):
        from forge_arena.forge.estimator import classify_difficulty
        tier = classify_difficulty(0.50, self.config)
        assert tier == DifficultyTier.LEARNABLE

    def test_boundary_too_easy(self):
        """Exactly at boundary → learnable (inclusive lower bound)."""
        from forge_arena.forge.estimator import classify_difficulty
        tier = classify_difficulty(0.85, self.config)
        assert tier == DifficultyTier.LEARNABLE

    def test_boundary_too_hard(self):
        from forge_arena.forge.estimator import classify_difficulty
        tier = classify_difficulty(0.20, self.config)
        assert tier == DifficultyTier.LEARNABLE


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler queue management
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskScheduler:
    def setup_method(self):
        from unittest.mock import MagicMock, AsyncMock
        from forge_arena.forge.scheduler import TaskScheduler

        config = MagicMock()
        config.queue_replenishment_threshold = 5
        config.batch_reestimation_interval = 50
        config.difficulty_thresholds.too_easy = 0.85
        config.difficulty_thresholds.too_hard = 0.20

        self.estimator = MagicMock()
        self.generator = MagicMock()
        self.scheduler = TaskScheduler(config, self.estimator, self.generator)

    def _make_task(self, task_id: str) -> object:
        from unittest.mock import MagicMock
        from forge_arena.models.tasks import TaskDomain, ObfuscationDepth
        task = MagicMock()
        task.id = task_id
        task.domain = TaskDomain.CUSTOMER_SUPPORT
        task.is_generated = False
        task.difficulty_tier = None
        return task

    def _make_snapshot(self, task_id: str, tier: DifficultyTier, pak: float):
        from unittest.mock import MagicMock
        snap = MagicMock()
        snap.task_id = task_id
        snap.difficulty_tier = tier
        snap.pass_at_k = pak
        return snap

    @pytest.mark.asyncio
    async def test_initialise_routes_learnable_to_active_queue(self):
        tasks = [self._make_task(f"t-{i}") for i in range(3)]
        snaps = [
            self._make_snapshot(f"t-{i}", DifficultyTier.LEARNABLE, 0.50)
            for i in range(3)
        ]
        self.estimator.batch_estimate.return_value = snaps
        await self.scheduler.initialise(tasks, lambda t: False)
        state = self.scheduler.get_queue_state()
        assert state.learnable_count == 3

    @pytest.mark.asyncio
    async def test_initialise_places_all_seed_tasks_in_active_queue(self):
        """Seed tasks bypass estimation and are placed directly in the learnable queue.

        Pre-estimating with a no-op policy (c=0) would classify every task
        as too-hard. Seed tasks are hand-authored for the learnable zone, so
        they skip estimation until real episodes are collected.
        """
        tasks = [self._make_task("easy-task")]
        await self.scheduler.initialise(tasks, lambda t: False)
        state = self.scheduler.get_queue_state()
        # Task goes straight to active queue; estimator is never called.
        assert state.learnable_count == 1
        assert state.too_easy_count == 0
        self.estimator.batch_estimate.assert_not_called()

    def test_request_task_returns_task(self):
        task = self._make_task("t-1")
        self.scheduler._active_queue.append(task)
        result = self.scheduler.request_task()
        assert result.id == "t-1"

    def test_request_task_empty_raises(self):
        from forge_arena.forge.scheduler import QueueEmptyError
        # No initialise() called — both _active_queue and _seed_bank are empty.
        with pytest.raises(QueueEmptyError):
            self.scheduler.request_task()

    def test_request_task_cycles_when_queue_exhausted(self):
        """After all seed tasks are consumed the queue refills from the seed bank."""
        import asyncio
        tasks = [self._make_task(f"seed-{i}") for i in range(3)]
        asyncio.run(self.scheduler.initialise(tasks, lambda t: False))

        # Consume all 3 tasks
        for _ in range(3):
            self.scheduler.request_task()

        # 4th call must not raise — it should cycle back through the seed bank
        result = self.scheduler.request_task()
        seed_ids = {t.id for t in tasks}
        assert result.id in seed_ids

    def test_batch_reestimate_does_not_wipe_queue_with_zero_accuracy_policy(self):
        """_batch_reestimate with always-False policy must not leave the queue empty.

        This guards against the episode-50 bug where lambda t: False caused
        pass@k = 0.0 for every task, routing all tasks to the too-hard archive
        and wiping the active queue.
        """
        import asyncio
        tasks = [self._make_task(f"t-{i}") for i in range(5)]
        snap_learnable = [
            self._make_snapshot(f"t-{i}", DifficultyTier.LEARNABLE, 0.50)
            for i in range(5)
        ]
        # Simulate always-wrong policy: estimator returns too-hard for every task
        from forge_arena.models.tasks import DifficultyTier as DT
        snap_too_hard = [
            self._make_snapshot(f"t-{i}", DT.TOO_HARD, 0.0)
            for i in range(5)
        ]
        self.estimator.batch_estimate.return_value = snap_too_hard
        asyncio.run(self.scheduler.initialise(tasks, lambda t: False))

        # Trigger batch re-estimation with the always-wrong policy
        asyncio.run(self.scheduler._batch_reestimate(lambda t: False))

        # Active queue must still be non-empty after the safety-net refill
        assert len(self.scheduler._active_queue) > 0

    def test_difficulty_history_empty_after_initialise(self):
        """No snapshot history is recorded at init time.

        History starts accumulating only after real episodes are collected
        via update(). Seed tasks are placed in the queue without estimation.
        """
        import asyncio
        tasks = [self._make_task("t-hist")]
        asyncio.run(self.scheduler.initialise(tasks, lambda t: False))
        curve = self.scheduler.get_difficulty_curve()
        # No estimation ran, so history is empty.
        assert "t-hist" not in curve


# ─────────────────────────────────────────────────────────────────────────────
# Generator quality validation
# ─────────────────────────────────────────────────────────────────────────────

def _make_task(
    corruption_type: CorruptionType | None = None,
    corruption_explanation: str = "",
    corrupted_worker_output: str = "",
) -> Task:
    """Minimal Task fixture for validation tests."""
    return Task(
        id="test-gen-001",
        domain=TaskDomain.CUSTOMER_SUPPORT,
        corruption_type=corruption_type,
        task_description="Summarise the refund policy.",
        source_material="Policy text.",
        clean_worker_output="30-day returns accepted.",
        corrupted_worker_output=corrupted_worker_output or None,
        corruption_location=None,
        corruption_explanation=corruption_explanation or None,
        ground_truth_correction=None,
        is_generated=True,
    )


class TestGeneratorValidation:
    """Unit tests for TaskVariantGenerator._validate_generated_task()."""

    def test_clean_task_always_valid(self):
        task = _make_task(corruption_type=None)
        assert TaskGenerator._validate_generated_task(task) is True

    def test_corrupted_task_with_trigger_vocab_in_explanation_valid(self):
        # "omits" is in FACTUAL_OMISSION trigger_vocab
        task = _make_task(
            corruption_type=CorruptionType.FACTUAL_OMISSION,
            corruption_explanation="The Worker omits the 30-day deadline.",
        )
        assert TaskGenerator._validate_generated_task(task) is True

    def test_corrupted_task_with_trigger_vocab_in_output_valid(self):
        # "consistently recommends" (exact phrase) is in BIAS_INJECTION trigger_vocab
        task = _make_task(
            corruption_type=CorruptionType.BIAS_INJECTION,
            corrupted_worker_output="The agent consistently recommends Product A for all clients.",
        )
        assert TaskGenerator._validate_generated_task(task) is True

    def test_corrupted_task_with_no_trigger_vocab_invalid(self):
        # Neither explanation nor output contains any AUTHORITY_FABRICATION vocab
        task = _make_task(
            corruption_type=CorruptionType.AUTHORITY_FABRICATION,
            corruption_explanation="Something is wrong with the answer.",
            corrupted_worker_output="The policy says returns are accepted.",
        )
        assert TaskGenerator._validate_generated_task(task) is False

    def test_corrupted_task_with_empty_fields_invalid(self):
        # Corrupt task with no text fields at all
        task = _make_task(corruption_type=CorruptionType.TEMPORAL_SHIFT)
        assert TaskGenerator._validate_generated_task(task) is False

    @pytest.mark.parametrize("ctype", list(CorruptionType))
    def test_trigger_vocab_detected_for_each_type(self, ctype):
        """Each corruption type's own trigger vocabulary should make validation pass."""
        from forge_arena.arena.corruptions.types import CORRUPTION_REGISTRY
        meta = CORRUPTION_REGISTRY[ctype]
        trigger_phrase = meta.trigger_vocab[0]
        task = _make_task(
            corruption_type=ctype,
            corruption_explanation=f"The Worker {trigger_phrase} the correct value.",
        )
        assert TaskGenerator._validate_generated_task(task) is True

