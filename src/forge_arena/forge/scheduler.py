"""Active task queue management — filters learnable tasks, triggers generation."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

from forge_arena.config import ForgeConfig
from forge_arena.forge.estimator import DifficultyEstimator, OverseerPolicy
from forge_arena.forge.generator import TaskGenerator
from forge_arena.models.rewards import DifficultySnapshot, ForgeQueueState, ForgeStats
from forge_arena.models.tasks import DifficultyTier, Task


class QueueEmptyError(RuntimeError):
    """Raised when the active queue is temporarily empty during replenishment."""


class TaskScheduler:
    """Manages the active task queue for the Forge.

    Responsibilities:
      - Maintain the ``active_queue`` of LEARNABLE tasks.
      - Filter TOO_EASY and TOO_HARD tasks to archives.
      - Trigger the TaskGenerator asynchronously when ``active_queue`` drops
        below ``queue_replenishment_threshold``.
      - Track difficulty history for all tasks.
    """

    def __init__(
        self,
        config: ForgeConfig,
        estimator: DifficultyEstimator,
        generator: TaskGenerator,
    ) -> None:
        self._config = config
        self._estimator = estimator
        self._generator = generator

        self._active_queue: deque[Task] = deque()
        self._too_easy_archive: list[Task] = []
        self._too_hard_archive: list[Task] = []
        self._pending_estimation: set[str] = set()
        self._seed_bank: list[Task] = []

        # History of all pass@k snapshots, keyed by task_id
        self._difficulty_history: dict[str, list[DifficultySnapshot]] = {}

        self._episode_count = 0
        self._replenishment_in_progress = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def initialise(self, seed_bank: list[Task], overseer_policy: OverseerPolicy) -> None:  # noqa: ARG002
        """Populate the active queue from seed tasks.

        Seed tasks are hand-crafted to sit in the learnable zone, so they are
        added directly to the active queue without synthetic pre-estimation.
        Pre-estimating against a dummy policy (c=0) would classify every task
        as too-hard and leave the queue empty before training begins.
        Re-estimation against the real Overseer policy begins after the first
        batch_reestimation_interval episodes via ``update()``.
        """
        self._seed_bank = seed_bank
        for task in seed_bank:
            task.difficulty_tier = DifficultyTier.LEARNABLE
            self._active_queue.append(task)

    def request_task(self) -> Task:
        """Return the next learnable task.

        If the active queue is empty (all seed tasks consumed in a long eval
        run) the original seed tasks are cycled back in before popping.
        QueueEmptyError is only raised in the degenerate case where the seed
        bank itself is empty (i.e. initialise() was never called).
        """
        if not self._active_queue:
            self._refill_from_seed_bank()
        if not self._active_queue:
            raise QueueEmptyError(
                "Active task queue is empty and seed bank is also empty. "
                "Call initialise() before requesting tasks."
            )
        return self._active_queue.popleft()

    async def update(self, overseer_policy: OverseerPolicy) -> None:
        """Called periodically after N episodes to re-estimate the active bank.

        If the queue drops below the replenishment threshold after re-estimation,
        triggers async task generation.
        """
        self._episode_count += 1
        if self._episode_count % self._config.batch_reestimation_interval == 0:
            await self._batch_reestimate(overseer_policy)

        if len(self._active_queue) < self._config.queue_replenishment_threshold:
            asyncio.create_task(self._replenish(overseer_policy))

    def get_queue_state(self) -> ForgeQueueState:
        seed_count = sum(1 for t in self._seed_bank if not t.is_generated)
        gen_count = sum(1 for t in self._seed_bank if t.is_generated)
        return ForgeQueueState(
            learnable_count=len(self._active_queue),
            too_easy_count=len(self._too_easy_archive),
            too_hard_count=len(self._too_hard_archive),
            pending_estimation_count=len(self._pending_estimation),
            generated_task_count=gen_count,
            seed_task_count=seed_count,
            replenishment_triggered=self._replenishment_in_progress,
        )

    def get_stats(self) -> ForgeStats:
        per_domain: dict[str, float] = {}
        for task_id, snapshots in self._difficulty_history.items():
            if snapshots:
                latest = snapshots[-1]
                # Group by task domain (requires looking up the task)
                for t in self._seed_bank:
                    if t.id == task_id and snapshots:
                        key = t.domain.value
                        per_domain.setdefault(key, []).append(latest.pass_at_k)  # type: ignore[arg-type]

        domain_avg = {k: sum(v) / len(v) for k, v in per_domain.items()}  # type: ignore[attr-defined]

        # Count tasks that transitioned between tiers
        transitions = self._count_tier_transitions()

        return ForgeStats(
            total_episodes=self._episode_count,
            generator_acceptance_rate=self._generator.acceptance_rate,
            difficulty_transitions=transitions,
            per_domain_pass_at_k=domain_avg,
        )

    def get_difficulty_curve(self) -> dict[str, list[DifficultySnapshot]]:
        """Return the full difficulty history for the /oversight/difficulty_curve endpoint."""
        return dict(self._difficulty_history)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _route_task(self, task: Task, tier: DifficultyTier) -> None:
        task.difficulty_tier = tier
        if tier == DifficultyTier.LEARNABLE:
            self._active_queue.append(task)
        elif tier == DifficultyTier.TOO_EASY:
            self._too_easy_archive.append(task)
        else:
            self._too_hard_archive.append(task)

    def _record_snapshot(self, snapshot: DifficultySnapshot) -> None:
        self._difficulty_history.setdefault(snapshot.task_id, []).append(snapshot)

    async def _batch_reestimate(self, overseer_policy: OverseerPolicy) -> None:
        """Re-estimate all tasks and rebuild the active queue."""
        all_tasks = list(self._active_queue) + self._too_easy_archive + self._too_hard_archive
        snapshots = self._estimator.batch_estimate(all_tasks, overseer_policy, self._episode_count)

        # Rebuild queues
        self._active_queue.clear()
        self._too_easy_archive.clear()
        self._too_hard_archive.clear()

        for task, snap in zip(all_tasks, snapshots):
            self._record_snapshot(snap)
            self._route_task(task, snap.difficulty_tier)

        # Safety net: if the policy classified every task as too-hard (e.g.
        # during early training when the overseer is near-random), refill from
        # the original seed bank so the queue is never left completely empty.
        if not self._active_queue:
            self._refill_from_seed_bank()

    async def _replenish(self, overseer_policy: OverseerPolicy) -> None:
        """Generate and estimate new tasks to replenish the active queue."""
        if self._replenishment_in_progress:
            return
        self._replenishment_in_progress = True
        try:
            # Find a seed task that was learnable or slightly too-easy as the generation base
            seed = self._pick_generation_seed()
            if seed is None:
                return

            snap_history = self._difficulty_history.get(seed.id, [])
            current_pak = snap_history[-1].pass_at_k if snap_history else 0.50

            variant = await self._generator.generate_variant(
                seed, DifficultyTier.LEARNABLE, current_pak
            )

            # Estimate the variant before adding it
            snap = self._estimator.estimate(variant, overseer_policy, self._episode_count)
            self._record_snapshot(snap)
            self._seed_bank.append(variant)

            accepted = snap.difficulty_tier == DifficultyTier.LEARNABLE
            self._generator.record_acceptance(accepted)

            if accepted:
                self._active_queue.append(variant)
        finally:
            self._replenishment_in_progress = False

    def _pick_generation_seed(self) -> Optional[Task]:
        # Prefer tasks from the too-easy archive (they became easy — generate harder variants)
        for t in self._too_easy_archive:
            return t
        # Fall back to any task in the seed bank
        return self._seed_bank[0] if self._seed_bank else None

    def _refill_from_seed_bank(self) -> None:
        """Reload original (non-generated) seed tasks back into the active queue.

        Called when the active queue is empty — either because all seed tasks
        have been consumed during a long eval run, or because _batch_reestimate
        classified every task as too-hard (which happens when the overseer
        policy is near-random early in training).  Only original seed tasks are
        cycled back; generated variants are excluded because they may have been
        estimated against an older policy and their difficulty scores are stale.
        """
        seed_tasks = [t for t in self._seed_bank if not t.is_generated]
        for task in seed_tasks:
            task.difficulty_tier = DifficultyTier.LEARNABLE
            self._active_queue.append(task)

    def _count_tier_transitions(self) -> int:
        count = 0
        for snapshots in self._difficulty_history.values():
            for i in range(1, len(snapshots)):
                if snapshots[i].difficulty_tier != snapshots[i - 1].difficulty_tier:
                    count += 1
        return count
