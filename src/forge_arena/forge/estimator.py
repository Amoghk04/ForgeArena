"""pass@k difficulty estimator using the unbiased HumanEval estimator."""

from __future__ import annotations

from math import comb
from typing import Callable

from forge_arena.config import ForgeConfig
from forge_arena.models.rewards import DifficultySnapshot
from forge_arena.models.tasks import DifficultyTier, Task


# Type alias for the Overseer policy callable.
# Called with a Task and returns True if the Overseer detected the corruption.
OverseerPolicy = Callable[[Task], bool]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the HumanEval paper.

    1 − C(n−c, k) / C(n, k)

    Args:
        n: Total number of completions sampled.
        c: Number of correct completions.
        k: Number of completions per task in the evaluation.

    Returns:
        Float in [0.0, 1.0].
    """
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if c < 0 or c > n:
        raise ValueError(f"c ({c}) must be in [0, {n}]")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def classify_difficulty(pass_at_k_value: float, config: ForgeConfig) -> DifficultyTier:
    """Classify a task's difficulty tier given its pass@k estimate."""
    if pass_at_k_value > config.difficulty_thresholds.too_easy:
        return DifficultyTier.TOO_EASY
    if pass_at_k_value < config.difficulty_thresholds.too_hard:
        return DifficultyTier.TOO_HARD
    return DifficultyTier.LEARNABLE


class DifficultyEstimator:
    """Measures pass@k for oversight tasks against the current Overseer policy.

    Two modes:
      - Batch: estimates all tasks in the bank and rebuilds the active queue.
      - Incremental: estimates a single task type when the queue runs low.

    The estimator does NOT manage the queue itself — that is the Scheduler's job.
    It only computes pass@k values and creates DifficultySnapshot records.
    """

    def __init__(self, config: ForgeConfig, episode_counter: list[int]) -> None:
        self._config = config
        self._episode_counter = episode_counter  # shared mutable reference from Scheduler

    def estimate(
        self,
        task: Task,
        overseer_policy: OverseerPolicy,
        current_episode: int,
    ) -> DifficultySnapshot:
        """Run pass@k estimation for a single task.

        Samples k completions from the Overseer policy and computes the
        unbiased pass@k estimate.

        Args:
            task: The task to estimate.
            overseer_policy: Callable — the current Overseer (sampled k times).
            current_episode: Current training episode count (for the snapshot).

        Returns:
            DifficultySnapshot with the computed pass@k and tier.
        """
        k = self._config.estimation_k
        n = self._config.estimation_n_samples  # n >> k so pass@k is non-binary
        results = [overseer_policy(task) for _ in range(n)]
        c = sum(results)

        pak = pass_at_k(n, c, k)
        tier = classify_difficulty(pak, self._config)

        return DifficultySnapshot(
            task_id=task.id,
            episode_step=current_episode,
            pass_at_k=pak,
            difficulty_tier=tier,
            n_samples=n,
            n_correct=c,
            k=k,
        )

    def batch_estimate(
        self,
        tasks: list[Task],
        overseer_policy: OverseerPolicy,
        current_episode: int,
    ) -> list[DifficultySnapshot]:
        """Run estimation for all tasks in the bank."""
        return [self.estimate(t, overseer_policy, current_episode) for t in tasks]
