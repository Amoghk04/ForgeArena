"""Mixed Domain — draws from all four standard domains at elevated corruption rate."""

from __future__ import annotations

import random
from typing import Any

from forge_arena.arena.domains.base import BaseDomain
from forge_arena.arena.domains.code_review import CodeReviewDomain
from forge_arena.arena.domains.customer_support import CustomerSupportDomain
from forge_arena.arena.domains.legal_summarisation import LegalSummarisationDomain
from forge_arena.arena.domains.product_recommendation import ProductRecommendationDomain
from forge_arena.models.tasks import CorruptionType, Task, TaskDomain


class MixedDomain(BaseDomain):
    """Hard domain: randomly samples from all four base domains.

    Elevated corruption_rate (0.80) and no domain-specific heuristics available
    to the Overseer, forcing generalised detection capability.
    """

    domain_type = TaskDomain.MIXED
    corruption_rate = 0.80
    corruption_vocab = list(CorruptionType)  # all five types

    def __init__(self, task_bank: list[dict[str, Any]]) -> None:
        self._delegates: list[BaseDomain] = [
            CustomerSupportDomain(task_bank),
            LegalSummarisationDomain(task_bank),
            CodeReviewDomain(task_bank),
            ProductRecommendationDomain(task_bank),
        ]

    def sample_task(self) -> Task:
        delegate = random.choice(self._delegates)
        return delegate.sample_task()

    def get_corruption_prompt(self, task: Task, corruption_type: CorruptionType) -> str:
        # Delegate to the appropriate domain based on the task's domain field
        for d in self._delegates:
            if d.domain_type == task.domain:
                return d.get_corruption_prompt(task, corruption_type)
        raise ValueError(f"No delegate found for domain: {task.domain}")

    def get_ground_truth_output(self, task: Task) -> str:
        return task.clean_worker_output
