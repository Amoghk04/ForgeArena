"""Customer Support domain implementation."""

from __future__ import annotations

import random
from typing import Any

from forge_arena.arena.corruptions.injector import build_corruption_prompt
from forge_arena.arena.domains.base import BaseDomain
from forge_arena.models.tasks import CorruptionType, Task, TaskDomain


class CustomerSupportDomain(BaseDomain):
    domain_type = TaskDomain.CUSTOMER_SUPPORT
    corruption_rate = 0.50
    corruption_vocab = [
        CorruptionType.FACTUAL_OMISSION,
        CorruptionType.TEMPORAL_SHIFT,
        CorruptionType.AUTHORITY_FABRICATION,
        CorruptionType.INSTRUCTION_OVERRIDE,
    ]

    def __init__(self, task_bank: list[dict[str, Any]]) -> None:
        self._task_bank = [Task(**t) for t in task_bank if t.get("domain") == "customer_support"]

    def sample_task(self) -> Task:
        if not self._task_bank:
            raise RuntimeError("CustomerSupportDomain: task bank is empty")
        return random.choice(self._task_bank)

    def get_corruption_prompt(self, task: Task, corruption_type: CorruptionType) -> str:
        return build_corruption_prompt(task, corruption_type)

    def get_ground_truth_output(self, task: Task) -> str:
        return task.clean_worker_output
