"""Legal Document Summarisation domain."""

from __future__ import annotations

import random
from typing import Any

from forge_arena.arena.corruptions.injector import build_corruption_prompt
from forge_arena.arena.domains.base import BaseDomain
from forge_arena.models.tasks import CorruptionType, Task, TaskDomain


class LegalSummarisationDomain(BaseDomain):
    domain_type = TaskDomain.LEGAL_SUMMARISATION
    corruption_rate = 0.50
    corruption_vocab = [
        CorruptionType.FACTUAL_OMISSION,
        CorruptionType.AUTHORITY_FABRICATION,
        CorruptionType.TEMPORAL_SHIFT,
        CorruptionType.BIAS_INJECTION,
    ]

    def __init__(self, task_bank: list[dict[str, Any]]) -> None:
        self._task_bank = [
            Task(**t) for t in task_bank if t.get("domain") == "legal_summarisation"
        ]

    def sample_task(self) -> Task:
        if not self._task_bank:
            raise RuntimeError("LegalSummarisationDomain: task bank is empty")
        return random.choice(self._task_bank)

    def get_corruption_prompt(self, task: Task, corruption_type: CorruptionType) -> str:
        return build_corruption_prompt(task, corruption_type)

    def get_ground_truth_output(self, task: Task) -> str:
        return task.clean_worker_output
