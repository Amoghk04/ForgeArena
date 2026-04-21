"""Abstract base class for all task domains."""

from __future__ import annotations

from abc import ABC, abstractmethod

from forge_arena.models.tasks import CorruptionType, Task, TaskDomain


class BaseDomain(ABC):
    """All Arena task domains implement this interface."""

    domain_type: TaskDomain
    corruption_rate: float = 0.50
    corruption_vocab: list[CorruptionType]

    @abstractmethod
    def sample_task(self) -> Task:
        """Sample and return a task from this domain's task bank."""
        ...

    @abstractmethod
    def get_corruption_prompt(self, task: Task, corruption_type: CorruptionType) -> str:
        """Return the system prompt fragment injected into the Worker.

        This is in addition to the corruption injector — domains may provide
        domain-specific framing for the corruption instruction.
        """
        ...

    @abstractmethod
    def get_ground_truth_output(self, task: Task) -> str:
        """Return the clean Worker output used by the correction grader."""
        ...

    def is_valid_corruption(self, corruption_type: CorruptionType) -> bool:
        return corruption_type in self.corruption_vocab
