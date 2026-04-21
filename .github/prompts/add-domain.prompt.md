---
description: "Add a new task domain to the Arena. Use when: extending the environment with a new task domain beyond the five built-in ones."
---

# Add a New Task Domain

## Overview

Task domains live in `src/forge_arena/arena/domains/`. Each domain provides the surface for Worker corruption and must implement the `BaseDomain` interface.

## Steps

### 1. Create the domain file

Create `src/forge_arena/arena/domains/<your_domain>.py`. Implement all `BaseDomain` abstract methods:

```python
from forge_arena.arena.domains.base import BaseDomain
from forge_arena.arena.corruptions.types import CorruptionType
from forge_arena.models.tasks import Task, TaskDomain

class YourDomain(BaseDomain):
    domain_type = TaskDomain.YOUR_DOMAIN
    corruption_rate = 0.50  # adjust as needed
    corruption_vocab = [
        CorruptionType.FACTUAL_OMISSION,
        CorruptionType.TEMPORAL_SHIFT,
        # ... which corruption types make sense for this domain
    ]

    def sample_task(self) -> Task:
        # Select a task from the domain's task bank (loaded from openenv.yaml / seed_tasks.json)
        ...

    def get_corruption_prompt(self, task: Task, corruption_type: CorruptionType) -> str:
        # Return the system prompt fragment injected into the Worker when corrupting
        ...

    def get_ground_truth_output(self, task: Task) -> str:
        # Return the clean Worker output used by the correction grader
        ...
```

### 2. Register the domain

Add the domain value to `TaskDomain` enum in `src/forge_arena/models/tasks.py`.

Register an instance in `src/forge_arena/arena/domains/__init__.py`:
```python
from .your_domain import YourDomain
DOMAIN_REGISTRY[TaskDomain.YOUR_DOMAIN] = YourDomain()
```

### 3. Add seed tasks

Add at least 10 hand-crafted tasks for the new domain in `tasks/seed_tasks.json`. Each task must have:
- `clean_worker_output`
- `corrupted_worker_output` for each corruption type in `corruption_vocab`
- `corruption_location` and `ground_truth_correction`

Use the `/generate-task` prompt to scaffold tasks.

### 4. Register in openenv.yaml

Add a `tasks` entry for the domain in `openenv.yaml` with `corruption_rate` and `pass_at_k_target`.

### 5. Write tests

Add domain-specific test cases in `tests/test_episode.py` covering:
- Successful episode with no corruption
- Episode with each corruption type in `corruption_vocab`
- Correct ground truth returned post-episode
