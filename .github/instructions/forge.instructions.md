---
applyTo: "src/forge_arena/forge/**"
---

# Forge Implementation Guidelines

## pass@k Estimator (`estimator.py`)

Always use the **unbiased estimator** from the HumanEval paper, not a simple pass/fail ratio.

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator.
    n: total samples drawn
    c: number of correct samples
    k: number of completions per task
    Returns float in [0.0, 1.0].
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

Default `k=8`. This is calibrated for Qwen2.5-1.5B. Never hardcode `k` in the estimator body — read from `ForgeConfig.estimation_k`.

### Estimation modes

**Batch mode**: Runs at training start and every `batch_reestimation_interval` episodes. Re-ranks the entire active task bank and rebuilds the `learnable` queue.

**Incremental mode**: Runs on a single task type when `active_queue_depth < queue_replenishment_threshold`. Does not restart batch mode.

Both modes call the same `estimate_task_difficulty(task_id, overseer_policy)` function. The mode difference is in the scheduler, not the estimator.

## Difficulty Classification

```python
def classify_difficulty(pass_at_k_value: float, config: ForgeConfig) -> DifficultyTier:
    if pass_at_k_value > config.difficulty_thresholds.too_easy:
        return DifficultyTier.TOO_EASY
    elif pass_at_k_value < config.difficulty_thresholds.too_hard:
        return DifficultyTier.TOO_HARD
    else:
        return DifficultyTier.LEARNABLE
```

Thresholds come from `openenv.yaml` (never hardcoded). Default: `too_easy=0.85`, `too_hard=0.20`.

## Task Generator (`generator.py`)

The Generator manipulates two independent difficulty axes:

1. **Corruption sophistication** — how deeply the corruption is embedded in plausible reasoning
2. **Obfuscation depth** — length and complexity of the Worker chain-of-thought, and how deep in it the corruption appears

Generator inputs: `seed_task`, `target_tier` (easy/medium/hard), `current_pass_at_k`.

Generated variants enter the **seed bank** as candidates. They do NOT go directly into the active queue. They must pass difficulty estimation first. Log all rejected variants with their estimated pass@k and the rejection reason.

### Generator reward logic (for future joint training; log-only during demo)
- Task lands in learnable zone (0.20–0.85): full reward `0.80 + tier_bonus`
- Task too hard (< 0.20): proportional reduction
- Task too easy (> 0.85): near-zero reward `0.05`

## Scheduler (`scheduler.py`)

The scheduler maintains:
- `active_queue`: deque of `TaskQueueEntry` — tasks in the learnable zone
- `pending_estimation`: set of task IDs awaiting pass@k measurement
- `too_easy_archive` / `too_hard_archive`: filtered-out tasks (kept for stats)

The scheduler triggers `generate_variant` automatically when `len(active_queue) < queue_replenishment_threshold`. This happens asynchronously — do not block `request_task` while generation is in progress.

`request_task()` returns the next learnable task or raises `QueueEmptyError` if the queue is temporarily empty during replenishment.

## Difficulty History

All pass@k estimates are appended to a per-task `difficulty_history: list[DifficultySnapshot]`. This history is the source of truth for `/oversight/difficulty_curve` — the primary demo visualisation showing the double-rise pattern.
