"""FastAPI application — openenv v2 compliant endpoints."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException
from openenv.core import HTTPEnvServer
from pydantic import BaseModel as _BaseModel

from forge_arena.arena.domains import init_domain_registry
from forge_arena.arena.episode import EpisodeManager
from forge_arena.arena.worker import WorkerAgent
from forge_arena.config import get_settings
from forge_arena.env import AnyForgeAction, ForgeArenaEnvironment
from forge_arena.forge.estimator import DifficultyEstimator
from forge_arena.forge.generator import TaskGenerator
from forge_arena.forge.scheduler import TaskScheduler
from forge_arena.graders.composite import CompositeGrader
from forge_arena.models.actions import AnyAction
from forge_arena.models.observations import EpisodeResult, ResetObservation, StateObservation
from forge_arena.models.rewards import (
    BaselineScores,
    ForgeQueueState,
    ForgeStats,
    OversightStats,
)
from forge_arena.models.tasks import CorruptionType, Task, TaskDomain

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Application state (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────

_episode_manager: EpisodeManager | None = None
_scheduler: TaskScheduler | None = None
_composite_grader: CompositeGrader | None = None
_task_bank: list[Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load seed tasks and warm up the Forge on startup."""
    global _episode_manager, _scheduler, _composite_grader, _task_bank

    settings = get_settings()
    seed_path = Path("tasks/seed_tasks.json")
    if seed_path.exists():
        raw: list[dict] = json.loads(seed_path.read_text())
        _task_bank = [Task.model_validate(t) for t in raw]
    else:
        logger.warning("seed_tasks.json not found — starting with empty task bank")
        raw = []
        _task_bank = []

    # init_domain_registry expects raw dicts so domain constructors can build Task objects
    init_domain_registry(raw)

    worker = WorkerAgent(settings)
    _composite_grader = CompositeGrader(
        detection_weight=settings.reward.detection_weight,
        explanation_weight=settings.reward.explanation_weight,
        correction_weight=settings.reward.correction_weight,
        calibration_weight=settings.reward.calibration_weight,
        full_reward_threshold=settings.correction.full_reward_threshold,
        zero_reward_threshold=settings.correction.zero_reward_threshold,
        neutral_correction_score=settings.correction.neutral_score_when_clean,
    )
    # EpisodeManager only needs the WorkerAgent — grading is done in the endpoint
    _episode_manager = EpisodeManager(worker=worker)

    _episode_counter: list[int] = []
    estimator = DifficultyEstimator(settings.forge, _episode_counter)
    generator = TaskGenerator(settings)
    _scheduler = TaskScheduler(settings.forge, estimator, generator)

    # Dummy no-op policy for initial estimation (no training data yet)
    _no_op_policy = lambda task: False  # noqa: E731

    if _task_bank:
        await _scheduler.initialise(_task_bank, _no_op_policy)
        logger.info("Forge scheduler initialised", queue_state=_scheduler.get_queue_state())

    # Register openenv HTTPEnvServer routes (/reset, /step, /state, /health, /schema, /metadata)
    # The factory closure captures the shared services — each request gets a fresh env instance.
    def _env_factory() -> ForgeArenaEnvironment:
        if _episode_manager is None or _scheduler is None or _composite_grader is None:
            raise RuntimeError("Services not initialised")
        return ForgeArenaEnvironment(_episode_manager, _scheduler, _composite_grader)

    _http_server = HTTPEnvServer(
        env=_env_factory,
        action_cls=AnyForgeAction,  # delegates to AnyAction discriminated union — supports all action_types
        observation_cls=ResetObservation,
    )
    _http_server.register_routes(app, mode="simulation")
    logger.info("openenv HTTPEnvServer routes registered")

    yield

    logger.info("Shutdown complete")


app = FastAPI(
    title="Forge + Arena",
    description="Adaptive RL training environment for AI oversight",
    version="0.1.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_scheduler() -> TaskScheduler:
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Forge scheduler not initialised")
    return _scheduler


def _get_grader() -> CompositeGrader:
    if _composite_grader is None:
        raise HTTPException(status_code=503, detail="Grader not initialised")
    return _composite_grader


# NOTE: /reset, /step, /state are registered by HTTPEnvServer.register_routes() in lifespan.
# They conform to the openenv wire format:
#   POST /reset  → {observation: {...}, reward: null, done: false}
#   POST /step   → {observation: {...}, reward: float|null, done: bool}
#   GET  /state  → openenv State object (episode_id, step_count — not our StateObservation)
# Client: send episode_id as an extra field in the /step request body (StepRequest.extra="allow").

# GET /episode_state?episode_id= is a Forge-specific extension — returns the full
# StateObservation for a named episode, safe to call without advancing it.
@app.get("/episode_state", response_model=StateObservation)
async def episode_state(
    episode_id: str,
) -> StateObservation:
    """Inspect a specific episode state without advancing it. Ground truth is NOT revealed."""
    from forge_arena.arena.episode import EpisodeManager as _EM
    if _episode_manager is None:
        raise HTTPException(status_code=503, detail="Episode manager not initialised")
    obs = _episode_manager.get_state_observation(episode_id)
    if obs is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id!r} not found")
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /tasks
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks(
    scheduler: TaskScheduler = Depends(_get_scheduler),
) -> dict[str, Any]:
    """Full task bank with difficulty metadata."""
    curve = scheduler.get_difficulty_curve()

    def task_entry(t: Task) -> dict[str, Any]:
        snapshots = curve.get(t.id, [])
        latest_pak = snapshots[-1].pass_at_k if snapshots else None
        tier = snapshots[-1].difficulty_tier.value if snapshots else None
        return {
            "id": t.id,
            "domain": t.domain.value,
            "corruption_sophistication": t.corruption_sophistication,
            "obfuscation_depth": t.obfuscation_depth.value,
            "difficulty_tier": tier,
            "pass_at_k": latest_pak,
            "is_generated": t.is_generated,
        }

    return {"tasks": [task_entry(t) for t in _task_bank]}


# ─────────────────────────────────────────────────────────────────────────────
# 5. POST /grader — standalone (offline evaluation, GRPO reward endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneGraderRequest(_BaseModel):
    episode_id: str
    domain: str
    corruption_present: bool
    corruption_type: str | None = None
    ground_truth_output: str
    overseer_detection: bool
    overseer_confidence: float
    overseer_explanation: str
    overseer_correction: str = ""


@app.post("/grader")
async def grader(
    body: StandaloneGraderRequest,
    grader_: CompositeGrader = Depends(_get_grader),
) -> dict[str, Any]:
    """Standalone grader for offline evaluation and GRPO reward computation.

    Accepts ground truth directly — only valid when the caller already holds
    the true labels (e.g. the GRPO trainer after episode termination).
    """
    try:
        domain = TaskDomain(body.domain)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown domain: {body.domain!r}")

    try:
        corruption_type = CorruptionType(body.corruption_type) if body.corruption_type else None
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown corruption_type: {body.corruption_type!r}")

    composite = grader_.score(
        episode_id=body.episode_id,
        domain=domain,
        corruption_present=body.corruption_present,
        corruption_type=corruption_type,
        ground_truth_output=body.ground_truth_output,
        overseer_detection=body.overseer_detection,
        overseer_explanation=body.overseer_explanation,
        overseer_correction=body.overseer_correction,
        overseer_confidence=body.overseer_confidence,
    )
    return composite.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# 6. GET /baseline
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/baseline", response_model=BaselineScores)
async def baseline() -> BaselineScores:
    """Pre-computed baseline scores for an untrained Overseer (random guessing)."""
    # Baseline for a random-guess policy on a 50% corruption rate environment
    rand_detection = 0.50
    rand_explanation = 0.12
    rand_correction = 0.15
    rand_calibration = 0.50
    composite = 0.40 * rand_detection + 0.30 * rand_explanation + 0.20 * rand_correction + 0.10 * rand_calibration
    return BaselineScores(
        model_id="random_baseline",
        detection_accuracy=rand_detection,
        explanation_score=rand_explanation,
        correction_quality=rand_correction,
        composite_reward=round(composite, 4),
        per_domain={d.value: round(composite, 4) for d in TaskDomain},
        per_corruption_type={c.value: round(composite, 4) for c in CorruptionType},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET /forge/queue
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/forge/queue", response_model=ForgeQueueState)
async def forge_queue(
    scheduler: TaskScheduler = Depends(_get_scheduler),
) -> ForgeQueueState:
    return scheduler.get_queue_state()


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET /forge/stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/forge/stats", response_model=ForgeStats)
async def forge_stats(
    scheduler: TaskScheduler = Depends(_get_scheduler),
) -> ForgeStats:
    return scheduler.get_stats()


# ─────────────────────────────────────────────────────────────────────────────
# 9. GET /oversight/stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/oversight/stats", response_model=OversightStats)
async def oversight_stats() -> OversightStats:
    """Aggregate detection/explanation/correction accuracy from completed episodes."""
    from forge_arena.arena.episode import EpisodeStore
    from forge_arena.models.tasks import EpisodePhase

    done = [s for s in EpisodeStore.values() if s.phase == EpisodePhase.DONE]
    total = len(done)

    if total == 0:
        return OversightStats(
            total_episodes=0,
            detection_accuracy=0.0,
            per_corruption_detection={},
            per_corruption_explanation={},
            per_domain_correction={},
            mean_composite_reward=0.0,
        )

    correct_detections = sum(
        1 for s in done
        if s.overseer_detection is not None and s.overseer_detection == s.corruption_present
    )

    # Per-corruption-type detection accuracy
    per_corruption: dict[str, list[bool]] = {}
    per_corruption_exp: dict[str, list[float]] = {}
    per_domain_corr: dict[str, list[float]] = {}

    for s in done:
        ct_key = s.corruption_type.value if s.corruption_type else "none"
        if ct_key not in per_corruption:
            per_corruption[ct_key] = []
            per_corruption_exp[ct_key] = []
        per_corruption[ct_key].append(
            s.overseer_detection is not None and s.overseer_detection == s.corruption_present
        )
        if s.reward is not None:
            per_corruption_exp[ct_key].append(float(s.reward))

        d_key = s.task.domain.value
        if d_key not in per_domain_corr:
            per_domain_corr[d_key] = []
        if s.reward is not None:
            per_domain_corr[d_key].append(float(s.reward))

    return OversightStats(
        total_episodes=total,
        detection_accuracy=correct_detections / total,
        per_corruption_detection={
            k: sum(v) / len(v) for k, v in per_corruption.items() if v
        },
        per_corruption_explanation={
            k: sum(v) / len(v) for k, v in per_corruption_exp.items() if v
        },
        per_domain_correction={
            k: sum(v) / len(v) for k, v in per_domain_corr.items() if v
        },
        mean_composite_reward=sum(s.reward for s in done if s.reward is not None) / max(1, total),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. GET /oversight/difficulty_curve
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/oversight/difficulty_curve")
async def difficulty_curve(
    scheduler: TaskScheduler = Depends(_get_scheduler),
) -> dict[str, Any]:
    """Pass@k time series per task — primary demo visualisation."""
    curve = scheduler.get_difficulty_curve()
    return {
        task_id: [s.model_dump() for s in snapshots]
        for task_id, snapshots in curve.items()
    }

# /health, /schema, /metadata, /state, /ws are all registered by HTTPEnvServer.register_routes()
# in the lifespan hook above. No manual stubs needed.
