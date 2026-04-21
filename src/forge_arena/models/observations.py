"""Pydantic models for observations returned by the environment to agents."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from forge_arena.models.tasks import CorruptionType, EpisodePhase, TaskDomain


class ResetObservation(BaseModel):
    """Returned by POST /reset. Contains only what the Overseer should see at task start."""

    episode_id: str
    task_description: str
    source_material: str
    domain: TaskDomain
    episode_done: bool = False
    phase: EpisodePhase = EpisodePhase.WORKER_RESPONDING


class WorkerObservation(BaseModel):
    """Returned mid-episode once the Worker has responded — Phase 3 entry."""

    episode_id: str
    task_description: str
    source_material: str
    domain: TaskDomain
    worker_cot: str
    worker_output: str
    episode_done: bool = False
    phase: EpisodePhase = EpisodePhase.OVERSEER_INSPECTING


class EpisodeResult(BaseModel):
    """Returned after overseer_inspect. Includes ground truth and reward."""

    episode_id: str
    task_description: str
    domain: TaskDomain
    worker_cot: str
    worker_output: str

    # Ground truth — safe to return now that episode is done
    corruption_present: bool
    corruption_type: Optional[CorruptionType]

    # Overseer submission
    overseer_detection: bool
    overseer_explanation: str
    overseer_correction: str
    overseer_confidence: float

    # Reward breakdown
    reward: float
    detection_score: float
    explanation_score: float
    correction_score: float
    calibration_score: float

    episode_done: bool = True
    phase: EpisodePhase = EpisodePhase.DONE


class StateObservation(BaseModel):
    """Returned by GET /state — safe inspection without advancing the episode."""

    episode_id: str
    phase: EpisodePhase
    domain: TaskDomain
    task_description: str
    worker_cot: Optional[str]
    worker_output: Optional[str]
    probe_used: bool
    episode_done: bool
    # Intentionally omits: corruption_present, corruption_type, reward
