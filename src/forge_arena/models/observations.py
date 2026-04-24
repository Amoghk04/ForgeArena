"""Pydantic models for observations returned by the environment to agents."""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from openenv.core import Observation as _OpenEnvObservation

from forge_arena.models.tasks import CorruptionType, EpisodePhase, TaskDomain


class ResetObservation(_OpenEnvObservation):
    """Returned by POST /reset. Contains task info and the Worker's response for the Overseer to inspect."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)

    episode_id: str
    task_description: str
    source_material: str
    domain: TaskDomain
    worker_cot: str
    worker_output: str
    done: bool = False
    phase: EpisodePhase = EpisodePhase.OVERSEER_INSPECTING


class WorkerObservation(_OpenEnvObservation):
    """Returned mid-episode once the Worker has responded — Phase 3 entry."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)

    episode_id: str
    task_description: str
    source_material: str
    domain: TaskDomain
    worker_cot: str
    worker_output: str
    done: bool = False
    phase: EpisodePhase = EpisodePhase.OVERSEER_INSPECTING


class EpisodeResult(_OpenEnvObservation):
    """Returned after overseer_inspect. Includes ground truth and reward."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)

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

    done: bool = True
    phase: EpisodePhase = EpisodePhase.DONE


class StateObservation(_OpenEnvObservation):
    """Returned by GET /state — safe inspection without advancing the episode."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)

    episode_id: str
    phase: EpisodePhase
    domain: TaskDomain
    task_description: str
    worker_cot: Optional[str]
    worker_output: Optional[str]
    probe_used: bool
    done: bool
    # Intentionally omits: corruption_present, corruption_type, reward
