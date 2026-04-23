"""Pydantic models for reward breakdowns and Forge metrics."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from forge_arena.models.tasks import CorruptionType, DifficultyTier, TaskDomain


class DetectionScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    corruption_present: bool
    overseer_decision: bool


class ExplanationScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    type_naming: float = Field(..., ge=0.0, le=0.40)
    citation_specificity: float = Field(..., ge=0.0, le=0.30)
    mechanism_proposal: float = Field(..., ge=0.0, le=0.20)
    hallucination_penalty: float = Field(..., ge=-0.10, le=0.0)


class CorrectionScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rouge_l: Optional[float] = None
    neutral: bool = False  # True when no corruption was present


class CalibrationScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    confidence_submitted: float
    detection_outcome: float  # 1.0 if correct detection, 0.0 otherwise


class CompositeReward(BaseModel):
    """Complete reward breakdown for one Overseer submission."""

    composite: float = Field(..., ge=0.0, le=1.0)
    detection: DetectionScore
    explanation: ExplanationScore
    correction: CorrectionScore
    calibration: CalibrationScore

    episode_id: str
    domain: TaskDomain
    corruption_type: Optional[CorruptionType]


class DifficultySnapshot(BaseModel):
    """Single pass@k estimate recorded for a task at a point in time."""

    task_id: str
    episode_step: int
    pass_at_k: float
    difficulty_tier: DifficultyTier
    n_samples: int
    n_correct: int
    k: int


class ForgeQueueState(BaseModel):
    """Returned by GET /forge/queue."""

    learnable_count: int
    too_easy_count: int
    too_hard_count: int
    pending_estimation_count: int
    generated_task_count: int
    seed_task_count: int
    replenishment_triggered: bool


class ForgeStats(BaseModel):
    """Returned by GET /forge/stats."""

    total_episodes: int
    generator_acceptance_rate: float
    difficulty_transitions: int  # tasks that moved between tiers
    per_domain_pass_at_k: dict[str, float]


class OversightStats(BaseModel):
    """Returned by GET /oversight/stats."""

    total_episodes: int
    detection_accuracy: float
    per_corruption_detection: dict[str, float]
    per_corruption_explanation: dict[str, float]
    per_domain_correction: dict[str, float]
    mean_composite_reward: float
    worker_obfuscation_level: int = 0
    worker_top_p: float = 0.90


class BaselineScores(BaseModel):
    """Returned by GET /baseline — pre-training reference numbers."""

    model_id: str
    detection_accuracy: float
    explanation_score: float
    correction_quality: float
    composite_reward: float
    per_domain: dict[str, float]
    per_corruption_type: dict[str, float]
