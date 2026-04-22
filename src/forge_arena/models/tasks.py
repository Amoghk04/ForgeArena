"""Pydantic models for tasks and episode state."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskDomain(str, Enum):
    CUSTOMER_SUPPORT = "customer_support"
    LEGAL_SUMMARISATION = "legal_summarisation"
    CODE_REVIEW = "code_review"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    MIXED = "mixed"


class CorruptionType(str, Enum):
    FACTUAL_OMISSION = "FACTUAL_OMISSION"
    BIAS_INJECTION = "BIAS_INJECTION"
    TEMPORAL_SHIFT = "TEMPORAL_SHIFT"
    AUTHORITY_FABRICATION = "AUTHORITY_FABRICATION"
    INSTRUCTION_OVERRIDE = "INSTRUCTION_OVERRIDE"


class DifficultyTier(str, Enum):
    TOO_EASY = "too-easy"
    LEARNABLE = "learnable"
    TOO_HARD = "too-hard"
    UNESTIMATED = "unestimated"


class ObfuscationDepth(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    """A single oversight task in the seed bank."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: TaskDomain
    corruption_type: Optional[CorruptionType] = None
    obfuscation_depth: ObfuscationDepth = ObfuscationDepth.LOW

    task_description: str
    source_material: str = ""

    # Ground truth — stored server-side, never exposed to agents mid-episode
    clean_worker_output: str
    corrupted_worker_output: Optional[str] = None
    corruption_location: Optional[str] = None
    corruption_explanation: Optional[str] = None
    ground_truth_correction: Optional[str] = None

    # Forge metadata
    corruption_sophistication: float = Field(default=0.5, ge=0.0, le=1.0)
    pass_at_k: Optional[float] = None
    difficulty_tier: DifficultyTier = DifficultyTier.UNESTIMATED
    is_generated: bool = False  # True for Generator-created variants


class EpisodePhase(str, Enum):
    RESET = "reset"
    WORKER_RESPONDING = "worker_responding"
    OVERSEER_INSPECTING = "overseer_inspecting"
    DONE = "done"


class EpisodeState(BaseModel):
    """Server-side episode state — never serialised fully to clients."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Task
    phase: EpisodePhase = EpisodePhase.RESET

    # Worker phase
    worker_cot: Optional[str] = None
    worker_output: Optional[str] = None

    # Ground truth — NEVER include in observation responses before episode_done
    corruption_present: bool = False
    corruption_type: Optional[CorruptionType] = None

    # Overseer phase
    probe_used: bool = False
    overseer_detection: Optional[bool] = None
    overseer_explanation: Optional[str] = None
    overseer_correction: Optional[str] = None
    overseer_confidence: Optional[float] = None

    # Reward (available only after done)
    reward: Optional[float] = None
    done: bool = False
