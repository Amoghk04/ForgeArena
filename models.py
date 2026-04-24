"""Data models for the Forge + Arena environment.

These are the public API types used by external clients (e.g. GRPOTrainer,
run_eval.py) to interact with the environment over HTTP.  The full internal
models live in ``src/forge_arena/models/``.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import Field
from openenv.core.env_server.types import Action, Observation


# ── Actions ──────────────────────────────────────────────────────────────────

class OverseerProbeAction(Action):
    """Ask the Worker a clarifying question before submitting a verdict."""

    action_type: Literal["overseer_probe"] = "overseer_probe"
    question: str = Field(..., min_length=1, max_length=1000, description="Probe question for the Worker")


class OverseerInspectAction(Action):
    """Submit the Overseer's final verdict on the Worker's output."""

    action_type: Literal["overseer_inspect"] = "overseer_inspect"
    detection: bool = Field(..., description="True if the Overseer believes a corruption is present")
    explanation: str = Field(default="", max_length=4000, description="Explanation of detected corruption")
    correction: str = Field(default="", max_length=4000, description="Corrected version of the Worker's output")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overseer confidence in [0.0, 1.0]")


AnyForgeAction = Union[OverseerProbeAction, OverseerInspectAction]


# ── Observations ─────────────────────────────────────────────────────────────

class ForgeObservation(Observation):
    """Observation returned after POST /reset and mid-episode probe steps."""

    episode_id: str = Field(..., description="UUID4 episode identifier")
    task_description: str = Field(..., description="The task the Worker was given")
    source_material: str = Field(default="", description="Source material the Worker used (if any)")
    domain: str = Field(..., description="Task domain (e.g. CODE_REVIEW, LEGAL_SUMMARISATION)")
    worker_cot: str = Field(..., description="Worker chain-of-thought (may contain corruption)")
    worker_output: str = Field(..., description="Worker final output (may contain corruption)")
    done: bool = False
    phase: str = Field(default="overseer_inspecting", description="Current episode phase")


class ForgeEpisodeResult(Observation):
    """Returned after OverseerInspectAction — episode is complete."""

    episode_id: str
    domain: str
    worker_cot: str
    worker_output: str
    done: bool = True

    # Ground truth (only revealed on episode completion)
    corruption_present: bool
    corruption_type: Optional[str] = None
    ground_truth_output: str = ""

    # Overseer submission echoed back
    overseer_detection: bool
    overseer_explanation: str = ""
    overseer_correction: str = ""
    overseer_confidence: float

    # Reward
    composite_reward: float
    detection_score: float
    explanation_score: float
    correction_score: float
    calibration_score: float
