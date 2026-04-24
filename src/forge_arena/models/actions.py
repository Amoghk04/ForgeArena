"""Pydantic models for agent actions submitted via POST /step."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openenv.core import Action as _OpenEnvAction


class ActionType(str, Enum):
    WORKER_RESPOND = "worker_respond"
    OVERSEER_PROBE = "overseer_probe"
    OVERSEER_INSPECT = "overseer_inspect"


class WorkerRespondAction(_OpenEnvAction):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)
    action_type: Literal[ActionType.WORKER_RESPOND] = ActionType.WORKER_RESPOND
    # Worker populates these — in our environment they are produced by the Worker model
    chain_of_thought: str
    output: str


class OverseerProbeAction(_OpenEnvAction):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)
    action_type: Literal[ActionType.OVERSEER_PROBE] = ActionType.OVERSEER_PROBE
    question: str = Field(..., min_length=1, max_length=1000)


class OverseerInspectAction(_OpenEnvAction):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)
    action_type: Literal[ActionType.OVERSEER_INSPECT] = ActionType.OVERSEER_INSPECT
    detection: bool
    explanation: str = Field(default="", max_length=4000)
    correction: str = Field(default="", max_length=4000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    # Set to True for dataset-collection dummy steps so the Forge scheduler
    # does not count them as real Overseer evaluations (which would corrupt
    # pass@k estimates and drain the learnable queue).
    dry_run: bool = False

    @field_validator("explanation")
    @classmethod
    def explanation_required_when_detected(cls, v: str, info: object) -> str:
        # Soft validation — callers should provide explanations when detection=True
        # Hard enforcement is in the grader (no explanation → zero explanation score)
        return v


# Discriminated union — FastAPI uses the `action_type` literal to disambiguate
AnyAction = Annotated[
    Union[WorkerRespondAction, OverseerProbeAction, OverseerInspectAction],
    Field(discriminator="action_type"),
]


class StepRequest(BaseModel):
    episode_id: str
    action: AnyAction


class ProbeResponse(BaseModel):
    """Worker's answer to an Overseer probe question."""

    episode_id: str
    question: str
    answer: str
    probe_used: bool = True
