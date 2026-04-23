"""Pydantic models for agent actions submitted via POST /step."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openenv.core import Action as _OpenEnvAction


class ActionType(str, Enum):
    WORKER_RESPOND = "worker_respond"
    OVERSEER_INSPECT = "overseer_inspect"


class WorkerRespondAction(_OpenEnvAction):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)
    action_type: Literal[ActionType.WORKER_RESPOND] = ActionType.WORKER_RESPOND
    # Worker populates these — in our environment they are produced by the Worker model
    chain_of_thought: str
    output: str


class OverseerInspectAction(_OpenEnvAction):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)
    action_type: Literal[ActionType.OVERSEER_INSPECT] = ActionType.OVERSEER_INSPECT
    detection: bool
    explanation: str = Field(default="", max_length=4000)
    correction: str = Field(default="", max_length=4000)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("explanation")
    @classmethod
    def explanation_required_when_detected(cls, v: str, info: object) -> str:
        # Soft validation — callers should provide explanations when detection=True
        # Hard enforcement is in the grader (no explanation → zero explanation score)
        return v


# Discriminated union — FastAPI uses the `action_type` literal to disambiguate
AnyAction = Annotated[
    Union[WorkerRespondAction, OverseerInspectAction],
    Field(discriminator="action_type"),
]


class StepRequest(BaseModel):
    episode_id: str
    action: AnyAction



