"""ForgeArenaEnvironment — openenv.core.Environment implementation.

Wraps the full Forge + Arena stack (EpisodeManager, TaskScheduler, CompositeGrader)
in the standard openenv Environment interface so that openenv-compatible training
harnesses (e.g. GRPOTrainer via openenv) can drive the loop through the standard
reset / step / state contract.

Design:
  - Factory-compatible: constructor takes shared services, NOT a pre-selected task.
    Task selection happens inside reset_async() via the scheduler.
  - Session-keyed: episode state lives in the global EpisodeStore (dict keyed by
    UUID episode_id). step_async() receives episode_id as a kwarg, matching how
    openenv HTTPEnvServer passes extra StepRequest fields.
  - Fully async: the Worker makes HF Inference API calls. Sync reset()/step() raise
    NotImplementedError — use reset_async()/step_async() or the HTTP endpoints.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core import Action as _OpenEnvAction
from openenv.core import Environment
from pydantic import ConfigDict, TypeAdapter

from forge_arena.arena.episode import EpisodeManager, EpisodeStore
from forge_arena.forge.scheduler import TaskScheduler
from forge_arena.graders.composite import CompositeGrader
from forge_arena.models.actions import AnyAction, OverseerInspectAction
from forge_arena.models.observations import (
    EpisodeResult,
    ResetObservation,
    StateObservation,
)

# ---------------------------------------------------------------------------
# AnyForgeAction — openenv-compatible root that delegates to the discriminated
# union so HTTPEnvServer can deserialise OverseerInspectAction from the wire payload.
# ---------------------------------------------------------------------------
_any_adapter: TypeAdapter[AnyAction] = TypeAdapter(AnyAction)


class AnyForgeAction(_OpenEnvAction):
    """Thin openenv Action subclass used only as the action_cls sentinel.

    model_validate() is overridden to delegate to the AnyAction discriminated
    union so that inspect actions deserialise correctly.
    """

    model_config = ConfigDict(extra="allow")

    @classmethod
    def model_validate(  # type: ignore[override]
        cls,
        obj: Any,
        *,
        strict: Optional[bool] = None,
        from_attributes: Optional[bool] = None,
        context: Optional[Any] = None,
    ) -> AnyAction:  # type: ignore[override]
        return _any_adapter.validate_python(obj)


class ForgeArenaEnvironment(Environment):
    """openenv Environment wrapping the Forge + Arena stack.

    Each HTTP request creates a fresh instance from the shared services. Episode
    state is persisted in the global EpisodeStore, so concurrent requests are
    fully isolated.

    SUPPORTS_CONCURRENT_SESSIONS = True because all mutable state is keyed by
    UUID episode_id in EpisodeStore — there is no shared per-instance state.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        episode_manager: EpisodeManager,
        scheduler: TaskScheduler,
        grader: CompositeGrader,
    ) -> None:
        super().__init__()
        self._manager = episode_manager
        self._scheduler = scheduler
        self._grader = grader
        # Set only when this instance initiates a reset (WebSocket / direct use).
        # HTTP-mode HTTPEnvServer creates a fresh env per request, so this is only
        # meaningful within a single request lifetime.
        self._episode_id: Optional[str] = None

    # ──────────────────────────────────────────────────────────────
    # Sync stubs — not supported (Worker requires async HF API calls)
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ResetObservation:  # type: ignore[override]
        raise NotImplementedError("ForgeArenaEnvironment is async-only. Use reset_async().")

    def step(self, action: AnyAction, timeout_s: Optional[float] = None, **kwargs) -> EpisodeResult:  # type: ignore[override]
        raise NotImplementedError("ForgeArenaEnvironment is async-only. Use step_async().")

    # ──────────────────────────────────────────────────────────────
    # Async implementation
    # ──────────────────────────────────────────────────────────────

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ResetObservation:
        """Select a task from the scheduler and start a new episode.

        Raises QueueEmptyError if no learnable tasks are available.
        """
        task = self._scheduler.request_task()
        obs = await self._manager.reset(task, task.domain)
        self._episode_id = obs.episode_id
        return obs

    async def step_async(
        self,
        action: AnyAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> EpisodeResult:
        """Advance the episode.

        ``episode_id`` must be present in ``kwargs`` (sent as an extra field on
        the openenv StepRequest body, which has ``extra="allow"``). Falls back to
        ``self._episode_id`` for WebSocket / direct-use sessions.

        OverseerInspectAction → grades submission, returns EpisodeResult (done=True).
        Fixed-length episodes: reset → overseer_inspect → done.
        """
        episode_id: Optional[str] = kwargs.get("episode_id") or self._episode_id
        if not episode_id:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail="episode_id is required in the step request body",
            )

        if isinstance(action, OverseerInspectAction):
            episode_state = EpisodeStore.get(episode_id)
            if episode_state is None:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=404, detail=f"Episode '{episode_id}' not found"
                )

            composite_reward_obj = self._grader.score(
                episode_id=episode_id,
                domain=episode_state.task.domain,
                corruption_present=episode_state.corruption_present,
                corruption_type=episode_state.corruption_type,
                ground_truth_output=episode_state.task.clean_worker_output,
                overseer_detection=action.detection,
                overseer_explanation=action.explanation,
                overseer_correction=action.correction,
                overseer_confidence=action.confidence,
                overseer_accuracy=self._manager.worker.rolling_detection_accuracy,
            )
            result = self._manager.step_inspect(
                episode_id=episode_id,
                action=action,
                composite_reward=composite_reward_obj.composite,
                detection_score=composite_reward_obj.detection.score,
                explanation_score=composite_reward_obj.explanation.score,
                correction_score=composite_reward_obj.correction.score,
                calibration_score=composite_reward_obj.calibration.score,
            )
            # Record detection outcome for adversarial temperature scaling
            overseer_correct = action.detection == episode_state.corruption_present
            self._manager.worker.record_detection_outcome(overseer_correct)
            # Advance the Forge scheduler after each terminal action
            await self._scheduler.update(lambda t: False)
            return result

        else:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported action_type: {type(action).__name__}",
            )

    # ──────────────────────────────────────────────────────────────
    # State property (required by Environment ABC)
    # ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> StateObservation:  # type: ignore[override]
        """Return current episode state for this env instance.

        Meaningful only when self._episode_id is set (i.e. reset_async was
        called on this same instance — WebSocket / direct use). For per-request
        HTTP access, the GET /state?episode_id= endpoint queries EpisodeStore
        directly and does not rely on this property.
        """
        if self._episode_id is not None:
            return self._manager.get_state_observation(self._episode_id)
        # Pre-reset placeholder — returns the openenv base State type
        from openenv.core.env_server.types import State as _OEState
        return _OEState()  # type: ignore[return-value]
