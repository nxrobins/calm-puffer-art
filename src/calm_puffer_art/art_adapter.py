from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import Any, Awaitable, Iterable, Mapping, Sequence

from .actions import (
    AdaptiveActionSpace,
    ActionCodec,
    action_codec_key,
    action_space_checkpoint_metadata,
)
from .runtime import (
    TrajectoryRingBuffer,
    VersionedTrajectoryBatch,
    WeightBroadcastChannel,
    restore_control_state as restore_runtime_control_state,
    train_result_dollar_seconds,
    train_result_score,
    useful_experience_count,
)
from .scheduler import (
    AdaptiveScheduler,
    SchedulerDecision,
    observe_stale_batch_feedback,
    scheduler_checkpoint_metadata,
)
from .types import (
    ActionUnit,
    Checkpoint,
    Message,
    PolicySnapshot,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


ART_RAW_GROUP_KEY = "art/raw_group"
ART_RAW_TRAJECTORY_KEY = "art/raw_trajectory"
ART_BACKEND_STATE_KEY = "art_backend/state"


class StaleArtBatchError(RuntimeError):
    """Raised when an enqueued ART train batch exceeds max policy lag."""


@dataclass(frozen=True)
class ArtAdapterConfig:
    """Configuration for dependency-free ART object conversion."""

    default_scenario_id: str = "art"
    scenario_metadata_key: str = "scenario_id"
    preserve_raw: bool = True


@dataclass(frozen=True)
class AsyncArtBackendConfig:
    """Configuration for the dependency-free ART backend-shaped wrapper."""

    train_queue_capacity: int = 3
    train_batch_groups: int = 1
    max_policy_lag: int = 2
    max_train_steps: int | None = None
    cost_per_second_usd: float = 1.0
    synchronous_fallback: bool = False

    def validate(self) -> None:
        if self.train_queue_capacity <= 0:
            raise ValueError("train_queue_capacity must be positive")
        if self.train_batch_groups <= 0:
            raise ValueError("train_batch_groups must be positive")
        if self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when set")
        if self.cost_per_second_usd < 0:
            raise ValueError("cost_per_second_usd must be non-negative")


def art_trajectory_to_local(
    trajectory: Any,
    *,
    scenario_id: str | None = None,
    config: ArtAdapterConfig | None = None,
) -> Trajectory:
    """Convert an ART-like Trajectory object into the local runtime record.

    The converter is structural rather than importing ART. Current ART
    trajectories expose `messages_and_choices`, `reward`,
    `initial_policy_version`, `final_policy_version`, `metrics`, and
    `metadata`; tests use fakes with the same shape.
    """

    cfg = config or ArtAdapterConfig()
    metadata = dict(_mapping_value(trajectory, "metadata"))
    metrics = _float_metrics(_mapping_value(trajectory, "metrics"))
    resolved_scenario_id = (
        scenario_id
        or str(metadata.get(cfg.scenario_metadata_key) or cfg.default_scenario_id)
    )
    messages_and_choices = list(_value(trajectory, "messages_and_choices", []))
    messages = _messages_from_art(messages_and_choices)
    actions = _actions_from_art(messages_and_choices)
    initial_version = _optional_int(_value(trajectory, "initial_policy_version", None))
    final_version = _optional_int(_value(trajectory, "final_policy_version", None))
    policy_step = _first_int(
        initial_version,
        final_version,
        _value(trajectory, "policy_version", 0),
    )
    metadata.setdefault("scheduler/arm_id", f"{resolved_scenario_id}|art")
    metadata.setdefault("scheduler/scenario_id", resolved_scenario_id)
    metadata.setdefault("art/initial_policy_version", initial_version)
    metadata.setdefault("art/final_policy_version", final_version)
    if cfg.preserve_raw:
        metadata[ART_RAW_TRAJECTORY_KEY] = trajectory
    return Trajectory(
        scenario_id=resolved_scenario_id,
        policy_step=policy_step,
        messages=messages,
        actions=actions,
        reward=float(_value(trajectory, "reward", 0.0) or 0.0),
        metrics=metrics,
        metadata=metadata,
        duration_s=float(metrics.get("duration", 0.0)),
    )


def art_group_to_local(
    group: Any,
    *,
    scenario_id: str | None = None,
    config: ArtAdapterConfig | None = None,
) -> TrajectoryGroup:
    """Convert an ART-like TrajectoryGroup into a local TrajectoryGroup."""

    cfg = config or ArtAdapterConfig()
    group_metadata = dict(_mapping_value(group, "metadata"))
    resolved_scenario_id = (
        scenario_id
        or str(group_metadata.get(cfg.scenario_metadata_key) or cfg.default_scenario_id)
    )
    art_trajectories = list(_value(group, "trajectories", group))
    trajectories = tuple(
        art_trajectory_to_local(
            trajectory,
            scenario_id=resolved_scenario_id,
            config=cfg,
        )
        for trajectory in art_trajectories
    )
    metadata: dict[str, Any] = dict(group_metadata)
    metadata["art/exceptions"] = float(len(_value(group, "exceptions", [])))
    if cfg.preserve_raw:
        metadata[ART_RAW_GROUP_KEY] = group
    return TrajectoryGroup(
        scenario_id=resolved_scenario_id,
        trajectories=trajectories,
        metrics=_float_metrics(_mapping_value(group, "metrics")),
        metadata=metadata,
    )


def art_groups_to_local(
    groups: Iterable[Any],
    *,
    scenario_id: str | None = None,
    config: ArtAdapterConfig | None = None,
) -> tuple[TrajectoryGroup, ...]:
    return tuple(
        art_group_to_local(group, scenario_id=scenario_id, config=config)
        for group in groups
    )


def local_group_to_art(group: TrajectoryGroup) -> Any:
    """Return the original ART group preserved on a converted group."""

    raw_group = group.metadata.get(ART_RAW_GROUP_KEY)
    if raw_group is not None:
        return raw_group
    raw_trajectories = [
        trajectory.metadata.get(ART_RAW_TRAJECTORY_KEY)
        for trajectory in group.trajectories
    ]
    if raw_trajectories and all(raw is not None for raw in raw_trajectories):
        return raw_trajectories
    raise ValueError("TrajectoryGroup does not contain preserved ART objects")


def local_groups_to_art(groups: Iterable[TrajectoryGroup]) -> list[Any]:
    return [local_group_to_art(group) for group in groups]


class ArtBackendTrainer:
    """TrainerBackend adapter that delegates converted groups to an ART backend.

    This keeps ART's loss/backend implementation outside the scaffold. The
    wrapped backend can be a real `art.Backend` or a fake with the same
    `train(model, trajectory_groups, **kwargs)` coroutine shape.
    """

    def __init__(
        self,
        *,
        backend: Any,
        model: Any,
        train_kwargs: Mapping[str, Any] | None = None,
        keep_current_policy: bool = True,
    ) -> None:
        self.backend = backend
        self.model = model
        self.train_kwargs = dict(train_kwargs or {})
        self.keep_current_policy = keep_current_policy

    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        art_groups = local_groups_to_art(groups)
        result = await _maybe_await(
            self.backend.train(
                self.model,
                art_groups,
                **self.train_kwargs,
            )
        )
        return train_result_from_art(
            result,
            fallback_policy=current.policy if self.keep_current_policy else None,
        )


@dataclass
class _PendingArtGroup:
    model: Any
    group: TrajectoryGroup
    kwargs: dict[str, Any]
    future: asyncio.Future[Any]


@dataclass(frozen=True)
class ArtRolloutAdmission:
    """Admission decision for an external ART rollout producer."""

    actor_id: int
    active_actor_count: int
    admitted: bool
    delay_s: float = 0.0
    delay_dollar_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtRolloutAssignment:
    """Atomic admission plus scheduler decision for an ART rollout producer."""

    admission: ArtRolloutAdmission
    decision: SchedulerDecision | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return self.admission.admitted and self.decision is not None


class AsyncArtBackend:
    """Backend-shaped ART wrapper backed by the local Puffer-style train ring.

    The wrapped backend performs the real ART training. This class provides the
    async queueing, bounded staleness, scheduler observation, and weight update
    broadcast around it without importing ART.
    """

    def __init__(
        self,
        *,
        backend: Any,
        config: AsyncArtBackendConfig | None = None,
        adapter_config: ArtAdapterConfig | None = None,
        scheduler: AdaptiveScheduler | None = None,
        action_space: AdaptiveActionSpace | None = None,
        weight_channel: WeightBroadcastChannel | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or AsyncArtBackendConfig()
        self.config.validate()
        self.adapter_config = adapter_config or ArtAdapterConfig()
        self.scheduler = scheduler
        self.action_space = action_space
        self.weight_channel = weight_channel or WeightBroadcastChannel()
        self.ring = TrajectoryRingBuffer(
            capacity=self.config.train_queue_capacity,
            max_policy_lag=self.config.max_policy_lag,
        )
        self._started_at = time.perf_counter()
        self._model: Any | None = None
        self._current_step = 0
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._submitted_batches = 0
        self._submitted_groups = 0
        self._submitted_train_groups = 0
        self._completed_batches = 0
        self._failed_batches = 0
        self._stale_batches = 0
        self._stale_pending_groups = 0
        self._stopped_admissions = 0
        self._trainer_wait_s = 0.0
        self._trainer_wait_dollar_seconds = 0.0
        self._trainer_dollar_seconds = 0.0
        self._actor_admission_delay_s = 0.0
        self._actor_admission_dollar_seconds = 0.0
        self._sample_dollar_seconds = 0.0
        self._failed_rollouts = 0
        self._published_policy_updates = 0
        self._published_policy_improvement = 0.0
        self._published_policy_reward_improving_experience = 0.0
        self._latest_published_policy_score = 0.0
        self._last_published_policy_score = 0.0
        self._pending_groups: list[_PendingArtGroup] = []
        self._pending_lock = asyncio.Lock()

    def _model_inference_name(self, model: Any, step: int | None = None) -> str:
        delegate = getattr(self.backend, "_model_inference_name", None)
        if delegate is not None:
            return delegate(model, step)
        name = str(getattr(model, "name", "model"))
        return name if step is None else f"{name}@{step}"

    async def register(self, model: Any) -> None:
        self._model = model
        delegate = getattr(self.backend, "register", None)
        if delegate is not None:
            await _maybe_await(delegate(model))
        self._ensure_worker()

    async def _get_step(self, model: Any) -> int:
        delegate = getattr(self.backend, "_get_step", None)
        if delegate is not None and self._current_step == 0:
            step = await _maybe_await(delegate(model))
            parsed = _optional_int(step)
            if parsed is not None:
                self._current_step = parsed
        return self._current_step

    def restore_control_state(
        self,
        source: Mapping[str, Any] | PolicySnapshot | Checkpoint | None,
    ) -> dict[str, bool]:
        """Restore scheduler/action-space memory and the served policy step.

        ART checkpoints carry plain metadata, while `PolicySnapshot` and
        `Checkpoint` objects also carry the step used by stale-policy checks.
        Restoring both keeps the scheduler's learned objective memory aligned
        with the async ring's policy-lag baseline after a process restart.
        """

        restored = dict(
            restore_runtime_control_state(
                source,
                scheduler=self.scheduler,
                action_space=self.action_space,
            )
        )
        restored["policy_step"] = False
        restored["art_backend"] = False
        step = _source_policy_step(source)
        if step is not None:
            self._current_step = max(self._current_step, step)
            self.ring.current_policy_step = self._current_step
            restored["policy_step"] = True
        metadata = _source_metadata(source)
        backend_state = metadata.get(ART_BACKEND_STATE_KEY)
        if isinstance(backend_state, Mapping):
            self.load_state_dict(backend_state)
            restored["art_backend"] = True
        return restored

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-friendly ART bridge accounting state."""

        return {
            "version": 1,
            "published_policy_updates": self._published_policy_updates,
            "published_policy_improvement": self._published_policy_improvement,
            "published_policy_reward_improving_experience": (
                self._published_policy_reward_improving_experience
            ),
            "latest_published_policy_score": self._latest_published_policy_score,
            "last_published_policy_score": self._last_published_policy_score,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load state produced by :meth:`state_dict`."""

        self._published_policy_updates = max(
            0,
            _state_int(
                state.get("published_policy_updates"),
                self._published_policy_updates,
            ),
        )
        self._published_policy_improvement = max(
            0.0,
            _state_float(
                state.get("published_policy_improvement"),
                self._published_policy_improvement,
            ),
        )
        self._published_policy_reward_improving_experience = max(
            0.0,
            _state_float(
                state.get("published_policy_reward_improving_experience"),
                self._published_policy_reward_improving_experience,
            ),
        )
        self._latest_published_policy_score = _state_float(
            state.get("latest_published_policy_score"),
            self._latest_published_policy_score,
        )
        self._last_published_policy_score = _state_float(
            state.get("last_published_policy_score"),
            self._last_published_policy_score,
        )

    async def admit_rollout(
        self,
        *,
        actor_id: int = 0,
        configured_actor_count: int = 1,
        trajectory_queue_pressure: float = 0.0,
        apply_delay: bool = True,
    ) -> ArtRolloutAdmission:
        """Apply scheduler actor-count and admission-delay control.

        External ART rollout pools can call this before `select_rollout()`.
        If `admitted` is false, the actor should skip this rollout attempt.
        If admitted, merge `metadata` into the eventual ART trajectory metadata.
        """

        configured = max(1, int(configured_actor_count))
        queue_pressure = max(0.0, float(trajectory_queue_pressure))
        if not self._should_continue_rollout_admission(
            trajectory_queue_pressure=queue_pressure,
        ):
            self._stopped_admissions += 1
            return ArtRolloutAdmission(
                actor_id=actor_id,
                active_actor_count=0,
                admitted=False,
                metadata={
                    "actor_id": actor_id,
                    "scheduler/active_actor_count": 0,
                    "scheduler/admitted": False,
                    "scheduler/stop_recommended": True,
                    "scheduler/stop_reason": "continuation_exhausted",
                },
            )
        active_count = self.active_actor_count(
            configured=configured,
            trajectory_queue_pressure=queue_pressure,
        )
        base_metadata: dict[str, Any] = {
            "actor_id": actor_id,
            "scheduler/active_actor_count": active_count,
            "scheduler/admitted": actor_id < active_count,
        }
        if actor_id >= active_count:
            return ArtRolloutAdmission(
                actor_id=actor_id,
                active_actor_count=active_count,
                admitted=False,
                metadata=base_metadata,
            )

        requested_delay_s = self.rollout_admission_delay_s(
            trajectory_queue_pressure=queue_pressure,
        )
        elapsed_s = 0.0
        if requested_delay_s > 0.0:
            if apply_delay:
                started = time.perf_counter()
                await asyncio.sleep(requested_delay_s)
                elapsed_s = time.perf_counter() - started
            else:
                elapsed_s = requested_delay_s
        delay_dollar_seconds = elapsed_s * self.config.cost_per_second_usd
        if delay_dollar_seconds > 0.0:
            self._actor_admission_delay_s += elapsed_s
            self._actor_admission_dollar_seconds += delay_dollar_seconds
            observer = getattr(self.scheduler, "observe_rollout_admission_delay", None)
            if observer is not None:
                observer(
                    seconds=elapsed_s,
                    dollar_seconds=delay_dollar_seconds,
                )
        admission_delay_ms = max(0, int(round(elapsed_s * 1000.0)))
        metadata = {
            **base_metadata,
            "scheduler/active_rollout_admission_delay_ms": admission_delay_ms,
            "scheduler/active_rollout_admission_delay_s": elapsed_s,
            "scheduler/admission_observed": delay_dollar_seconds > 0.0,
        }
        if delay_dollar_seconds > 0.0:
            metadata["cost/actor_admission_dollar_seconds"] = (
                delay_dollar_seconds
            )
        return ArtRolloutAdmission(
            actor_id=actor_id,
            active_actor_count=active_count,
            admitted=True,
            delay_s=elapsed_s,
            delay_dollar_seconds=delay_dollar_seconds,
            metadata=metadata,
        )

    async def admit_and_select_rollout(
        self,
        *,
        scenarios: Sequence[Scenario],
        action_codec: ActionCodec | None = None,
        action_codecs: Sequence[ActionCodec] | None = None,
        actor_id: int = 0,
        configured_actor_count: int = 1,
        trajectory_queue_pressure: float = 0.0,
        apply_delay: bool = True,
    ) -> ArtRolloutAssignment:
        """Atomically admit, reserve, and describe an external ART rollout.

        This is the preferred bridge call for high-throughput producer pools:
        it applies continuation and actor-count controls, then immediately
        selects a scheduler arm so projected in-flight rollout spend is
        reserved before the caller starts the rollout.
        """

        admission = await self.admit_rollout(
            actor_id=actor_id,
            configured_actor_count=configured_actor_count,
            trajectory_queue_pressure=trajectory_queue_pressure,
            apply_delay=apply_delay,
        )
        if not admission.admitted:
            return ArtRolloutAssignment(
                admission=admission,
                metadata=admission.metadata,
            )
        if not self._should_continue_rollout_admission(
            trajectory_queue_pressure=trajectory_queue_pressure,
        ):
            self._stopped_admissions += 1
            metadata = {
                **admission.metadata,
                "scheduler/admitted": False,
                "scheduler/stop_recommended": True,
                "scheduler/stop_reason": "continuation_exhausted",
            }
            rejected = ArtRolloutAdmission(
                actor_id=actor_id,
                active_actor_count=0,
                admitted=False,
                delay_s=admission.delay_s,
                delay_dollar_seconds=admission.delay_dollar_seconds,
                metadata=metadata,
            )
            return ArtRolloutAssignment(admission=rejected, metadata=metadata)

        decision = self.select_rollout(
            scenarios=scenarios,
            action_codec=action_codec,
            action_codecs=action_codecs,
            actor_id=actor_id,
            trajectory_queue_pressure=trajectory_queue_pressure,
        )
        if self._selected_rollout_exceeds_accounted_budget():
            self._cancel_rollout_decision(
                decision,
                metadata=admission.metadata,
            )
            self._stopped_admissions += 1
            metadata = {
                **admission.metadata,
                "scheduler/admitted": False,
                "scheduler/stop_recommended": True,
                "scheduler/stop_reason": "projected_budget_exhausted",
            }
            rejected = ArtRolloutAdmission(
                actor_id=actor_id,
                active_actor_count=0,
                admitted=False,
                delay_s=admission.delay_s,
                delay_dollar_seconds=admission.delay_dollar_seconds,
                metadata=metadata,
            )
            return ArtRolloutAssignment(admission=rejected, metadata=metadata)
        metadata = art_rollout_metadata(decision, extra=admission.metadata)
        return ArtRolloutAssignment(
            admission=admission,
            decision=decision,
            metadata=metadata,
        )

    def record_rollout_failure(
        self,
        assignment: ArtRolloutAssignment,
        *,
        exception: BaseException | str | None = None,
        dollar_seconds: float | None = None,
        queue_wait_dollar_seconds: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Observe an admitted external rollout that failed before submission.

        External producer pools should call this when an assignment returned by
        :meth:`admit_and_select_rollout` is abandoned or crashes before a real
        ART trajectory can be submitted. The scheduler treats it as failed
        experience, releases the in-flight reservation, and accounts the spend.
        """

        if not assignment.admitted or assignment.decision is None:
            raise ValueError("assignment must be admitted and carry a decision")
        decision = assignment.decision
        failure_metadata = dict(assignment.metadata)
        if metadata is not None:
            failure_metadata.update(metadata)
        failure_metadata.setdefault("scenario_id", decision.scenario.id)
        failure_metadata.setdefault("scheduler/scenario_id", decision.scenario.id)
        failure_metadata.setdefault("scheduler/arm_id", decision.arm_id)
        failure_metadata.setdefault(
            "scheduler/action_codec",
            action_codec_key(decision.action_codec),
        )
        failure_metadata["scheduler/rollout_failed_before_submit"] = True
        failure_metadata.setdefault("failure/mode", "rollout_failed_before_submit")

        rollout_cost = _failure_rollout_dollar_seconds(
            failure_metadata,
            dollar_seconds=dollar_seconds,
        )
        queue_wait_cost = _validated_nonnegative_float(
            queue_wait_dollar_seconds,
            name="queue_wait_dollar_seconds",
        )
        metrics = {"rollout/dollar_seconds": rollout_cost}
        if queue_wait_cost > 0.0:
            metrics["cost/actor_queue_wait_dollar_seconds"] = queue_wait_cost

        policy_step = _optional_int(failure_metadata.get("scheduler/policy_step"))
        if policy_step is None:
            policy_step = self._current_step
        failure = Trajectory(
            scenario_id=decision.scenario.id,
            policy_step=policy_step,
            messages=[],
            actions=[],
            reward=0.0,
            metrics=metrics,
            metadata=failure_metadata,
            duration_s=0.0,
            exception=_failure_exception_text(exception),
        )

        self._failed_rollouts += 1
        self._sample_dollar_seconds += _trajectory_sample_dollar_seconds(failure)
        if self.scheduler is not None:
            self.scheduler.observe_rollout(
                failure,
                accepted=False,
                dollar_seconds=rollout_cost,
                queue_wait_dollar_seconds=queue_wait_cost,
            )
            if self.action_space is not None:
                self.action_space.update_from_metrics(
                    self.scheduler.metrics(),
                    allow_promotions=False,
                    allow_demotions=True,
                )
        return failure

    def active_actor_count(
        self,
        *,
        configured: int,
        trajectory_queue_pressure: float = 0.0,
    ) -> int:
        configured = max(1, int(configured))
        if self.scheduler is None:
            return configured
        controller = getattr(self.scheduler, "active_actor_count", None)
        if controller is None:
            return configured
        return min(
            configured,
            max(
                1,
                int(
                    controller(
                        configured=configured,
                        trajectory_queue_pressure=max(
                            0.0,
                            float(trajectory_queue_pressure),
                        ),
                        train_queue_pressure=self._train_queue_pressure(),
                        policy_step=self._current_step,
                    )
                ),
            ),
        )

    def rollout_admission_delay_s(
        self,
        *,
        trajectory_queue_pressure: float = 0.0,
    ) -> float:
        if self.scheduler is None:
            return 0.0
        controller = getattr(self.scheduler, "rollout_admission_delay_s", None)
        if controller is None:
            return 0.0
        return max(
            0.0,
            float(
                controller(
                    trajectory_queue_pressure=max(
                        0.0,
                        float(trajectory_queue_pressure),
                    ),
                    train_queue_pressure=self._train_queue_pressure(),
                    policy_step=self._current_step,
                )
            ),
        )

    def select_rollout(
        self,
        *,
        scenarios: Sequence[Scenario],
        action_codec: ActionCodec | None = None,
        action_codecs: Sequence[ActionCodec] | None = None,
        actor_id: int = 0,
        trajectory_queue_pressure: float = 0.0,
    ) -> SchedulerDecision:
        """Choose scenario and action granularity for an external ART rollout.

        ART rollout producers can call this before constructing a trajectory,
        then attach :func:`art_rollout_metadata` to the produced ART trajectory
        so submitted samples are credited to the chosen scheduler arm.
        """

        codecs = self._selectable_action_codecs(
            action_codec=action_codec,
            action_codecs=action_codecs,
        )
        if not scenarios:
            raise ValueError("at least one scenario is required")
        if self.scheduler is not None:
            return self.scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=codecs,
                actor_id=actor_id,
                policy_step=self._current_step,
                trajectory_queue_pressure=max(0.0, trajectory_queue_pressure),
                train_queue_pressure=self._train_queue_pressure(),
                configured_train_batch_groups=self.config.train_batch_groups,
                configured_max_policy_lag=self.config.max_policy_lag,
            )
        scenario = scenarios[0]
        codec = codecs[0]
        return SchedulerDecision(
            scenario=scenario,
            action_codec=codec,
            arm_id=f"{scenario.id}|{action_codec_key(codec)}",
            target_train_batch_groups=self.config.train_batch_groups,
            max_policy_lag=self.config.max_policy_lag,
            metadata={
                "actor_id": actor_id,
                "policy_step": self._current_step,
                "trajectory_queue_pressure": max(0.0, trajectory_queue_pressure),
                "train_queue_pressure": self._train_queue_pressure(),
                "score": 0.0,
                "objective_score": 0.0,
                "exploration_score": 0.0,
                "coverage_forced": False,
            },
        )

    async def submit_train(
        self,
        model: Any,
        trajectory_groups: Iterable[Any],
        **kwargs: Any,
    ) -> asyncio.Future[Any]:
        """Enqueue ART trajectory groups and return a future for the train result.

        This is the nonblocking path: callers pay backpressure only until the
        bounded ring accepts the batch, then they can keep producing rollouts
        while the background trainer consumes the batch.
        """

        if self.config.synchronous_fallback:
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            art_groups = list(trajectory_groups)
            local_groups = art_groups_to_local(
                art_groups,
                config=self.adapter_config,
            )
            active_max_policy_lag = self._max_policy_lag()
            self.ring.max_policy_lag = active_max_policy_lag
            self.ring.current_policy_step = self._current_step
            self._tag_batch_control_metadata(
                local_groups,
                target_train_batch_groups=len(local_groups),
                max_policy_lag=active_max_policy_lag,
            )
            self._submitted_batches += 1
            self._submitted_train_groups += len(art_groups)
            if (
                local_groups
                and self._groups_max_lag(local_groups) > active_max_policy_lag
            ):
                self._observe_submitted_rollouts(
                    local_groups,
                    accepted=False,
                    allow_action_space_promotions=False,
                )
                self._stale_batches += 1
                self._failed_batches += 1
                observe_stale_batch_feedback(
                    self.scheduler,
                    groups=local_groups,
                    policy_step=self._current_step,
                    reason="art_sync_stale",
                )
                self._refresh_action_space_from_scheduler(
                    allow_promotions=False,
                    allow_demotions=True,
                )
                future.set_exception(
                    StaleArtBatchError(
                        "ART batch exceeded max_policy_lag before synchronous training"
                    )
                )
                return future
            self._observe_submitted_rollouts(local_groups)
            started = time.perf_counter()
            policy_step = self._current_step
            try:
                result = await _maybe_await(
                    self.backend.train(model, art_groups, **kwargs)
                )
                duration_s = time.perf_counter() - started
                local_result = train_result_from_art(result, fallback_policy=model)
                train_dollar_seconds = train_result_dollar_seconds(
                    local_result,
                    duration_s=duration_s,
                    cost_per_second_usd=self.config.cost_per_second_usd,
                )
                self._trainer_dollar_seconds += train_dollar_seconds
                next_step = _optional_int(local_result.metadata.get("art/step"))
                self._current_step = (
                    max(self._current_step + 1, next_step)
                    if next_step is not None
                    else self._current_step + 1
                )
                self.ring.current_policy_step = self._current_step
                if self.scheduler is not None:
                    self.scheduler.observe_train(
                        groups=local_groups,
                        result=local_result,
                        duration_s=duration_s,
                        dollar_seconds=train_dollar_seconds,
                        policy_step=policy_step,
                    )
                    if self.action_space is not None:
                        self.action_space.update_from_metrics(
                            self.scheduler.metrics()
                        )
                self._schedule_stale_pending_discard()
                self._record_published_update(local_result, local_groups)
                checkpoint_metadata = dict(local_result.metadata)
                checkpoint_metadata.update(scheduler_checkpoint_metadata(self.scheduler))
                checkpoint_metadata.update(
                    action_space_checkpoint_metadata(self.action_space)
                )
                checkpoint_metadata[ART_BACKEND_STATE_KEY] = self.state_dict()
                snapshot = PolicySnapshot(
                    step=self._current_step,
                    policy=model,
                    checkpoint_id=(
                        local_result.checkpoint_id
                        or f"art-step-{self._current_step}"
                    ),
                    created_at=time.time(),
                    metadata=checkpoint_metadata,
                )
                await self.weight_channel.publish(snapshot)
            except BaseException as exc:
                self._failed_batches += 1
                future.set_exception(exc)
            else:
                self._completed_batches += 1
                future.set_result(result)
            return future
        if self._closed:
            raise RuntimeError("AsyncArtBackend is closed")
        if self._model is None:
            await self.register(model)
        self._ensure_worker()

        local_groups = art_groups_to_local(
            trajectory_groups,
            config=self.adapter_config,
        )
        future = asyncio.get_running_loop().create_future()
        active_max_policy_lag = self._max_policy_lag()
        self._tag_batch_control_metadata(
            local_groups,
            target_train_batch_groups=len(local_groups),
            max_policy_lag=active_max_policy_lag,
        )
        if local_groups and self._groups_max_lag(local_groups) > active_max_policy_lag:
            self._submitted_batches += 1
            self._submitted_train_groups += len(local_groups)
            self._observe_submitted_rollouts(
                local_groups,
                accepted=False,
                allow_action_space_promotions=False,
            )
            self._stale_batches += 1
            self._failed_batches += 1
            observe_stale_batch_feedback(
                self.scheduler,
                groups=local_groups,
                policy_step=self._current_step,
                reason="art_async_stale_on_submit",
            )
            self._refresh_action_space_from_scheduler(
                allow_promotions=False,
                allow_demotions=True,
            )
            future.set_exception(
                StaleArtBatchError(
                    "ART batch exceeded max_policy_lag before async training"
                )
            )
            return future
        self._observe_submitted_rollouts(local_groups)
        await self._submit_local_batch(
            model=model,
            groups=local_groups,
            futures=(future,),
            kwargs=dict(kwargs),
        )
        return future

    async def submit_group(
        self,
        model: Any,
        trajectory_group: Any,
        **kwargs: Any,
    ) -> asyncio.Future[Any]:
        """Submit one ART TrajectoryGroup for scheduler-controlled batching."""

        if self.config.synchronous_fallback:
            return await self.submit_train(model, [trajectory_group], **kwargs)
        if self._closed:
            raise RuntimeError("AsyncArtBackend is closed")
        if self._model is None:
            await self.register(model)
        self._ensure_worker()

        local_group = art_group_to_local(
            trajectory_group,
            config=self.adapter_config,
        )
        future = asyncio.get_running_loop().create_future()
        pending = _PendingArtGroup(
            model=model,
            group=local_group,
            kwargs=dict(kwargs),
            future=future,
        )
        async with self._pending_lock:
            self._discard_stale_pending_locked()
            self._submitted_groups += 1
            active_max_policy_lag = self._max_policy_lag()
            if self._pending_group_max_lag(pending) > active_max_policy_lag:
                target = self._target_train_batch_groups(
                    pending_groups=len(self._pending_groups) + 1
                )
                self._tag_batch_control_metadata(
                    (local_group,),
                    target_train_batch_groups=target,
                    max_policy_lag=active_max_policy_lag,
                )
                self._observe_submitted_rollouts(
                    (local_group,),
                    accepted=False,
                    allow_action_space_promotions=False,
                )
                self._discard_pending_group(
                    pending,
                    reason="art_pending_group_stale_on_submit",
                )
                return future
            if self._pending_groups and not self._compatible_pending_group(pending):
                await self._flush_pending_locked()
            target = self._target_train_batch_groups(
                pending_groups=len(self._pending_groups) + 1
            )
            self._tag_batch_control_metadata(
                (local_group,),
                target_train_batch_groups=target,
                max_policy_lag=active_max_policy_lag,
            )
            self._observe_submitted_rollouts((local_group,))
            self._pending_groups.append(pending)
            if len(self._pending_groups) >= target:
                await self._flush_pending_locked()
        return future

    async def flush_pending_groups(self) -> int:
        """Flush partial ART group batches that have not reached cadence yet."""

        async with self._pending_lock:
            return await self._flush_pending_locked()

    async def train(
        self,
        model: Any,
        trajectory_groups: Iterable[Any],
        **kwargs: Any,
    ) -> Any:
        future = await self.submit_train(model, trajectory_groups, **kwargs)
        return await future

    async def close(self) -> None:
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        delegate = getattr(self.backend, "close", None)
        if delegate is not None:
            await _maybe_await(delegate())

    def stats(self) -> dict[str, float]:
        wall_s = max(time.perf_counter() - self._started_at, 1e-9)
        wall_dollar_seconds = wall_s * self.config.cost_per_second_usd
        scheduler_stats = (
            self.scheduler.metrics() if self.scheduler is not None else {}
        )
        accounted_dollar_seconds = self._accounted_dollar_seconds(scheduler_stats)
        if wall_dollar_seconds > 0.0:
            published_objective = (
                self._published_policy_reward_improving_experience
                / max(wall_dollar_seconds, 1e-9)
            )
        else:
            published_objective = 0.0
        if accounted_dollar_seconds > 0.0:
            accounted_published_objective = (
                self._published_policy_reward_improving_experience
                / max(accounted_dollar_seconds, 1e-9)
            )
        else:
            accounted_published_objective = 0.0
        stats = self.ring.stats()
        stats["art_backend/wall_clock_s"] = wall_s
        stats["art_backend/wall_clock_dollar_seconds"] = wall_dollar_seconds
        stats["art_backend/accounted_dollar_seconds"] = accounted_dollar_seconds
        stats["art_backend/current_step"] = float(self._current_step)
        stats["art_backend/current_max_policy_lag"] = float(self.ring.max_policy_lag)
        stats["art_backend/closed"] = 1.0 if self._closed else 0.0
        stats["art_backend/submitted_batches"] = float(self._submitted_batches)
        stats["art_backend/submitted_groups"] = float(self._submitted_groups)
        stats["art_backend/submitted_train_groups"] = float(
            self._submitted_train_groups
        )
        stats["art_backend/completed_batches"] = float(self._completed_batches)
        stats["art_backend/failed_batches"] = float(self._failed_batches)
        stats["art_backend/stale_batches"] = float(self._stale_batches)
        stats["art_backend/stale_pending_groups"] = float(
            self._stale_pending_groups
        )
        stats["art_backend/stopped_admissions"] = float(self._stopped_admissions)
        stats["art_backend/trainer_wait_s"] = self._trainer_wait_s
        stats["art_backend/trainer_wait_dollar_seconds"] = (
            self._trainer_wait_dollar_seconds
        )
        stats["art_backend/trainer_dollar_seconds"] = self._trainer_dollar_seconds
        stats["art_backend/actor_admission_delay_s"] = (
            self._actor_admission_delay_s
        )
        stats["art_backend/actor_admission_dollar_seconds"] = (
            self._actor_admission_dollar_seconds
        )
        stats["art_backend/sample_dollar_seconds"] = self._sample_dollar_seconds
        stats["art_backend/failed_rollouts"] = float(self._failed_rollouts)
        stats["art_backend/pending_groups"] = float(len(self._pending_groups))
        stats["art_backend/submitted_batches_per_s"] = (
            self._submitted_batches / wall_s
        )
        stats["art_backend/submitted_train_groups_per_s"] = (
            self._submitted_train_groups / wall_s
        )
        stats["art_backend/completed_batches_per_s"] = (
            self._completed_batches / wall_s
        )
        stats["art_backend/sample_dollar_seconds_per_s"] = (
            self._sample_dollar_seconds / wall_s
        )
        stats["art_backend/published_policy_updates"] = float(
            self._published_policy_updates
        )
        stats["art_backend/published_policy_improvement"] = (
            self._published_policy_improvement
        )
        stats[
            "art_backend/published_policy_reward_improving_experience"
        ] = self._published_policy_reward_improving_experience
        stats["art_backend/latest_published_policy_score"] = (
            self._latest_published_policy_score
        )
        stats[
            "art_backend/published_policy_reward_improving_experience_per_dollar_second"
        ] = published_objective
        stats[
            "art_backend/accounted_published_policy_reward_improving_experience_per_dollar_second"
        ] = accounted_published_objective
        if scheduler_stats:
            stats.update(scheduler_stats)
        if self.action_space is not None:
            stats.update(self.action_space.metrics())
        return stats

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._trainer_loop())

    async def _trainer_loop(self) -> None:
        while not self._closed:
            self.ring.max_policy_lag = self._max_policy_lag()
            train_wait_started = time.perf_counter()
            batch = await self.ring.get(
                current_policy_step=self._current_step,
                priority_scorer=self._batch_priority_scorer(),
            )
            train_wait_s = time.perf_counter() - train_wait_started
            train_wait_dollar_seconds = (
                train_wait_s * self.config.cost_per_second_usd
            )
            self._trainer_wait_s += train_wait_s
            self._trainer_wait_dollar_seconds += train_wait_dollar_seconds
            futures = self._batch_futures(batch)
            if not futures:
                continue
            self._tag_batch_control_metadata(
                batch.groups,
                max_policy_lag=self.ring.max_policy_lag,
            )
            model = batch.metadata.get("art/model", self._model)
            kwargs = dict(batch.metadata.get("art/train_kwargs", {}))
            started = time.perf_counter()
            policy_step = self._current_step
            try:
                raw_result = await _maybe_await(
                    self.backend.train(
                        model,
                        local_groups_to_art(batch.groups),
                        **kwargs,
                    )
                )
                duration_s = time.perf_counter() - started
                local_result = train_result_from_art(raw_result, fallback_policy=model)
                train_dollar_seconds = train_result_dollar_seconds(
                    local_result,
                    duration_s=duration_s,
                    cost_per_second_usd=self.config.cost_per_second_usd,
                )
                self._trainer_dollar_seconds += train_dollar_seconds
                next_step = _optional_int(local_result.metadata.get("art/step"))
                self._current_step = (
                    max(self._current_step + 1, next_step)
                    if next_step is not None
                    else self._current_step + 1
                )
                self.ring.current_policy_step = self._current_step
                if self.scheduler is not None:
                    self.scheduler.observe_train(
                        groups=batch.groups,
                        result=local_result,
                        duration_s=duration_s,
                        dollar_seconds=(
                            train_dollar_seconds + train_wait_dollar_seconds
                        ),
                        policy_step=policy_step,
                    )
                    if self.action_space is not None:
                        self.action_space.update_from_metrics(
                            self.scheduler.metrics()
                        )
                self._schedule_stale_pending_discard()
                self._record_published_update(local_result, batch.groups)
                checkpoint_metadata = dict(local_result.metadata)
                checkpoint_metadata.update(scheduler_checkpoint_metadata(self.scheduler))
                checkpoint_metadata.update(
                    action_space_checkpoint_metadata(self.action_space)
                )
                checkpoint_metadata[ART_BACKEND_STATE_KEY] = self.state_dict()
                snapshot = PolicySnapshot(
                    step=self._current_step,
                    policy=model,
                    checkpoint_id=(
                        local_result.checkpoint_id
                        or f"art-step-{self._current_step}"
                    ),
                    created_at=time.time(),
                    metadata=checkpoint_metadata,
                )
                await self.weight_channel.publish(snapshot)
                self._completed_batches += 1
                for future in futures:
                    if not future.done():
                        future.set_result(raw_result)
            except BaseException as exc:
                self._failed_batches += 1
                for future in futures:
                    if not future.done():
                        future.set_exception(exc)

    def _record_published_update(
        self,
        result: TrainResult,
        groups: Sequence[TrajectoryGroup],
    ) -> None:
        score = train_result_score(result, groups)
        improvement = max(0.0, score - self._last_published_policy_score)
        experience = useful_experience_count(groups)
        self._published_policy_updates += 1
        self._published_policy_improvement += improvement
        self._published_policy_reward_improving_experience += (
            improvement * experience
        )
        self._last_published_policy_score = score
        self._latest_published_policy_score = score

    def _should_continue_rollout_admission(
        self,
        *,
        trajectory_queue_pressure: float,
    ) -> bool:
        if self.scheduler is None:
            return True
        should_continue = getattr(self.scheduler, "should_continue_training", None)
        if should_continue is None:
            return True
        max_train_steps = (
            self.config.max_train_steps
            if self.config.max_train_steps is not None
            else max(self._current_step + 1, 1_000_000_000)
        )
        return bool(
            should_continue(
                policy_step=self._current_step,
                max_train_steps=max_train_steps,
                pending_train_batches=self.ring.pending_batches
                + len(self._pending_groups),
                train_queue_pressure=max(
                    self._train_queue_pressure(),
                    max(0.0, min(1.0, trajectory_queue_pressure)),
                ),
            )
        )

    def _selected_rollout_exceeds_accounted_budget(self) -> bool:
        if self.scheduler is None:
            return False
        metrics = self.scheduler.metrics()
        budget = _state_float(
            metrics.get("scheduler/budget/max_accounted_dollar_seconds"),
            0.0,
        )
        if budget <= 0.0:
            return False
        projected = _state_float(
            metrics.get("scheduler/budget/projected_accounted_dollar_seconds"),
            0.0,
        )
        return projected > budget

    def _cancel_rollout_decision(
        self,
        decision: SchedulerDecision,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.scheduler is None:
            return
        cancel = getattr(self.scheduler, "cancel_rollout_decision", None)
        if cancel is not None:
            if metadata:
                decision = replace(
                    decision,
                    metadata={**decision.metadata, **metadata},
                )
            cancel(decision)

    def _accounted_dollar_seconds(
        self,
        scheduler_stats: Mapping[str, float],
    ) -> float:
        scheduler_accounted = sum(
            max(0.0, float(scheduler_stats.get(key, 0.0)))
            for key in (
                "scheduler/costs/rollout_dollar_seconds",
                "scheduler/costs/queue_wait_dollar_seconds",
                "scheduler/costs/rollout_admission_dollar_seconds",
                "scheduler/costs/train_dollar_seconds",
            )
        )
        if scheduler_accounted > 0.0:
            return scheduler_accounted
        return (
            self._sample_dollar_seconds
            + self._trainer_dollar_seconds
            + self._trainer_wait_dollar_seconds
        )

    def _score_groups(self, groups: Sequence[TrajectoryGroup]) -> float:
        if self.scheduler is None:
            return 0.0
        return self.scheduler.score_train_groups(groups, policy_step=self._current_step)

    def _observe_submitted_rollouts(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        accepted: bool = True,
        allow_action_space_promotions: bool = True,
    ) -> None:
        self._sample_dollar_seconds += _groups_sample_dollar_seconds(groups)
        if self.scheduler is None:
            return
        observe_rollout = getattr(self.scheduler, "observe_rollout", None)
        if observe_rollout is None:
            return
        observe_admission_delay = getattr(
            self.scheduler,
            "observe_rollout_admission_delay",
            None,
        )
        for group in groups:
            for trajectory in group.trajectories:
                rollout_cost, queue_wait_cost = _trajectory_rollout_cost_parts(
                    trajectory,
                    cost_per_second_usd=self.config.cost_per_second_usd,
                )
                admission_cost = _trajectory_admission_dollar_seconds(trajectory)
                admission_observed = bool(
                    trajectory.metadata.get("scheduler/admission_observed")
                )
                if (
                    admission_cost > 0.0
                    and observe_admission_delay is not None
                    and not admission_observed
                ):
                    observe_admission_delay(
                        seconds=_trajectory_admission_seconds(
                            trajectory,
                            admission_cost=admission_cost,
                            cost_per_second_usd=self.config.cost_per_second_usd,
                        ),
                        dollar_seconds=admission_cost,
                    )
                observe_rollout(
                    trajectory,
                    accepted=accepted and trajectory.exception is None,
                    dollar_seconds=rollout_cost,
                    queue_wait_dollar_seconds=queue_wait_cost,
                )
        if self.action_space is not None:
            self.action_space.update_from_metrics(
                self.scheduler.metrics(),
                allow_promotions=allow_action_space_promotions,
                allow_demotions=False,
            )

    def _batch_priority_scorer(self):
        if self.scheduler is None:
            return None

        def score_batch(batch: VersionedTrajectoryBatch, policy_step: int) -> float:
            return self.scheduler.score_train_groups(
                batch.groups,
                policy_step=policy_step,
            )

        return score_batch

    async def _submit_local_batch(
        self,
        *,
        model: Any,
        groups: Sequence[TrajectoryGroup],
        futures: Sequence[asyncio.Future[Any]],
        kwargs: Mapping[str, Any],
    ) -> None:
        active_max_policy_lag = self._max_policy_lag()
        self._tag_batch_control_metadata(
            groups,
            target_train_batch_groups=len(groups),
            max_policy_lag=active_max_policy_lag,
        )
        batch = VersionedTrajectoryBatch(
            groups=tuple(groups),
            assembled_at_step=self._current_step,
            priority_score=self._score_groups(groups),
            metadata={
                "art/model": model,
                "art/train_kwargs": dict(kwargs),
                "art/result_futures": tuple(futures),
            },
            on_discard=self._discard_submitted_batch,
        )
        self._submitted_batches += 1
        self._submitted_train_groups += len(groups)
        await self.ring.put(batch)

    async def _flush_pending_locked(self) -> int:
        self._discard_stale_pending_locked()
        if not self._pending_groups:
            return 0
        pending = tuple(self._pending_groups)
        self._pending_groups.clear()
        first = pending[0]
        await self._submit_local_batch(
            model=first.model,
            groups=tuple(item.group for item in pending),
            futures=tuple(item.future for item in pending),
            kwargs=first.kwargs,
        )
        return 1

    async def _discard_stale_pending_groups(self) -> int:
        async with self._pending_lock:
            return self._discard_stale_pending_locked()

    def _schedule_stale_pending_discard(self) -> None:
        if not self._pending_groups:
            return
        task = asyncio.create_task(self._discard_stale_pending_groups())
        task.add_done_callback(_consume_task_exception)

    def _discard_stale_pending_locked(self) -> int:
        if not self._pending_groups:
            return 0
        active_max_policy_lag = self._max_policy_lag()
        kept: list[_PendingArtGroup] = []
        discarded = 0
        for pending in self._pending_groups:
            if self._pending_group_max_lag(pending) > active_max_policy_lag:
                self._discard_pending_group(
                    pending,
                    reason="art_pending_group_stale",
                )
                discarded += 1
            else:
                kept.append(pending)
        self._pending_groups = kept
        return discarded

    def _discard_pending_group(
        self,
        pending: _PendingArtGroup,
        *,
        reason: str,
    ) -> None:
        self._stale_pending_groups += 1
        observe_stale_batch_feedback(
            self.scheduler,
            groups=(pending.group,),
            policy_step=self._current_step,
            reason=reason,
        )
        self._refresh_action_space_from_scheduler(
            allow_promotions=False,
            allow_demotions=True,
        )
        if not pending.future.done():
            pending.future.set_exception(
                StaleArtBatchError(
                    "ART pending group exceeded max_policy_lag before batching"
                )
            )

    def _pending_group_max_lag(self, pending: _PendingArtGroup) -> int:
        policy_steps = [
            trajectory.policy_step
            for trajectory in pending.group.trajectories
        ]
        if not policy_steps:
            return 0
        return max(0, self._current_step - min(policy_steps))

    def _groups_max_lag(self, groups: Sequence[TrajectoryGroup]) -> int:
        policy_steps = [
            trajectory.policy_step
            for group in groups
            for trajectory in group.trajectories
        ]
        if not policy_steps:
            return 0
        return max(0, self._current_step - min(policy_steps))

    def _compatible_pending_group(self, pending: _PendingArtGroup) -> bool:
        if not self._pending_groups:
            return True
        current = self._pending_groups[0]
        return current.model is pending.model and current.kwargs == pending.kwargs

    def _target_train_batch_groups(self, *, pending_groups: int | None = None) -> int:
        if self.scheduler is None:
            return self.config.train_batch_groups
        return max(
            1,
            self.scheduler.target_train_batch_groups(
                configured=self.config.train_batch_groups,
                pending_groups=(
                    len(self._pending_groups)
                    if pending_groups is None
                    else pending_groups
                ),
                train_queue_pressure=self._train_queue_pressure(),
                policy_step=self._current_step,
            ),
        )

    def _max_policy_lag(self) -> int:
        if self.scheduler is None:
            return self.config.max_policy_lag
        return max(
            0,
            self.scheduler.max_policy_lag(
                configured=self.config.max_policy_lag,
                train_queue_pressure=self._train_queue_pressure(),
                policy_step=self._current_step,
            ),
        )

    def _train_queue_pressure(self) -> float:
        return min(1.0, self.ring.pending_batches / self.ring.capacity)

    def _selectable_action_codecs(
        self,
        *,
        action_codec: ActionCodec | None,
        action_codecs: Sequence[ActionCodec] | None,
    ) -> tuple[ActionCodec, ...]:
        if (
            self.action_space is not None
            and action_codec is None
            and action_codecs is None
        ):
            codecs = self.action_space.codecs
        elif action_codecs is not None:
            codecs = tuple(action_codecs)
        elif action_codec is not None:
            codecs = (action_codec,)
        else:
            codecs = ()
        if self.action_space is not None:
            for codec in codecs:
                self.action_space.add_codec(codec)
            codecs = self.action_space.codecs
        if not codecs:
            raise ValueError("at least one action codec is required")
        return codecs

    @staticmethod
    def _batch_futures(batch: VersionedTrajectoryBatch) -> tuple[asyncio.Future[Any], ...]:
        futures = batch.metadata.get("art/result_futures")
        if isinstance(futures, tuple) and all(
            isinstance(future, asyncio.Future) for future in futures
        ):
            return futures
        future = batch.metadata.get("art/result_future")
        if isinstance(future, asyncio.Future):
            return (future,)
        return ()

    def _discard_submitted_batch(self, batch: VersionedTrajectoryBatch) -> None:
        self._stale_batches += 1
        self._failed_batches += 1
        observe_stale_batch_feedback(
            self.scheduler,
            groups=batch.groups,
            policy_step=self._current_step,
            reason="art_train_ring_stale",
        )
        self._refresh_action_space_from_scheduler(
            allow_promotions=False,
            allow_demotions=True,
        )
        for future in self._batch_futures(batch):
            if not future.done():
                future.set_exception(
                    StaleArtBatchError(
                        "ART batch exceeded max_policy_lag before training"
                    )
                )

    def _refresh_action_space_from_scheduler(
        self,
        *,
        allow_promotions: bool,
        allow_demotions: bool,
    ) -> None:
        if self.scheduler is None or self.action_space is None:
            return
        if (
            allow_demotions
            and not allow_promotions
            and not self.action_space.demote_on_stale_feedback
        ):
            return
        self.action_space.update_from_metrics(
            self.scheduler.metrics(),
            allow_promotions=allow_promotions,
            allow_demotions=allow_demotions,
        )

    @staticmethod
    def _tag_batch_control_metadata(
        groups: Sequence[TrajectoryGroup],
        *,
        target_train_batch_groups: int | None = None,
        max_policy_lag: int | None = None,
    ) -> None:
        for group in groups:
            for trajectory in group.trajectories:
                if target_train_batch_groups is not None:
                    trajectory.metadata["scheduler/active_target_train_batch_groups"] = (
                        target_train_batch_groups
                    )
                if max_policy_lag is not None:
                    trajectory.metadata["scheduler/active_max_policy_lag"] = (
                        max_policy_lag
                    )


def train_result_from_art(
    result: Any,
    *,
    fallback_policy: Any | None = None,
) -> TrainResult:
    metrics = _float_metrics(_mapping_value(result, "metrics"))
    step = _optional_int(_value(result, "step", None))
    checkpoint_path = _value(result, "checkpoint_path", None)
    artifact_name = _value(result, "artifact_name", None)
    checkpoint_id = _checkpoint_id(
        step=step,
        checkpoint_path=checkpoint_path,
        artifact_name=artifact_name,
    )
    metadata: dict[str, Any] = {}
    if step is not None:
        metadata["art/step"] = step
    if checkpoint_path is not None:
        metadata["art/checkpoint_path"] = str(checkpoint_path)
    if artifact_name is not None:
        metadata["art/artifact_name"] = str(artifact_name)
    return TrainResult(
        policy=fallback_policy,
        metrics=metrics,
        checkpoint_id=checkpoint_id,
        metadata=metadata,
    )


def art_rollout_metadata(
    decision: SchedulerDecision,
    *,
    actor_id: int | None = None,
    policy_step: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ART trajectory metadata that preserves a rollout decision.

    The returned mapping is intentionally plain so it can be merged into ART's
    own trajectory metadata without importing ART in this package.
    """

    decision_actor_id = actor_id
    if decision_actor_id is None:
        parsed_actor = _optional_int(decision.metadata.get("actor_id"))
        decision_actor_id = parsed_actor
    decision_policy_step = policy_step
    if decision_policy_step is None:
        parsed_step = _optional_int(decision.metadata.get("policy_step"))
        decision_policy_step = parsed_step

    metadata: dict[str, Any] = {
        "scenario_id": decision.scenario.id,
        "scheduler/arm_id": decision.arm_id,
        "scheduler/scenario_id": decision.scenario.id,
        "scheduler/action_codec": action_codec_key(decision.action_codec),
        "scheduler/target_train_batch_groups": decision.target_train_batch_groups,
        "scheduler/max_policy_lag": decision.max_policy_lag,
    }
    if decision_actor_id is not None:
        metadata["actor_id"] = decision_actor_id
    if decision_policy_step is not None:
        metadata["scheduler/policy_step"] = decision_policy_step
        metadata.setdefault("art/initial_policy_version", decision_policy_step)
    for key in (
        "score",
        "objective_score",
        "exploration_score",
        "coverage_forced",
        "coverage_target",
        "coverage_share",
        "coverage_deficit",
        "coverage_cost_share",
        "coverage_cost_limit",
        "coverage_cost_limited",
        "expected_rollout_dollar_seconds",
        "estimated_rollout_dollar_seconds",
        "reserved_rollout_dollar_seconds",
        "unobserved_rollout_cost_penalty",
        "unobserved_rollout_cost_estimated",
    ):
        value = decision.metadata.get(key)
        if isinstance(value, bool):
            metadata[f"scheduler/decision/{key}"] = value
        elif isinstance(value, (int, float)) and isfinite(float(value)):
            metadata[f"scheduler/decision/{key}"] = float(value)
    if extra is not None:
        metadata.update(extra)
    return metadata


def _messages_from_art(messages_and_choices: Sequence[Any]) -> list[Message]:
    messages: list[Message] = []
    for item in messages_and_choices:
        message = _choice_message(item)
        if message is None:
            message = item
        default_role = "assistant" if _choice_message(item) else "user"
        role = str(_value(message, "role", default_role))
        content = _value(message, "content", "")
        messages.append(
            Message(role=role, content="" if content is None else str(content))
        )
    return messages


def _actions_from_art(messages_and_choices: Sequence[Any]) -> list[ActionUnit]:
    actions: list[ActionUnit] = []
    for index, item in enumerate(messages_and_choices):
        message = _choice_message(item)
        if message is None:
            continue
        content = _value(message, "content", "") or ""
        text = str(content)
        actions.append(
            ActionUnit(
                kind="art_choice",
                payload=text,
                token_count=len(text.split()),
                text=text,
                metadata={"choice_index": index},
            )
        )
    return actions


def _choice_message(item: Any) -> Any | None:
    return _value(item, "message", None)


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(BaseException):
        task.exception()


def _checkpoint_id(
    *,
    step: int | None,
    checkpoint_path: Any | None,
    artifact_name: Any | None,
) -> str | None:
    if checkpoint_path is not None:
        return Path(str(checkpoint_path)).name or str(checkpoint_path)
    if artifact_name is not None:
        return str(artifact_name)
    if step is not None:
        return f"art-step-{step}"
    return None


def _mapping_value(obj: Any, name: str) -> Mapping[str, Any]:
    value = _value(obj, name, {})
    return value if isinstance(value, Mapping) else {}


def _float_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            metrics[str(key)] = float(value)
            continue
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(float_value):
            metrics[str(key)] = float_value
    return metrics


def _groups_sample_dollar_seconds(groups: Sequence[TrajectoryGroup]) -> float:
    return sum(
        _trajectory_sample_dollar_seconds(trajectory)
        for group in groups
        for trajectory in group.trajectories
    )


def _trajectory_sample_dollar_seconds(trajectory: Trajectory) -> float:
    explicit_total = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/dollar_seconds",),
    )
    if explicit_total is None:
        explicit_total = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/dollar_seconds",),
        )
    if explicit_total is not None:
        return explicit_total

    rollout_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("rollout/dollar_seconds",),
    )
    if rollout_cost is None:
        rollout_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("rollout/dollar_seconds",),
        )
    queue_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
    )
    if queue_cost is None:
        queue_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
        )
    admission_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/actor_admission_dollar_seconds", "admission/dollar_seconds"),
    )
    if admission_cost is None:
        admission_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/actor_admission_dollar_seconds", "admission/dollar_seconds"),
        )
    return (rollout_cost or 0.0) + (queue_cost or 0.0) + (admission_cost or 0.0)


def _trajectory_rollout_cost_parts(
    trajectory: Trajectory,
    *,
    cost_per_second_usd: float,
) -> tuple[float, float]:
    explicit_total = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/dollar_seconds",),
    )
    if explicit_total is None:
        explicit_total = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/dollar_seconds",),
        )
    admission_cost = _trajectory_admission_dollar_seconds(trajectory)
    if explicit_total is not None:
        return max(0.0, explicit_total - admission_cost), 0.0

    rollout_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("rollout/dollar_seconds",),
    )
    if rollout_cost is None:
        rollout_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("rollout/dollar_seconds",),
        )
    if rollout_cost is None:
        rollout_cost = max(0.0, trajectory.duration_s * cost_per_second_usd)

    queue_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
    )
    if queue_cost is None:
        queue_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
        )
    return rollout_cost, queue_cost or 0.0


def _trajectory_admission_dollar_seconds(trajectory: Trajectory) -> float:
    admission_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/actor_admission_dollar_seconds", "admission/dollar_seconds"),
    )
    if admission_cost is None:
        admission_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/actor_admission_dollar_seconds", "admission/dollar_seconds"),
        )
    return admission_cost or 0.0


def _trajectory_admission_seconds(
    trajectory: Trajectory,
    *,
    admission_cost: float,
    cost_per_second_usd: float,
) -> float:
    seconds = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("scheduler/active_rollout_admission_delay_s", "admission/s"),
    )
    if seconds is None:
        seconds = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("scheduler/active_rollout_admission_delay_s", "admission/s"),
        )
    if seconds is not None:
        return seconds
    if cost_per_second_usd > 0.0:
        return admission_cost / cost_per_second_usd
    return 0.0


def _failure_rollout_dollar_seconds(
    metadata: Mapping[str, Any],
    *,
    dollar_seconds: float | None,
) -> float:
    if dollar_seconds is not None:
        return _validated_nonnegative_float(
            dollar_seconds,
            name="dollar_seconds",
        )
    reserved = _first_nonnegative_mapping_float(
        metadata,
        (
            "scheduler/decision/reserved_rollout_dollar_seconds",
            "scheduler/decision/estimated_rollout_dollar_seconds",
            "scheduler/decision/expected_rollout_dollar_seconds",
        ),
    )
    return reserved or 0.0


def _validated_nonnegative_float(value: float, *, name: str) -> float:
    candidate = float(value)
    if not isfinite(candidate) or candidate < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return candidate


def _failure_exception_text(exception: BaseException | str | None) -> str:
    if exception is None:
        return "rollout_failed_before_submit"
    if isinstance(exception, BaseException):
        text = str(exception)
        if text:
            return f"{type(exception).__name__}: {text}"
        return type(exception).__name__
    return str(exception) or "rollout_failed_before_submit"


def _first_nonnegative_mapping_float(
    values: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(candidate) and candidate >= 0.0:
            return candidate
    return None


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _source_policy_step(
    source: Mapping[str, Any] | PolicySnapshot | Checkpoint | None,
) -> int | None:
    if source is None:
        return None
    if isinstance(source, (PolicySnapshot, Checkpoint)):
        return _optional_int(source.step)
    if not isinstance(source, Mapping):
        return None
    for key in ("step", "policy_step", "art/step"):
        parsed = _optional_int(source.get(key))
        if parsed is not None:
            return parsed
    nested = source.get("metadata")
    if isinstance(nested, Mapping):
        for key in ("step", "policy_step", "art/step"):
            parsed = _optional_int(nested.get(key))
            if parsed is not None:
                return parsed
    return None


def _source_metadata(
    source: Mapping[str, Any] | PolicySnapshot | Checkpoint | None,
) -> Mapping[str, Any]:
    if source is None:
        return {}
    if isinstance(source, (PolicySnapshot, Checkpoint)):
        return source.metadata
    if not isinstance(source, Mapping):
        return {}
    nested = source.get("metadata")
    if isinstance(nested, Mapping):
        return nested
    return source


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return 0


def _state_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if isfinite(candidate) else default
