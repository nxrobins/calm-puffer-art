from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from dataclasses import dataclass, field
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
    train_result_dollar_seconds,
)
from .scheduler import (
    AdaptiveScheduler,
    SchedulerDecision,
    observe_stale_batch_feedback,
    scheduler_checkpoint_metadata,
)
from .types import (
    ActionUnit,
    Message,
    PolicySnapshot,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


ART_RAW_GROUP_KEY = "art/raw_group"
ART_RAW_TRAJECTORY_KEY = "art/raw_trajectory"


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
    cost_per_second_usd: float = 1.0
    synchronous_fallback: bool = False

    def validate(self) -> None:
        if self.train_queue_capacity <= 0:
            raise ValueError("train_queue_capacity must be positive")
        if self.train_batch_groups <= 0:
            raise ValueError("train_batch_groups must be positive")
        if self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
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
        self._model: Any | None = None
        self._current_step = 0
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._submitted_batches = 0
        self._submitted_groups = 0
        self._completed_batches = 0
        self._failed_batches = 0
        self._stale_batches = 0
        self._trainer_wait_s = 0.0
        self._trainer_wait_dollar_seconds = 0.0
        self._actor_admission_delay_s = 0.0
        self._actor_admission_dollar_seconds = 0.0
        self._sample_dollar_seconds = 0.0
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
            self._submitted_batches += 1
            try:
                result = await _maybe_await(
                    self.backend.train(model, list(trajectory_groups), **kwargs)
                )
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
        self._observe_submitted_rollouts(local_groups)
        future = asyncio.get_running_loop().create_future()
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
        self._observe_submitted_rollouts((local_group,))
        future = asyncio.get_running_loop().create_future()
        pending = _PendingArtGroup(
            model=model,
            group=local_group,
            kwargs=dict(kwargs),
            future=future,
        )
        async with self._pending_lock:
            if self._pending_groups and not self._compatible_pending_group(pending):
                await self._flush_pending_locked()
            self._pending_groups.append(pending)
            self._submitted_groups += 1
            target = self._target_train_batch_groups()
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
        stats = self.ring.stats()
        stats["art_backend/current_step"] = float(self._current_step)
        stats["art_backend/current_max_policy_lag"] = float(self.ring.max_policy_lag)
        stats["art_backend/closed"] = 1.0 if self._closed else 0.0
        stats["art_backend/submitted_batches"] = float(self._submitted_batches)
        stats["art_backend/submitted_groups"] = float(self._submitted_groups)
        stats["art_backend/completed_batches"] = float(self._completed_batches)
        stats["art_backend/failed_batches"] = float(self._failed_batches)
        stats["art_backend/stale_batches"] = float(self._stale_batches)
        stats["art_backend/trainer_wait_s"] = self._trainer_wait_s
        stats["art_backend/trainer_wait_dollar_seconds"] = (
            self._trainer_wait_dollar_seconds
        )
        stats["art_backend/actor_admission_delay_s"] = (
            self._actor_admission_delay_s
        )
        stats["art_backend/actor_admission_dollar_seconds"] = (
            self._actor_admission_dollar_seconds
        )
        stats["art_backend/sample_dollar_seconds"] = self._sample_dollar_seconds
        stats["art_backend/pending_groups"] = float(len(self._pending_groups))
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
                checkpoint_metadata = dict(local_result.metadata)
                checkpoint_metadata.update(scheduler_checkpoint_metadata(self.scheduler))
                checkpoint_metadata.update(
                    action_space_checkpoint_metadata(self.action_space)
                )
                await self.weight_channel.publish(
                    PolicySnapshot(
                        step=self._current_step,
                        policy=model,
                        checkpoint_id=(
                            local_result.checkpoint_id
                            or f"art-step-{self._current_step}"
                        ),
                        created_at=time.time(),
                        metadata=checkpoint_metadata,
                    )
                )
                self._completed_batches += 1
                for future in futures:
                    if not future.done():
                        future.set_result(raw_result)
            except BaseException as exc:
                self._failed_batches += 1
                for future in futures:
                    if not future.done():
                        future.set_exception(exc)

    def _score_groups(self, groups: Sequence[TrajectoryGroup]) -> float:
        if self.scheduler is None:
            return 0.0
        return self.scheduler.score_train_groups(groups, policy_step=self._current_step)

    def _observe_submitted_rollouts(
        self,
        groups: Sequence[TrajectoryGroup],
    ) -> None:
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
                    accepted=trajectory.exception is None,
                    dollar_seconds=rollout_cost,
                    queue_wait_dollar_seconds=queue_wait_cost,
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
        self._sample_dollar_seconds += _groups_sample_dollar_seconds(groups)
        await self.ring.put(batch)

    async def _flush_pending_locked(self) -> int:
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

    def _compatible_pending_group(self, pending: _PendingArtGroup) -> bool:
        if not self._pending_groups:
            return True
        current = self._pending_groups[0]
        return current.model is pending.model and current.kwargs == pending.kwargs

    def _target_train_batch_groups(self) -> int:
        if self.scheduler is None:
            return self.config.train_batch_groups
        return max(
            1,
            self.scheduler.target_train_batch_groups(
                configured=self.config.train_batch_groups,
                pending_groups=len(self._pending_groups),
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
        for future in self._batch_futures(batch):
            if not future.done():
                future.set_exception(
                    StaleArtBatchError(
                        "ART batch exceeded max_policy_lag before training"
                    )
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


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return 0
