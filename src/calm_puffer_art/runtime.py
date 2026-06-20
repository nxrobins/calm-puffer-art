from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Mapping as MappingABC
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from math import isfinite
from statistics import fmean
from typing import Any, Callable, Mapping, Protocol, Sequence

from .actions import (
    ACTION_SPACE_STATE_KEY,
    ActionCodec,
    AdaptiveActionSpace,
    action_space_checkpoint_metadata,
    semantic_bandwidth,
)
from .scheduler import (
    AdaptiveScheduler,
    SCHEDULER_STATE_KEY,
    SchedulerDecision,
    action_quality,
    observe_stale_batch_feedback,
    scheduler_checkpoint_metadata,
)
from .types import (
    Checkpoint,
    Message,
    PolicySnapshot,
    PromotionDecision,
    RunSummary,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
    mean,
)


class AgentPolicy(Protocol):
    """Policy object used by user-defined workflows."""

    async def act(
        self,
        messages: Sequence[Message],
        *,
        scenario: Scenario,
        codec: ActionCodec,
    ):
        ...


class TrainerBackend(Protocol):
    """Consumes ART-like trajectory groups and returns a new checkpoint."""

    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        ...


class PromotionEvaluator(Protocol):
    """Programmable gate for publishing trained candidate policies."""

    async def __call__(
        self,
        *,
        current: PolicySnapshot,
        result: TrainResult,
        groups: Sequence[TrajectoryGroup],
    ) -> PromotionDecision:
        ...


@dataclass
class MetricPromotionEvaluator:
    """Promotes candidates when a result metric improves enough."""

    metric_key: str = "train/reward"
    min_delta: float = 0.0
    initial_score: float = 0.0
    best_score: float = field(init=False)

    def __post_init__(self) -> None:
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.best_score = self.initial_score

    async def __call__(
        self,
        *,
        current: PolicySnapshot,
        result: TrainResult,
        groups: Sequence[TrajectoryGroup],
    ) -> PromotionDecision:
        score = _result_metric_score(
            result,
            groups,
            preferred_key=self.metric_key,
        )
        baseline = self.best_score
        improvement = score - baseline
        promoted = improvement >= self.min_delta
        if promoted:
            self.best_score = score
        return PromotionDecision(
            promoted=promoted,
            score=score,
            baseline_score=baseline,
            improvement=improvement,
            reason=(
                "metric_improved"
                if promoted
                else "metric_below_promotion_threshold"
            ),
            metrics={f"promotion/metric/{self.metric_key}": score},
        )


@dataclass
class RolloutPromotionEvaluator:
    """Promotes candidates using held-out rollout workflow scores."""

    scenarios: Sequence[Scenario]
    workflow: RolloutWorkflow
    action_codec: ActionCodec
    min_delta: float = 0.0
    initial_score: float = 0.0
    cost_per_second_usd: float = 1.0
    actor_id: int = -1
    best_score: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("at least one evaluation scenario is required")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.cost_per_second_usd < 0:
            raise ValueError("cost_per_second_usd must be non-negative")
        self.scenarios = tuple(self.scenarios)
        self.best_score = self.initial_score

    async def __call__(
        self,
        *,
        current: PolicySnapshot,
        result: TrainResult,
        groups: Sequence[TrajectoryGroup],
    ) -> PromotionDecision:
        candidate_policy = (
            result.policy
            if result.policy is not None
            else current.policy
        )
        started = time.perf_counter()
        rewards: list[float] = []
        dollar_seconds = 0.0
        action_units = 0
        source_tokens = 0
        failures = 0

        for index, scenario in enumerate(self.scenarios):
            trajectory_started = time.perf_counter()
            context = RolloutContext(
                actor_id=self.actor_id,
                policy_step=current.step + 1,
                action_codec=self.action_codec,
                scheduler_arm_id=f"{scenario.id}|promotion_eval",
                decision_metadata={
                    "promotion/eval": True,
                    "promotion/eval_index": index,
                },
            )
            try:
                trajectory = await self.workflow(
                    candidate_policy,
                    scenario,
                    context,
                )
                trajectory.duration_s = time.perf_counter() - trajectory_started
            except Exception as exc:
                failures += 1
                trajectory = Trajectory(
                    scenario_id=scenario.id,
                    policy_step=current.step + 1,
                    messages=[],
                    actions=[],
                    reward=0.0,
                    duration_s=time.perf_counter() - trajectory_started,
                    exception=f"{type(exc).__name__}: {exc}",
                    metadata={
                        "promotion/eval": True,
                        "promotion/eval_index": index,
                    },
                )

            quality = action_quality(trajectory)
            rewards.append(trajectory.reward * quality)
            action_units += trajectory.action_units
            source_tokens += trajectory.token_count
            dollar_seconds += _trajectory_eval_dollar_seconds(
                trajectory,
                cost_per_second_usd=self.cost_per_second_usd,
            )

        score = mean(rewards)
        baseline = self.best_score
        improvement = score - baseline
        promoted = improvement >= self.min_delta
        if promoted:
            self.best_score = score
        duration_s = time.perf_counter() - started
        if dollar_seconds <= 0.0:
            dollar_seconds = duration_s * self.cost_per_second_usd
        return PromotionDecision(
            promoted=promoted,
            score=score,
            baseline_score=baseline,
            improvement=improvement,
            dollar_seconds=dollar_seconds,
            reason=(
                "eval_improved"
                if promoted
                else "eval_below_promotion_threshold"
            ),
            metrics={
                "promotion/eval/reward_mean": score,
                "promotion/eval/trajectories": float(len(self.scenarios)),
                "promotion/eval/failures": float(failures),
                "promotion/eval/action_units": float(action_units),
                "promotion/eval/source_tokens": float(source_tokens),
                "promotion/eval/duration_s": duration_s,
                "promotion/eval/dollar_seconds": dollar_seconds,
            },
        )


@dataclass(frozen=True)
class RolloutContext:
    """Runtime metadata passed to user-defined rollout workflows."""

    actor_id: int
    policy_step: int
    action_codec: ActionCodec
    scheduler_arm_id: str | None = None
    decision_metadata: Mapping[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)


class RolloutWorkflow(Protocol):
    """Callable that performs one arbitrary agentic workflow."""

    async def __call__(
        self,
        policy: AgentPolicy,
        scenario: Scenario,
        context: RolloutContext,
    ) -> Trajectory:
        ...


@dataclass(frozen=True)
class ControlPlaneConfig:
    num_actors: int = 8
    group_size: int = 4
    train_batch_groups: int = 4
    max_train_steps: int = 8
    queue_max_trajectories: int = 128
    train_queue_capacity: int = 3
    max_policy_lag: int = 1
    cost_per_second_usd: float = 1.0

    def validate(self) -> None:
        positive = {
            "num_actors": self.num_actors,
            "group_size": self.group_size,
            "train_batch_groups": self.train_batch_groups,
            "max_train_steps": self.max_train_steps,
            "queue_max_trajectories": self.queue_max_trajectories,
            "train_queue_capacity": self.train_queue_capacity,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        if self.cost_per_second_usd < 0:
            raise ValueError("cost_per_second_usd must be non-negative")


@dataclass(frozen=True)
class GroupAddResult:
    accepted: bool
    groups: tuple[TrajectoryGroup, ...] = ()


@dataclass(frozen=True)
class VersionedTrajectoryBatch:
    """Fixed train payload with explicit collection-version metadata."""

    groups: tuple[TrajectoryGroup, ...]
    assembled_at_step: int
    priority_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    on_discard: Callable[["VersionedTrajectoryBatch"], None] | None = None
    created_at: float = field(default_factory=time.perf_counter)
    min_policy_step: int = field(init=False)
    max_policy_step: int = field(init=False)
    num_trajectories: int = field(init=False)
    source_tokens: int = field(init=False)
    action_units: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("VersionedTrajectoryBatch requires at least one group")
        trajectories = [
            trajectory
            for group in self.groups
            for trajectory in group.trajectories
        ]
        if not trajectories:
            raise ValueError("VersionedTrajectoryBatch requires trajectories")
        policy_steps = [trajectory.policy_step for trajectory in trajectories]
        object.__setattr__(self, "min_policy_step", min(policy_steps))
        object.__setattr__(self, "max_policy_step", max(policy_steps))
        object.__setattr__(self, "num_trajectories", len(trajectories))
        object.__setattr__(
            self,
            "source_tokens",
            sum(trajectory.token_count for trajectory in trajectories),
        )
        object.__setattr__(
            self,
            "action_units",
            sum(trajectory.action_units for trajectory in trajectories),
        )

    @property
    def mean_reward(self) -> float:
        return mean([group.mean_reward for group in self.groups])

    def max_lag_at(self, policy_step: int) -> int:
        return max(0, policy_step - self.min_policy_step)


class TrajectoryRingBuffer:
    """Bounded train-batch queue with Puffer-style backpressure and staleness."""

    def __init__(self, *, capacity: int, max_policy_lag: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        self.capacity = capacity
        self.max_policy_lag = max_policy_lag
        self.current_policy_step = 0
        self.total_produced = 0
        self.total_consumed = 0
        self.total_discarded = 0
        self.backpressure_events = 0
        self.priority_consumptions = 0
        self.consumed_priority_total = 0.0
        self._batches: deque[VersionedTrajectoryBatch] = deque()
        self._condition = asyncio.Condition()

    async def put(self, batch: VersionedTrajectoryBatch) -> None:
        async with self._condition:
            while len(self._batches) >= self.capacity:
                self.backpressure_events += 1
                await self._condition.wait()
            self._batches.append(batch)
            self.total_produced += 1
            self._condition.notify_all()

    async def get(self, *, current_policy_step: int) -> VersionedTrajectoryBatch:
        async with self._condition:
            self.current_policy_step = current_policy_step
            while True:
                while not self._batches:
                    await self._condition.wait()

                stale_count = self._discard_stale_locked(current_policy_step)
                if stale_count:
                    self._condition.notify_all()
                    if not self._batches:
                        continue

                batch = self._pop_highest_priority_locked()

                self.total_consumed += 1
                self.consumed_priority_total += batch.priority_score
                self._condition.notify_all()
                return batch

    @property
    def pending_batches(self) -> int:
        return len(self._batches)

    @property
    def pending_groups(self) -> int:
        return sum(len(batch.groups) for batch in self._batches)

    def stats(self) -> dict[str, float]:
        max_pending_priority = (
            max(batch.priority_score for batch in self._batches)
            if self._batches
            else 0.0
        )
        return {
            "capacity": float(self.capacity),
            "pending_batches": float(self.pending_batches),
            "pending_groups": float(self.pending_groups),
            "produced_batches": float(self.total_produced),
            "consumed_batches": float(self.total_consumed),
            "discarded_batches": float(self.total_discarded),
            "backpressure_events": float(self.backpressure_events),
            "current_policy_step": float(self.current_policy_step),
            "priority_consumptions": float(self.priority_consumptions),
            "consumed_priority_total": self.consumed_priority_total,
            "max_pending_priority": max_pending_priority,
        }

    def _discard_stale_locked(self, current_policy_step: int) -> int:
        kept: deque[VersionedTrajectoryBatch] = deque()
        stale_count = 0
        while self._batches:
            batch = self._batches.popleft()
            if batch.max_lag_at(current_policy_step) > self.max_policy_lag:
                if batch.on_discard is not None:
                    batch.on_discard(batch)
                stale_count += 1
            else:
                kept.append(batch)
        self._batches = kept
        self.total_discarded += stale_count
        return stale_count

    def _pop_highest_priority_locked(self) -> VersionedTrajectoryBatch:
        best_index = 0
        best_batch = self._batches[0]
        for index, batch in enumerate(self._batches):
            if (
                batch.priority_score > best_batch.priority_score
                or (
                    batch.priority_score == best_batch.priority_score
                    and batch.created_at < best_batch.created_at
                )
            ):
                best_index = index
                best_batch = batch
        if best_index != 0:
            self.priority_consumptions += 1
        del self._batches[best_index]
        return best_batch


@dataclass(frozen=True)
class WeightUpdate:
    """Published checkpoint event for inference workers or adapters."""

    step: int
    checkpoint_id: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class WeightBroadcastChannel:
    """Broadcasts checkpoint updates without coupling actors to the trainer."""

    def __init__(self) -> None:
        self._latest: WeightUpdate | None = None
        self._subscribers: list[asyncio.Queue[WeightUpdate]] = []
        self._condition = asyncio.Condition()
        self.broadcast_count = 0

    def subscribe(
        self,
        *,
        replay_latest: bool = True,
        maxsize: int = 0,
    ) -> asyncio.Queue[WeightUpdate]:
        queue: asyncio.Queue[WeightUpdate] = asyncio.Queue(maxsize=maxsize)
        if replay_latest and self._latest is not None:
            queue.put_nowait(self._latest)
        self._subscribers.append(queue)
        return queue

    async def publish(self, snapshot: PolicySnapshot) -> WeightUpdate:
        update = WeightUpdate(
            step=snapshot.step,
            checkpoint_id=snapshot.checkpoint_id,
            created_at=snapshot.created_at,
            metadata=dict(snapshot.metadata),
        )
        async with self._condition:
            self._latest = update
            self.broadcast_count += 1
            subscribers = tuple(self._subscribers)
            self._condition.notify_all()

        for queue in subscribers:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(update)
        return update

    async def wait_for_step(self, step: int) -> WeightUpdate:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._latest is not None and self._latest.step >= step
            )
            assert self._latest is not None
            return self._latest


class TrajectoryGrouper:
    """Builds same-scenario groups while enforcing bounded policy staleness."""

    def __init__(self, group_size: int) -> None:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.group_size = group_size
        self._pending: dict[str, deque[Trajectory]] = defaultdict(deque)
        self.stale_dropped = 0

    def add(
        self,
        trajectory: Trajectory,
        *,
        latest_step: int,
        max_policy_lag: int,
    ) -> GroupAddResult:
        if latest_step - trajectory.policy_step > max_policy_lag:
            self.stale_dropped += 1
            return GroupAddResult(accepted=False)

        bucket = self._pending[trajectory.scenario_id]
        bucket.append(trajectory)
        groups: list[TrajectoryGroup] = []
        while len(bucket) >= self.group_size:
            trajectories = tuple(bucket.popleft() for _ in range(self.group_size))
            groups.append(
                TrajectoryGroup(
                    scenario_id=trajectory.scenario_id,
                    trajectories=trajectories,
                    metrics={
                        "group/mean_policy_step": mean(
                            [float(t.policy_step) for t in trajectories]
                        ),
                        "group/mean_reward": mean([t.reward for t in trajectories]),
                    },
                )
            )
        return GroupAddResult(accepted=True, groups=tuple(groups))

    @property
    def pending_trajectories(self) -> int:
        return sum(len(bucket) for bucket in self._pending.values())


class PolicyRegistry:
    def __init__(self, initial_policy: AgentPolicy | PolicySnapshot) -> None:
        now = time.time()
        if isinstance(initial_policy, PolicySnapshot):
            self._snapshot = initial_policy
            self._checkpoints: list[Checkpoint] = [
                Checkpoint(
                    step=initial_policy.step,
                    checkpoint_id=initial_policy.checkpoint_id,
                    created_at=initial_policy.created_at,
                    metadata=dict(initial_policy.metadata),
                )
            ]
        else:
            self._snapshot = PolicySnapshot(
                step=0,
                policy=initial_policy,
                checkpoint_id="step-0",
                created_at=now,
            )
            self._checkpoints = [
                Checkpoint(step=0, checkpoint_id="step-0", created_at=now)
            ]
        self._condition = asyncio.Condition()

    async def snapshot(self) -> PolicySnapshot:
        async with self._condition:
            return self._snapshot

    async def publish(self, result: TrainResult) -> PolicySnapshot:
        async with self._condition:
            next_step = self._snapshot.step + 1
            next_policy = result.policy if result.policy is not None else self._snapshot.policy
            checkpoint_id = result.checkpoint_id or f"step-{next_step}"
            now = time.time()
            self._snapshot = PolicySnapshot(
                step=next_step,
                policy=next_policy,
                checkpoint_id=checkpoint_id,
                created_at=now,
                metadata=dict(result.metadata),
            )
            self._checkpoints.append(
                Checkpoint(
                    step=next_step,
                    checkpoint_id=checkpoint_id,
                    created_at=now,
                    metrics=dict(result.metrics),
                    metadata=dict(result.metadata),
                )
            )
            self._condition.notify_all()
            return self._snapshot

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(self._checkpoints)


class ScenarioSampler:
    def __init__(self, scenarios: Sequence[Scenario]) -> None:
        if not scenarios:
            raise ValueError("at least one scenario is required")
        self._scenarios = tuple(scenarios)
        self._index = 0
        self._lock = asyncio.Lock()

    async def next(self) -> Scenario:
        async with self._lock:
            scenario = self._scenarios[self._index % len(self._scenarios)]
            self._index += 1
            return scenario


class RuntimeTelemetry:
    def __init__(self, *, cost_per_second_usd: float) -> None:
        self.cost_per_second_usd = cost_per_second_usd
        self.started_at = time.perf_counter()
        self.trajectories_seen = 0
        self.trajectories_accepted = 0
        self.trajectories_failed = 0
        self.groups_trained = 0
        self.train_steps = 0
        self.action_units = 0
        self.source_tokens = 0
        self.rollout_s = 0.0
        self.rollout_dollar_seconds = 0.0
        self.actor_queue_wait_s = 0.0
        self.trainer_s = 0.0
        self.trainer_dollar_seconds = 0.0
        self.rewards: list[float] = []
        self.train_rewards: list[float] = []
        self.action_quality_total = 0.0
        self.action_quality_count = 0
        self.unsafe_trajectories = 0
        self.promotion_evaluations = 0
        self.promotions = 0
        self.promotion_rejections = 0
        self.promotion_eval_dollar_seconds = 0.0
        self.latest_promotion_score = 0.0
        self.latest_promotion_baseline_score = 0.0
        self.latest_promotion_improvement = 0.0
        self.latest_promotion_promoted = False

    def record_actor_queue_wait(self, seconds: float) -> None:
        self.actor_queue_wait_s += max(0.0, seconds)

    def record_trajectory(
        self,
        trajectory: Trajectory,
        *,
        accepted: bool,
        dollar_seconds: float | None = None,
    ) -> None:
        self.trajectories_seen += 1
        self.rollout_s += max(0.0, trajectory.duration_s)
        if dollar_seconds is None:
            dollar_seconds = max(0.0, trajectory.duration_s) * self.cost_per_second_usd
        self.rollout_dollar_seconds += max(0.0, dollar_seconds)
        quality = action_quality(trajectory)
        self.action_quality_total += quality
        self.action_quality_count += 1
        if quality <= 0.0:
            self.unsafe_trajectories += 1
        if accepted:
            self.trajectories_accepted += 1
            self.action_units += trajectory.action_units
            self.source_tokens += trajectory.token_count
            effective_reward = trajectory.reward * quality
            if isfinite(effective_reward):
                self.rewards.append(effective_reward)
        if trajectory.exception:
            self.trajectories_failed += 1

    def record_train(
        self,
        groups: Sequence[TrajectoryGroup],
        result: TrainResult,
        *,
        duration_s: float,
        dollar_seconds: float | None = None,
    ) -> None:
        self.train_steps += 1
        self.groups_trained += len(groups)
        self.trainer_s += max(0.0, duration_s)
        if dollar_seconds is None:
            dollar_seconds = max(0.0, duration_s) * self.cost_per_second_usd
        self.trainer_dollar_seconds += max(0.0, dollar_seconds)
        group_rewards = [group.mean_reward for group in groups]
        self.train_rewards.append(mean(group_rewards))
        metric_reward = result.metrics.get("train/reward")
        if metric_reward is not None and isfinite(metric_reward):
            self.train_rewards.append(float(metric_reward))

    def record_promotion(self, decision: PromotionDecision) -> None:
        self.promotion_evaluations += 1
        if decision.promoted:
            self.promotions += 1
        else:
            self.promotion_rejections += 1
        self.promotion_eval_dollar_seconds += max(0.0, decision.dollar_seconds)
        self.latest_promotion_score = decision.score
        self.latest_promotion_baseline_score = decision.baseline_score
        self.latest_promotion_improvement = decision.improvement
        self.latest_promotion_promoted = decision.promoted

    def metrics(self, *, stale_dropped: int) -> dict[str, float]:
        wall_s = max(time.perf_counter() - self.started_at, 1e-9)
        first_reward, last_reward = self._reward_windows()
        reward_delta = last_reward - first_reward
        dollar_seconds = wall_s * self.cost_per_second_usd
        rollout_dollar_seconds = self.rollout_dollar_seconds
        trainer_dollar_seconds = self.trainer_dollar_seconds
        queue_wait_dollar_seconds = (
            self.actor_queue_wait_s * self.cost_per_second_usd
        )
        accounted_dollar_seconds = (
            rollout_dollar_seconds
            + trainer_dollar_seconds
            + queue_wait_dollar_seconds
            + self.promotion_eval_dollar_seconds
        )
        if dollar_seconds > 0:
            reward_experience = (
                max(0.0, reward_delta)
                * self.trajectories_accepted
                / max(dollar_seconds, 1e-9)
            )
        else:
            reward_experience = 0.0
        if accounted_dollar_seconds > 0:
            accounted_reward_experience = (
                max(0.0, reward_delta)
                * self.trajectories_accepted
                / max(accounted_dollar_seconds, 1e-9)
            )
        else:
            accounted_reward_experience = 0.0

        return {
            "time/wall_clock_s": wall_s,
            "time/rollout_s": self.rollout_s,
            "time/trainer_s": self.trainer_s,
            "time/actor_queue_wait_s": self.actor_queue_wait_s,
            "data/trajectories_seen": float(self.trajectories_seen),
            "data/trajectories_accepted": float(self.trajectories_accepted),
            "data/trajectories_failed": float(self.trajectories_failed),
            "data/unsafe_trajectories": float(self.unsafe_trajectories),
            "data/stale_trajectories_dropped": float(stale_dropped),
            "data/groups_trained": float(self.groups_trained),
            "data/train_steps": float(self.train_steps),
            "data/checkpoints_promoted": float(self.promotions),
            "throughput/accepted_trajectories_per_s": self.trajectories_accepted
            / wall_s,
            "throughput/action_units_per_s": self.action_units / wall_s,
            "throughput/source_tokens_per_s": self.source_tokens / wall_s,
            "actions/semantic_bandwidth_tokens_per_decision": self.source_tokens
            / self.action_units
            if self.action_units
            else 0.0,
            "actions/quality_mean": self.action_quality_total
            / self.action_quality_count
            if self.action_quality_count
            else 0.0,
            "reward/first_window_mean": first_reward,
            "reward/last_window_mean": last_reward,
            "reward/delta": reward_delta,
            "costs/runtime_dollar_seconds": dollar_seconds,
            "costs/wall_clock_dollar_seconds": dollar_seconds,
            "costs/rollout_dollar_seconds": rollout_dollar_seconds,
            "costs/trainer_dollar_seconds": trainer_dollar_seconds,
            "costs/promotion_eval_dollar_seconds": (
                self.promotion_eval_dollar_seconds
            ),
            "costs/actor_queue_wait_dollar_seconds": queue_wait_dollar_seconds,
            "costs/accounted_dollar_seconds": accounted_dollar_seconds,
            "promotion/evaluations": float(self.promotion_evaluations),
            "promotion/promoted": float(self.promotions),
            "promotion/rejected": float(self.promotion_rejections),
            "promotion/rate": (
                self.promotions / self.promotion_evaluations
                if self.promotion_evaluations
                else 0.0
            ),
            "promotion/latest_score": self.latest_promotion_score,
            "promotion/latest_baseline_score": (
                self.latest_promotion_baseline_score
            ),
            "promotion/latest_improvement": self.latest_promotion_improvement,
            "promotion/latest_promoted": (
                1.0 if self.latest_promotion_promoted else 0.0
            ),
            "north_star/reward_improving_experience_per_dollar_second": reward_experience,
            "north_star/accounted_reward_improving_experience_per_dollar_second": (
                accounted_reward_experience
            ),
        }

    def _reward_windows(self) -> tuple[float, float]:
        values = self.rewards or self.train_rewards
        if not values:
            return 0.0, 0.0
        window = max(1, min(10, len(values) // 3 or 1))
        return fmean(values[:window]), fmean(values[-window:])


class ControlPlane:
    """Continuous actor/trainer scheduler for ART-shaped workflows."""

    def __init__(self, config: ControlPlaneConfig | None = None) -> None:
        self.config = config or ControlPlaneConfig()
        self.config.validate()

    async def run(
        self,
        *,
        scenarios: Sequence[Scenario],
        initial_policy: AgentPolicy | PolicySnapshot,
        trainer: TrainerBackend,
        workflow: RolloutWorkflow,
        action_codec: ActionCodec | None = None,
        action_codecs: Sequence[ActionCodec] | None = None,
        action_space: AdaptiveActionSpace | None = None,
        weight_channel: WeightBroadcastChannel | None = None,
        scheduler: AdaptiveScheduler | None = None,
        promotion_evaluator: PromotionEvaluator | None = None,
    ) -> RunSummary:
        restore_control_state(
            initial_policy,
            scheduler=scheduler,
            action_space=action_space,
        )
        if action_space is not None and action_codec is None and action_codecs is None:
            codecs = action_space.codecs
        else:
            codecs = self._resolve_action_codecs(action_codec, action_codecs)
        if action_space is not None:
            for codec in codecs:
                action_space.add_codec(codec)
            codecs = action_space.codecs
        registry = PolicyRegistry(initial_policy)
        sampler = ScenarioSampler(scenarios)
        grouper = TrajectoryGrouper(self.config.group_size)
        telemetry = RuntimeTelemetry(cost_per_second_usd=self.config.cost_per_second_usd)
        trajectory_queue: asyncio.Queue[Trajectory] = asyncio.Queue(
            maxsize=self.config.queue_max_trajectories
        )
        train_ring = TrajectoryRingBuffer(
            capacity=self.config.train_queue_capacity,
            max_policy_lag=self.config.max_policy_lag,
        )
        broadcaster = weight_channel or WeightBroadcastChannel()
        stop = asyncio.Event()
        ready_groups: deque[TrajectoryGroup] = deque()
        actors = [
            asyncio.create_task(
                self._actor_loop(
                    actor_id=actor_id,
                    stop=stop,
                    registry=registry,
                    sampler=sampler,
                    scenarios=tuple(scenarios),
                    workflow=workflow,
                    action_codecs=codecs,
                    action_space=action_space,
                    trajectory_queue=trajectory_queue,
                    telemetry=telemetry,
                    train_ring=train_ring,
                    scheduler=scheduler,
                )
            )
            for actor_id in range(self.config.num_actors)
        ]
        batcher = asyncio.create_task(
            self._batcher_loop(
                stop=stop,
                registry=registry,
                grouper=grouper,
                trajectory_queue=trajectory_queue,
                train_ring=train_ring,
                telemetry=telemetry,
                ready_groups=ready_groups,
                scheduler=scheduler,
            )
        )

        try:
            for _ in range(self.config.max_train_steps):
                current = await registry.snapshot()
                if not self._should_continue_training(
                    scheduler=scheduler,
                    train_ring=train_ring,
                    policy_step=current.step,
                ):
                    break
                train_ring.max_policy_lag = self._max_policy_lag(
                    scheduler=scheduler,
                    train_ring=train_ring,
                    policy_step=current.step,
                )
                batch = await train_ring.get(current_policy_step=current.step)
                self._tag_batch_control_metadata(
                    batch.groups,
                    max_policy_lag=train_ring.max_policy_lag,
                )
                train_started = time.perf_counter()
                result = await trainer.train(current, batch.groups)
                train_duration_s = time.perf_counter() - train_started
                train_dollar_seconds = train_result_dollar_seconds(
                    result,
                    duration_s=train_duration_s,
                    cost_per_second_usd=self.config.cost_per_second_usd,
                )
                promotion = await self._evaluate_promotion(
                    evaluator=promotion_evaluator,
                    current=current,
                    result=result,
                    groups=batch.groups,
                )
                result = _with_promotion_metadata(result, promotion)
                telemetry.record_train(
                    batch.groups,
                    result,
                    duration_s=train_duration_s,
                    dollar_seconds=train_dollar_seconds,
                )
                telemetry.record_promotion(promotion)
                if scheduler is not None:
                    scheduler.observe_train(
                        groups=batch.groups,
                        result=result,
                        duration_s=train_duration_s,
                        dollar_seconds=train_dollar_seconds,
                        policy_step=current.step,
                    )
                    if action_space is not None:
                        action_space.update_from_metrics(scheduler.metrics())
                if promotion.promoted:
                    latest = await registry.publish(
                        _with_control_checkpoint_metadata(
                            result,
                            scheduler=scheduler,
                            action_space=action_space,
                        )
                    )
                    train_ring.current_policy_step = latest.step
                    await broadcaster.publish(latest)
                else:
                    train_ring.current_policy_step = current.step
        finally:
            stop.set()
            for actor in actors:
                actor.cancel()
            batcher.cancel()
            await asyncio.gather(*actors, batcher, return_exceptions=True)

        latest = await registry.snapshot()
        metrics = telemetry.metrics(stale_dropped=grouper.stale_dropped)
        metrics["data/stale_train_batches_dropped"] = float(train_ring.total_discarded)
        for key, value in train_ring.stats().items():
            metrics[f"train_queue/{key}"] = float(value)
        metrics["weights/broadcasts"] = float(broadcaster.broadcast_count)
        if scheduler is not None:
            metrics.update(scheduler.metrics())
        if action_space is not None:
            metrics.update(action_space.metrics())
        return RunSummary(
            latest_step=latest.step,
            checkpoints=registry.checkpoints,
            metrics=metrics,
            pending_trajectories=grouper.pending_trajectories
            + trajectory_queue.qsize(),
            pending_groups=len(ready_groups) + train_ring.pending_groups,
        )

    async def _evaluate_promotion(
        self,
        *,
        evaluator: PromotionEvaluator | None,
        current: PolicySnapshot,
        result: TrainResult,
        groups: Sequence[TrajectoryGroup],
    ) -> PromotionDecision:
        if evaluator is None:
            score = _result_metric_score(result, groups)
            return PromotionDecision(
                promoted=True,
                score=score,
                baseline_score=0.0,
                improvement=max(0.0, score),
                reason="promote_all",
            )

        started = time.perf_counter()
        decision_or_awaitable = evaluator(
            current=current,
            result=result,
            groups=groups,
        )
        decision = (
            await decision_or_awaitable
            if inspect.isawaitable(decision_or_awaitable)
            else decision_or_awaitable
        )
        if not isinstance(decision, PromotionDecision):
            raise TypeError("promotion_evaluator must return PromotionDecision")
        if decision.dollar_seconds > 0.0:
            return decision
        return replace(
            decision,
            dollar_seconds=(
                (time.perf_counter() - started)
                * self.config.cost_per_second_usd
            ),
        )

    async def _batcher_loop(
        self,
        *,
        stop: asyncio.Event,
        registry: PolicyRegistry,
        grouper: TrajectoryGrouper,
        trajectory_queue: asyncio.Queue[Trajectory],
        train_ring: TrajectoryRingBuffer,
        telemetry: RuntimeTelemetry,
        ready_groups: deque[TrajectoryGroup],
        scheduler: AdaptiveScheduler | None,
    ) -> None:
        while not stop.is_set():
            trajectory = await trajectory_queue.get()
            latest = await registry.snapshot()
            max_policy_lag = self._max_policy_lag(
                scheduler=scheduler,
                train_ring=train_ring,
                policy_step=latest.step,
            )
            result = grouper.add(
                trajectory,
                latest_step=latest.step,
                max_policy_lag=max_policy_lag,
            )
            rollout_dollar_seconds = self._trajectory_dollar_seconds(trajectory)
            queue_wait_dollar_seconds = self._trajectory_queue_wait_dollar_seconds(
                trajectory
            )
            telemetry.record_trajectory(
                trajectory,
                accepted=result.accepted,
                dollar_seconds=rollout_dollar_seconds,
            )
            if scheduler is not None:
                scheduler.observe_rollout(
                    trajectory,
                    accepted=result.accepted,
                    dollar_seconds=rollout_dollar_seconds,
                    queue_wait_dollar_seconds=queue_wait_dollar_seconds,
                )
            ready_groups.extend(result.groups)

            target_batch_groups = self._target_train_batch_groups(
                scheduler=scheduler,
                ready_groups=ready_groups,
                train_ring=train_ring,
                policy_step=latest.step,
            )
            while len(ready_groups) >= target_batch_groups:
                groups = tuple(
                    ready_groups.popleft()
                    for _ in range(target_batch_groups)
                )
                self._tag_batch_control_metadata(
                    groups,
                    target_train_batch_groups=target_batch_groups,
                )
                latest = await registry.snapshot()
                await train_ring.put(
                    VersionedTrajectoryBatch(
                        groups=groups,
                        assembled_at_step=latest.step,
                        priority_score=self._score_train_groups(
                            scheduler=scheduler,
                            groups=groups,
                            policy_step=latest.step,
                        ),
                        on_discard=self._stale_batch_callback(
                            scheduler=scheduler,
                            train_ring=train_ring,
                            reason="runtime_train_ring_stale",
                        ),
                    )
                )

    async def _actor_loop(
        self,
        *,
        actor_id: int,
        stop: asyncio.Event,
        registry: PolicyRegistry,
        sampler: ScenarioSampler,
        scenarios: Sequence[Scenario],
        workflow: RolloutWorkflow,
        action_codecs: Sequence[ActionCodec],
        action_space: AdaptiveActionSpace | None,
        trajectory_queue: asyncio.Queue[Trajectory],
        telemetry: RuntimeTelemetry,
        train_ring: TrajectoryRingBuffer,
        scheduler: AdaptiveScheduler | None,
    ) -> None:
        while not stop.is_set():
            snapshot = await registry.snapshot()
            decision = await self._select_rollout(
                scheduler=scheduler,
                sampler=sampler,
                scenarios=scenarios,
                action_codecs=(
                    action_space.codecs if action_space is not None else action_codecs
                ),
                actor_id=actor_id,
                policy_step=snapshot.step,
                trajectory_queue=trajectory_queue,
                train_ring=train_ring,
            )
            context = RolloutContext(
                actor_id=actor_id,
                policy_step=snapshot.step,
                action_codec=decision.action_codec,
                scheduler_arm_id=decision.arm_id,
                decision_metadata=decision.metadata,
            )
            started = time.perf_counter()
            try:
                trajectory = await workflow(snapshot.policy, decision.scenario, context)
                trajectory.duration_s = time.perf_counter() - started
            except Exception as exc:
                trajectory = Trajectory(
                    scenario_id=decision.scenario.id,
                    policy_step=snapshot.step,
                    messages=[],
                    actions=[],
                    reward=0.0,
                    duration_s=time.perf_counter() - started,
                    exception=f"{type(exc).__name__}: {exc}",
                    metadata={"actor_id": actor_id},
                )
            trajectory.metadata.setdefault("actor_id", actor_id)
            trajectory.metadata.setdefault("scheduler/arm_id", decision.arm_id)
            trajectory.metadata.setdefault("scheduler/scenario_id", decision.scenario.id)
            trajectory.metadata.setdefault(
                "scheduler/action_codec",
                getattr(decision.action_codec, "name", decision.action_codec.__class__.__name__),
            )
            trajectory.metadata.setdefault(
                "scheduler/target_train_batch_groups",
                decision.target_train_batch_groups,
            )
            trajectory.metadata.setdefault(
                "scheduler/max_policy_lag",
                decision.max_policy_lag,
            )

            queue_started = time.perf_counter()
            queue_wait_s = await self._put_trajectory_with_queue_cost(
                trajectory_queue,
                trajectory,
                started_at=queue_started,
            )
            telemetry.record_actor_queue_wait(queue_wait_s)

    def _resolve_action_codecs(
        self,
        action_codec: ActionCodec | None,
        action_codecs: Sequence[ActionCodec] | None,
    ) -> tuple[ActionCodec, ...]:
        if action_codecs is not None:
            codecs = tuple(action_codecs)
        elif action_codec is not None:
            codecs = (action_codec,)
        else:
            codecs = ()
        if not codecs:
            raise ValueError("at least one action codec is required")
        return codecs

    async def _select_rollout(
        self,
        *,
        scheduler: AdaptiveScheduler | None,
        sampler: ScenarioSampler,
        scenarios: Sequence[Scenario],
        action_codecs: Sequence[ActionCodec],
        actor_id: int,
        policy_step: int,
        trajectory_queue: asyncio.Queue[Trajectory],
        train_ring: TrajectoryRingBuffer,
    ) -> SchedulerDecision:
        if scheduler is not None:
            return scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=action_codecs,
                actor_id=actor_id,
                policy_step=policy_step,
                trajectory_queue_pressure=self._queue_pressure(trajectory_queue),
                train_queue_pressure=self._train_queue_pressure(train_ring),
                configured_train_batch_groups=self.config.train_batch_groups,
                configured_max_policy_lag=self.config.max_policy_lag,
            )
        scenario = await sampler.next()
        codec = action_codecs[0]
        return SchedulerDecision(
            scenario=scenario,
            action_codec=codec,
            arm_id=f"{scenario.id}|{getattr(codec, 'name', codec.__class__.__name__)}",
            target_train_batch_groups=self.config.train_batch_groups,
            max_policy_lag=self.config.max_policy_lag,
        )

    def _target_train_batch_groups(
        self,
        *,
        scheduler: AdaptiveScheduler | None,
        ready_groups: deque[TrajectoryGroup],
        train_ring: TrajectoryRingBuffer,
        policy_step: int,
    ) -> int:
        if scheduler is None:
            return self.config.train_batch_groups
        return max(
            1,
            scheduler.target_train_batch_groups(
                configured=self.config.train_batch_groups,
                pending_groups=len(ready_groups),
                train_queue_pressure=self._train_queue_pressure(train_ring),
                policy_step=policy_step,
            ),
        )

    def _max_policy_lag(
        self,
        *,
        scheduler: AdaptiveScheduler | None,
        train_ring: TrajectoryRingBuffer,
        policy_step: int,
    ) -> int:
        if scheduler is None:
            return self.config.max_policy_lag
        return max(
            0,
            scheduler.max_policy_lag(
                configured=self.config.max_policy_lag,
                train_queue_pressure=self._train_queue_pressure(train_ring),
                policy_step=policy_step,
            ),
        )

    def _score_train_groups(
        self,
        *,
        scheduler: AdaptiveScheduler | None,
        groups: Sequence[TrajectoryGroup],
        policy_step: int,
    ) -> float:
        if scheduler is None:
            return 0.0
        return scheduler.score_train_groups(groups, policy_step=policy_step)

    @staticmethod
    def _stale_batch_callback(
        *,
        scheduler: AdaptiveScheduler | None,
        train_ring: TrajectoryRingBuffer,
        reason: str,
    ) -> Callable[[VersionedTrajectoryBatch], None] | None:
        if scheduler is None:
            return None

        def on_discard(batch: VersionedTrajectoryBatch) -> None:
            observe_stale_batch_feedback(
                scheduler,
                groups=batch.groups,
                policy_step=train_ring.current_policy_step,
                reason=reason,
            )

        return on_discard

    def _should_continue_training(
        self,
        *,
        scheduler: AdaptiveScheduler | None,
        train_ring: TrajectoryRingBuffer,
        policy_step: int,
    ) -> bool:
        if scheduler is None:
            return policy_step < self.config.max_train_steps
        return scheduler.should_continue_training(
            policy_step=policy_step,
            max_train_steps=self.config.max_train_steps,
            pending_train_batches=train_ring.pending_batches,
            train_queue_pressure=self._train_queue_pressure(train_ring),
        )

    def _trajectory_dollar_seconds(self, trajectory: Trajectory) -> float:
        explicit_cost = _first_nonnegative_float(
            trajectory.metrics,
            ("cost/dollar_seconds", "rollout/dollar_seconds"),
        )
        if explicit_cost is None:
            explicit_cost = _first_nonnegative_float(
                trajectory.metadata,
                ("cost/dollar_seconds", "rollout/dollar_seconds"),
            )
        if explicit_cost is not None:
            return explicit_cost
        return max(0.0, trajectory.duration_s) * self.config.cost_per_second_usd

    def _trajectory_queue_wait_dollar_seconds(self, trajectory: Trajectory) -> float:
        explicit_cost = _first_nonnegative_float(
            trajectory.metrics,
            ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
        )
        if explicit_cost is None:
            explicit_cost = _first_nonnegative_float(
                trajectory.metadata,
                (
                    "cost/actor_queue_wait_dollar_seconds",
                    "queue_wait/dollar_seconds",
                ),
            )
        return explicit_cost or 0.0

    async def _put_trajectory_with_queue_cost(
        self,
        queue: asyncio.Queue[Trajectory],
        trajectory: Trajectory,
        *,
        started_at: float,
    ) -> float:
        try:
            queue.put_nowait(trajectory)
            return 0.0
        except asyncio.QueueFull:
            pass

        while True:
            queue_wait_s = time.perf_counter() - started_at
            queue_wait_dollar_seconds = (
                queue_wait_s * self.config.cost_per_second_usd
            )
            if queue_wait_dollar_seconds > 0.0:
                trajectory.metrics["cost/actor_queue_wait_dollar_seconds"] = (
                    queue_wait_dollar_seconds
                )
            try:
                queue.put_nowait(trajectory)
                return queue_wait_s
            except asyncio.QueueFull:
                await asyncio.sleep(0)

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

    @staticmethod
    def _queue_pressure(queue: asyncio.Queue[Trajectory]) -> float:
        maxsize = queue.maxsize
        if maxsize <= 0:
            return 0.0
        return min(1.0, queue.qsize() / maxsize)

    @staticmethod
    def _train_queue_pressure(train_ring: TrajectoryRingBuffer) -> float:
        return min(1.0, train_ring.pending_batches / train_ring.capacity)


def trajectory_semantic_bandwidth(trajectory: Trajectory) -> float:
    return semantic_bandwidth(trajectory.actions)


def train_result_dollar_seconds(
    result: TrainResult,
    *,
    duration_s: float,
    cost_per_second_usd: float,
) -> float:
    explicit_cost = _first_nonnegative_float(
        result.metrics,
        ("cost/dollar_seconds", "train/dollar_seconds", "trainer/dollar_seconds"),
    )
    if explicit_cost is None:
        explicit_cost = _first_nonnegative_float(
            result.metadata,
            (
                "cost/dollar_seconds",
                "train/dollar_seconds",
                "trainer/dollar_seconds",
            ),
        )
    if explicit_cost is not None:
        return explicit_cost
    return max(0.0, duration_s) * cost_per_second_usd


def _trajectory_eval_dollar_seconds(
    trajectory: Trajectory,
    *,
    cost_per_second_usd: float,
) -> float:
    explicit_cost = _first_nonnegative_float(
        trajectory.metrics,
        ("cost/dollar_seconds", "eval/dollar_seconds", "rollout/dollar_seconds"),
    )
    if explicit_cost is None:
        explicit_cost = _first_nonnegative_float(
            trajectory.metadata,
            (
                "cost/dollar_seconds",
                "eval/dollar_seconds",
                "rollout/dollar_seconds",
            ),
        )
    if explicit_cost is not None:
        return explicit_cost
    return max(0.0, trajectory.duration_s) * cost_per_second_usd


def _with_promotion_metadata(
    result: TrainResult,
    decision: PromotionDecision,
) -> TrainResult:
    metrics = dict(result.metrics)
    effective_score = decision.score if decision.promoted else decision.baseline_score
    metrics.update(
        {
            "promotion/promoted": 1.0 if decision.promoted else 0.0,
            "promotion/score": effective_score,
            "promotion/candidate_score": decision.score,
            "promotion/baseline_score": decision.baseline_score,
            "promotion/improvement": decision.improvement,
            "promotion/dollar_seconds": max(0.0, decision.dollar_seconds),
        }
    )
    for key, value in decision.metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and isfinite(float(value)):
            metrics[str(key)] = float(value)

    metadata = dict(result.metadata)
    metadata.update(
        {
            "promotion/promoted": decision.promoted,
            "promotion/reason": decision.reason,
            "promotion/score": decision.score,
            "promotion/effective_score": effective_score,
            "promotion/baseline_score": decision.baseline_score,
            "promotion/improvement": decision.improvement,
            "promotion/dollar_seconds": max(0.0, decision.dollar_seconds),
        }
    )
    return TrainResult(
        policy=result.policy,
        metrics=metrics,
        checkpoint_id=result.checkpoint_id,
        metadata=metadata,
    )


def _result_metric_score(
    result: TrainResult,
    groups: Sequence[TrajectoryGroup],
    *,
    preferred_key: str = "train/reward",
) -> float:
    for key in (preferred_key, "promotion/score", "eval/reward", "train/reward"):
        value = result.metrics.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and isfinite(float(value)):
            return float(value)
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if isfinite(parsed):
                return parsed
    return mean([group.mean_reward for group in groups])


def _first_nonnegative_float(
    values: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and isfinite(float(value)):
            return max(0.0, float(value))
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if isfinite(parsed):
                return max(0.0, parsed)
    return None


def _with_control_checkpoint_metadata(
    result: TrainResult,
    *,
    scheduler: AdaptiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
) -> TrainResult:
    control_metadata = {
        **scheduler_checkpoint_metadata(scheduler),
        **action_space_checkpoint_metadata(action_space),
    }
    if not control_metadata:
        return result
    metadata = dict(result.metadata)
    metadata.update(control_metadata)
    return TrainResult(
        policy=result.policy,
        metrics=result.metrics,
        checkpoint_id=result.checkpoint_id,
        metadata=metadata,
    )


def restore_control_state(
    source: Mapping[str, Any] | PolicySnapshot | Checkpoint | None,
    *,
    scheduler: AdaptiveScheduler | None = None,
    action_space: AdaptiveActionSpace | None = None,
) -> dict[str, bool]:
    """Restore scheduler/action-space state from checkpoint-style metadata."""

    metadata = _checkpoint_metadata(source)
    restored = {"scheduler": False, "action_space": False}
    scheduler_state = metadata.get(SCHEDULER_STATE_KEY)
    scheduler_loader = getattr(scheduler, "load_state_dict", None)
    if isinstance(scheduler_state, Mapping) and scheduler_loader is not None:
        scheduler_loader(scheduler_state)
        restored["scheduler"] = True

    action_space_state = metadata.get(ACTION_SPACE_STATE_KEY)
    action_space_loader = getattr(action_space, "load_state_dict", None)
    if isinstance(action_space_state, Mapping) and action_space_loader is not None:
        action_space_loader(action_space_state)
        restored["action_space"] = True
    return restored


def _checkpoint_metadata(
    source: Mapping[str, Any] | PolicySnapshot | Checkpoint | None,
) -> Mapping[str, Any]:
    if source is None:
        return {}
    if isinstance(source, (PolicySnapshot, Checkpoint)):
        return source.metadata
    if not isinstance(source, MappingABC):
        return {}
    nested = source.get("metadata")
    if isinstance(nested, MappingABC):
        return nested
    return source
