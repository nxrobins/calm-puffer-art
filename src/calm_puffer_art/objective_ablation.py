from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from .actions import (
    ActionCodec,
    ActionUnit,
    AdaptiveActionSpace,
    ChunkActionCodec,
    TokenActionCodec,
)
from .art_adapter import AsyncArtBackend, AsyncArtBackendConfig
from .runtime import ControlPlane, ControlPlaneConfig, RolloutContext
from .scheduler import ObjectiveScheduler
from .types import (
    Message,
    PolicySnapshot,
    RunSummary,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


NORTH_STAR = (
    "north_star/published_policy_reward_improving_experience_per_dollar_second"
)
ACCOUNTED_NORTH_STAR = (
    "north_star/accounted_published_policy_reward_improving_experience_per_dollar_second"
)
ART_NORTH_STAR = (
    "art_backend/published_policy_reward_improving_experience_per_dollar_second"
)
ART_ACCOUNTED_NORTH_STAR = (
    "art_backend/accounted_published_policy_reward_improving_experience_per_dollar_second"
)


@dataclass(frozen=True)
class AblationPolicy:
    async def act(
        self,
        messages: Sequence[Message],
        *,
        scenario: Scenario,
        codec: ActionCodec,
    ):
        return codec.encode(f"{scenario.id} {codec.name}")


class MeanRewardTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [
            trajectory.reward
            for group in groups
            for trajectory in group.trajectories
        ]
        return TrainResult(
            policy=current.policy,
            checkpoint_id=f"ablation-step-{current.step + 1}",
            metrics={
                "train/reward": fmean(rewards),
                "train/dollar_seconds": 1.0,
            },
        )


@dataclass(frozen=True)
class _AblationArtMessage:
    role: str
    content: str


@dataclass(frozen=True)
class _AblationArtChoice:
    message: _AblationArtMessage


@dataclass
class _AblationArtTrajectory:
    messages_and_choices: list[Any]
    reward: float
    initial_policy_version: int
    final_policy_version: int
    metrics: dict[str, float]
    metadata: dict[str, Any]


@dataclass
class _AblationArtGroup:
    trajectories: list[_AblationArtTrajectory]
    metadata: dict[str, Any]
    metrics: dict[str, float] | None = None

    def __iter__(self):
        return iter(self.trajectories)


@dataclass(frozen=True)
class _AblationArtTrainResult:
    step: int
    metrics: dict[str, float]
    checkpoint_path: str


class _AblationArtBackend:
    def __init__(self) -> None:
        self.step = 0
        self.calls = 0

    async def register(self, model: Any) -> None:
        return None

    async def _get_step(self, model: Any) -> int:
        return self.step

    async def train(
        self,
        model: Any,
        trajectory_groups: Sequence[_AblationArtGroup],
        **kwargs: Any,
    ) -> _AblationArtTrainResult:
        self.step += 1
        self.calls += 1
        rewards = [
            trajectory.reward
            for group in trajectory_groups
            for trajectory in group.trajectories
        ]
        reward = fmean(rewards) if rewards else 0.0
        return _AblationArtTrainResult(
            step=self.step,
            metrics={
                "train/reward": reward,
                "train/dollar_seconds": 1.0,
            },
            checkpoint_path=f".art/ablation/model/step_{self.step}",
        )


async def ablation_rollout(
    policy: AblationPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content="pick useful action bandwidth")]
    actions = await policy.act(messages, scenario=scenario, codec=context.action_codec)
    reward = float(scenario.payload.get(context.action_codec.name, 0.0))
    rollout_cost = float(
        scenario.payload.get(f"{context.action_codec.name}_dollar_seconds", 1.0)
    )
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
        metrics={"rollout/dollar_seconds": rollout_cost},
    )


async def action_space_ablation_rollout(
    policy: AblationPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content="pick useful semantic bandwidth")]
    actions = context.action_codec.encode(
        "alpha beta gamma delta epsilon zeta eta theta"
    )
    if isinstance(context.action_codec, ChunkActionCodec):
        codec_key = f"chunk_{context.action_codec.chunk_size}"
    else:
        codec_key = context.action_codec.name
    reward = float(scenario.payload.get(codec_key, 0.0))
    rollout_cost = float(scenario.payload.get(f"{codec_key}_dollar_seconds", 1.0))
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
        metrics={"rollout/dollar_seconds": rollout_cost},
    )


async def run_static_ablation() -> RunSummary:
    return await _run(scheduler=None)


async def run_objective_ablation() -> RunSummary:
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        min_actor_count=1,
        max_actor_count=2,
        exploration_bonus=0.0,
    )
    return await _run(scheduler=scheduler)


async def run_fixed_action_space_ablation() -> RunSummary:
    return await _run_action_space(
        scheduler=_objective_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
    )


async def run_adaptive_action_space_ablation() -> RunSummary:
    return await _run_action_space(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )


async def run_static_closed_loop_ablation() -> RunSummary:
    return await _run_closed_loop(
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
    )


async def run_objective_closed_loop_ablation() -> RunSummary:
    return await _run_closed_loop(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )


async def run_static_art_bridge_ablation() -> dict[str, float]:
    return await _run_art_bridge(
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
    )


async def run_objective_art_bridge_ablation() -> dict[str, float]:
    return await _run_art_bridge(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )


async def run_ablation() -> dict[str, Any]:
    static = await run_static_ablation()
    objective = await run_objective_ablation()
    static_score = float(static.metrics[NORTH_STAR])
    objective_score = float(objective.metrics[NORTH_STAR])
    accounted_static_score = float(static.metrics[ACCOUNTED_NORTH_STAR])
    accounted_objective_score = float(objective.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "north_star_absolute": objective_score - static_score,
            "north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
            "accounted_north_star_absolute": (
                accounted_objective_score - accounted_static_score
            ),
            "accounted_north_star_ratio": (
                accounted_objective_score / accounted_static_score
                if accounted_static_score > 0.0
                else None
            ),
        },
    }


async def run_action_space_ablation() -> dict[str, Any]:
    fixed = await run_fixed_action_space_ablation()
    adaptive = await run_adaptive_action_space_ablation()
    fixed_score = float(fixed.metrics[ACCOUNTED_NORTH_STAR])
    adaptive_score = float(adaptive.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "fixed": summary_metrics(fixed),
        "adaptive": summary_metrics(adaptive),
        "lift": {
            "accounted_north_star_absolute": adaptive_score - fixed_score,
            "accounted_north_star_ratio": (
                adaptive_score / fixed_score if fixed_score > 0.0 else None
            ),
        },
    }


async def run_closed_loop_ablation() -> dict[str, Any]:
    static = await run_static_closed_loop_ablation()
    objective = await run_objective_closed_loop_ablation()
    static_score = float(static.metrics[ACCOUNTED_NORTH_STAR])
    objective_score = float(objective.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "accounted_north_star_absolute": objective_score - static_score,
            "accounted_north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def run_art_bridge_ablation() -> dict[str, Any]:
    static = await run_static_art_bridge_ablation()
    objective = await run_objective_art_bridge_ablation()
    static_score = float(static[ART_ACCOUNTED_NORTH_STAR])
    objective_score = float(objective[ART_ACCOUNTED_NORTH_STAR])
    return {
        "static": static,
        "objective": objective,
        "lift": {
            "accounted_north_star_absolute": objective_score - static_score,
            "accounted_north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def _run(scheduler: ObjectiveScheduler | None) -> RunSummary:
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=2,
            group_size=1,
            train_batch_groups=2,
            max_train_steps=8,
            queue_max_trajectories=4,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=[
            Scenario(
                id="bandwidth",
                payload={
                    "token": 0.1,
                    "chunk": 1.0,
                    "token_dollar_seconds": 1.0,
                    "chunk_dollar_seconds": 1.5,
                },
            )
        ],
        initial_policy=AblationPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=ablation_rollout,
        action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
        scheduler=scheduler,
    )


async def _run_action_space(
    *,
    scheduler: ObjectiveScheduler,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec] | None,
) -> RunSummary:
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=1,
            group_size=1,
            train_batch_groups=1,
            max_train_steps=8,
            queue_max_trajectories=4,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=[
            Scenario(
                id="semantic",
                payload={
                    "token": 0.1,
                    "chunk_2": 1.0,
                    "chunk_4": 4.0,
                    "token_dollar_seconds": 1.0,
                    "chunk_2_dollar_seconds": 1.0,
                    "chunk_4_dollar_seconds": 1.0,
                },
            )
        ],
        initial_policy=AblationPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=action_space_ablation_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


async def _run_closed_loop(
    *,
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec] | None,
) -> RunSummary:
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=2,
            group_size=1,
            train_batch_groups=1,
            max_train_steps=8,
            queue_max_trajectories=4,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=[
            Scenario(
                id="closed_loop",
                payload={
                    "token": 0.1,
                    "chunk_2": 1.0,
                    "chunk_4": 4.0,
                    "token_dollar_seconds": 1.0,
                    "chunk_2_dollar_seconds": 1.0,
                    "chunk_4_dollar_seconds": 1.0,
                },
            )
        ],
        initial_policy=AblationPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=action_space_ablation_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


async def _run_art_bridge(
    *,
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec] | None,
) -> dict[str, float]:
    backend = AsyncArtBackend(
        backend=_AblationArtBackend(),
        config=AsyncArtBackendConfig(
            train_queue_capacity=2,
            train_batch_groups=1,
            max_policy_lag=2,
            max_train_steps=8,
            cost_per_second_usd=1.0,
        ),
        scheduler=scheduler,
        action_space=action_space,
    )
    scenarios = [
        Scenario(
            id="art_bridge",
            payload={
                "token": 0.1,
                "chunk_2": 1.0,
                "chunk_4": 4.0,
                "token_dollar_seconds": 1.0,
                "chunk_2_dollar_seconds": 1.0,
                "chunk_4_dollar_seconds": 1.0,
            },
        )
    ]
    futures = []
    await backend.register("art-model")
    rollout_limit = 8 if scheduler is None else 64
    try:
        for actor_id in range(rollout_limit):
            assignment = await backend.admit_and_select_rollout(
                scenarios=scenarios,
                action_codecs=action_codecs,
                actor_id=actor_id % 2,
                configured_actor_count=2,
                trajectory_queue_pressure=backend.ring.pending_batches
                / backend.ring.capacity,
            )
            if not assignment.admitted or assignment.decision is None:
                break
            group = _art_bridge_group_from_assignment(assignment.metadata)
            futures.append(await backend.submit_group("art-model", group))
            await asyncio.sleep(0)
        await backend.flush_pending_groups()
        if futures:
            await asyncio.gather(*futures)
        return bridge_summary_metrics(backend.stats())
    finally:
        await backend.close()


def _art_bridge_group_from_assignment(
    metadata: Mapping[str, Any],
) -> _AblationArtGroup:
    scenario_id = str(metadata.get("scheduler/scenario_id", "art_bridge"))
    text = "alpha beta gamma delta epsilon zeta eta theta"
    codec_key = str(metadata.get("scheduler/action_codec", "token"))
    actions = _bridge_actions_for_codec(codec_key, text)
    reward_key = _payload_key_for_codec(codec_key)
    payload = {
        "token": 0.1,
        "chunk_2": 1.0,
        "chunk_4": 4.0,
        "token_dollar_seconds": 1.0,
        "chunk_2_dollar_seconds": 1.0,
        "chunk_4_dollar_seconds": 1.0,
    }
    reward = float(payload.get(reward_key, 0.0))
    rollout_cost = float(payload.get(f"{reward_key}_dollar_seconds", 1.0))
    policy_step = int(float(metadata.get("scheduler/policy_step", 0)))
    trajectory_metadata = {
        **metadata,
        "scenario_id": scenario_id,
    }
    trajectory = _AblationArtTrajectory(
        messages_and_choices=[
            _AblationArtMessage(role="user", content="pick useful ART action"),
            *[
                _AblationArtChoice(
                    _AblationArtMessage(role="assistant", content=action.text)
                )
                for action in actions
            ],
        ],
        reward=reward,
        initial_policy_version=policy_step,
        final_policy_version=policy_step,
        metrics={"rollout/dollar_seconds": rollout_cost},
        metadata=trajectory_metadata,
    )
    return _AblationArtGroup(
        trajectories=[trajectory],
        metadata={"scenario_id": scenario_id},
        metrics={},
    )


def _bridge_actions_for_codec(codec_key: str, text: str) -> tuple[ActionUnit, ...]:
    if codec_key == "chunk(chunk_size=4)":
        return tuple(ChunkActionCodec(chunk_size=4).encode(text))
    if codec_key == "chunk(chunk_size=2)":
        return tuple(ChunkActionCodec(chunk_size=2).encode(text))
    return tuple(TokenActionCodec().encode(text))


def _payload_key_for_codec(codec_key: str) -> str:
    if codec_key == "chunk(chunk_size=4)":
        return "chunk_4"
    if codec_key == "chunk(chunk_size=2)":
        return "chunk_2"
    return "token"


def _objective_scheduler() -> ObjectiveScheduler:
    return ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        min_actor_count=1,
        max_actor_count=2,
        exploration_bonus=0.0,
    )


def summary_metrics(summary: RunSummary) -> dict[str, float]:
    keys = [
        NORTH_STAR,
        ACCOUNTED_NORTH_STAR,
        "reward/delta",
        "data/groups_trained",
        "data/train_steps",
        "actions/semantic_bandwidth_tokens_per_decision",
        "scheduler/arm/bandwidth_chunk_chunk_size_2/pulls",
        "scheduler/arm/bandwidth_chunk_chunk_size_2/mean_rollout_dollar_seconds",
        "scheduler/arm/bandwidth_chunk_chunk_size_2/total_improvement_per_dollar_second",
        "scheduler/arm/bandwidth_token/pulls",
        "scheduler/arm/bandwidth_token/mean_rollout_dollar_seconds",
        "scheduler/arm/bandwidth_token/total_improvement_per_dollar_second",
        "scheduler/arm/semantic_chunk_chunk_size_2/pulls",
        "scheduler/arm/semantic_chunk_chunk_size_2/mean_rollout_dollar_seconds",
        "scheduler/arm/semantic_chunk_chunk_size_2/total_improvement_per_dollar_second",
        "scheduler/arm/semantic_chunk_chunk_size_4/pulls",
        "scheduler/arm/semantic_chunk_chunk_size_4/mean_rollout_dollar_seconds",
        "scheduler/arm/semantic_chunk_chunk_size_4/total_improvement_per_dollar_second",
        "scheduler/arm/semantic_token/pulls",
        "scheduler/arm/semantic_token/mean_rollout_dollar_seconds",
        "scheduler/arm/semantic_token/total_improvement_per_dollar_second",
        "scheduler/arm/closed_loop_chunk_chunk_size_2/pulls",
        "scheduler/arm/closed_loop_chunk_chunk_size_2/mean_rollout_dollar_seconds",
        "scheduler/arm/closed_loop_chunk_chunk_size_2/total_improvement_per_dollar_second",
        "scheduler/arm/closed_loop_chunk_chunk_size_4/pulls",
        "scheduler/arm/closed_loop_chunk_chunk_size_4/mean_rollout_dollar_seconds",
        "scheduler/arm/closed_loop_chunk_chunk_size_4/total_improvement_per_dollar_second",
        "scheduler/arm/closed_loop_token/pulls",
        "scheduler/arm/closed_loop_token/mean_rollout_dollar_seconds",
        "scheduler/arm/closed_loop_token/total_improvement_per_dollar_second",
        "action_space/active_codecs",
        "action_space/promotions",
        "action_space/max_chunk_size",
        "action_space/codec/chunk_chunk_size_4/active",
        "action_space/decision/decisions",
        "action_space/decision/post_decision_observations",
        "action_space/decision/realized_objective_payoff",
        "action_space/decision/mean_realized_objective_payoff_per_decision",
        "action_space/decision/"
        "mean_realized_objective_payoff_per_post_decision_observation",
        "action_space/decision/realized_source_token_throughput_payoff",
        "action_space/decision/"
        "mean_realized_source_token_throughput_payoff_per_decision",
        "action_space/decision/"
        "mean_realized_source_token_throughput_payoff_per_post_decision_observation",
        "scheduler/control/cadence_1/train_updates",
        "scheduler/control/cadence_1/mean_objective_per_decision",
        "scheduler/control/cadence_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_1/train_updates",
        "scheduler/control/policy_lag_1/mean_objective_per_decision",
        "scheduler/control/policy_lag_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_2/train_updates",
        "scheduler/control/policy_lag_2/mean_objective_per_decision",
        "scheduler/control/policy_lag_2/mean_objective_per_feedback_update",
        "scheduler/control/admission_delay_ms_0/rollout_updates",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_decision",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_1/rollout_updates",
        "scheduler/control/actor_count_1/mean_objective_per_decision",
        "scheduler/control/actor_count_1/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_2/rollout_updates",
        "scheduler/control/actor_count_2/mean_objective_per_decision",
        "scheduler/control/actor_count_2/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_1/score",
        "scheduler/control/actor_count_2/score",
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
        "scheduler/joint_action/feedback_tuples",
        "scheduler/joint_action/positive_objective_tuples",
        "scheduler/joint_action/total_objective",
        "scheduler/joint_action/mean_objective_per_decision",
        "scheduler/joint_action/mean_objective_per_feedback_update",
        "scheduler/last_train_batch_joint_action_score",
    ]
    return {
        key: float(summary.metrics[key])
        for key in keys
        if key in summary.metrics
    }


def bridge_summary_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    keys = [
        ART_NORTH_STAR,
        ART_ACCOUNTED_NORTH_STAR,
        "art_backend/accounted_dollar_seconds",
        "art_backend/sample_dollar_seconds",
        "art_backend/trainer_dollar_seconds",
        "art_backend/submitted_groups",
        "art_backend/submitted_train_groups",
        "art_backend/completed_batches",
        "art_backend/published_policy_updates",
        "art_backend/published_policy_reward_improving_experience",
        "actions/semantic_bandwidth_tokens_per_decision",
        "action_space/active_codecs",
        "action_space/promotions",
        "action_space/max_chunk_size",
        "action_space/codec/chunk_chunk_size_4/active",
        "action_space/decision/decisions",
        "action_space/decision/post_decision_observations",
        "action_space/decision/realized_objective_payoff",
        "action_space/decision/mean_realized_objective_payoff_per_decision",
        "action_space/decision/"
        "mean_realized_objective_payoff_per_post_decision_observation",
        "scheduler/arm/art_bridge_chunk_chunk_size_2/pulls",
        "scheduler/arm/art_bridge_chunk_chunk_size_4/pulls",
        "scheduler/arm/art_bridge_token/pulls",
        "scheduler/control/cadence_1/train_updates",
        "scheduler/control/cadence_1/mean_objective_per_decision",
        "scheduler/control/cadence_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_1/train_updates",
        "scheduler/control/policy_lag_1/mean_objective_per_decision",
        "scheduler/control/policy_lag_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_2/train_updates",
        "scheduler/control/policy_lag_2/mean_objective_per_decision",
        "scheduler/control/policy_lag_2/mean_objective_per_feedback_update",
        "scheduler/control/admission_delay_ms_0/rollout_updates",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_decision",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_1/rollout_updates",
        "scheduler/control/actor_count_1/mean_objective_per_decision",
        "scheduler/control/actor_count_1/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_2/rollout_updates",
        "scheduler/control/actor_count_2/mean_objective_per_decision",
        "scheduler/control/actor_count_2/mean_objective_per_feedback_update",
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
        "scheduler/joint_action/positive_objective_tuples",
        "scheduler/joint_action/total_objective",
        "scheduler/joint_action/mean_objective_per_decision",
        "scheduler/joint_action/mean_objective_per_feedback_update",
        "scheduler/last_train_batch_joint_action_score",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}
