from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence

from .actions import (
    ActionCodec,
    AdaptiveActionSpace,
    ChunkActionCodec,
    TokenActionCodec,
)
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
        "action_space/decision/realized_source_token_throughput_payoff",
        "scheduler/control/cadence_1/train_updates",
        "scheduler/control/policy_lag_2/train_updates",
        "scheduler/control/actor_count_1/rollout_updates",
        "scheduler/control/actor_count_2/rollout_updates",
        "scheduler/control/actor_count_1/score",
        "scheduler/control/actor_count_2/score",
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
        "scheduler/joint_action/feedback_tuples",
        "scheduler/joint_action/positive_objective_tuples",
        "scheduler/joint_action/total_objective",
        "scheduler/last_train_batch_joint_action_score",
    ]
    return {
        key: float(summary.metrics[key])
        for key in keys
        if key in summary.metrics
    }
