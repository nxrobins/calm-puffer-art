from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence

from .actions import ActionCodec, ChunkActionCodec, TokenActionCodec
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


NORTH_STAR = "north_star/reward_improving_experience_per_dollar_second"
ACCOUNTED_NORTH_STAR = (
    "north_star/accounted_reward_improving_experience_per_dollar_second"
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
            metrics={"train/reward": fmean(rewards)},
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
        metrics={"cost/dollar_seconds": rollout_cost},
    )


async def run_static_ablation() -> RunSummary:
    return await _run(scheduler=None)


async def run_objective_ablation() -> RunSummary:
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        exploration_bonus=0.0,
    )
    return await _run(scheduler=scheduler)


async def run_ablation() -> dict[str, Any]:
    static = await run_static_ablation()
    objective = await run_objective_ablation()
    static_score = float(static.metrics[NORTH_STAR])
    objective_score = float(objective.metrics[NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "north_star_absolute": objective_score - static_score,
            "north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def _run(scheduler: ObjectiveScheduler | None) -> RunSummary:
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=1,
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
        "scheduler/control/cadence_1/train_updates",
        "scheduler/control/policy_lag_2/train_updates",
    ]
    return {
        key: float(summary.metrics[key])
        for key in keys
        if key in summary.metrics
    }
