from __future__ import annotations

import asyncio
import json
from statistics import fmean
from typing import Sequence

from calm_puffer_art import (
    AdaptiveActionSpace,
    ActionCodec,
    ControlPlane,
    ControlPlaneConfig,
    Message,
    ObjectiveScheduler,
    PolicySnapshot,
    RolloutContext,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


class EchoPolicy:
    async def act(
        self,
        messages: Sequence[Message],
        *,
        scenario: Scenario,
        codec: ActionCodec,
    ):
        return codec.encode("alpha beta gamma delta")


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
            checkpoint_id=f"step-{current.step + 1}",
            metrics={"train/reward": fmean(rewards)},
        )


async def rollout(
    policy: EchoPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content="adapt action bandwidth")]
    actions = await policy.act(messages, scenario=scenario, codec=context.action_codec)
    chunk_size = int(getattr(context.action_codec, "chunk_size", 1))
    reward = {1: 0.1, 2: 1.0, 4: 1.25}.get(chunk_size, 0.0)
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
    )


async def main() -> None:
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=1,
        min_policy_lag=1,
        max_policy_lag=2,
        exploration_bonus=0.0,
    )
    action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=1,
            group_size=1,
            train_batch_groups=1,
            max_train_steps=6,
            queue_max_trajectories=4,
            train_queue_capacity=2,
            max_policy_lag=2,
        )
    )
    summary = await runtime.run(
        scenarios=[Scenario(id="bandwidth")],
        initial_policy=EchoPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=rollout,
        action_space=action_space,
        scheduler=scheduler,
    )
    interesting = {
        key: value
        for key, value in summary.metrics.items()
        if key.startswith("action_space/")
        or key.startswith("scheduler/arm/bandwidth_")
        or key.startswith("north_star/")
        or key == "reward/delta"
    }
    print(json.dumps(interesting, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
