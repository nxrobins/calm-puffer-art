from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

from calm_puffer_art import (
    ActionCodec,
    ChunkActionCodec,
    ControlPlane,
    ControlPlaneConfig,
    Message,
    ObjectiveScheduler,
    PolicySnapshot,
    RolloutContext,
    Scenario,
    TokenActionCodec,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


@dataclass(frozen=True)
class StaticPolicy:
    async def act(
        self,
        messages: Sequence[Message],
        *,
        scenario: Scenario,
        codec: ActionCodec,
    ):
        return codec.encode(f"{scenario.id} via {codec.name}")


class EchoTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [trajectory.reward for group in groups for trajectory in group.trajectories]
        return TrainResult(
            policy=current.policy,
            checkpoint_id=f"objective-step-{current.step + 1}",
            metrics={"train/reward": fmean(rewards)},
        )


async def rollout(
    policy: StaticPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content=f"try {scenario.id}")]
    actions = await policy.act(messages, scenario=scenario, codec=context.action_codec)
    reward = float(scenario.payload.get(context.action_codec.name, 0.0))
    safe = bool(scenario.payload.get(f"{context.action_codec.name}_safe", True))
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
        metadata={
            "action/safe": safe,
            "action/quality": 1.0 if safe else 0.0,
            "verifier/passed": safe,
        },
    )


async def main() -> None:
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=3,
        exploration_bonus=0.0,
    )
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=1,
            group_size=1,
            train_batch_groups=2,
            max_train_steps=16,
            queue_max_trajectories=8,
            train_queue_capacity=2,
            max_policy_lag=3,
            cost_per_second_usd=1.0,
        )
    )
    summary = await runtime.run(
        scenarios=[
            Scenario(id="easy", payload={"token": 0.1, "chunk": 1.0}),
            Scenario(
                id="risky",
                payload={"token": 0.3, "chunk": 2.0, "chunk_safe": False},
            ),
        ],
        initial_policy=StaticPolicy(),
        trainer=EchoTrainer(),
        workflow=rollout,
        action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
        scheduler=scheduler,
    )

    interesting = {
        key: value
        for key, value in summary.metrics.items()
        if key.startswith("scheduler/")
        or key.startswith("train_queue/priority")
        or key == "train_queue/consumed_priority_total"
        or key.startswith("north_star/")
        or key in {"reward/delta", "data/groups_trained"}
    }
    print(json.dumps(interesting, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
