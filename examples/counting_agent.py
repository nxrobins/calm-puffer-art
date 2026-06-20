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
    PolicySnapshot,
    RolloutContext,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


@dataclass(frozen=True)
class CountingPolicy:
    level: int

    async def act(
        self,
        messages: Sequence[Message],
        *,
        scenario: Scenario,
        codec: ActionCodec,
    ):
        target = int(scenario.payload["target"])
        guess = min(self.level, target)
        return codec.encode(f"answer {guess}")


class CountingTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [trajectory.reward for group in groups for trajectory in group.trajectories]
        current_policy = current.policy
        next_policy = CountingPolicy(level=current_policy.level + 1)
        return TrainResult(
            policy=next_policy,
            checkpoint_id=f"counting-level-{next_policy.level}",
            metrics={"train/reward": fmean(rewards), "policy/level": float(next_policy.level)},
        )


async def rollout(policy: CountingPolicy, scenario: Scenario, context: RolloutContext) -> Trajectory:
    messages = [
        Message(role="system", content="Return an answer command."),
        Message(role="user", content=f"Count to {scenario.payload['target']}."),
    ]
    actions = await policy.act(messages, scenario=scenario, codec=context.action_codec)
    decoded = context.action_codec.decode(actions)
    guess = int(decoded.split()[-1])
    target = int(scenario.payload["target"])
    reward = guess / target
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages + [Message(role="assistant", content=decoded)],
        actions=actions,
        reward=reward,
        metrics={"guess": float(guess), "target": float(target)},
        metadata={"actor_id": context.actor_id},
    )


async def main() -> None:
    scenarios = [Scenario(id=f"count-{target}", payload={"target": target}) for target in (3, 4, 5)]
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=6,
            group_size=3,
            train_batch_groups=2,
            max_train_steps=5,
            queue_max_trajectories=12,
            max_policy_lag=1,
            cost_per_second_usd=0.50,
        )
    )
    summary = await runtime.run(
        scenarios=scenarios,
        initial_policy=CountingPolicy(level=0),
        trainer=CountingTrainer(),
        workflow=rollout,
        action_codec=ChunkActionCodec(chunk_size=2),
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
