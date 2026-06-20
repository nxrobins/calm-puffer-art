import asyncio
import time
import unittest
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

from calm_puffer_art import (
    ACTION_SPACE_STATE_KEY,
    ActionCodec,
    AdaptiveActionSpace,
    ChunkActionCodec,
    ControlPlane,
    ControlPlaneConfig,
    MetricPromotionEvaluator,
    Message,
    ObjectiveScheduler,
    PolicySnapshot,
    PROMOTION_STATE_KEY,
    PromotionDecision,
    RolloutContext,
    RolloutPromotionEvaluator,
    SCHEDULER_STATE_KEY,
    Scenario,
    TokenActionCodec,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
    TrajectoryRingBuffer,
    VersionedTrajectoryBatch,
    WeightBroadcastChannel,
    action_space_checkpoint_metadata,
    promotion_checkpoint_metadata,
    restore_control_state,
    scheduler_checkpoint_metadata,
    train_result_dollar_seconds,
)
from calm_puffer_art.actions import action_codec_key
from calm_puffer_art.runtime import RuntimeTelemetry, TrajectoryGrouper


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
        return codec.encode(f"answer {min(self.level, target)}")


class CountingTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [trajectory.reward for group in groups for trajectory in group.trajectories]
        next_policy = CountingPolicy(level=current.policy.level + 1)
        return TrainResult(
            policy=next_policy,
            checkpoint_id=f"level-{next_policy.level}",
            metrics={"train/reward": fmean(rewards), "policy/level": float(next_policy.level)},
        )


class NoopTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [trajectory.reward for group in groups for trajectory in group.trajectories]
        return TrainResult(
            policy=current.policy,
            checkpoint_id=f"step-{current.step + 1}",
            metrics={"train/reward": fmean(rewards)},
        )


class CostedTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [trajectory.reward for group in groups for trajectory in group.trajectories]
        return TrainResult(
            policy=current.policy,
            checkpoint_id=f"step-{current.step + 1}",
            metrics={
                "train/reward": fmean(rewards),
                "train/dollar_seconds": 42.0,
            },
        )


class FixedCostTrainer:
    def __init__(self, *, score: float, dollar_seconds: float) -> None:
        self.score = score
        self.dollar_seconds = dollar_seconds

    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        return TrainResult(
            policy=current.policy,
            checkpoint_id=f"step-{current.step + 1}",
            metrics={
                "train/reward": self.score,
                "train/dollar_seconds": self.dollar_seconds,
            },
        )


class SequencedMetricTrainer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = tuple(scores)
        self.calls = 0

    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return TrainResult(
            policy=CountingPolicy(level=current.policy.level + self.calls),
            checkpoint_id=f"candidate-{self.calls}",
            metrics={
                "train/reward": score,
                "eval/reward": score,
            },
        )


class FixedCostPromotionEvaluator:
    def __init__(self, *, score: float, dollar_seconds: float) -> None:
        self.score = score
        self.dollar_seconds = dollar_seconds

    async def __call__(
        self,
        *,
        current: PolicySnapshot,
        result: TrainResult,
        groups: Sequence[TrajectoryGroup],
    ) -> PromotionDecision:
        return PromotionDecision(
            promoted=True,
            score=self.score,
            baseline_score=0.0,
            improvement=self.score,
            dollar_seconds=self.dollar_seconds,
            reason="fixed_cost_eval",
        )


class ActorCapScheduler:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.calls = []

    def active_actor_count(self, **kwargs):
        self.calls.append(kwargs)
        return self.cap


async def counting_rollout(
    policy: CountingPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content=f"target={scenario.payload['target']}")]
    actions = await policy.act(messages, scenario=scenario, codec=context.action_codec)
    guess = int(context.action_codec.decode(actions).split()[-1])
    target = int(scenario.payload["target"])
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=guess / target,
    )


async def costed_counting_rollout(
    policy: CountingPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    trajectory = await counting_rollout(policy, scenario, context)
    trajectory.metrics["eval/dollar_seconds"] = 2.0
    return trajectory


async def adaptive_rollout(
    policy: CountingPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content=f"scenario={scenario.id}")]
    actions = context.action_codec.encode(f"{scenario.id} {context.action_codec.name}")
    reward = float(scenario.payload.get(context.action_codec.name, 0.0))
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
    )


async def flat_rollout(
    policy: CountingPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    actions = context.action_codec.encode("flat")
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=[Message(role="user", content="flat")],
        actions=actions,
        reward=0.0,
    )


async def adaptive_chunk_size_rollout(
    policy: CountingPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    chunk_size = int(getattr(context.action_codec, "chunk_size", 1))
    actions = context.action_codec.encode("alpha beta gamma delta")
    reward = {1: 0.1, 2: 1.0, 4: 1.2}.get(chunk_size, 0.0)
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=[Message(role="user", content="adapt chunks")],
        actions=actions,
        reward=reward,
    )


class RuntimeTests(unittest.TestCase):
    def test_control_plane_trains_continuously_and_improves_reward(self):
        async def run():
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=4,
                    group_size=2,
                    train_batch_groups=2,
                    max_train_steps=4,
                    queue_max_trajectories=8,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            summary = await runtime.run(
                scenarios=[
                    Scenario(id="a", payload={"target": 4}),
                    Scenario(id="b", payload={"target": 4}),
                ],
                initial_policy=CountingPolicy(level=0),
                trainer=CountingTrainer(),
                workflow=counting_rollout,
                action_codec=ChunkActionCodec(chunk_size=2),
                weight_channel=channel,
            )
            emitted = []
            while not updates.empty():
                emitted.append(updates.get_nowait())
            return summary, emitted

        summary, emitted = asyncio.run(run())

        self.assertEqual(summary.latest_step, 4)
        self.assertEqual(len(summary.checkpoints), 5)
        self.assertEqual([update.step for update in emitted], [1, 2, 3, 4])
        self.assertGreater(summary.metrics["reward/delta"], 0.0)
        self.assertGreater(
            summary.metrics["north_star/reward_improving_experience_per_dollar_second"],
            0.0,
        )
        self.assertEqual(summary.metrics["data/groups_trained"], 8.0)
        self.assertEqual(summary.metrics["train_queue/consumed_batches"], 4.0)
        self.assertEqual(summary.metrics["weights/broadcasts"], 4.0)

    def test_control_plane_wires_objective_scheduler_into_rollouts(self):
        async def run():
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
                    max_train_steps=5,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=3,
                    cost_per_second_usd=1.0,
                )
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            summary = await runtime.run(
                scenarios=[
                    Scenario(id="easy", payload={"token": 0.1, "chunk": 1.0}),
                    Scenario(id="hard", payload={"token": 0.0, "chunk": 0.2}),
                ],
                initial_policy=CountingPolicy(level=1),
                trainer=NoopTrainer(),
                workflow=adaptive_rollout,
                action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
                scheduler=scheduler,
                weight_channel=channel,
            )
            emitted = []
            while not updates.empty():
                emitted.append(updates.get_nowait())
            return summary, emitted

        summary, emitted = asyncio.run(run())
        arm_metric = "scheduler/arm/easy_chunk_chunk_size_2/pulls"

        self.assertEqual(summary.latest_step, 5)
        self.assertIn("scheduler/global_marginal_objective_ema", summary.metrics)
        self.assertIn(arm_metric, summary.metrics)
        self.assertGreater(summary.metrics[arm_metric], 1.0)
        self.assertEqual(summary.metrics["scheduler/last_target_train_batch_groups"], 1.0)
        self.assertEqual(summary.metrics["scheduler/last_max_policy_lag"], 3.0)
        cadence_credit = [
            value
            for key, value in summary.metrics.items()
            if key.startswith("scheduler/control/cadence_")
            and key.endswith("/train_updates")
        ]
        lag_credit = [
            value
            for key, value in summary.metrics.items()
            if key.startswith("scheduler/control/policy_lag_")
            and key.endswith("/train_updates")
        ]
        actor_count_credit = [
            value
            for key, value in summary.metrics.items()
            if key.startswith("scheduler/control/actor_count_")
            and key.endswith("/rollout_updates")
        ]
        self.assertTrue(any(value > 0.0 for value in cadence_credit))
        self.assertTrue(any(value > 0.0 for value in lag_credit))
        self.assertTrue(any(value > 0.0 for value in actor_count_credit))
        scheduler_state = summary.checkpoints[-1].metadata[SCHEDULER_STATE_KEY]
        self.assertEqual(scheduler_state["version"], 1)
        self.assertIn("easy|chunk(chunk_size=2)", scheduler_state["arms"])
        self.assertGreater(
            scheduler_state["learning_state"]["train_dollar_seconds"],
            0.0,
        )
        self.assertEqual(
            emitted[-1].metadata[SCHEDULER_STATE_KEY]["learning_state"][
                "total_pulls"
            ],
            scheduler_state["learning_state"]["total_pulls"],
        )

    def test_control_plane_promotion_gate_rejects_unimproved_candidate(self):
        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=0,
                max_policy_lag=1,
                exploration_bonus=0.0,
            )
            trainer = SequencedMetricTrainer([0.1, 1.0])
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=2,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            summary = await runtime.run(
                scenarios=[Scenario(id="flat")],
                initial_policy=CountingPolicy(level=0),
                trainer=trainer,
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                scheduler=scheduler,
                weight_channel=channel,
                promotion_evaluator=MetricPromotionEvaluator(
                    metric_key="eval/reward",
                    min_delta=0.5,
                    initial_score=0.0,
                ),
            )
            emitted = []
            while not updates.empty():
                emitted.append(updates.get_nowait())
            return summary, emitted

        summary, emitted = asyncio.run(run())

        self.assertEqual(summary.metrics["data/train_steps"], 2.0)
        self.assertEqual(summary.metrics["data/checkpoints_promoted"], 1.0)
        self.assertEqual(summary.metrics["promotion/evaluations"], 2.0)
        self.assertEqual(summary.metrics["promotion/promoted"], 1.0)
        self.assertEqual(summary.metrics["promotion/rejected"], 1.0)
        self.assertEqual(summary.metrics["weights/broadcasts"], 1.0)
        self.assertEqual(summary.latest_step, 1)
        self.assertEqual(len(summary.checkpoints), 2)
        self.assertEqual([update.step for update in emitted], [1])
        self.assertEqual(summary.checkpoints[-1].checkpoint_id, "candidate-2")
        self.assertTrue(summary.checkpoints[-1].metadata["promotion/promoted"])
        self.assertEqual(
            summary.checkpoints[-1].metadata["promotion/reason"],
            "metric_improved",
        )
        self.assertEqual(
            summary.metrics["scheduler/train_last_reward_improvement"],
            1.0,
        )

    def test_control_plane_resumes_promotion_evaluator_state(self):
        async def first_run():
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=1,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            return await runtime.run(
                scenarios=[Scenario(id="flat")],
                initial_policy=CountingPolicy(level=0),
                trainer=SequencedMetricTrainer([1.0]),
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                promotion_evaluator=MetricPromotionEvaluator(
                    metric_key="eval/reward",
                    min_delta=0.5,
                    initial_score=0.0,
                ),
            )

        async def second_run(snapshot):
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=2,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            summary = await runtime.run(
                scenarios=[Scenario(id="flat")],
                initial_policy=snapshot,
                trainer=SequencedMetricTrainer([1.2]),
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                weight_channel=channel,
                promotion_evaluator=MetricPromotionEvaluator(
                    metric_key="eval/reward",
                    min_delta=0.5,
                    initial_score=0.0,
                ),
            )
            emitted = []
            while not updates.empty():
                emitted.append(updates.get_nowait())
            return summary, emitted

        first = asyncio.run(first_run())
        promotion_state = first.checkpoints[-1].metadata[PROMOTION_STATE_KEY]
        snapshot = PolicySnapshot(
            step=first.latest_step,
            policy=CountingPolicy(level=1),
            checkpoint_id=first.checkpoints[-1].checkpoint_id,
            created_at=first.checkpoints[-1].created_at,
            metadata=first.checkpoints[-1].metadata,
        )
        second, emitted = asyncio.run(second_run(snapshot))

        self.assertEqual(promotion_state["learning_state"]["best_score"], 1.0)
        self.assertEqual(second.latest_step, 1)
        self.assertEqual(len(second.checkpoints), 1)
        self.assertEqual(emitted, [])
        self.assertEqual(second.metrics["promotion/evaluations"], 2.0)
        self.assertEqual(second.metrics["promotion/rejected"], 2.0)
        self.assertEqual(second.metrics["promotion/promoted"], 0.0)
        self.assertEqual(second.metrics["promotion/latest_baseline_score"], 1.0)
        self.assertEqual(second.metrics["promotion/latest_score"], 1.2)

    def test_control_plane_rollout_promotion_evaluator_scores_heldout_workflow(self):
        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=0,
                max_policy_lag=1,
                exploration_bonus=0.0,
            )
            trainer = SequencedMetricTrainer([0.0, 0.0])
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=2,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            summary = await runtime.run(
                scenarios=[Scenario(id="train-flat")],
                initial_policy=CountingPolicy(level=0),
                trainer=trainer,
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                scheduler=scheduler,
                weight_channel=channel,
                promotion_evaluator=RolloutPromotionEvaluator(
                    scenarios=[Scenario(id="heldout", payload={"target": 2})],
                    workflow=costed_counting_rollout,
                    action_codec=TokenActionCodec(),
                    min_delta=0.75,
                    initial_score=0.0,
                    cost_per_second_usd=1.0,
                ),
            )
            emitted = []
            while not updates.empty():
                emitted.append(updates.get_nowait())
            return summary, emitted

        summary, emitted = asyncio.run(run())

        self.assertEqual(summary.metrics["data/train_steps"], 2.0)
        self.assertEqual(summary.metrics["data/checkpoints_promoted"], 1.0)
        self.assertEqual(summary.metrics["promotion/evaluations"], 2.0)
        self.assertEqual(summary.metrics["promotion/promoted"], 1.0)
        self.assertEqual(summary.metrics["promotion/rejected"], 1.0)
        self.assertEqual(summary.latest_step, 1)
        self.assertEqual(len(summary.checkpoints), 2)
        self.assertEqual([update.step for update in emitted], [1])
        self.assertEqual(summary.checkpoints[-1].checkpoint_id, "candidate-2")
        self.assertEqual(
            summary.checkpoints[-1].metrics["promotion/eval/reward_mean"],
            1.0,
        )
        self.assertEqual(
            summary.checkpoints[-1].metadata["promotion/reason"],
            "eval_improved",
        )
        self.assertGreater(
            summary.metrics["costs/promotion_eval_dollar_seconds"],
            0.0,
        )
        self.assertEqual(summary.metrics["costs/promotion_eval_dollar_seconds"], 4.0)
        self.assertEqual(summary.metrics["scheduler/arm/heldout_token/pulls"], 2.0)
        self.assertEqual(summary.metrics["scheduler/arm/heldout_token/accepted"], 2.0)
        self.assertEqual(
            summary.metrics["scheduler/arm/heldout_token/rollout_dollar_seconds"],
            4.0,
        )
        self.assertEqual(
            summary.metrics["scheduler/train_last_reward_improvement"],
            1.0,
        )

    def test_control_plane_promotes_adaptive_action_space_codecs(self):
        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=1,
                max_policy_lag=2,
                exploration_bonus=0.0,
            )
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
            )
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=6,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                    cost_per_second_usd=1.0,
                )
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            summary = await runtime.run(
                scenarios=[Scenario(id="adapt")],
                initial_policy=CountingPolicy(level=1),
                trainer=NoopTrainer(),
                workflow=adaptive_chunk_size_rollout,
                action_space=action_space,
                scheduler=scheduler,
                weight_channel=channel,
            )
            emitted = []
            while not updates.empty():
                emitted.append(updates.get_nowait())
            return summary, emitted

        summary, emitted = asyncio.run(run())

        self.assertEqual(
            summary.metrics["action_space/codec/chunk_chunk_size_4/active"],
            1.0,
        )
        self.assertGreaterEqual(summary.metrics["action_space/promotions"], 1.0)
        self.assertIn("scheduler/arm/adapt_chunk_chunk_size_4/pulls", summary.metrics)
        action_space_state = summary.checkpoints[-1].metadata[ACTION_SPACE_STATE_KEY]
        self.assertEqual(action_space_state["version"], 1)
        self.assertIn(
            "chunk(chunk_size=4)",
            [codec["key"] for codec in action_space_state["active_codecs"]],
        )
        self.assertEqual(
            emitted[-1].metadata[ACTION_SPACE_STATE_KEY]["learning_state"][
                "promotions"
            ],
            action_space_state["learning_state"]["promotions"],
        )

    def test_control_plane_demotes_promoted_chunk_when_parent_has_better_objective(self):
        async def low_value_promoted_chunk_rollout(
            policy: CountingPolicy,
            scenario: Scenario,
            context: RolloutContext,
        ) -> Trajectory:
            chunk_size = int(getattr(context.action_codec, "chunk_size", 1))
            actions = context.action_codec.encode("alpha beta gamma delta")
            reward = {1: 0.1, 2: 1.0, 4: 0.6}.get(chunk_size, 0.0)
            return Trajectory(
                scenario_id=scenario.id,
                policy_step=context.policy_step,
                messages=[Message(role="user", content="adapt chunks")],
                actions=actions,
                reward=reward,
            )

        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=1,
                max_policy_lag=2,
                exploration_bonus=0.0,
            )
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
                demotion_parent_margin=0.0,
            )
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=10,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                    cost_per_second_usd=1.0,
                )
            )
            return await runtime.run(
                scenarios=[Scenario(id="adapt")],
                initial_policy=CountingPolicy(level=1),
                trainer=NoopTrainer(),
                workflow=low_value_promoted_chunk_rollout,
                action_space=action_space,
                scheduler=scheduler,
            )

        summary = asyncio.run(run())

        self.assertEqual(
            summary.metrics["action_space/codec/chunk_chunk_size_4/disabled"],
            1.0,
        )
        self.assertNotIn(
            "action_space/codec/chunk_chunk_size_4/active",
            summary.metrics,
        )
        self.assertGreaterEqual(summary.metrics["action_space/promotions"], 1.0)
        self.assertEqual(summary.metrics["action_space/demotions"], 1.0)
        action_space_state = summary.checkpoints[-1].metadata[ACTION_SPACE_STATE_KEY]
        self.assertIn(
            "chunk(chunk_size=4)",
            action_space_state["disabled_codec_keys"],
        )
        self.assertNotIn(
            "chunk(chunk_size=4)",
            [codec["key"] for codec in action_space_state["active_codecs"]],
        )

    def test_control_plane_promotes_and_demotes_latent_patch_candidate(self):
        async def low_value_latent_rollout(
            policy: CountingPolicy,
            scenario: Scenario,
            context: RolloutContext,
        ) -> Trajectory:
            name = getattr(context.action_codec, "name", "")
            reward = 0.2 if name == "latent_patch" else 1.0 if name == "chunk" else 0.1
            return Trajectory(
                scenario_id=scenario.id,
                policy_step=context.policy_step,
                messages=[Message(role="user", content="adapt latent patches")],
                actions=context.action_codec.encode("alpha beta gamma delta"),
                reward=reward,
            )

        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=1,
                max_policy_lag=2,
                exploration_bonus=0.0,
            )
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=2,
                promote_latent_patches=True,
                latent_patch_latent_size=3,
                demotion_parent_margin=0.0,
            )
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=10,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                    cost_per_second_usd=1.0,
                )
            )
            return await runtime.run(
                scenarios=[Scenario(id="adapt")],
                initial_policy=CountingPolicy(level=1),
                trainer=NoopTrainer(),
                workflow=low_value_latent_rollout,
                action_space=action_space,
                scheduler=scheduler,
            )

        summary = asyncio.run(run())

        latent_metric_key = "latent_patch_latent_size_3_patch_size_2"
        self.assertEqual(summary.metrics["action_space/promotions"], 1.0)
        self.assertEqual(summary.metrics["action_space/demotions"], 1.0)
        self.assertEqual(
            summary.metrics[f"action_space/codec/{latent_metric_key}/disabled"],
            1.0,
        )
        self.assertNotIn(
            f"action_space/codec/{latent_metric_key}/active",
            summary.metrics,
        )
        self.assertNotIn(
            "action_space/codec/chunk_chunk_size_4/active",
            summary.metrics,
        )
        self.assertGreater(
            summary.metrics[
                "scheduler/arm/adapt_latent_patch_latent_size_3_patch_size_2/pulls"
            ],
            0.0,
        )
        action_space_state = summary.checkpoints[-1].metadata[ACTION_SPACE_STATE_KEY]
        self.assertIn(
            "latent_patch(latent_size=3,patch_size=2)",
            action_space_state["disabled_codec_keys"],
        )

    def test_restore_control_state_loads_scheduler_and_action_space_metadata(self):
        source_scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        source_scheduler.observe_rollout(
            Trajectory(
                scenario_id="resume",
                policy_step=3,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "resume|chunk(chunk_size=4)"},
            ),
            accepted=True,
            dollar_seconds=2.0,
        )
        source_action_space = AdaptiveActionSpace(
            min_chunk_size=4,
            max_chunk_size=4,
            include_token=False,
        )
        source_promotion = MetricPromotionEvaluator(
            metric_key="eval/reward",
            min_delta=0.5,
            initial_score=0.0,
        )
        source_promotion.best_score = 2.0
        metadata = {
            **scheduler_checkpoint_metadata(source_scheduler),
            **action_space_checkpoint_metadata(source_action_space),
            **promotion_checkpoint_metadata(source_promotion),
        }
        restored_scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        restored_action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)
        restored_promotion = MetricPromotionEvaluator(
            metric_key="train/reward",
            min_delta=0.0,
            initial_score=0.0,
        )

        restored = restore_control_state(
            {"metadata": metadata},
            scheduler=restored_scheduler,
            action_space=restored_action_space,
            promotion_evaluator=restored_promotion,
        )

        self.assertEqual(
            restored,
            {"scheduler": True, "action_space": True, "promotion": True},
        )
        self.assertEqual(
            restored_scheduler.metrics()["scheduler/total_rollout_observations"],
            1.0,
        )
        self.assertEqual(restored_action_space.min_chunk_size, 4)
        self.assertEqual(
            [action_codec_key(codec) for codec in restored_action_space.codecs],
            ["chunk(chunk_size=4)"],
        )
        self.assertEqual(restored_promotion.metric_key, "eval/reward")
        self.assertEqual(restored_promotion.min_delta, 0.5)
        self.assertEqual(restored_promotion.best_score, 2.0)

    def test_control_plane_resumes_from_policy_snapshot_control_state(self):
        async def run():
            source_scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            source_scheduler.observe_rollout(
                Trajectory(
                    scenario_id="adapt",
                    policy_step=7,
                    messages=[],
                    actions=[],
                    reward=1.2,
                    metadata={"scheduler/arm_id": "adapt|chunk(chunk_size=4)"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )
            source_action_space = AdaptiveActionSpace(
                min_chunk_size=4,
                max_chunk_size=4,
                include_token=False,
            )
            metadata = {
                **scheduler_checkpoint_metadata(source_scheduler),
                **action_space_checkpoint_metadata(source_action_space),
            }
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)
            snapshot = PolicySnapshot(
                step=7,
                policy=CountingPolicy(level=1),
                checkpoint_id="resume-7",
                created_at=1.0,
                metadata=metadata,
            )
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
                scenarios=[Scenario(id="adapt")],
                initial_policy=snapshot,
                trainer=NoopTrainer(),
                workflow=adaptive_chunk_size_rollout,
                action_space=action_space,
                scheduler=scheduler,
            )

        summary = asyncio.run(run())

        self.assertEqual(summary.checkpoints[0].step, 7)
        self.assertEqual(summary.checkpoints[0].checkpoint_id, "resume-7")
        self.assertEqual(summary.latest_step, 8)
        self.assertEqual(
            summary.metrics["action_space/codec/chunk_chunk_size_4/active"],
            1.0,
        )
        self.assertNotIn(
            "action_space/codec/chunk_chunk_size_2/active",
            summary.metrics,
        )
        self.assertGreater(
            summary.metrics["scheduler/total_rollout_decisions"],
            1.0,
        )
        self.assertIn(SCHEDULER_STATE_KEY, summary.checkpoints[-1].metadata)
        self.assertIn(ACTION_SPACE_STATE_KEY, summary.checkpoints[-1].metadata)

    def test_control_plane_stops_early_when_scheduler_roi_is_exhausted(self):
        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=1,
                max_policy_lag=1,
                exploration_bonus=0.0,
                min_train_steps=1,
                roi_patience=1,
                min_train_objective=0.0,
            )
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=5,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            return await runtime.run(
                scenarios=[Scenario(id="flat")],
                initial_policy=CountingPolicy(level=0),
                trainer=NoopTrainer(),
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                scheduler=scheduler,
            )

        summary = asyncio.run(run())

        self.assertLess(summary.latest_step, 5)
        self.assertEqual(summary.latest_step, 1)
        self.assertEqual(summary.metrics["scheduler/stop_recommended"], 1.0)

    def test_control_plane_uses_explicit_train_dollar_seconds(self):
        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=1,
                max_policy_lag=1,
                exploration_bonus=0.0,
            )
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=1,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1000.0,
                )
            )
            summary = await runtime.run(
                scenarios=[Scenario(id="costed")],
                initial_policy=CountingPolicy(level=1),
                trainer=CostedTrainer(),
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                scheduler=scheduler,
            )
            return summary

        summary = asyncio.run(run())

        self.assertEqual(summary.latest_step, 1)
        self.assertEqual(summary.metrics["costs/trainer_dollar_seconds"], 42.0)
        expected_scheduler_cost = (
            summary.metrics["costs/trainer_dollar_seconds"]
            + summary.metrics["costs/trainer_wait_dollar_seconds"]
        )
        self.assertAlmostEqual(
            summary.metrics["scheduler/costs/train_dollar_seconds"],
            expected_scheduler_cost,
        )
        self.assertAlmostEqual(
            summary.checkpoints[-1].metadata[SCHEDULER_STATE_KEY][
                "learning_state"
            ]["train_dollar_seconds"],
            expected_scheduler_cost,
        )

    def test_scheduler_train_objective_includes_promotion_eval_cost(self):
        async def run():
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=1,
                max_policy_lag=1,
                exploration_bonus=0.0,
            )
            runtime = ControlPlane(
                ControlPlaneConfig(
                    num_actors=1,
                    group_size=1,
                    train_batch_groups=1,
                    max_train_steps=1,
                    queue_max_trajectories=4,
                    train_queue_capacity=2,
                    max_policy_lag=1,
                    cost_per_second_usd=1.0,
                )
            )
            summary = await runtime.run(
                scenarios=[Scenario(id="costed-promotion")],
                initial_policy=CountingPolicy(level=1),
                trainer=FixedCostTrainer(score=1.0, dollar_seconds=2.0),
                workflow=flat_rollout,
                action_codecs=[TokenActionCodec()],
                scheduler=scheduler,
                promotion_evaluator=FixedCostPromotionEvaluator(
                    score=1.0,
                    dollar_seconds=8.0,
                ),
            )
            return summary

        summary = asyncio.run(run())

        self.assertEqual(summary.latest_step, 1)
        self.assertEqual(summary.metrics["costs/trainer_dollar_seconds"], 2.0)
        self.assertEqual(summary.metrics["costs/promotion_eval_dollar_seconds"], 8.0)
        expected_scheduler_cost = (
            summary.metrics["costs/trainer_dollar_seconds"]
            + summary.metrics["costs/promotion_eval_dollar_seconds"]
            + summary.metrics["costs/trainer_wait_dollar_seconds"]
        )
        self.assertAlmostEqual(
            summary.metrics["scheduler/costs/train_dollar_seconds"],
            expected_scheduler_cost,
        )
        self.assertAlmostEqual(
            summary.metrics["scheduler/train_last_objective"],
            1.0 / expected_scheduler_cost,
        )
        self.assertAlmostEqual(
            summary.checkpoints[-1].metadata[SCHEDULER_STATE_KEY][
                "learning_state"
            ]["train_dollar_seconds"],
            expected_scheduler_cost,
        )

    def test_grouper_drops_trajectories_that_exceed_policy_lag(self):
        grouper = TrajectoryGrouper(group_size=2)
        stale = Trajectory(
            scenario_id="s",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
        )
        fresh = Trajectory(
            scenario_id="s",
            policy_step=3,
            messages=[],
            actions=[],
            reward=1.0,
        )

        stale_result = grouper.add(stale, latest_step=3, max_policy_lag=1)
        first_fresh = grouper.add(fresh, latest_step=3, max_policy_lag=1)
        second_fresh = grouper.add(fresh, latest_step=3, max_policy_lag=1)

        self.assertFalse(stale_result.accepted)
        self.assertTrue(first_fresh.accepted)
        self.assertEqual(grouper.stale_dropped, 1)
        self.assertEqual(len(second_fresh.groups), 1)

    def test_runtime_telemetry_uses_effective_reward_for_unsafe_actions(self):
        telemetry = RuntimeTelemetry(cost_per_second_usd=1.0)
        telemetry.record_trajectory(
            Trajectory(
                scenario_id="unsafe",
                policy_step=0,
                messages=[],
                actions=[],
                reward=10.0,
                metadata={"action/safe": False, "action/quality": 0.0},
            ),
            accepted=True,
        )

        metrics = telemetry.metrics(stale_dropped=0)

        self.assertEqual(metrics["actions/quality_mean"], 0.0)
        self.assertEqual(metrics["data/unsafe_trajectories"], 1.0)
        self.assertEqual(metrics["reward/last_window_mean"], 0.0)

    def test_runtime_telemetry_attributes_accounted_costs(self):
        telemetry = RuntimeTelemetry(cost_per_second_usd=2.0)
        trajectory = Trajectory(
            scenario_id="costed",
            policy_step=0,
            messages=[],
            actions=TokenActionCodec().encode("answer"),
            reward=0.0,
            duration_s=2.0,
            metadata={"scheduler/arm_id": "costed|token"},
        )

        telemetry.record_actor_queue_wait(1.0)
        telemetry.record_trajectory(trajectory, accepted=True)
        telemetry.record_train(
            [TrajectoryGroup(scenario_id="costed", trajectories=(trajectory,))],
            TrainResult(metrics={"train/reward": 0.0}),
            duration_s=3.0,
        )

        metrics = telemetry.metrics(stale_dropped=0)

        self.assertEqual(metrics["time/rollout_s"], 2.0)
        self.assertEqual(metrics["costs/rollout_dollar_seconds"], 4.0)
        self.assertEqual(metrics["costs/trainer_dollar_seconds"], 6.0)
        self.assertEqual(metrics["costs/actor_queue_wait_dollar_seconds"], 2.0)
        self.assertEqual(metrics["costs/accounted_dollar_seconds"], 12.0)
        self.assertIn(
            "north_star/accounted_reward_improving_experience_per_dollar_second",
            metrics,
        )

    def test_runtime_telemetry_attributes_trainer_wait_cost(self):
        telemetry = RuntimeTelemetry(cost_per_second_usd=5.0)

        telemetry.record_train_wait(2.0)
        metrics = telemetry.metrics(stale_dropped=0)

        self.assertEqual(metrics["time/trainer_wait_s"], 2.0)
        self.assertEqual(metrics["costs/trainer_wait_dollar_seconds"], 10.0)
        self.assertEqual(metrics["costs/accounted_dollar_seconds"], 10.0)

    def test_runtime_telemetry_accepts_explicit_rollout_dollar_seconds(self):
        telemetry = RuntimeTelemetry(cost_per_second_usd=100.0)
        trajectory = Trajectory(
            scenario_id="costed",
            policy_step=0,
            messages=[],
            actions=TokenActionCodec().encode("answer"),
            reward=1.0,
            duration_s=2.0,
        )

        telemetry.record_trajectory(
            trajectory,
            accepted=True,
            dollar_seconds=0.5,
        )
        metrics = telemetry.metrics(stale_dropped=0)

        self.assertEqual(metrics["time/rollout_s"], 2.0)
        self.assertEqual(metrics["costs/rollout_dollar_seconds"], 0.5)
        self.assertEqual(metrics["costs/accounted_dollar_seconds"], 0.5)

    def test_runtime_stamps_actor_queue_wait_cost_before_enqueue(self):
        async def run():
            runtime = ControlPlane(ControlPlaneConfig(cost_per_second_usd=10.0))
            queue: asyncio.Queue[Trajectory] = asyncio.Queue(maxsize=1)
            await queue.put(
                Trajectory(
                    scenario_id="blocking",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=0.0,
                )
            )
            trajectory = Trajectory(
                scenario_id="costed",
                policy_step=0,
                messages=[],
                actions=TokenActionCodec().encode("answer"),
                reward=1.0,
            )

            task = asyncio.create_task(
                runtime._put_trajectory_with_queue_cost(
                    queue,
                    trajectory,
                    started_at=time.perf_counter(),
                )
            )
            await asyncio.sleep(0.001)
            queue.get_nowait()
            wait_s = await asyncio.wait_for(task, timeout=1.0)
            queued = queue.get_nowait()
            return wait_s, trajectory, queued

        wait_s, trajectory, queued = asyncio.run(run())

        self.assertGreater(wait_s, 0.0)
        self.assertIs(queued, trajectory)
        self.assertGreater(
            queued.metrics["cost/actor_queue_wait_dollar_seconds"],
            0.0,
        )

    def test_runtime_applies_scheduler_rollout_admission_delay_before_work(self):
        async def run():
            runtime = ControlPlane(ControlPlaneConfig(cost_per_second_usd=10.0))
            scheduler = ObjectiveScheduler(
                max_rollout_admission_delay_s=0.001,
                rollout_admission_pressure_threshold=0.0,
                exploration_bonus=0.0,
            )
            queue: asyncio.Queue[Trajectory] = asyncio.Queue(maxsize=1)
            await queue.put(
                Trajectory(
                    scenario_id="blocking",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=0.0,
                )
            )
            train_ring = TrajectoryRingBuffer(capacity=1, max_policy_lag=1)
            telemetry = RuntimeTelemetry(cost_per_second_usd=10.0)

            elapsed = await runtime._apply_rollout_admission_delay(
                scheduler=scheduler,
                trajectory_queue=queue,
                train_ring=train_ring,
                telemetry=telemetry,
                policy_step=0,
            )
            return elapsed, telemetry.metrics(stale_dropped=0), scheduler.metrics()

        elapsed, telemetry_metrics, scheduler_metrics = asyncio.run(run())

        self.assertGreater(elapsed, 0.0)
        self.assertGreater(
            telemetry_metrics["time/actor_admission_delay_s"],
            0.0,
        )
        self.assertGreater(
            telemetry_metrics["costs/actor_admission_delay_dollar_seconds"],
            0.0,
        )
        self.assertAlmostEqual(
            telemetry_metrics["costs/accounted_dollar_seconds"],
            telemetry_metrics["costs/actor_admission_delay_dollar_seconds"],
        )
        self.assertEqual(scheduler_metrics["scheduler/admission/decisions"], 1.0)
        self.assertGreater(
            scheduler_metrics["scheduler/admission/total_delay_s"],
            0.0,
        )
        self.assertGreater(
            scheduler_metrics[
                "scheduler/costs/rollout_admission_dollar_seconds"
            ],
            0.0,
        )

    def test_runtime_clamps_scheduler_active_actor_count(self):
        runtime = ControlPlane(ControlPlaneConfig(num_actors=4))
        trajectory_queue: asyncio.Queue[Trajectory] = asyncio.Queue(maxsize=2)
        trajectory_queue.put_nowait(
            Trajectory(
                scenario_id="queued",
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.0,
            )
        )
        train_ring = TrajectoryRingBuffer(capacity=2, max_policy_lag=1)

        low_scheduler = ActorCapScheduler(cap=0)
        high_scheduler = ActorCapScheduler(cap=99)

        low_count = runtime._active_actor_count(
            scheduler=low_scheduler,
            trajectory_queue=trajectory_queue,
            train_ring=train_ring,
            policy_step=3,
        )
        high_count = runtime._active_actor_count(
            scheduler=high_scheduler,
            trajectory_queue=trajectory_queue,
            train_ring=train_ring,
            policy_step=4,
        )

        self.assertEqual(low_count, 1)
        self.assertEqual(high_count, 4)
        self.assertEqual(low_scheduler.calls[0]["configured"], 4)
        self.assertEqual(low_scheduler.calls[0]["policy_step"], 3)
        self.assertGreater(low_scheduler.calls[0]["trajectory_queue_pressure"], 0.0)

    def test_runtime_telemetry_accepts_explicit_train_dollar_seconds(self):
        telemetry = RuntimeTelemetry(cost_per_second_usd=100.0)
        trajectory = Trajectory(
            scenario_id="costed",
            policy_step=0,
            messages=[],
            actions=TokenActionCodec().encode("answer"),
            reward=1.0,
        )

        telemetry.record_train(
            [TrajectoryGroup(scenario_id="costed", trajectories=(trajectory,))],
            TrainResult(metrics={"train/reward": 1.0}),
            duration_s=2.0,
            dollar_seconds=3.5,
        )
        metrics = telemetry.metrics(stale_dropped=0)

        self.assertEqual(metrics["time/trainer_s"], 2.0)
        self.assertEqual(metrics["costs/trainer_dollar_seconds"], 3.5)
        self.assertEqual(metrics["costs/accounted_dollar_seconds"], 3.5)

    def test_train_result_dollar_seconds_prefers_explicit_cost_metrics(self):
        result = TrainResult(
            metrics={"train/dollar_seconds": 7.0},
            metadata={"trainer/dollar_seconds": 9.0},
        )

        self.assertEqual(
            train_result_dollar_seconds(
                result,
                duration_s=2.0,
                cost_per_second_usd=100.0,
            ),
            7.0,
        )

    def test_ring_buffer_discards_stale_batch_and_unblocks_producer(self):
        async def run():
            ring = TrajectoryRingBuffer(capacity=1, max_policy_lag=1)
            discarded = []
            fresh = make_batch(policy_step=3, scenario_id="s")
            stale = make_batch(
                policy_step=0,
                scenario_id="s",
                on_discard=discarded.append,
            )

            await ring.put(stale)
            put_fresh = asyncio.create_task(ring.put(fresh))
            await asyncio.sleep(0)
            self.assertFalse(put_fresh.done())

            received = await asyncio.wait_for(
                ring.get(current_policy_step=3),
                timeout=1.0,
            )
            await asyncio.wait_for(put_fresh, timeout=1.0)
            return ring, received, discarded, stale

        ring, received, discarded, stale = asyncio.run(run())

        self.assertEqual(received.min_policy_step, 3)
        self.assertEqual(ring.total_discarded, 1)
        self.assertEqual(ring.total_consumed, 1)
        self.assertEqual(ring.backpressure_events, 1)
        self.assertEqual(len(discarded), 1)
        self.assertIs(discarded[0], stale)
        self.assertEqual(discarded[0].min_policy_step, 0)

    def test_ring_buffer_consumes_highest_priority_ready_batch(self):
        async def run():
            ring = TrajectoryRingBuffer(capacity=3, max_policy_lag=10)
            low = make_batch(policy_step=0, scenario_id="low", priority_score=0.1)
            high = make_batch(policy_step=0, scenario_id="high", priority_score=10.0)

            await ring.put(low)
            await ring.put(high)
            received = await ring.get(current_policy_step=0)
            return ring, received

        ring, received = asyncio.run(run())

        self.assertEqual(received.groups[0].scenario_id, "high")
        self.assertEqual(ring.priority_consumptions, 1)
        self.assertEqual(ring.pending_batches, 1)

    def test_ring_buffer_rescores_priority_at_consume_time(self):
        async def run():
            scheduler = ObjectiveScheduler(
                exploration_bonus=0.0,
                staleness_priority_weight=1.0,
            )
            ring = TrajectoryRingBuffer(capacity=3, max_policy_lag=10)
            fresh = make_batch(
                policy_step=4,
                scenario_id="fresh",
                priority_score=10.0,
                reward=1.0,
                metadata={
                    "scheduler/arm_id": "fresh|token",
                    "scheduler/active_max_policy_lag": 4,
                },
            )
            near_stale = make_batch(
                policy_step=0,
                scenario_id="near-stale",
                priority_score=1.0,
                reward=1.0,
                metadata={
                    "scheduler/arm_id": "near-stale|token",
                    "scheduler/active_max_policy_lag": 4,
                },
            )

            await ring.put(fresh)
            await ring.put(near_stale)
            received = await ring.get(
                current_policy_step=4,
                priority_scorer=lambda batch, policy_step: (
                    scheduler.score_train_groups(
                        batch.groups,
                        policy_step=policy_step,
                    )
                ),
            )
            return ring, received

        ring, received = asyncio.run(run())

        self.assertEqual(received.groups[0].scenario_id, "near-stale")
        self.assertEqual(ring.priority_consumptions, 1)
        self.assertEqual(ring.consumed_priority_total, 2.0)

    def test_weight_broadcast_channel_replays_latest_and_waits(self):
        async def run():
            channel = WeightBroadcastChannel()
            first_subscriber = channel.subscribe()
            first = await channel.publish(
                PolicySnapshot(
                    step=1,
                    policy=CountingPolicy(level=1),
                    checkpoint_id="level-1",
                    created_at=1.0,
                )
            )
            late_subscriber = channel.subscribe()
            waited = await channel.wait_for_step(1)
            return (
                first,
                waited,
                first_subscriber.get_nowait(),
                late_subscriber.get_nowait(),
                channel.broadcast_count,
            )

        first, waited, first_received, late_received, broadcasts = asyncio.run(run())

        self.assertEqual(first.step, 1)
        self.assertEqual(waited, first)
        self.assertEqual(first_received, first)
        self.assertEqual(late_received, first)
        self.assertEqual(broadcasts, 1)


def make_batch(
    *,
    policy_step: int,
    scenario_id: str,
    priority_score: float = 0.0,
    reward: float | None = None,
    metadata: dict | None = None,
    on_discard=None,
) -> VersionedTrajectoryBatch:
    trajectory = Trajectory(
        scenario_id=scenario_id,
        policy_step=policy_step,
        messages=[],
        actions=[],
        reward=float(policy_step) if reward is None else reward,
        metadata=dict(metadata or {}),
    )
    group = TrajectoryGroup(scenario_id=scenario_id, trajectories=(trajectory,))
    return VersionedTrajectoryBatch(
        groups=(group,),
        assembled_at_step=policy_step,
        priority_score=priority_score,
        on_discard=on_discard,
    )


if __name__ == "__main__":
    unittest.main()
