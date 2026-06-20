import asyncio
import unittest
from dataclasses import dataclass, field

from calm_puffer_art import (
    ART_RAW_GROUP_KEY,
    ART_RAW_TRAJECTORY_KEY,
    ArtBackendTrainer,
    AsyncArtBackend,
    AsyncArtBackendConfig,
    ObjectiveScheduler,
    PolicySnapshot,
    SCHEDULER_STATE_KEY,
    StaleArtBatchError,
    TrajectoryGroup,
    WeightBroadcastChannel,
    art_group_to_local,
    local_group_to_art,
    train_result_from_art,
)


@dataclass
class FakeMessage:
    role: str
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeArtTrajectory:
    messages_and_choices: list
    reward: float
    initial_policy_version: int | None = None
    final_policy_version: int | None = None
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeArtGroup:
    trajectories: list[FakeArtTrajectory]
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    exceptions: list[BaseException] = field(default_factory=list)

    def __iter__(self):
        return iter(self.trajectories)


@dataclass
class FakeArtTrainResult:
    step: int
    metrics: dict
    checkpoint_path: str | None = None
    artifact_name: str | None = None


class FakeArtBackend:
    def __init__(self) -> None:
        self.calls = []
        self.registered = []
        self.closed = False
        self.step = 0
        self.block_event: asyncio.Event | None = None

    async def register(self, model):
        self.registered.append(model)

    async def _get_step(self, model):
        return self.step

    async def train(self, model, trajectory_groups, **kwargs):
        if self.block_event is not None:
            await self.block_event.wait()
        self.step += 1
        self.calls.append((model, trajectory_groups, kwargs))
        return FakeArtTrainResult(
            step=self.step,
            metrics={"train/reward": 0.75, "ignored": "not-float"},
            checkpoint_path=f".art/project/models/model/step_{self.step}",
        )

    async def close(self):
        self.closed = True


class CostedFakeArtBackend(FakeArtBackend):
    async def train(self, model, trajectory_groups, **kwargs):
        self.step += 1
        self.calls.append((model, trajectory_groups, kwargs))
        return FakeArtTrainResult(
            step=self.step,
            metrics={
                "train/reward": 0.75,
                "trainer/dollar_seconds": 17.0,
            },
            checkpoint_path=f".art/project/models/model/step_{self.step}",
        )


class FixedCadenceScheduler:
    def __init__(self, target: int, lag: int | None = None) -> None:
        self.target = target
        self.lag = lag
        self.observed_batches = []
        self.stale_batches = []
        self.scored_batches = []
        self.lag_calls = []
        self.observed_dollar_seconds = []

    def target_train_batch_groups(self, **kwargs):
        return self.target

    def max_policy_lag(self, **kwargs):
        self.lag_calls.append(kwargs)
        if self.lag is None:
            return kwargs["configured"]
        return self.lag

    def score_train_groups(self, groups, *, policy_step):
        self.scored_batches.append(len(groups))
        return float(len(groups))

    def observe_train(self, *, groups, result, duration_s, dollar_seconds, policy_step):
        self.observed_batches.append(len(groups))
        self.observed_dollar_seconds.append(dollar_seconds)

    def observe_stale_batch(self, *, groups, policy_step, reason):
        self.stale_batches.append(
            {
                "groups": len(groups),
                "policy_step": policy_step,
                "reason": reason,
            }
        )

    def metrics(self):
        return {}


class ArtAdapterTests(unittest.TestCase):
    def test_art_group_to_local_preserves_versions_messages_actions_and_raw_group(self):
        art_trajectory = FakeArtTrajectory(
            messages_and_choices=[
                {"role": "user", "content": "question"},
                FakeChoice(FakeMessage(role="assistant", content="answer one")),
            ],
            reward=0.75,
            initial_policy_version=3,
            final_policy_version=4,
            metrics={"duration": 1.5, "non_numeric": "skip"},
            metadata={"scenario_id": "math", "source": "art"},
        )
        art_group = FakeArtGroup(
            trajectories=[art_trajectory],
            metadata={"scenario_id": "math"},
            metrics={"group_score": 0.5},
        )

        group = art_group_to_local(art_group)
        trajectory = group.trajectories[0]

        self.assertEqual(group.scenario_id, "math")
        self.assertEqual(group.metrics["group_score"], 0.5)
        self.assertIs(group.metadata[ART_RAW_GROUP_KEY], art_group)
        self.assertEqual(trajectory.policy_step, 3)
        self.assertEqual(trajectory.reward, 0.75)
        self.assertEqual(trajectory.duration_s, 1.5)
        self.assertEqual([message.role for message in trajectory.messages], ["user", "assistant"])
        self.assertEqual(trajectory.actions[0].kind, "art_choice")
        self.assertEqual(trajectory.actions[0].text, "answer one")
        self.assertEqual(trajectory.metadata["scheduler/arm_id"], "math|art")
        self.assertEqual(trajectory.metadata["scheduler/scenario_id"], "math")
        self.assertIs(trajectory.metadata[ART_RAW_TRAJECTORY_KEY], art_trajectory)
        self.assertIs(local_group_to_art(group), art_group)

    def test_art_backend_trainer_delegates_raw_groups_and_maps_result(self):
        async def run():
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=1,
                    )
                ]
            )
            group = art_group_to_local(art_group)
            backend = FakeArtBackend()
            trainer = ArtBackendTrainer(
                backend=backend,
                model="art-model",
                train_kwargs={"config": "train-config"},
            )
            result = await trainer.train(
                PolicySnapshot(
                    step=0,
                    policy="served-policy",
                    checkpoint_id="step-0",
                    created_at=0.0,
                ),
                [group],
            )
            return backend, result, art_group

        backend, result, art_group = asyncio.run(run())

        self.assertEqual(backend.calls[0][0], "art-model")
        self.assertIs(backend.calls[0][1][0], art_group)
        self.assertEqual(backend.calls[0][2], {"config": "train-config"})
        self.assertEqual(result.policy, "served-policy")
        self.assertEqual(result.metrics, {"train/reward": 0.75})
        self.assertEqual(result.checkpoint_id, "step_1")
        self.assertEqual(result.metadata["art/step"], 1)

    def test_async_art_backend_wraps_backend_protocol_and_broadcasts_updates(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                weight_channel=channel,
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metadata={"scenario_id": "art-task"},
                    )
                ],
                metadata={"scenario_id": "art-task"},
            )

            await async_backend.register("art-model")
            result = await async_backend.train(
                "art-model",
                [art_group],
                config="train-config",
            )
            update = updates.get_nowait()
            stats = async_backend.stats()
            await async_backend.close()
            return backend, result, update, stats, scheduler.metrics()

        backend, result, update, stats, metrics = asyncio.run(run())

        self.assertEqual(backend.registered, ["art-model"])
        self.assertEqual(backend.calls[0][0], "art-model")
        self.assertEqual(backend.calls[0][2], {"config": "train-config"})
        self.assertEqual(result.step, 1)
        self.assertEqual(update.step, 1)
        self.assertEqual(update.checkpoint_id, "step_1")
        self.assertEqual(update.metadata["art/step"], 1)
        self.assertEqual(update.metadata[SCHEDULER_STATE_KEY]["version"], 1)
        self.assertIn("art-task|art", update.metadata[SCHEDULER_STATE_KEY]["arms"])
        self.assertGreater(
            update.metadata[SCHEDULER_STATE_KEY]["learning_state"][
                "train_dollar_seconds"
            ],
            0.0,
        )
        self.assertEqual(stats["consumed_batches"], 1.0)
        self.assertEqual(stats["art_backend/current_step"], 1.0)
        self.assertEqual(metrics["scheduler/train_reward_ema"], 0.75)
        self.assertEqual(
            metrics["scheduler/control/cadence_1/train_updates"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/control/policy_lag_2/train_updates"],
            1.0,
        )
        self.assertTrue(backend.closed)

    def test_async_art_backend_synchronous_fallback_calls_backend_directly(self):
        async def run():
            backend = FakeArtBackend()
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(synchronous_fallback=True),
            )
            art_group = FakeArtGroup(
                trajectories=[FakeArtTrajectory(messages_and_choices=[], reward=1.0)]
            )
            result = await async_backend.train("art-model", [art_group], mode="sync")
            return backend, result

        backend, result = asyncio.run(run())

        self.assertEqual(result.step, 1)
        self.assertEqual(backend.calls[0][2], {"mode": "sync"})

    def test_async_art_backend_uses_explicit_train_dollar_seconds(self):
        async def run():
            backend = CostedFakeArtBackend()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(cost_per_second_usd=1000.0),
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metadata={"scenario_id": "costed-art"},
                    )
                ],
                metadata={"scenario_id": "costed-art"},
            )

            await async_backend.register("art-model")
            result = await async_backend.train("art-model", [art_group])
            metrics = scheduler.metrics()
            stats = async_backend.stats()
            await async_backend.close()
            return result, metrics, stats

        result, metrics, stats = asyncio.run(run())

        self.assertEqual(result.step, 1)
        self.assertGreaterEqual(
            stats["art_backend/trainer_wait_dollar_seconds"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["scheduler/costs/train_dollar_seconds"],
            17.0 + stats["art_backend/trainer_wait_dollar_seconds"],
        )

    def test_async_art_backend_charges_trainer_wait_to_scheduler(self):
        async def run():
            backend = CostedFakeArtBackend()
            scheduler = FixedCadenceScheduler(target=1)
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(cost_per_second_usd=1000.0),
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ],
                metadata={"scenario_id": "waited-art"},
            )

            await async_backend.register("art-model")
            await asyncio.sleep(0.001)
            result = await async_backend.train("art-model", [art_group])
            stats = async_backend.stats()
            await async_backend.close()
            return result, stats, scheduler

        result, stats, scheduler = asyncio.run(run())

        self.assertEqual(result.step, 1)
        self.assertGreater(stats["art_backend/trainer_wait_dollar_seconds"], 0.0)
        self.assertEqual(scheduler.observed_batches, [1])
        self.assertAlmostEqual(
            scheduler.observed_dollar_seconds[0],
            17.0 + stats["art_backend/trainer_wait_dollar_seconds"],
        )

    def test_async_art_backend_submit_train_returns_future_before_training_finishes(self):
        async def run():
            backend = FakeArtBackend()
            backend.block_event = asyncio.Event()
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(train_queue_capacity=2, max_policy_lag=2),
            )
            first = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )
            second = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )

            first_future = await async_backend.submit_train(
                "art-model",
                [first],
                batch="first",
            )
            second_future = await async_backend.submit_train(
                "art-model",
                [second],
                batch="second",
            )
            await asyncio.sleep(0)
            before = async_backend.stats()
            first_done_before = first_future.done()
            second_done_before = second_future.done()

            backend.block_event.set()
            first_result = await first_future
            second_result = await second_future
            after = async_backend.stats()
            await async_backend.close()
            return (
                backend,
                before,
                after,
                first_done_before,
                second_done_before,
                first_result,
                second_result,
            )

        (
            backend,
            before,
            after,
            first_done_before,
            second_done_before,
            first_result,
            second_result,
        ) = asyncio.run(run())

        self.assertFalse(first_done_before)
        self.assertFalse(second_done_before)
        self.assertEqual(before["art_backend/submitted_batches"], 2.0)
        self.assertEqual(before["art_backend/completed_batches"], 0.0)
        self.assertEqual(first_result.step, 1)
        self.assertEqual(second_result.step, 2)
        self.assertEqual(after["art_backend/completed_batches"], 2.0)
        self.assertEqual(after["art_backend/failed_batches"], 0.0)
        self.assertEqual(
            [call[2]["batch"] for call in backend.calls],
            ["first", "second"],
        )

    def test_async_art_backend_submit_group_batches_by_configured_cadence(self):
        async def run():
            backend = FakeArtBackend()
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=2,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                ),
            )
            first = FakeArtGroup(
                trajectories=[FakeArtTrajectory(messages_and_choices=[], reward=0.2)]
            )
            second = FakeArtGroup(
                trajectories=[FakeArtTrajectory(messages_and_choices=[], reward=0.4)]
            )

            first_future = await async_backend.submit_group("art-model", first)
            await asyncio.sleep(0)
            before = async_backend.stats()
            second_future = await async_backend.submit_group("art-model", second)
            first_result = await first_future
            second_result = await second_future
            after = async_backend.stats()
            await async_backend.close()
            return backend, before, after, first_result, second_result

        backend, before, after, first_result, second_result = asyncio.run(run())

        self.assertFalse(first_result is None)
        self.assertEqual(first_result, second_result)
        self.assertEqual(before["art_backend/pending_groups"], 1.0)
        self.assertEqual(before["art_backend/submitted_batches"], 0.0)
        self.assertEqual(after["art_backend/submitted_groups"], 2.0)
        self.assertEqual(after["art_backend/submitted_batches"], 1.0)
        self.assertEqual(after["art_backend/completed_batches"], 1.0)
        self.assertEqual(len(backend.calls[0][1]), 2)

    def test_async_art_backend_flushes_partial_group_batch(self):
        async def run():
            backend = FakeArtBackend()
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=3,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                ),
            )
            group = FakeArtGroup(
                trajectories=[FakeArtTrajectory(messages_and_choices=[], reward=0.2)]
            )

            future = await async_backend.submit_group("art-model", group)
            before = async_backend.stats()
            flushed = await async_backend.flush_pending_groups()
            result = await future
            after = async_backend.stats()
            await async_backend.close()
            return before, flushed, result, after, backend

        before, flushed, result, after, backend = asyncio.run(run())

        self.assertFalse(result is None)
        self.assertEqual(before["art_backend/pending_groups"], 1.0)
        self.assertEqual(flushed, 1)
        self.assertEqual(after["art_backend/pending_groups"], 0.0)
        self.assertEqual(after["art_backend/completed_batches"], 1.0)
        self.assertEqual(len(backend.calls[0][1]), 1)

    def test_async_art_backend_scheduler_controls_group_batch_cadence(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = FixedCadenceScheduler(target=2)
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=3,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                ),
                scheduler=scheduler,
            )
            first = FakeArtGroup(
                trajectories=[FakeArtTrajectory(messages_and_choices=[], reward=0.2)]
            )
            second = FakeArtGroup(
                trajectories=[FakeArtTrajectory(messages_and_choices=[], reward=0.4)]
            )

            first_future = await async_backend.submit_group("art-model", first)
            second_future = await async_backend.submit_group("art-model", second)
            await first_future
            await second_future
            stats = async_backend.stats()
            await async_backend.close()
            return backend, scheduler, stats

        backend, scheduler, stats = asyncio.run(run())

        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][1]), 2)
        self.assertEqual(scheduler.scored_batches, [2, 2])
        self.assertEqual(scheduler.observed_batches, [2])
        self.assertEqual(stats["art_backend/submitted_batches"], 1.0)

    def test_async_art_backend_scheduler_controls_policy_lag_staleness(self):
        async def run():
            backend = FakeArtBackend()
            backend.block_event = asyncio.Event()
            scheduler = FixedCadenceScheduler(target=1, lag=0)
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_queue_capacity=2,
                    max_policy_lag=2,
                ),
                scheduler=scheduler,
            )
            fresh = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )
            stale = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )

            first_future = await async_backend.submit_train("art-model", [fresh])
            await asyncio.sleep(0)
            second_future = await async_backend.submit_train("art-model", [stale])
            backend.block_event.set()
            first_result = await first_future
            with self.assertRaises(StaleArtBatchError):
                await second_future
            stats = async_backend.stats()
            await async_backend.close()
            return first_result, scheduler, stats

        first_result, scheduler, stats = asyncio.run(run())

        self.assertEqual(first_result.step, 1)
        self.assertEqual(stats["art_backend/current_step"], 1.0)
        self.assertEqual(stats["art_backend/current_max_policy_lag"], 0.0)
        self.assertEqual(stats["discarded_batches"], 1.0)
        self.assertEqual(stats["art_backend/stale_batches"], 1.0)
        self.assertEqual(stats["art_backend/failed_batches"], 1.0)
        self.assertEqual(
            scheduler.stale_batches,
            [
                {
                    "groups": 1,
                    "policy_step": 1,
                    "reason": "art_train_ring_stale",
                }
            ],
        )
        self.assertGreaterEqual(len(scheduler.lag_calls), 2)
        self.assertEqual(scheduler.lag_calls[-1]["policy_step"], 1)

    def test_async_art_backend_raises_when_enqueued_batch_goes_stale(self):
        async def run():
            backend = FakeArtBackend()
            backend.block_event = asyncio.Event()
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_queue_capacity=2,
                    max_policy_lag=0,
                ),
            )
            fresh = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )
            stale = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )

            first = asyncio.create_task(async_backend.train("art-model", [fresh]))
            await asyncio.sleep(0)
            second = asyncio.create_task(async_backend.train("art-model", [stale]))
            await asyncio.sleep(0)
            backend.block_event.set()
            first_result = await first
            with self.assertRaises(StaleArtBatchError):
                await second
            stats = async_backend.stats()
            await async_backend.close()
            return first_result, stats

        first_result, stats = asyncio.run(run())

        self.assertEqual(first_result.step, 1)
        self.assertEqual(stats["discarded_batches"], 1.0)
        self.assertEqual(stats["art_backend/stale_batches"], 1.0)
        self.assertEqual(stats["art_backend/failed_batches"], 1.0)

    def test_train_result_from_art_uses_artifact_or_step_checkpoint_ids(self):
        artifact = train_result_from_art(
            FakeArtTrainResult(
                step=8,
                metrics={"train/reward": 0.5},
                artifact_name="entity/project/model:step8",
            )
        )
        step_only = train_result_from_art(
            FakeArtTrainResult(step=9, metrics={"train/reward": 0.25})
        )

        self.assertEqual(artifact.checkpoint_id, "entity/project/model:step8")
        self.assertEqual(step_only.checkpoint_id, "art-step-9")

    def test_local_group_without_art_metadata_cannot_be_delegated(self):
        group = TrajectoryGroup(scenario_id="local", trajectories=())

        with self.assertRaises(ValueError):
            local_group_to_art(group)


if __name__ == "__main__":
    unittest.main()
