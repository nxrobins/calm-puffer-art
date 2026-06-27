import asyncio
import unittest
from dataclasses import dataclass, field, replace

from calm_puffer_art import (
    ACTION_SPACE_STATE_KEY,
    ART_BACKEND_STATE_KEY,
    ART_RAW_GROUP_KEY,
    ART_RAW_TRAJECTORY_KEY,
    AdaptiveActionSpace,
    ArtBackendTrainer,
    AsyncArtBackend,
    AsyncArtBackendConfig,
    ChunkActionCodec,
    ObjectiveScheduler,
    PolicySnapshot,
    SCHEDULER_STATE_KEY,
    Scenario,
    StaleArtBatchError,
    TokenActionCodec,
    Trajectory,
    TrajectoryGroup,
    WeightBroadcastChannel,
    action_codec_key,
    action_space_signature,
    art_rollout_metadata,
    art_group_to_local,
    local_group_to_art,
    scheduling_action_key,
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
        self.assertEqual(
            update.metadata[ART_BACKEND_STATE_KEY]["published_policy_updates"],
            1,
        )
        self.assertEqual(
            update.metadata[ART_BACKEND_STATE_KEY][
                "published_policy_improvement"
            ],
            0.75,
        )
        state_decisions = update.metadata[ART_BACKEND_STATE_KEY][
            "publication_decision_stats"
        ]
        self.assertEqual(
            state_decisions[
                "action=publish|reason=async_train_result"
            ]["decisions"],
            1,
        )
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
        self.assertGreater(stats["art_backend/wall_clock_s"], 0.0)
        self.assertEqual(stats["art_backend/submitted_train_groups"], 1.0)
        self.assertGreater(stats["art_backend/submitted_train_groups_per_s"], 0.0)
        self.assertGreater(stats["art_backend/completed_batches_per_s"], 0.0)
        self.assertEqual(stats["art_backend/published_policy_updates"], 1.0)
        self.assertEqual(stats["art_backend/published_policy_improvement"], 0.75)
        self.assertEqual(
            stats["art_backend/published_policy_reward_improving_experience"],
            0.75,
        )
        self.assertEqual(stats["art_backend/publication/decision/keys"], 1.0)
        self.assertEqual(stats["art_backend/publication/decision/decisions"], 1.0)
        self.assertEqual(stats["art_backend/publication/decision/published"], 1.0)
        self.assertEqual(
            stats[
                "art_backend/publication/decision/"
                "positive_reward_improving_keys"
            ],
            1.0,
        )
        self.assertEqual(
            stats[
                "art_backend/publication/decision/"
                "total_published_policy_improvement"
            ],
            0.75,
        )
        self.assertEqual(
            stats[
                "art_backend/publication/decision/"
                "realized_reward_improving_experience"
            ],
            0.75,
        )
        decision_prefix = (
            "art_backend/publication/decision/"
            "action_publish_reason_async_train_result"
        )
        self.assertEqual(stats[f"{decision_prefix}/decisions"], 1.0)
        self.assertEqual(
            stats[f"{decision_prefix}/realized_reward_improving_experience"],
            0.75,
        )
        self.assertEqual(stats["art_backend/latest_published_policy_score"], 0.75)
        self.assertGreater(
            stats[
                "art_backend/"
                "published_policy_reward_improving_experience_per_dollar_second"
            ],
            0.0,
        )
        self.assertGreater(
            stats[
                "art_backend/"
                "accounted_published_policy_reward_improving_experience_per_dollar_second"
            ],
            0.0,
        )
        self.assertEqual(stats["scheduler/train_reward_ema"], 0.75)
        self.assertEqual(
            stats["scheduler/control/cadence_1/train_updates"],
            1.0,
        )
        self.assertEqual(metrics["scheduler/train_reward_ema"], 0.75)
        self.assertEqual(
            metrics["scheduler/control/cadence_1/train_updates"],
            1.0,
        )
        credited_lag_updates = [
            value
            for key, value in metrics.items()
            if key.startswith("scheduler/control/policy_lag_")
            and key.endswith("/train_updates")
            and value > 0.0
        ]
        self.assertEqual(credited_lag_updates, [1.0])
        self.assertTrue(backend.closed)

    def test_async_art_backend_reports_semantic_bandwidth_without_scheduler(self):
        async def run():
            async_backend = AsyncArtBackend(
                backend=FakeArtBackend(),
                config=AsyncArtBackendConfig(synchronous_fallback=True),
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(FakeMessage("assistant", "alpha beta")),
                            FakeChoice(FakeMessage("assistant", "gamma delta")),
                        ],
                        reward=1.0,
                        initial_policy_version=0,
                        metadata={"scenario_id": "art-task"},
                    )
                ],
                metadata={"scenario_id": "art-task"},
            )

            await async_backend.train("art-model", [art_group])
            stats = async_backend.stats()
            await async_backend.close()
            return stats

        stats = asyncio.run(run())

        self.assertEqual(stats["art_backend/action_units"], 2.0)
        self.assertEqual(stats["art_backend/source_tokens"], 4.0)
        self.assertEqual(
            stats["actions/semantic_bandwidth_tokens_per_decision"],
            2.0,
        )

    def test_async_art_backend_snapshots_action_space_state_after_train_feedback(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=2,
                promote_latent_patches=True,
                latent_patch_latent_size=3,
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                action_space=action_space,
                weight_channel=channel,
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(
                                FakeMessage(
                                    role="assistant",
                                    content="alpha beta gamma delta",
                                )
                            )
                        ],
                        reward=1.0,
                        initial_policy_version=0,
                        metadata={
                            "scenario_id": "art-task",
                            "scheduler/arm_id": "art-task|chunk(chunk_size=2)",
                        },
                    )
                ],
                metadata={"scenario_id": "art-task"},
            )

            await async_backend.register("art-model")
            await async_backend.train("art-model", [art_group])
            update = updates.get_nowait()
            stats = async_backend.stats()
            await async_backend.close()
            return update, stats, scheduler.metrics()

        update, stats, metrics = asyncio.run(run())

        action_space_state = update.metadata[ACTION_SPACE_STATE_KEY]
        self.assertIn(
            "latent_patch(latent_size=3,patch_size=2)",
            [codec["key"] for codec in action_space_state["active_codecs"]],
        )
        self.assertEqual(action_space_state["learning_state"]["promotions"], 1)
        self.assertEqual(stats["action_space/promotions"], 1.0)
        self.assertEqual(
            stats["action_space/codec/latent_patch_latent_size_3_patch_size_2/active"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/art_task_chunk_chunk_size_2/pulls"],
            1.0,
        )
        self.assertEqual(
            metrics[
                "scheduler/arm/art_task_chunk_chunk_size_2/"
                "semantic_bandwidth_tokens_per_decision"
            ],
            4.0,
        )

    def test_async_art_backend_restores_checkpointed_control_state(self):
        async def run():
            first_backend = FakeArtBackend()
            first_scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            first_action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
            )
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            first_async_backend = AsyncArtBackend(
                backend=first_backend,
                scheduler=first_scheduler,
                action_space=first_action_space,
                weight_channel=channel,
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(
                                FakeMessage(
                                    role="assistant",
                                    content="alpha beta gamma delta",
                                )
                            )
                        ],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 1.0},
                        metadata={
                            "scenario_id": "resume-art",
                            "scheduler/arm_id": (
                                "resume-art|chunk(chunk_size=2)"
                            ),
                        },
                    )
                ],
                metadata={"scenario_id": "resume-art"},
            )

            await first_async_backend.register("art-model")
            await first_async_backend.train("art-model", [art_group])
            update = updates.get_nowait()
            await first_async_backend.close()

            resumed_scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            resumed_action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
            )
            resumed_async_backend = AsyncArtBackend(
                backend=FakeArtBackend(),
                scheduler=resumed_scheduler,
                action_space=resumed_action_space,
            )
            snapshot = PolicySnapshot(
                step=update.step,
                policy="art-model",
                checkpoint_id=update.checkpoint_id,
                created_at=update.created_at,
                metadata=update.metadata,
            )

            restored = resumed_async_backend.restore_control_state(snapshot)
            restored_metrics = resumed_scheduler.metrics()
            await resumed_async_backend.register("art-model")
            resolved_step = await resumed_async_backend._get_step("art-model")
            first = resumed_async_backend.select_rollout(
                scenarios=[Scenario(id="resume-art")],
                actor_id=0,
            )
            second = resumed_async_backend.select_rollout(
                scenarios=[Scenario(id="resume-art")],
                actor_id=1,
            )
            resumed_stats = resumed_async_backend.stats()
            await resumed_async_backend.close()
            return (
                update,
                restored,
                restored_metrics,
                resolved_step,
                first,
                second,
                resumed_stats,
            )

        (
            update,
            restored,
            restored_metrics,
            resolved_step,
            first,
            second,
            resumed_stats,
        ) = asyncio.run(run())

        self.assertEqual(
            restored,
            {
                "scheduler": True,
                "action_space": True,
                "promotion": False,
                "policy_step": True,
                "art_backend": True,
            },
        )
        self.assertEqual(resolved_step, update.step)
        self.assertEqual(
            resumed_stats["art_backend/current_step"],
            float(update.step),
        )
        self.assertEqual(
            resumed_stats["action_space/codec/chunk_chunk_size_4/active"],
            1.0,
        )
        self.assertEqual(
            restored_metrics["scheduler/arm/resume_art_chunk_chunk_size_2/pulls"],
            1.0,
        )
        selected_codec_keys = {
            action_codec_key(first.action_codec),
            action_codec_key(second.action_codec),
        }
        self.assertIn("chunk(chunk_size=4)", selected_codec_keys)

    def test_async_art_backend_restores_published_policy_accounting(self):
        async def run():
            first_backend = FakeArtBackend()
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            first_async_backend = AsyncArtBackend(
                backend=first_backend,
                weight_channel=channel,
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metadata={"scenario_id": "published-art"},
                    )
                ],
                metadata={"scenario_id": "published-art"},
            )

            await first_async_backend.register("art-model")
            await first_async_backend.train("art-model", [art_group])
            update = updates.get_nowait()
            await first_async_backend.close()

            resumed_backend = FakeArtBackend()
            resumed_async_backend = AsyncArtBackend(backend=resumed_backend)
            snapshot = PolicySnapshot(
                step=update.step,
                policy="art-model",
                checkpoint_id=update.checkpoint_id,
                created_at=update.created_at,
                metadata=update.metadata,
            )
            restored = resumed_async_backend.restore_control_state(snapshot)
            await resumed_async_backend.register("art-model")
            await resumed_async_backend.train("art-model", [art_group])
            stats = resumed_async_backend.stats()
            await resumed_async_backend.close()
            return restored, stats

        restored, stats = asyncio.run(run())

        self.assertTrue(restored["art_backend"])
        self.assertEqual(stats["art_backend/published_policy_updates"], 2.0)
        self.assertEqual(stats["art_backend/published_policy_improvement"], 0.75)
        self.assertEqual(
            stats["art_backend/published_policy_reward_improving_experience"],
            0.75,
        )
        self.assertEqual(stats["art_backend/publication/decision/decisions"], 2.0)
        self.assertEqual(stats["art_backend/publication/decision/published"], 2.0)
        self.assertEqual(
            stats[
                "art_backend/publication/decision/"
                "realized_reward_improving_experience"
            ],
            0.75,
        )
        self.assertEqual(stats["art_backend/latest_published_policy_score"], 0.75)

    def test_async_art_backend_selects_external_art_rollout_and_metadata(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=2)
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                action_space=action_space,
                config=AsyncArtBackendConfig(
                    train_batch_groups=2,
                    max_policy_lag=3,
                ),
            )
            expected_signature = action_space_signature(action_space)
            decision = async_backend.select_rollout(
                scenarios=[Scenario(id="external-select")],
                action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
                actor_id=7,
                trajectory_queue_pressure=0.25,
            )
            metadata = art_rollout_metadata(decision)
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(
                                FakeMessage(
                                    role="assistant",
                                    content="alpha beta",
                                )
                            )
                        ],
                        reward=1.0,
                        initial_policy_version=metadata[
                            "art/initial_policy_version"
                        ],
                        metrics={"cost/dollar_seconds": 3.0},
                        metadata=metadata,
                    )
                ],
                metadata={"scenario_id": "external-select"},
            )

            await async_backend.register("art-model")
            await async_backend.train("art-model", [art_group])
            metrics = scheduler.metrics()
            await async_backend.close()
            return decision, metadata, metrics, expected_signature

        decision, metadata, metrics, expected_signature = asyncio.run(run())

        metric_arm = "scheduler/arm/external_select_token"
        self.assertEqual(decision.arm_id, "external-select|token")
        self.assertEqual(metadata["scheduler/arm_id"], decision.arm_id)
        self.assertEqual(metadata["scheduler/scenario_id"], "external-select")
        self.assertEqual(metadata["scheduler/action_codec"], "token")
        self.assertEqual(metadata["actor_id"], 7)
        self.assertEqual(metadata["scheduler/target_train_batch_groups"], 2)
        self.assertEqual(metadata["scheduler/max_policy_lag"], 3)
        self.assertEqual(metadata["scheduler/action_space_key"], expected_signature)
        self.assertIn(
            "scheduler/decision/estimated_rollout_dollar_seconds",
            metadata,
        )
        self.assertIn(
            "scheduler/decision/unobserved_rollout_cost_penalty",
            metadata,
        )
        self.assertIn(
            "scheduler/decision/unobserved_rollout_cost_estimated",
            metadata,
        )
        self.assertEqual(metrics[f"{metric_arm}/decisions"], 1.0)
        self.assertEqual(metrics[f"{metric_arm}/pulls"], 1.0)
        self.assertEqual(metrics[f"{metric_arm}/inflight"], 0.0)
        self.assertEqual(
            metrics[f"{metric_arm}/mean_rollout_dollar_seconds"],
            3.0,
        )

    def test_art_rollout_metadata_preserves_action_space_joint_key(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        decision = scheduler.select_rollout(
            scenarios=[Scenario(id="external-select")],
            action_codecs=[TokenActionCodec()],
            actor_id=7,
            policy_step=0,
            trajectory_queue_pressure=0.25,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=3,
            active_actor_count=4,
            rollout_admission_delay_ms=25,
            action_space_key="bridge-space",
        )
        decision = replace(
            decision,
            metadata={
                **decision.metadata,
                "coverage_control_key": "forced|arm=external_select_token",
            },
        )

        metadata = art_rollout_metadata(decision)

        self.assertEqual(metadata["scheduler/action_space_key"], "bridge_space")
        self.assertIn("scheduler/cadence_response_key", metadata)
        self.assertIn("scheduler/policy_lag_response_key", metadata)
        self.assertEqual(
            metadata["scheduler/coverage_control_key"],
            "forced|arm=external_select_token",
        )
        self.assertEqual(
            metadata["scheduler/joint_action_key"],
            scheduling_action_key(
                arm_id=decision.arm_id,
                target_train_batch_groups=decision.target_train_batch_groups,
                max_policy_lag=decision.max_policy_lag,
                active_actor_count=4,
                admission_delay_ms=25,
                action_space_key="bridge-space",
            ),
        )

    def test_async_art_backend_applies_external_actor_admission_control(self):
        async def run():
            backend = CostedFakeArtBackend()
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_actor_count=1,
                max_actor_count=1,
                min_policy_lag=1,
                max_policy_lag=1,
                exploration_bonus=0.0,
                control_exploration_bonus=0.0,
                max_rollout_admission_delay_s=0.001,
                rollout_admission_pressure_threshold=0.0,
                rollout_admission_positive_signal_scale=1.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(
                    train_batch_groups=1,
                    max_policy_lag=1,
                    cost_per_second_usd=1000.0,
                ),
            )

            rejected = await async_backend.admit_rollout(
                actor_id=1,
                configured_actor_count=2,
                trajectory_queue_pressure=1.0,
                apply_delay=False,
            )
            admitted = await async_backend.admit_rollout(
                actor_id=0,
                configured_actor_count=2,
                trajectory_queue_pressure=1.0,
                apply_delay=False,
            )
            decision = async_backend.select_rollout(
                scenarios=[Scenario(id="admission-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=0,
                trajectory_queue_pressure=1.0,
            )
            metadata = art_rollout_metadata(decision, extra=admitted.metadata)
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(
                                FakeMessage(
                                    role="assistant",
                                    content="alpha beta",
                                )
                            )
                        ],
                        reward=1.0,
                        initial_policy_version=metadata[
                            "art/initial_policy_version"
                        ],
                        metrics={"rollout/dollar_seconds": 2.0},
                        metadata=metadata,
                    )
                ],
                metadata={"scenario_id": "admission-art"},
            )

            await async_backend.register("art-model")
            await async_backend.train("art-model", [art_group])
            metrics = scheduler.metrics()
            stats = async_backend.stats()
            await async_backend.close()
            return rejected, admitted, metadata, metrics, stats

        rejected, admitted, metadata, metrics, stats = asyncio.run(run())

        self.assertFalse(rejected.admitted)
        self.assertEqual(rejected.active_actor_count, 1)
        self.assertTrue(admitted.admitted)
        self.assertEqual(admitted.active_actor_count, 1)
        self.assertEqual(admitted.metadata["scheduler/active_actor_count"], 1)
        self.assertEqual(
            admitted.metadata["scheduler/active_rollout_admission_delay_ms"],
            1,
        )
        self.assertTrue(admitted.metadata["scheduler/admission_observed"])
        self.assertAlmostEqual(admitted.delay_dollar_seconds, 1.0)
        self.assertEqual(metadata["scheduler/active_actor_count"], 1)
        self.assertEqual(metadata["cost/actor_admission_dollar_seconds"], 1.0)
        self.assertAlmostEqual(
            stats["art_backend/actor_admission_dollar_seconds"],
            admitted.delay_dollar_seconds,
        )
        self.assertAlmostEqual(
            metrics["scheduler/costs/rollout_admission_dollar_seconds"],
            admitted.delay_dollar_seconds,
        )
        self.assertAlmostEqual(
            metrics["scheduler/arm/admission_art_token/admission_dollar_seconds"],
            admitted.delay_dollar_seconds,
        )
        self.assertEqual(
            metrics["scheduler/control/admission_delay_ms_1/rollout_updates"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/control/actor_count_1/rollout_updates"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/control/actor_count_1/decisions"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["scheduler/actor/actor_0/admission_dollar_seconds"],
            admitted.delay_dollar_seconds,
        )

    def test_async_art_backend_admission_stops_when_budget_exhausted(self):
        async def run():
            backend = CostedFakeArtBackend()
            scheduler = ObjectiveScheduler(
                exploration_bonus=0.0,
                max_accounted_dollar_seconds=1.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(
                    train_batch_groups=1,
                    max_policy_lag=2,
                ),
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metadata={"scenario_id": "budget-art"},
                    )
                ],
                metadata={"scenario_id": "budget-art"},
            )

            await async_backend.train("art-model", [art_group])
            admission = await async_backend.admit_rollout(
                actor_id=0,
                configured_actor_count=4,
                trajectory_queue_pressure=0.0,
                apply_delay=False,
            )
            stats = async_backend.stats()
            await async_backend.close()
            return admission, stats

        admission, stats = asyncio.run(run())

        self.assertFalse(admission.admitted)
        self.assertEqual(admission.active_actor_count, 0)
        self.assertEqual(admission.metadata["scheduler/stop_recommended"], True)
        self.assertEqual(
            admission.metadata["scheduler/stop_reason"],
            "continuation_exhausted",
        )
        self.assertEqual(stats["art_backend/stopped_admissions"], 1.0)
        self.assertEqual(stats["scheduler/budget/accounted_exhausted"], 1.0)
        self.assertEqual(stats["scheduler/stop_recommended"], 1.0)

    def test_async_art_backend_admit_and_select_reserves_budget(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                exploration_bonus=0.0,
                max_accounted_dollar_seconds=3.0,
            )
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id="budget-art",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=1.0,
                    metadata={"scheduler/arm_id": "budget-art|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(train_batch_groups=1),
            )

            first = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=0,
                configured_actor_count=4,
                apply_delay=False,
            )
            second = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=1,
                configured_actor_count=4,
                apply_delay=False,
            )
            third = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=2,
                configured_actor_count=4,
                apply_delay=False,
            )
            stats = async_backend.stats()
            await async_backend.close()
            return first, second, third, stats

        first, second, third, stats = asyncio.run(run())

        self.assertTrue(first.admitted)
        self.assertTrue(second.admitted)
        self.assertFalse(third.admitted)
        self.assertIsNotNone(first.decision)
        self.assertIsNotNone(second.decision)
        self.assertIsNone(third.decision)
        self.assertEqual(first.metadata["scheduler/arm_id"], "budget-art|token")
        self.assertEqual(
            first.metadata["scheduler/joint_action_key"],
            scheduling_action_key(
                arm_id=first.decision.arm_id,
                target_train_batch_groups=first.decision.target_train_batch_groups,
                max_policy_lag=first.decision.max_policy_lag,
                active_actor_count=first.metadata["scheduler/active_actor_count"],
                admission_delay_ms=first.metadata[
                    "scheduler/active_rollout_admission_delay_ms"
                ],
            ),
        )
        self.assertEqual(
            first.metadata["scheduler/decision/reserved_rollout_dollar_seconds"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/total_rollout_decisions"],
            2.0,
        )
        self.assertEqual(
            stats["scheduler/budget/accounted_dollar_seconds"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/budget/reserved_inflight_rollout_dollar_seconds"],
            2.0,
        )
        self.assertEqual(
            stats["scheduler/budget/projected_accounted_dollar_seconds"],
            3.0,
        )
        self.assertEqual(stats["scheduler/budget/accounted_exhausted"], 1.0)
        self.assertEqual(stats["art_backend/stopped_admissions"], 1.0)

    def test_async_art_backend_rejects_assignment_that_would_exceed_budget(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                exploration_bonus=0.0,
                max_accounted_dollar_seconds=2.5,
            )
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id="budget-art",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=1.0,
                    metadata={"scheduler/arm_id": "budget-art|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(train_batch_groups=1),
            )

            first = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=0,
                configured_actor_count=4,
                apply_delay=False,
            )
            second = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=1,
                configured_actor_count=4,
                apply_delay=False,
            )
            stats = async_backend.stats()
            await async_backend.close()
            return first, second, stats

        first, second, stats = asyncio.run(run())

        self.assertTrue(first.admitted)
        self.assertFalse(second.admitted)
        self.assertIsNotNone(first.decision)
        self.assertIsNone(second.decision)
        self.assertEqual(
            second.metadata["scheduler/stop_reason"],
            "projected_budget_exhausted",
        )
        self.assertEqual(
            stats["scheduler/total_rollout_decisions"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/total_inflight_rollouts"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/budget/reserved_inflight_rollout_dollar_seconds"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/budget/projected_accounted_dollar_seconds"],
            2.0,
        )
        self.assertEqual(stats["scheduler/budget/accounted_exhausted"], 0.0)
        self.assertEqual(stats["scheduler/control/cadence_1/decisions"], 1.0)
        self.assertEqual(stats["scheduler/control/policy_lag_1/decisions"], 1.0)
        self.assertEqual(stats["scheduler/control/actor_count_4/decisions"], 1.0)
        self.assertEqual(stats["scheduler/admission/decisions"], 1.0)
        self.assertEqual(
            stats["scheduler/control/admission_delay_ms_0/decisions"],
            1.0,
        )
        self.assertEqual(stats["art_backend/stopped_admissions"], 1.0)

    def test_async_art_backend_records_failed_rollout_assignment(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                exploration_bonus=0.0,
                max_accounted_dollar_seconds=5.0,
            )
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id="budget-art",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=1.0,
                    metadata={"scheduler/arm_id": "budget-art|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(train_batch_groups=1),
            )

            assignment = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=0,
                configured_actor_count=2,
                apply_delay=False,
            )
            before_failure = async_backend.stats()
            failure = async_backend.record_rollout_failure(
                assignment,
                exception=RuntimeError("actor died"),
            )
            after_failure = async_backend.stats()
            next_assignment = await async_backend.admit_and_select_rollout(
                scenarios=[Scenario(id="budget-art")],
                action_codecs=[TokenActionCodec()],
                actor_id=0,
                configured_actor_count=2,
                apply_delay=False,
            )
            after_reselect = async_backend.stats()
            await async_backend.close()
            return (
                assignment,
                before_failure,
                failure,
                after_failure,
                next_assignment,
                after_reselect,
            )

        (
            assignment,
            before_failure,
            failure,
            after_failure,
            next_assignment,
            after_reselect,
        ) = asyncio.run(run())

        self.assertTrue(assignment.admitted)
        self.assertEqual(
            before_failure[
                "scheduler/budget/reserved_inflight_rollout_dollar_seconds"
            ],
            1.0,
        )
        self.assertEqual(before_failure["scheduler/total_inflight_rollouts"], 1.0)
        self.assertEqual(failure.scenario_id, "budget-art")
        self.assertEqual(failure.reward, 0.0)
        self.assertEqual(failure.metrics["rollout/dollar_seconds"], 1.0)
        self.assertEqual(
            failure.metadata["scheduler/rollout_failed_before_submit"],
            True,
        )
        self.assertIn("actor died", failure.exception or "")
        self.assertEqual(
            after_failure[
                "scheduler/budget/reserved_inflight_rollout_dollar_seconds"
            ],
            0.0,
        )
        self.assertEqual(after_failure["scheduler/total_inflight_rollouts"], 0.0)
        self.assertEqual(
            after_failure["scheduler/budget/accounted_dollar_seconds"],
            2.0,
        )
        self.assertEqual(
            after_failure["scheduler/budget/projected_accounted_dollar_seconds"],
            2.0,
        )
        self.assertEqual(after_failure["scheduler/failure_rollouts"], 1.0)
        self.assertEqual(after_failure["art_backend/failed_rollouts"], 1.0)
        self.assertEqual(after_failure["art_backend/sample_dollar_seconds"], 1.0)
        self.assertTrue(next_assignment.admitted)
        self.assertEqual(
            after_reselect[
                "scheduler/budget/reserved_inflight_rollout_dollar_seconds"
            ],
            1.0,
        )

    def test_async_art_backend_select_rollout_uses_promoted_action_space_codecs(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                action_space=action_space,
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(
                                FakeMessage(
                                    role="assistant",
                                    content="alpha beta gamma delta",
                                )
                            )
                        ],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 1.0},
                        metadata={
                            "scenario_id": "adapt",
                            "scheduler/arm_id": "adapt|chunk(chunk_size=2)",
                        },
                    )
                ],
                metadata={"scenario_id": "adapt"},
            )

            await async_backend.register("art-model")
            await async_backend.train("art-model", [art_group])
            expected_signature = action_space_signature(action_space)
            first = async_backend.select_rollout(
                scenarios=[Scenario(id="adapt")],
                actor_id=0,
                active_actor_count=1,
                rollout_admission_delay_ms=0,
            )
            second = async_backend.select_rollout(
                scenarios=[Scenario(id="adapt")],
                actor_id=1,
                active_actor_count=1,
                rollout_admission_delay_ms=0,
            )
            metrics = scheduler.metrics()
            await async_backend.close()
            return first, second, metrics, async_backend.stats(), expected_signature

        first, second, metrics, stats, expected_signature = asyncio.run(run())

        selected_codec_keys = {
            action_codec_key(first.action_codec),
            action_codec_key(second.action_codec),
        }
        self.assertIn("chunk(chunk_size=4)", selected_codec_keys)
        self.assertEqual(
            stats["action_space/codec/chunk_chunk_size_4/active"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/adapt_chunk_chunk_size_4/decisions"],
            1.0,
        )
        self.assertEqual(first.metadata["action_space_key"], expected_signature)
        self.assertIn(
            f"|action_space={expected_signature}",
            first.metadata["joint_action_key"],
        )

    def test_async_art_backend_promotes_action_space_from_submitted_rollouts(self):
        async def run():
            backend = FakeArtBackend()
            backend.block_event = asyncio.Event()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
                demotion_min_pulls=999,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                action_space=action_space,
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[
                            FakeChoice(
                                FakeMessage(
                                    role="assistant",
                                    content="alpha beta gamma delta",
                                )
                            )
                        ],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 1.0},
                        metadata={
                            "scenario_id": "adapt",
                            "scheduler/arm_id": "adapt|chunk(chunk_size=2)",
                        },
                    )
                ],
                metadata={"scenario_id": "adapt"},
            )

            await async_backend.register("art-model")
            future = await async_backend.submit_train("art-model", [art_group])
            first = async_backend.select_rollout(
                scenarios=[Scenario(id="adapt")],
                actor_id=0,
            )
            second = async_backend.select_rollout(
                scenarios=[Scenario(id="adapt")],
                actor_id=1,
            )
            backend.block_event.set()
            await future
            stats = async_backend.stats()
            await async_backend.close()
            return first, second, stats

        first, second, stats = asyncio.run(run())

        selected_codec_keys = {
            action_codec_key(first.action_codec),
            action_codec_key(second.action_codec),
        }
        self.assertIn("chunk(chunk_size=4)", selected_codec_keys)
        self.assertEqual(
            stats["action_space/codec/chunk_chunk_size_4/active"],
            1.0,
        )

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

    def test_async_art_backend_synchronous_fallback_observes_objective_feedback(self):
        async def run():
            backend = CostedFakeArtBackend()
            scheduler = ObjectiveScheduler(exploration_bonus=0.0)
            channel = WeightBroadcastChannel()
            updates = channel.subscribe()
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                weight_channel=channel,
                config=AsyncArtBackendConfig(
                    synchronous_fallback=True,
                    cost_per_second_usd=1000.0,
                ),
            )
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 11.0},
                        metadata={"scenario_id": "sync-art"},
                    )
                ],
                metadata={"scenario_id": "sync-art"},
            )

            result = await async_backend.train(
                "art-model",
                [art_group],
                mode="sync",
            )
            metrics = scheduler.metrics()
            stats = async_backend.stats()
            update = updates.get_nowait()
            await async_backend.close()
            return backend, result, metrics, stats, update, art_group

        backend, result, metrics, stats, update, art_group = asyncio.run(run())

        self.assertEqual(result.step, 1)
        self.assertIs(backend.calls[0][1][0], art_group)
        self.assertEqual(backend.calls[0][2], {"mode": "sync"})
        self.assertEqual(stats["art_backend/current_step"], 1.0)
        self.assertEqual(stats["art_backend/submitted_batches"], 1.0)
        self.assertEqual(stats["art_backend/completed_batches"], 1.0)
        self.assertEqual(stats["art_backend/sample_dollar_seconds"], 11.0)
        self.assertEqual(stats["art_backend/trainer_dollar_seconds"], 17.0)
        self.assertEqual(stats["art_backend/trainer_wait_dollar_seconds"], 0.0)
        self.assertEqual(stats["art_backend/published_policy_updates"], 1.0)
        self.assertEqual(metrics["scheduler/arm/sync_art_art/pulls"], 1.0)
        self.assertEqual(
            metrics["scheduler/arm/sync_art_art/mean_rollout_dollar_seconds"],
            11.0,
        )
        self.assertEqual(metrics["scheduler/costs/rollout_dollar_seconds"], 11.0)
        self.assertEqual(metrics["scheduler/costs/train_dollar_seconds"], 17.0)
        self.assertEqual(metrics["scheduler/accounted_last_dollar_seconds"], 28.0)
        self.assertEqual(
            metrics["scheduler/control/cadence_1/train_updates"],
            1.0,
        )
        self.assertEqual(metrics["scheduler/train_selection/decisions"], 1.0)
        self.assertEqual(metrics["scheduler/train_selection/train_updates"], 1.0)
        self.assertGreater(metrics["scheduler/train_selection/total_objective"], 0.0)
        self.assertEqual(update.step, 1)
        self.assertIn(SCHEDULER_STATE_KEY, update.metadata)
        self.assertIn(ART_BACKEND_STATE_KEY, update.metadata)

    def test_async_art_backend_synchronous_fallback_rejects_stale_batch(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = FixedCadenceScheduler(target=1, lag=0)
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                config=AsyncArtBackendConfig(
                    synchronous_fallback=True,
                    max_policy_lag=0,
                ),
            )
            async_backend._current_step = 1
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 5.0},
                    )
                ]
            )

            future = await async_backend.submit_train("art-model", [art_group])
            with self.assertRaises(StaleArtBatchError):
                await future
            stats = async_backend.stats()
            await async_backend.close()
            return backend, scheduler, stats

        backend, scheduler, stats = asyncio.run(run())

        self.assertEqual(len(backend.calls), 0)
        self.assertEqual(stats["art_backend/submitted_batches"], 1.0)
        self.assertEqual(stats["art_backend/completed_batches"], 0.0)
        self.assertEqual(stats["art_backend/failed_batches"], 1.0)
        self.assertEqual(stats["art_backend/stale_batches"], 1.0)
        self.assertEqual(stats["art_backend/sample_dollar_seconds"], 5.0)
        self.assertEqual(
            scheduler.stale_batches,
            [{"groups": 1, "policy_step": 1, "reason": "art_sync_stale"}],
        )

    def test_async_art_backend_synchronous_stale_demotes_action_space(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                min_policy_lag=0,
                max_policy_lag=0,
                exploration_bonus=0.0,
                rollout_objective_weight=0.0,
            )
            action_space = AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
                demotion_min_pulls=2,
                demote_on_stale_feedback=True,
            )
            action_space.add_codec(ChunkActionCodec(chunk_size=4))
            async_backend = AsyncArtBackend(
                backend=backend,
                scheduler=scheduler,
                action_space=action_space,
                config=AsyncArtBackendConfig(
                    synchronous_fallback=True,
                    max_policy_lag=0,
                ),
            )
            async_backend._current_step = 1
            art_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 1.0},
                        metadata={
                            "scenario_id": "stale-action",
                            "scheduler/arm_id": (
                                "stale-action|chunk(chunk_size=4)"
                            ),
                        },
                    ),
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 1.0},
                        metadata={
                            "scenario_id": "stale-action",
                            "scheduler/arm_id": (
                                "stale-action|chunk(chunk_size=4)"
                            ),
                        },
                    ),
                ],
                metadata={"scenario_id": "stale-action"},
            )

            future = await async_backend.submit_train("art-model", [art_group])
            with self.assertRaises(StaleArtBatchError):
                await future
            stats = async_backend.stats()
            await async_backend.close()
            return backend, stats

        backend, stats = asyncio.run(run())

        self.assertEqual(len(backend.calls), 0)
        self.assertEqual(stats["art_backend/stale_batches"], 1.0)
        self.assertEqual(stats["action_space/demotions"], 1.0)
        self.assertEqual(
            stats["action_space/codec/chunk_chunk_size_4/disabled"],
            1.0,
        )
        self.assertNotIn(
            "action_space/codec/chunk_chunk_size_4/active",
            stats,
        )
        self.assertLess(
            stats[
                "scheduler/arm/stale_action_chunk_chunk_size_4/objective_score"
            ],
            0.0,
        )

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

    def test_async_art_backend_observes_art_sample_cost_before_train_feedback(self):
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
                        metrics={"cost/dollar_seconds": 11.0},
                        metadata={"scenario_id": "sample-cost-art"},
                    )
                ],
                metadata={"scenario_id": "sample-cost-art"},
            )

            await async_backend.register("art-model")
            result = await async_backend.train("art-model", [art_group])
            metrics = scheduler.metrics()
            stats = async_backend.stats()
            await async_backend.close()
            return result, metrics, stats

        result, metrics, stats = asyncio.run(run())

        self.assertEqual(result.step, 1)
        self.assertEqual(stats["art_backend/sample_dollar_seconds"], 11.0)
        self.assertEqual(
            metrics["scheduler/arm/sample_cost_art_art/pulls"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/sample_cost_art_art/mean_rollout_dollar_seconds"],
            11.0,
        )
        self.assertEqual(
            metrics["scheduler/costs/rollout_dollar_seconds"],
            11.0,
        )
        self.assertAlmostEqual(
            metrics["scheduler/costs/train_dollar_seconds"],
            17.0 + stats["art_backend/trainer_wait_dollar_seconds"],
        )
        self.assertAlmostEqual(
            metrics["scheduler/accounted_last_dollar_seconds"],
            28.0 + stats["art_backend/trainer_wait_dollar_seconds"],
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
        self.assertEqual(after["art_backend/published_policy_updates"], 2.0)
        self.assertEqual(after["art_backend/published_policy_improvement"], 0.75)
        self.assertEqual(
            after["art_backend/published_policy_reward_improving_experience"],
            0.75,
        )
        self.assertEqual(after["art_backend/latest_published_policy_score"], 0.75)
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

    def test_async_art_backend_drops_stale_pending_group_on_submit(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=2,
                max_train_batch_groups=2,
                min_policy_lag=0,
                max_policy_lag=0,
                exploration_bonus=0.0,
                control_exploration_bonus=0.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=2,
                    train_queue_capacity=2,
                    max_policy_lag=0,
                ),
                scheduler=scheduler,
            )
            await async_backend.register("art-model")
            async_backend._current_step = 2
            old_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 3.0},
                    )
                ],
                metadata={"scenario_id": "stale-submit"},
            )

            future = await async_backend.submit_group("art-model", old_group)
            with self.assertRaises(StaleArtBatchError):
                await future
            stats = async_backend.stats()
            await async_backend.close()
            return backend, stats

        backend, stats = asyncio.run(run())

        self.assertEqual(len(backend.calls), 0)
        self.assertEqual(stats["art_backend/submitted_groups"], 1.0)
        self.assertEqual(stats["art_backend/submitted_batches"], 0.0)
        self.assertEqual(stats["art_backend/pending_groups"], 0.0)
        self.assertEqual(stats["art_backend/stale_pending_groups"], 1.0)
        self.assertEqual(stats["art_backend/sample_dollar_seconds"], 3.0)
        self.assertEqual(stats["scheduler/arm/stale_submit_art/pulls"], 1.0)
        self.assertEqual(stats["scheduler/arm/stale_submit_art/accepted"], 0.0)
        self.assertEqual(
            stats["scheduler/arm/stale_submit_art/total_positive_improvement"],
            0.0,
        )
        self.assertEqual(
            stats["scheduler/arm/stale_submit_art/stale_updates"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/control/cadence_2/stale_updates"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/control/policy_lag_0/stale_updates"],
            1.0,
        )

    def test_async_art_backend_rejects_stale_submit_train_without_rollout_credit(
        self,
    ):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=1,
                max_train_batch_groups=1,
                min_policy_lag=0,
                max_policy_lag=0,
                exploration_bonus=0.0,
                control_exploration_bonus=0.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=1,
                    train_queue_capacity=2,
                    max_policy_lag=0,
                ),
                scheduler=scheduler,
            )
            await async_backend.register("art-model")
            async_backend._current_step = 2
            old_group = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=10.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 5.0},
                    )
                ],
                metadata={"scenario_id": "stale-train"},
            )

            future = await async_backend.submit_train("art-model", [old_group])
            with self.assertRaises(StaleArtBatchError):
                await future
            stats = async_backend.stats()
            await async_backend.close()
            return backend, stats

        backend, stats = asyncio.run(run())

        self.assertEqual(len(backend.calls), 0)
        self.assertEqual(stats["art_backend/submitted_batches"], 1.0)
        self.assertEqual(stats["art_backend/completed_batches"], 0.0)
        self.assertEqual(stats["art_backend/failed_batches"], 1.0)
        self.assertEqual(stats["art_backend/stale_batches"], 1.0)
        self.assertEqual(stats["art_backend/sample_dollar_seconds"], 5.0)
        self.assertEqual(stats["scheduler/arm/stale_train_art/pulls"], 1.0)
        self.assertEqual(stats["scheduler/arm/stale_train_art/accepted"], 0.0)
        self.assertEqual(
            stats["scheduler/arm/stale_train_art/total_positive_improvement"],
            0.0,
        )
        self.assertEqual(stats["scheduler/arm/stale_train_art/stale_updates"], 1.0)
        self.assertEqual(
            stats["scheduler/control/cadence_1/stale_updates"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/control/policy_lag_0/stale_updates"],
            1.0,
        )

    def test_async_art_backend_debits_pending_group_stale_after_policy_update(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = FixedCadenceScheduler(target=2, lag=0)
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=2,
                    train_queue_capacity=2,
                    max_policy_lag=0,
                ),
                scheduler=scheduler,
            )
            pending = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 4.0},
                    )
                ]
            )
            trained = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ]
            )

            await async_backend.register("art-model")
            pending_future = await async_backend.submit_group("art-model", pending)
            train_result = await async_backend.train("art-model", [trained])
            with self.assertRaises(StaleArtBatchError):
                await asyncio.wait_for(pending_future, timeout=1.0)
            stats = async_backend.stats()
            await async_backend.close()
            return train_result, scheduler, stats

        train_result, scheduler, stats = asyncio.run(run())

        self.assertEqual(train_result.step, 1)
        self.assertEqual(stats["art_backend/current_step"], 1.0)
        self.assertEqual(stats["art_backend/pending_groups"], 0.0)
        self.assertEqual(stats["art_backend/stale_pending_groups"], 1.0)
        self.assertEqual(stats["art_backend/sample_dollar_seconds"], 4.0)
        self.assertEqual(
            scheduler.stale_batches,
            [
                {
                    "groups": 1,
                    "policy_step": 1,
                    "reason": "art_pending_group_stale",
                }
            ],
        )

    def test_async_art_backend_pending_stale_debits_scheduler_controls(self):
        async def run():
            backend = FakeArtBackend()
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=2,
                max_train_batch_groups=2,
                min_policy_lag=0,
                max_policy_lag=0,
                exploration_bonus=0.0,
                control_exploration_bonus=0.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=2,
                    train_queue_capacity=2,
                    max_policy_lag=0,
                ),
                scheduler=scheduler,
            )
            pending = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 4.0},
                    )
                ],
                metadata={"scenario_id": "pending-stale-control"},
            )
            trained = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=1.0,
                        initial_policy_version=0,
                    )
                ],
                metadata={"scenario_id": "fresh-control"},
            )

            await async_backend.register("art-model")
            pending_future = await async_backend.submit_group("art-model", pending)
            await async_backend.train("art-model", [trained])
            with self.assertRaises(StaleArtBatchError):
                await asyncio.wait_for(pending_future, timeout=1.0)
            stats = async_backend.stats()
            await async_backend.close()
            return stats

        stats = asyncio.run(run())

        self.assertEqual(stats["art_backend/stale_pending_groups"], 1.0)
        self.assertEqual(
            stats["scheduler/arm/pending_stale_control_art/stale_updates"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/control/cadence_2/stale_updates"],
            1.0,
        )
        self.assertEqual(
            stats["scheduler/control/policy_lag_0/stale_updates"],
            1.0,
        )

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

    def test_async_art_backend_cadence_batch_credits_accounted_sample_cost(self):
        async def run():
            backend = CostedFakeArtBackend()
            scheduler = ObjectiveScheduler(
                min_train_batch_groups=2,
                max_train_batch_groups=2,
                exploration_bonus=0.0,
                control_exploration_bonus=0.0,
            )
            async_backend = AsyncArtBackend(
                backend=backend,
                config=AsyncArtBackendConfig(
                    train_batch_groups=3,
                    train_queue_capacity=2,
                    max_policy_lag=2,
                    cost_per_second_usd=1000.0,
                ),
                scheduler=scheduler,
            )
            first = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=0.8,
                        initial_policy_version=0,
                        metrics={"cost/dollar_seconds": 5.0},
                        metadata={"scenario_id": "external-art"},
                    )
                ],
                metadata={"scenario_id": "external-art"},
            )
            second = FakeArtGroup(
                trajectories=[
                    FakeArtTrajectory(
                        messages_and_choices=[],
                        reward=0.6,
                        initial_policy_version=0,
                        metrics={
                            "rollout/dollar_seconds": 4.0,
                            "queue_wait/dollar_seconds": 2.0,
                            "admission/dollar_seconds": 1.0,
                        },
                        metadata={"scenario_id": "external-art"},
                    )
                ],
                metadata={"scenario_id": "external-art"},
            )

            await async_backend.register("art-model")
            first_future = await async_backend.submit_group("art-model", first)
            before = async_backend.stats()
            second_future = await async_backend.submit_group("art-model", second)
            first_result = await first_future
            second_result = await second_future
            stats = async_backend.stats()
            metrics = scheduler.metrics()
            await async_backend.close()
            return backend, before, stats, metrics, first_result, second_result

        (
            backend,
            before,
            stats,
            metrics,
            first_result,
            second_result,
        ) = asyncio.run(run())

        expected_sample_cost = 12.0
        expected_train_cost = (
            17.0
            + stats["art_backend/trainer_wait_dollar_seconds"]
        )
        expected_accounted_cost = expected_train_cost + expected_sample_cost
        self.assertEqual(first_result, second_result)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][1]), 2)
        self.assertEqual(before["art_backend/pending_groups"], 1.0)
        self.assertEqual(before["art_backend/submitted_batches"], 0.0)
        self.assertEqual(stats["art_backend/submitted_groups"], 2.0)
        self.assertEqual(stats["art_backend/submitted_batches"], 1.0)
        self.assertEqual(stats["art_backend/completed_batches"], 1.0)
        self.assertEqual(
            stats["art_backend/sample_dollar_seconds"],
            expected_sample_cost,
        )
        self.assertEqual(metrics["scheduler/costs/rollout_dollar_seconds"], 9.0)
        self.assertEqual(metrics["scheduler/costs/queue_wait_dollar_seconds"], 2.0)
        self.assertEqual(
            metrics["scheduler/costs/rollout_admission_dollar_seconds"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["scheduler/costs/train_dollar_seconds"],
            expected_train_cost,
        )
        self.assertAlmostEqual(
            metrics["scheduler/accounted_last_dollar_seconds"],
            expected_accounted_cost,
        )
        self.assertEqual(metrics["scheduler/arm/external_art_art/pulls"], 2.0)
        self.assertEqual(
            metrics["scheduler/arm/external_art_art/mean_sample_dollar_seconds"],
            expected_sample_cost / 2.0,
        )
        self.assertEqual(
            metrics["scheduler/control/train_objective_accounted"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/control/cadence_2/train_updates"],
            1.0,
        )
        self.assertGreater(metrics["scheduler/control/cadence_2/objective_ema"], 0.0)

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
