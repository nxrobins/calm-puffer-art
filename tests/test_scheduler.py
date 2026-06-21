import unittest

from calm_puffer_art import (
    ActionUnit,
    ChunkActionCodec,
    ObjectiveScheduler,
    Scenario,
    TokenActionCodec,
    Trajectory,
    TrajectoryGroup,
    TrainResult,
    action_quality,
    trajectory_failure_modes,
    trajectory_reconstruction_accuracy,
)


class ObjectiveSchedulerTests(unittest.TestCase):
    def test_scheduler_explores_then_prefers_best_marginal_objective_arm(self):
        scenarios = [Scenario(id="easy"), Scenario(id="hard")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)

        rewards_by_arm = {
            "easy|token": 0.1,
            "easy|chunk(chunk_size=2)": 1.0,
            "hard|token": 0.0,
            "hard|chunk(chunk_size=2)": 0.2,
        }

        decisions = []
        for _ in range(4):
            decision = scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=codecs,
                actor_id=0,
                policy_step=0,
                trajectory_queue_pressure=0.0,
                train_queue_pressure=0.0,
                configured_train_batch_groups=2,
                configured_max_policy_lag=2,
            )
            decisions.append(decision.arm_id)
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id=decision.scenario.id,
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=rewards_by_arm[decision.arm_id],
                    metadata={"scheduler/arm_id": decision.arm_id},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )

        next_decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )

        self.assertEqual(
            decisions,
            [
                "easy|token",
                "easy|chunk(chunk_size=2)",
                "hard|token",
                "hard|chunk(chunk_size=2)",
            ],
        )
        self.assertEqual(next_decision.arm_id, "easy|chunk(chunk_size=2)")

    def test_reward_scale_normalization_changes_cross_scale_rollout_choice(self):
        scenarios = [Scenario(id="large"), Scenario(id="small")]
        codecs = [TokenActionCodec()]

        def observe(
            scheduler: ObjectiveScheduler,
            scenario_id: str,
            reward: float,
        ) -> None:
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id=scenario_id,
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=reward,
                    metadata={"scheduler/arm_id": f"{scenario_id}|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )

        raw = ObjectiveScheduler(exploration_bonus=0.0, ema_alpha=1.0)
        normalized = ObjectiveScheduler(
            exploration_bonus=0.0,
            ema_alpha=1.0,
            reward_scale_normalization="arm_range",
        )
        for scheduler in (raw, normalized):
            observe(scheduler, "large", 1000.0)
            observe(scheduler, "large", 1010.0)
            observe(scheduler, "small", 0.5)

        raw_decision = raw.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        normalized_decision = normalized.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        metrics = normalized.metrics()

        self.assertEqual(raw_decision.arm_id, "large|token")
        self.assertEqual(normalized_decision.arm_id, "small|token")
        self.assertAlmostEqual(
            metrics["scheduler/arm/large_token/last_reward_scale"],
            1010.0,
        )
        self.assertAlmostEqual(
            metrics[
                "scheduler/arm/large_token/"
                "last_normalized_positive_improvement"
            ],
            10.0 / 1010.0,
        )

    def test_reward_scale_normalization_changes_cross_scale_train_credit(self):
        scenarios = [Scenario(id="large"), Scenario(id="small")]
        codecs = [TokenActionCodec()]

        def rollout_group(
            scheduler: ObjectiveScheduler,
            scenario_id: str,
        ) -> TrajectoryGroup:
            trajectory = Trajectory(
                scenario_id=scenario_id,
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.0,
                metadata={"scheduler/arm_id": f"{scenario_id}|token"},
            )
            scheduler.observe_rollout(
                trajectory,
                accepted=True,
                dollar_seconds=1.0,
            )
            return TrajectoryGroup(
                scenario_id=scenario_id,
                trajectories=(trajectory,),
            )

        def observe_train(
            scheduler: ObjectiveScheduler,
            group: TrajectoryGroup,
            reward: float,
            policy_step: int,
        ) -> None:
            scheduler.observe_train(
                groups=[group],
                result=TrainResult(metrics={"train/reward": reward}),
                duration_s=1.0,
                dollar_seconds=1.0,
                policy_step=policy_step,
            )

        raw = ObjectiveScheduler(
            exploration_bonus=0.0,
            ema_alpha=1.0,
            rollout_objective_weight=0.0,
        )
        normalized = ObjectiveScheduler(
            exploration_bonus=0.0,
            ema_alpha=1.0,
            rollout_objective_weight=0.0,
            reward_scale_normalization="arm_range",
        )
        raw_large = rollout_group(raw, "large")
        raw_small = rollout_group(raw, "small")
        normalized_large = rollout_group(normalized, "large")
        normalized_small = rollout_group(normalized, "small")

        observe_train(raw, raw_large, 1000.0, 0)
        observe_train(raw, raw_large, 1010.0, 1)
        observe_train(raw, raw_small, 0.5, 2)
        observe_train(normalized, normalized_large, 1000.0, 0)
        observe_train(normalized, normalized_large, 1010.0, 1)
        observe_train(normalized, normalized_small, 0.5, 2)

        raw_decision = raw.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=3,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        normalized_decision = normalized.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=3,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        metrics = normalized.metrics()

        self.assertEqual(raw_decision.arm_id, "large|token")
        self.assertEqual(normalized_decision.arm_id, "small|token")
        self.assertAlmostEqual(
            metrics["scheduler/arm/large_token/last_train_reward_scale"],
            1010.0,
        )
        self.assertAlmostEqual(
            metrics[
                "scheduler/arm/large_token/"
                "last_normalized_train_reward_improvement"
            ],
            10.0 / 1010.0,
        )
        self.assertAlmostEqual(
            metrics["scheduler/train_last_reward_improving_experience"],
            0.5,
        )
        self.assertAlmostEqual(
            metrics[
                "scheduler/train_last_control_reward_improving_experience"
            ],
            0.5,
        )
        restored = ObjectiveScheduler()
        restored.load_state_dict(normalized.state_dict())
        restored_metrics = restored.metrics()

        self.assertEqual(
            restored.state_dict()["config"]["reward_scale_normalization"],
            "arm_range",
        )
        self.assertAlmostEqual(
            restored_metrics[
                "scheduler/arm/large_token/last_train_reward_scale"
            ],
            1010.0,
        )

    def test_scheduler_reports_action_logprob_contract_metrics(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        trajectory = Trajectory(
            scenario_id="prob",
            policy_step=0,
            messages=[],
            actions=[
                ActionUnit(
                    kind="chunk",
                    payload=("alpha", "beta"),
                    token_count=2,
                    old_logprob=-2.0,
                    new_logprob=-1.75,
                    reference_logprob=-2.25,
                ),
                ActionUnit(
                    kind="chunk",
                    payload=("gamma", "delta"),
                    token_count=2,
                ),
            ],
            reward=1.0,
            metadata={"scheduler/arm_id": "prob|chunk(chunk_size=2)"},
        )

        scheduler.observe_rollout(
            trajectory,
            accepted=True,
            dollar_seconds=1.0,
        )
        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        metrics = restored.metrics()
        prefix = "scheduler/arm/prob_chunk_chunk_size_2"

        self.assertAlmostEqual(metrics[f"{prefix}/old_logprob_coverage"], 0.5)
        self.assertAlmostEqual(metrics[f"{prefix}/new_logprob_coverage"], 0.5)
        self.assertAlmostEqual(
            metrics[f"{prefix}/reference_logprob_coverage"],
            0.5,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/old_new_logprob_delta_mean"],
            0.25,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/old_new_logprob_abs_delta_mean"],
            0.25,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/old_reference_logprob_delta_mean"],
            0.25,
        )
        self.assertGreater(metrics[f"{prefix}/importance_ratio_mean"], 1.0)

    def test_scheduler_reserves_inflight_untried_arms_for_async_actors(self):
        scenarios = [Scenario(id="easy"), Scenario(id="hard")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)

        decisions = [
            scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=codecs,
                actor_id=actor_id,
                policy_step=0,
                trajectory_queue_pressure=0.0,
                train_queue_pressure=0.0,
                configured_train_batch_groups=2,
                configured_max_policy_lag=2,
            )
            for actor_id in range(4)
        ]

        self.assertEqual(
            [decision.arm_id for decision in decisions],
            [
                "easy|token",
                "easy|chunk(chunk_size=2)",
                "hard|token",
                "hard|chunk(chunk_size=2)",
            ],
        )
        metrics = scheduler.metrics()
        self.assertEqual(metrics["scheduler/total_rollout_decisions"], 4.0)
        self.assertEqual(metrics["scheduler/total_rollout_observations"], 0.0)
        self.assertEqual(metrics["scheduler/total_inflight_rollouts"], 4.0)
        self.assertEqual(metrics["scheduler/arm/easy_token/inflight"], 1.0)
        self.assertEqual(
            metrics["scheduler/arm/easy_chunk_chunk_size_2/inflight"],
            1.0,
        )
        self.assertEqual(
            scheduler.state_dict()["arms"]["easy|token"]["inflight"],
            0,
        )

        scheduler.observe_rollout(
            Trajectory(
                scenario_id=decisions[0].scenario.id,
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.1,
                metadata={"scheduler/arm_id": decisions[0].arm_id},
            ),
            accepted=True,
            dollar_seconds=1.0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/total_rollout_decisions"], 4.0)
        self.assertEqual(metrics["scheduler/total_rollout_observations"], 1.0)
        self.assertEqual(metrics["scheduler/total_inflight_rollouts"], 3.0)
        self.assertEqual(metrics["scheduler/arm/easy_token/inflight"], 0.0)
        self.assertEqual(metrics["scheduler/arm/easy_token/pulls"], 1.0)

    def test_unobserved_rollout_exploration_prefers_lower_estimated_cost(self):
        scenarios = [Scenario(id="expensive"), Scenario(id="cheap")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)

        for scenario_id, cost in (("expensive", 10.0), ("cheap", 1.0)):
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id=scenario_id,
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=0.1,
                    metadata={"scheduler/arm_id": f"{scenario_id}|token"},
                ),
                accepted=True,
                dollar_seconds=cost,
            )

        decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        metrics = scheduler.metrics()

        self.assertEqual(decision.arm_id, "cheap|chunk(chunk_size=2)")
        self.assertAlmostEqual(
            decision.metadata["estimated_rollout_dollar_seconds"],
            1.0,
        )
        self.assertTrue(decision.metadata["unobserved_rollout_cost_estimated"])
        self.assertLess(
            metrics[
                "scheduler/arm/cheap_chunk_chunk_size_2/"
                "unobserved_rollout_cost_penalty"
            ],
            metrics[
                "scheduler/arm/expensive_chunk_chunk_size_2/"
                "unobserved_rollout_cost_penalty"
            ],
        )
        self.assertAlmostEqual(
            metrics["scheduler/last_rollout_estimated_dollar_seconds"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/last_rollout_unobserved_cost_estimated"],
            1.0,
        )

    def test_scheduler_defaults_to_marginal_objective_over_raw_reward_efficiency(self):
        scenarios = [Scenario(id="stale-high"), Scenario(id="fresh-improver")]
        codecs = [TokenActionCodec()]

        default_scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        raw_reward_scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            reward_efficiency_weight=1.0,
        )

        for scheduler in (default_scheduler, raw_reward_scheduler):
            for _ in range(30):
                scheduler.observe_rollout(
                    Trajectory(
                        scenario_id="stale-high",
                        policy_step=0,
                        messages=[],
                        actions=[],
                        reward=100.0,
                        metadata={"scheduler/arm_id": "stale-high|token"},
                    ),
                    accepted=True,
                    dollar_seconds=1.0,
                )
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id="fresh-improver",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=6.0,
                    metadata={"scheduler/arm_id": "fresh-improver|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )

        default_decision = default_scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )
        raw_reward_decision = raw_reward_scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )

        self.assertEqual(default_decision.arm_id, "fresh-improver|token")
        self.assertEqual(raw_reward_decision.arm_id, "stale-high|token")
        self.assertEqual(
            default_scheduler.metrics()["scheduler/weights/reward_efficiency"],
            0.0,
        )
        self.assertGreater(
            default_scheduler.metrics()[
                "scheduler/arm/fresh_improver_token/objective_score"
            ],
            default_scheduler.metrics()[
                "scheduler/arm/stale_high_token/objective_score"
            ],
        )

    def test_scheduler_prefers_lower_reward_arm_when_cost_normalized_objective_is_better(self):
        scenarios = [Scenario(id="expensive"), Scenario(id="cheap")]
        codecs = [TokenActionCodec()]
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)

        scheduler.observe_rollout(
            Trajectory(
                scenario_id="expensive",
                policy_step=0,
                messages=[],
                actions=[],
                reward=10.0,
                metadata={"scheduler/arm_id": "expensive|token"},
            ),
            accepted=True,
            dollar_seconds=100.0,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="cheap",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "cheap|token"},
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )
        metrics = scheduler.metrics()

        self.assertEqual(decision.arm_id, "cheap|token")
        self.assertEqual(
            metrics["scheduler/arm/expensive_token/mean_rollout_dollar_seconds"],
            100.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/cheap_token/mean_rollout_dollar_seconds"],
            1.0,
        )
        self.assertGreater(
            metrics[
                "scheduler/arm/cheap_token/total_improvement_per_dollar_second"
            ],
            metrics[
                "scheduler/arm/expensive_token/total_improvement_per_dollar_second"
            ],
        )

    def test_confidence_penalty_prefers_steadier_objective_gain(self):
        scenarios = [Scenario(id="spiky"), Scenario(id="steady")]
        codecs = [TokenActionCodec()]
        unpenalized = ObjectiveScheduler(exploration_bonus=0.0, ema_alpha=0.5)
        penalized = ObjectiveScheduler(
            exploration_bonus=0.0,
            ema_alpha=0.5,
            confidence_penalty_weight=2.0,
        )

        for scheduler in (unpenalized, penalized):
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id="spiky",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=0.0,
                    metadata={"scheduler/arm_id": "spiky|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id="steady",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=0.0,
                    metadata={"scheduler/arm_id": "steady|token"},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )
            for policy_step, train_reward in ((0, 10.0), (1, 10.0)):
                scheduler.observe_train(
                    groups=[
                        TrajectoryGroup(
                            scenario_id="spiky",
                            trajectories=(
                                Trajectory(
                                    scenario_id="spiky",
                                    policy_step=0,
                                    messages=[],
                                    actions=[],
                                    reward=1.0,
                                    metadata={"scheduler/arm_id": "spiky|token"},
                                ),
                            ),
                        )
                    ],
                    result=TrainResult(metrics={"train/reward": train_reward}),
                    duration_s=1.0,
                    dollar_seconds=1.0,
                    policy_step=policy_step,
                )
            for policy_step, train_reward in ((0, 1.0), (1, 2.0)):
                scheduler.observe_train(
                    groups=[
                        TrajectoryGroup(
                            scenario_id="steady",
                            trajectories=(
                                Trajectory(
                                    scenario_id="steady",
                                    policy_step=0,
                                    messages=[],
                                    actions=[],
                                    reward=1.0,
                                    metadata={"scheduler/arm_id": "steady|token"},
                                ),
                            ),
                        )
                    ],
                    result=TrainResult(metrics={"train/reward": train_reward}),
                    duration_s=1.0,
                    dollar_seconds=1.0,
                    policy_step=policy_step,
                )

        raw_decision = unpenalized.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=2,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        penalized_decision = penalized.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=2,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        metrics = penalized.metrics()

        self.assertEqual(raw_decision.arm_id, "spiky|token")
        self.assertEqual(penalized_decision.arm_id, "steady|token")
        self.assertGreater(
            metrics["scheduler/arm/spiky_token/raw_objective_score"],
            metrics["scheduler/arm/steady_token/raw_objective_score"],
        )
        self.assertLess(
            metrics["scheduler/arm/spiky_token/objective_score"],
            metrics["scheduler/arm/steady_token/objective_score"],
        )
        self.assertGreater(
            metrics["scheduler/arm/spiky_token/confidence_penalty"],
            metrics["scheduler/arm/steady_token/confidence_penalty"],
        )

    def test_rollout_objective_includes_queue_wait_cost(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        no_wait = Trajectory(
            scenario_id="no-wait",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "no-wait|token"},
        )
        waited = Trajectory(
            scenario_id="waited",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "waited|token"},
        )

        scheduler.observe_rollout(no_wait, accepted=True, dollar_seconds=1.0)
        scheduler.observe_rollout(
            waited,
            accepted=True,
            dollar_seconds=1.0,
            queue_wait_dollar_seconds=9.0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/costs/queue_wait_dollar_seconds"], 9.0)
        self.assertEqual(
            metrics["scheduler/arm/waited_token/queue_wait_dollar_seconds"],
            9.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/waited_token/mean_rollout_dollar_seconds"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/waited_token/mean_sample_dollar_seconds"],
            10.0,
        )
        self.assertGreater(
            metrics["scheduler/arm/no_wait_token/objective_score"],
            metrics["scheduler/arm/waited_token/objective_score"],
        )

    def test_rollout_metrics_expose_arm_semantic_bandwidth(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        actions = ChunkActionCodec(chunk_size=2).encode("alpha beta gamma delta")

        scheduler.observe_rollout(
            Trajectory(
                scenario_id="bandwidth",
                policy_step=0,
                messages=[],
                actions=actions,
                reward=1.0,
                metadata={
                    "scheduler/arm_id": "bandwidth|chunk(chunk_size=2)",
                },
            ),
            accepted=True,
            dollar_seconds=2.0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(
            metrics["scheduler/arm/bandwidth_chunk_chunk_size_2/action_units"],
            2.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/bandwidth_chunk_chunk_size_2/source_tokens"],
            4.0,
        )
        self.assertEqual(
            metrics[
                "scheduler/arm/bandwidth_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision"
            ],
            2.0,
        )
        self.assertEqual(
            metrics[
                "scheduler/arm/bandwidth_chunk_chunk_size_2/source_tokens_per_dollar_second"
            ],
            2.0,
        )

    def test_rollout_admission_delay_backs_off_saturated_low_signal_sampling(self):
        scheduler = ObjectiveScheduler(
            max_rollout_admission_delay_s=0.2,
            rollout_admission_pressure_threshold=0.5,
            exploration_bonus=0.0,
        )

        delay = scheduler.rollout_admission_delay_s(
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        scheduler.observe_rollout_admission_delay(
            seconds=delay,
            dollar_seconds=delay * 3.0,
        )
        metrics = scheduler.metrics()

        self.assertAlmostEqual(delay, 0.2)
        self.assertEqual(metrics["scheduler/admission/decisions"], 1.0)
        self.assertAlmostEqual(metrics["scheduler/admission/total_delay_s"], 0.2)
        self.assertAlmostEqual(
            metrics["scheduler/costs/rollout_admission_dollar_seconds"],
            0.6,
        )
        self.assertAlmostEqual(metrics["scheduler/costs/total_dollar_seconds"], 0.6)

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        restored_metrics = restored.metrics()

        self.assertAlmostEqual(
            restored_metrics["scheduler/admission/total_delay_s"],
            metrics["scheduler/admission/total_delay_s"],
        )
        self.assertAlmostEqual(
            restored_metrics[
                "scheduler/costs/rollout_admission_dollar_seconds"
            ],
            metrics["scheduler/costs/rollout_admission_dollar_seconds"],
        )

    def test_rollout_admission_delay_preserves_more_sampling_with_objective_signal(self):
        no_signal = ObjectiveScheduler(
            max_rollout_admission_delay_s=0.2,
            rollout_admission_pressure_threshold=0.5,
            exploration_bonus=0.0,
        )
        positive_signal = ObjectiveScheduler(
            max_rollout_admission_delay_s=0.2,
            rollout_admission_pressure_threshold=0.5,
            rollout_admission_positive_signal_scale=0.25,
            exploration_bonus=0.0,
        )
        positive_signal.observe_rollout(
            Trajectory(
                scenario_id="useful",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "useful|token"},
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        no_signal_delay = no_signal.rollout_admission_delay_s(
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        positive_signal_delay = positive_signal.rollout_admission_delay_s(
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=1,
        )

        self.assertAlmostEqual(no_signal_delay, 0.2)
        self.assertAlmostEqual(positive_signal_delay, 0.05)
        self.assertLess(positive_signal_delay, no_signal_delay)

    def test_rollout_admission_delay_explores_and_reuses_objective_controls(self):
        scheduler = ObjectiveScheduler(
            max_rollout_admission_delay_s=0.2,
            rollout_admission_pressure_threshold=0.5,
            exploration_bonus=0.0,
            control_exploration_bonus=0.1,
        )

        first_delay = scheduler.rollout_admission_delay_s(
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        first_delay_ms = int(round(first_delay * 1000.0))
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="admission",
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.0,
                metadata={
                    "scheduler/arm_id": "admission|token",
                    "scheduler/active_rollout_admission_delay_ms": first_delay_ms,
                },
                metrics={
                    "cost/actor_admission_dollar_seconds": first_delay,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        second_delay = scheduler.rollout_admission_delay_s(
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        second_delay_ms = int(round(second_delay * 1000.0))
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="admission",
                policy_step=1,
                messages=[],
                actions=[],
                reward=5.0,
                metadata={
                    "scheduler/arm_id": "admission|token",
                    "scheduler/active_rollout_admission_delay_ms": second_delay_ms,
                },
                metrics={
                    "cost/actor_admission_dollar_seconds": second_delay,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        third_delay = scheduler.rollout_admission_delay_s(
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        metrics = scheduler.metrics()

        self.assertAlmostEqual(first_delay, 0.2)
        self.assertNotEqual(second_delay_ms, first_delay_ms)
        self.assertAlmostEqual(third_delay, second_delay)
        self.assertEqual(
            metrics[
                f"scheduler/control/admission_delay_ms_{first_delay_ms}/rollout_updates"
            ],
            1.0,
        )
        self.assertEqual(
            metrics[
                f"scheduler/control/admission_delay_ms_{second_delay_ms}/rollout_updates"
            ],
            1.0,
        )
        self.assertGreater(
            metrics[f"scheduler/control/admission_delay_ms_{second_delay_ms}/score"],
            metrics[f"scheduler/control/admission_delay_ms_{first_delay_ms}/score"],
        )
        self.assertAlmostEqual(
            metrics["scheduler/arm/admission_token/admission_dollar_seconds"],
            first_delay + second_delay,
        )

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        restored_metrics = restored.metrics()
        self.assertAlmostEqual(
            restored_metrics[
                f"scheduler/control/admission_delay_ms_{second_delay_ms}/score"
            ],
            metrics[f"scheduler/control/admission_delay_ms_{second_delay_ms}/score"],
        )

    def test_rollout_failure_modes_track_reconstruction_drift(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            reconstruction_drift_threshold=0.95,
        )
        trajectory = Trajectory(
            scenario_id="code",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "code|chunk(chunk_size=4)",
                "reconstruction/accuracy": 0.9,
            },
        )

        self.assertEqual(
            trajectory_failure_modes(
                trajectory,
                reconstruction_drift_threshold=0.95,
            ),
            ("reconstruction_drift",),
        )
        self.assertEqual(trajectory_reconstruction_accuracy(trajectory), 0.9)

        scheduler.observe_rollout(
            trajectory,
            accepted=True,
            dollar_seconds=1.0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/failure_rollouts"], 1.0)
        self.assertEqual(metrics["scheduler/failure/reconstruction_drift"], 1.0)
        self.assertEqual(
            metrics[
                "scheduler/arm/code_chunk_chunk_size_4/failure/reconstruction_drift"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/code_chunk_chunk_size_4/failure_rate"],
            1.0,
        )
        self.assertEqual(metrics["scheduler/reconstruction_observations"], 1.0)
        self.assertAlmostEqual(
            metrics["scheduler/reconstruction_accuracy_mean"],
            0.9,
        )
        self.assertAlmostEqual(
            metrics["scheduler/reconstruction_max_drift"],
            0.1,
        )
        self.assertEqual(
            metrics[
                "scheduler/arm/code_chunk_chunk_size_4/reconstruction_observations"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            metrics[
                "scheduler/arm/code_chunk_chunk_size_4/reconstruction_accuracy_ema"
            ],
            0.9,
        )
        self.assertAlmostEqual(
            metrics[
                "scheduler/arm/code_chunk_chunk_size_4/reconstruction_accuracy_mean"
            ],
            0.9,
        )
        self.assertAlmostEqual(
            metrics[
                "scheduler/arm/code_chunk_chunk_size_4/reconstruction_accuracy_min"
            ],
            0.9,
        )
        self.assertAlmostEqual(
            metrics["scheduler/arm/code_chunk_chunk_size_4/reconstruction_drift_ema"],
            0.1,
        )
        self.assertAlmostEqual(
            metrics["scheduler/arm/code_chunk_chunk_size_4/reconstruction_max_drift"],
            0.1,
        )
        self.assertEqual(
            metrics["scheduler/arm/code_chunk_chunk_size_4/unsafe"],
            0.0,
        )

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        restored_metrics = restored.metrics()

        self.assertEqual(
            restored_metrics[
                "scheduler/arm/code_chunk_chunk_size_4/failure/reconstruction_drift"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            restored_metrics[
                "scheduler/arm/code_chunk_chunk_size_4/reconstruction_max_drift"
            ],
            0.1,
        )

    def test_rollout_failure_modes_accept_domain_verifier_modes(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        trajectory = Trajectory(
            scenario_id="tool",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "tool|chunk(chunk_size=2)",
                "verifier/failure_mode": "Tool timeout",
                "failure/modes": ["syntax_error", "Tool timeout", ""],
            },
        )

        self.assertEqual(
            trajectory_failure_modes(trajectory),
            ("syntax_error", "Tool_timeout"),
        )
        self.assertEqual(action_quality(trajectory), 0.0)

        scheduler.observe_rollout(
            trajectory,
            accepted=True,
            dollar_seconds=1.0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/failure_rollouts"], 1.0)
        self.assertEqual(metrics["scheduler/failure/Tool_timeout"], 1.0)
        self.assertEqual(metrics["scheduler/failure/syntax_error"], 1.0)
        self.assertEqual(
            metrics[
                "scheduler/arm/tool_chunk_chunk_size_2/failure/Tool_timeout"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/tool_chunk_chunk_size_2/action_quality_ema"],
            0.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/tool_chunk_chunk_size_2/effective_reward_ema"],
            0.0,
        )

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        restored_metrics = restored.metrics()

        self.assertEqual(
            restored_metrics[
                "scheduler/arm/tool_chunk_chunk_size_2/failure/syntax_error"
            ],
            1.0,
        )

    def test_positive_objective_tightens_cadence_and_policy_lag(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
        )

        before_batch_groups = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        before_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=0,
        )

        scheduler.observe_rollout(
            Trajectory(
                scenario_id="easy",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "easy|chunk(chunk_size=2)"},
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        after_batch_groups = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        after_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=1,
        )

        self.assertEqual(before_batch_groups, 3)
        self.assertEqual(before_lag, 2)
        self.assertEqual(after_batch_groups, 1)
        self.assertEqual(after_lag, 0)

    def test_train_pressure_widens_cadence_without_objective_signal(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            exploration_bonus=0.0,
        )

        target = scheduler.target_train_batch_groups(
            configured=2,
            pending_groups=0,
            train_queue_pressure=0.9,
            policy_step=0,
        )

        self.assertEqual(target, 4)

    def test_train_pressure_uses_cadence_feedback_after_stale_waste(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            exploration_bonus=0.0,
        )
        stale = Trajectory(
            scenario_id="pressure",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "pressure|token",
                "scheduler/active_target_train_batch_groups": 4,
            },
        )

        scheduler.observe_stale_batch(
            groups=[
                TrajectoryGroup(
                    scenario_id="pressure",
                    trajectories=(stale,),
                )
            ],
            policy_step=1,
            reason="pressure_cadence_waste",
        )
        target = scheduler.target_train_batch_groups(
            configured=2,
            pending_groups=0,
            train_queue_pressure=0.9,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertNotEqual(target, 4)
        self.assertLess(
            metrics["scheduler/control/cadence_4/objective_ema"],
            0.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/cadence_{target}/decisions"],
            1.0,
        )

    def test_control_selection_explores_untried_runtime_values(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
        )

        first_target = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        second_target = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        first_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        second_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertEqual(first_target, 3)
        self.assertIn(second_target, {1, 2, 4})
        self.assertNotEqual(second_target, first_target)
        self.assertEqual(first_lag, 2)
        self.assertIn(second_lag, {0, 1, 3})
        self.assertNotEqual(second_lag, first_lag)
        self.assertEqual(
            metrics[f"scheduler/control/cadence_{second_target}/decisions"],
            1.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/policy_lag_{second_lag}/decisions"],
            1.0,
        )
        self.assertGreater(metrics["scheduler/weights/control_exploration"], 0.0)

    def test_control_selection_reuses_higher_objective_runtime_values(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
        )
        fast = Trajectory(
            scenario_id="fast",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "fast|token",
                "scheduler/active_target_train_batch_groups": 2,
                "scheduler/active_max_policy_lag": 1,
            },
        )
        slow = Trajectory(
            scenario_id="slow",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "slow|token",
                "scheduler/active_target_train_batch_groups": 4,
                "scheduler/active_max_policy_lag": 3,
            },
        )

        scheduler.observe_rollout(fast, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="fast", trajectories=(fast,))],
            result=TrainResult(metrics={"train/reward": 2.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )
        scheduler.observe_rollout(slow, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="slow", trajectories=(slow,))],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=10.0,
            policy_step=1,
        )

        target = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        metrics = scheduler.metrics()

        self.assertEqual(target, 2)
        self.assertEqual(lag, 1)
        self.assertGreater(
            metrics["scheduler/control/cadence_2/objective_ema"],
            metrics["scheduler/control/cadence_4/objective_ema"],
        )
        self.assertGreater(
            metrics["scheduler/control/policy_lag_1/objective_ema"],
            metrics["scheduler/control/policy_lag_3/objective_ema"],
        )
        self.assertGreater(
            metrics["scheduler/control/cadence_2/score"],
            metrics["scheduler/control/cadence_4/score"],
        )

    def test_rollout_feedback_credits_and_reuses_cadence_and_lag_controls(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
            control_exploration_bonus=0.1,
            rollout_cadence_lag_control_weight=1.0,
        )

        first_target = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        first_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="controls",
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.0,
                metadata={
                    "scheduler/arm_id": "controls|token",
                    "scheduler/active_target_train_batch_groups": first_target,
                    "scheduler/active_max_policy_lag": first_lag,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        second_target = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        second_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="controls",
                policy_step=1,
                messages=[],
                actions=[],
                reward=5.0,
                metadata={
                    "scheduler/arm_id": "controls|token",
                    "scheduler/active_target_train_batch_groups": second_target,
                    "scheduler/active_max_policy_lag": second_lag,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        third_target = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        third_lag = scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        metrics = scheduler.metrics()

        self.assertEqual(first_target, 3)
        self.assertEqual(first_lag, 2)
        self.assertNotEqual(second_target, first_target)
        self.assertNotEqual(second_lag, first_lag)
        self.assertEqual(third_target, second_target)
        self.assertEqual(third_lag, second_lag)
        self.assertEqual(
            metrics[f"scheduler/control/cadence_{second_target}/rollout_updates"],
            1.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/policy_lag_{second_lag}/rollout_updates"],
            1.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/cadence_{second_target}/train_updates"],
            0.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/policy_lag_{second_lag}/train_updates"],
            0.0,
        )
        self.assertGreater(
            metrics[f"scheduler/control/cadence_{second_target}/score"],
            metrics[f"scheduler/control/cadence_{first_target}/score"],
        )
        self.assertGreater(
            metrics[f"scheduler/control/policy_lag_{second_lag}/score"],
            metrics[f"scheduler/control/policy_lag_{first_lag}/score"],
        )

    def test_actor_count_control_explores_and_reuses_objective_values(self):
        scheduler = ObjectiveScheduler(
            min_actor_count=1,
            max_actor_count=4,
            exploration_bonus=0.0,
            control_exploration_bonus=0.1,
        )

        first_count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="actors",
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.0,
                metadata={
                    "scheduler/arm_id": "actors|token",
                    "scheduler/active_actor_count": first_count,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        second_count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="actors",
                policy_step=1,
                messages=[],
                actions=[],
                reward=5.0,
                metadata={
                    "scheduler/arm_id": "actors|token",
                    "scheduler/active_actor_count": second_count,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        third_count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        metrics = scheduler.metrics()

        self.assertEqual(first_count, 4)
        self.assertNotEqual(second_count, first_count)
        self.assertEqual(third_count, second_count)
        self.assertEqual(
            metrics[f"scheduler/control/actor_count_{first_count}/rollout_updates"],
            1.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/actor_count_{second_count}/rollout_updates"],
            1.0,
        )
        self.assertGreater(
            metrics[f"scheduler/control/actor_count_{second_count}/score"],
            metrics[f"scheduler/control/actor_count_{first_count}/score"],
        )

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        restored_metrics = restored.metrics()
        self.assertAlmostEqual(
            restored_metrics[f"scheduler/control/actor_count_{second_count}/score"],
            metrics[f"scheduler/control/actor_count_{second_count}/score"],
        )

    def test_actor_slots_track_rollout_cost_and_objective(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        decision = scheduler.select_rollout(
            scenarios=[Scenario(id="actors")],
            action_codecs=[TokenActionCodec()],
            actor_id=2,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )
        metrics_after_decision = scheduler.metrics()

        self.assertEqual(
            metrics_after_decision["scheduler/actor/actor_2/decisions"],
            1.0,
        )
        self.assertEqual(
            metrics_after_decision["scheduler/actor/actor_2/inflight"],
            1.0,
        )

        scheduler.observe_rollout(
            Trajectory(
                scenario_id="actors",
                policy_step=0,
                messages=[],
                actions=[
                    ActionUnit(
                        kind="token",
                        payload="alpha",
                        token_count=3,
                    )
                ],
                reward=2.0,
                metadata={
                    "actor_id": 2,
                    "scheduler/arm_id": decision.arm_id,
                    "cost/actor_admission_dollar_seconds": 0.25,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
            queue_wait_dollar_seconds=0.5,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/actor/actor_2/inflight"], 0.0)
        self.assertEqual(metrics["scheduler/actor/actor_2/pulls"], 1.0)
        self.assertEqual(metrics["scheduler/actor/actor_2/accepted"], 1.0)
        self.assertEqual(
            metrics["scheduler/actor/actor_2/sample_dollar_seconds"],
            1.75,
        )
        self.assertEqual(
            metrics["scheduler/actor/actor_2/queue_wait_dollar_seconds"],
            0.5,
        )
        self.assertEqual(
            metrics["scheduler/actor/actor_2/admission_dollar_seconds"],
            0.25,
        )
        self.assertEqual(metrics["scheduler/actor/actor_2/action_units"], 1.0)
        self.assertEqual(metrics["scheduler/actor/actor_2/source_tokens"], 3.0)
        self.assertGreater(
            metrics["scheduler/actor/actor_2/rollout_objective_ema"],
            0.0,
        )

    def test_actor_slots_receive_train_and_stale_objective_credit(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0, ema_alpha=1.0)
        actor_0 = Trajectory(
            scenario_id="actors",
            policy_step=0,
            messages=[],
            actions=[],
            reward=2.0,
            metadata={
                "actor_id": 0,
                "scheduler/arm_id": "actors|token",
                "scheduler/active_actor_count": 2,
            },
        )
        actor_1 = Trajectory(
            scenario_id="actors",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "actor_id": 1,
                "scheduler/arm_id": "actors|token",
                "scheduler/active_actor_count": 2,
            },
        )
        group = TrajectoryGroup(
            scenario_id="actors",
            trajectories=(actor_0, actor_1),
        )

        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 3.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )
        scheduler.observe_stale_batch(
            groups=[group],
            policy_step=2,
            reason="actor-test",
        )
        metrics = scheduler.metrics()

        self.assertGreater(
            metrics["scheduler/actor/actor_0/total_train_objective"],
            metrics["scheduler/actor/actor_1/total_train_objective"],
        )
        self.assertEqual(metrics["scheduler/actor/actor_0/train_updates"], 1.0)
        self.assertEqual(metrics["scheduler/actor/actor_1/train_updates"], 1.0)
        self.assertEqual(metrics["scheduler/actor/actor_0/stale_updates"], 1.0)
        self.assertEqual(metrics["scheduler/actor/actor_1/stale_updates"], 1.0)
        self.assertLess(
            metrics["scheduler/actor/actor_0/total_stale_penalty_objective"],
            0.0,
        )
        self.assertGreater(
            metrics["scheduler/actor/actor_0/stale_experience"],
            metrics["scheduler/actor/actor_1/stale_experience"],
        )

    def test_actor_count_control_backs_off_saturated_low_signal_sampling(self):
        scheduler = ObjectiveScheduler(
            min_actor_count=1,
            max_actor_count=4,
            exploration_bonus=0.0,
        )

        count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(count, 1)
        self.assertEqual(
            metrics["scheduler/control/actor_count_1/decisions"],
            1.0,
        )

    def test_actor_count_pressure_uses_feedback_after_stale_waste(self):
        scheduler = ObjectiveScheduler(
            min_actor_count=1,
            max_actor_count=4,
            exploration_bonus=0.0,
        )
        stale = Trajectory(
            scenario_id="actors",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "actors|token",
                "scheduler/active_actor_count": 1,
            },
        )

        scheduler.observe_stale_batch(
            groups=[
                TrajectoryGroup(
                    scenario_id="actors",
                    trajectories=(stale,),
                )
            ],
            policy_step=1,
            reason="actor_pressure_waste",
        )
        count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=1.0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertNotEqual(count, 1)
        self.assertLess(
            metrics["scheduler/control/actor_count_1/objective_ema"],
            0.0,
        )
        self.assertEqual(
            metrics[f"scheduler/control/actor_count_{count}/decisions"],
            1.0,
        )

    def test_actor_count_backs_off_after_zero_roi_train_updates(self):
        scheduler = ObjectiveScheduler(
            min_actor_count=1,
            max_actor_count=4,
            exploration_bonus=0.0,
            ema_alpha=1.0,
            min_train_objective=0.0,
        )

        first_count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="roi",
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.0,
                metadata={"scheduler/arm_id": "roi|token"},
            ),
            accepted=True,
            dollar_seconds=1.0,
        )
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="roi",
                    trajectories=(
                        Trajectory(
                            scenario_id="roi",
                            policy_step=0,
                            messages=[],
                            actions=[],
                            reward=0.0,
                            metadata={"scheduler/arm_id": "roi|token"},
                        ),
                    ),
                )
            ],
            result=TrainResult(metrics={"train/reward": 0.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=1,
        )

        second_count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertEqual(first_count, 4)
        self.assertEqual(second_count, 1)
        self.assertEqual(
            metrics["scheduler/control/actor_count_1/decisions"],
            1.0,
        )

    def test_actor_count_low_roi_uses_feedback_after_stale_waste(self):
        scheduler = ObjectiveScheduler(
            min_actor_count=1,
            max_actor_count=4,
            exploration_bonus=0.0,
            ema_alpha=1.0,
            min_train_objective=0.0,
        )
        stale = Trajectory(
            scenario_id="actors",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "actors|token",
                "scheduler/active_actor_count": 1,
            },
        )
        zero_roi = Trajectory(
            scenario_id="roi",
            policy_step=0,
            messages=[],
            actions=[],
            reward=0.0,
            metadata={"scheduler/arm_id": "roi|token"},
        )

        scheduler.observe_stale_batch(
            groups=[
                TrajectoryGroup(
                    scenario_id="actors",
                    trajectories=(stale,),
                )
            ],
            policy_step=1,
            reason="actor_low_roi_waste",
        )
        scheduler.observe_rollout(zero_roi, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="roi", trajectories=(zero_roi,))],
            result=TrainResult(metrics={"train/reward": 0.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=1,
        )
        count = scheduler.active_actor_count(
            configured=4,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            policy_step=2,
        )

        self.assertNotEqual(count, 1)

    def test_train_objective_tightens_cadence_and_lag_under_pressure(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
        )
        useful = Trajectory(
            scenario_id="train",
            policy_step=0,
            messages=[],
            actions=[],
            reward=0.0,
            metadata={"scheduler/arm_id": "train|token"},
        )

        scheduler.observe_rollout(useful, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="train", trajectories=(useful,))],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        self.assertEqual(
            scheduler.target_train_batch_groups(
                configured=2,
                pending_groups=0,
                train_queue_pressure=0.9,
                policy_step=1,
            ),
            1,
        )
        self.assertEqual(
            scheduler.max_policy_lag(
                configured=2,
                train_queue_pressure=0.0,
                policy_step=1,
            ),
            0,
        )

    def test_train_objective_credits_and_reuses_cadence_and_lag_controls(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
        )
        trajectory = Trajectory(
            scenario_id="control",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "control|token",
                "scheduler/active_target_train_batch_groups": 4,
                "scheduler/active_max_policy_lag": 3,
            },
        )
        scheduler.observe_rollout(trajectory, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="control",
                    trajectories=(trajectory,),
                )
            ],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        target = scheduler.target_train_batch_groups(
            configured=2,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        lag = scheduler.max_policy_lag(
            configured=1,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertEqual(target, 4)
        self.assertEqual(lag, 3)
        self.assertEqual(
            metrics["scheduler/control/cadence_4/train_updates"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/control/policy_lag_3/train_updates"],
            1.0,
        )
        self.assertGreater(
            metrics["scheduler/control/cadence_4/objective_ema"],
            0.0,
        )
        self.assertGreater(
            metrics["scheduler/control/policy_lag_3/objective_ema"],
            0.0,
        )

    def test_control_train_credit_uses_accounted_interval_objective_by_default(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=2,
            min_policy_lag=0,
            max_policy_lag=1,
            ema_alpha=1.0,
            exploration_bonus=0.0,
            control_exploration_bonus=0.0,
        )
        cheap = Trajectory(
            scenario_id="cheap-control",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "cheap-control|token",
                "scheduler/active_target_train_batch_groups": 1,
                "scheduler/active_max_policy_lag": 0,
            },
        )
        expensive = Trajectory(
            scenario_id="expensive-control",
            policy_step=1,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "expensive-control|token",
                "scheduler/active_target_train_batch_groups": 2,
                "scheduler/active_max_policy_lag": 1,
            },
        )

        scheduler.observe_rollout(cheap, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="cheap-control",
                    trajectories=(cheap,),
                )
            ],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )
        scheduler.observe_rollout(expensive, accepted=True, dollar_seconds=99.0)
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="expensive-control",
                    trajectories=(expensive,),
                )
            ],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=1,
        )

        target = scheduler.target_train_batch_groups(
            configured=2,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        lag = scheduler.max_policy_lag(
            configured=1,
            train_queue_pressure=0.0,
            policy_step=2,
        )
        metrics = scheduler.metrics()

        self.assertEqual(target, 1)
        self.assertEqual(lag, 0)
        self.assertEqual(metrics["scheduler/control/train_objective_accounted"], 1.0)
        self.assertGreater(
            metrics["scheduler/control/cadence_1/objective_ema"],
            metrics["scheduler/control/cadence_2/objective_ema"],
        )
        self.assertGreater(
            metrics["scheduler/control/policy_lag_0/objective_ema"],
            metrics["scheduler/control/policy_lag_1/objective_ema"],
        )

    def test_accounted_train_control_credit_follows_improving_arm_in_mixed_batch(
        self,
    ):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            ema_alpha=1.0,
            exploration_bonus=0.0,
            control_exploration_bonus=0.0,
        )
        high_reward_baseline = Trajectory(
            scenario_id="high-static",
            policy_step=0,
            messages=[],
            actions=[],
            reward=10.0,
            metadata={"scheduler/arm_id": "high-static|token"},
        )
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="high-static",
                    trajectories=(high_reward_baseline,),
                )
            ],
            result=TrainResult(metrics={"train/reward": 10.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        high_reward_static = Trajectory(
            scenario_id="high-static",
            policy_step=1,
            messages=[],
            actions=[],
            reward=10.0,
            metadata={
                "scheduler/arm_id": "high-static|token",
                "scheduler/active_target_train_batch_groups": 4,
            },
        )
        low_reward_improving = Trajectory(
            scenario_id="low-improving",
            policy_step=1,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "low-improving|token",
                "scheduler/active_target_train_batch_groups": 1,
            },
        )
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="mixed-control",
                    trajectories=(high_reward_static, low_reward_improving),
                )
            ],
            result=TrainResult(metrics={"train/reward": 10.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/accounted_last_objective"], 10.0)
        self.assertEqual(
            metrics["scheduler/arm/high_static_token/last_train_reward_improvement"],
            0.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/low_improving_token/last_train_reward_improvement"],
            10.0,
        )
        self.assertEqual(
            metrics["scheduler/control/cadence_4/total_objective"],
            0.0,
        )
        self.assertEqual(
            metrics["scheduler/control/cadence_1/total_objective"],
            10.0,
        )
        self.assertEqual(
            scheduler.target_train_batch_groups(
                configured=2,
                pending_groups=0,
                train_queue_pressure=0.0,
                policy_step=2,
            ),
            1,
        )

    def test_stale_batch_feedback_penalizes_arms_and_controls(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            ema_alpha=1.0,
            exploration_bonus=0.0,
        )
        trajectory = Trajectory(
            scenario_id="control",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "control|token",
                "scheduler/active_target_train_batch_groups": 4,
                "scheduler/active_max_policy_lag": 3,
            },
        )
        group = TrajectoryGroup(
            scenario_id="control",
            trajectories=(trajectory,),
        )

        scheduler.observe_rollout(trajectory, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        self.assertEqual(
            scheduler.target_train_batch_groups(
                configured=2,
                pending_groups=0,
                train_queue_pressure=0.0,
                policy_step=1,
            ),
            4,
        )
        scheduler.observe_stale_batch(
            groups=[group],
            policy_step=1,
            reason="test_stale",
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/stale_batches"], 1.0)
        self.assertEqual(metrics["scheduler/stale_trajectories"], 1.0)
        self.assertEqual(metrics["scheduler/stale_experience"], 1.0)
        self.assertEqual(metrics["scheduler/arm/control_token/stale_updates"], 1.0)
        self.assertEqual(
            metrics["scheduler/control/cadence_4/stale_updates"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/control/policy_lag_3/stale_updates"],
            1.0,
        )
        self.assertLess(
            metrics["scheduler/control/cadence_4/objective_ema"],
            0.0,
        )
        self.assertLess(
            metrics["scheduler/control/policy_lag_3/objective_ema"],
            0.0,
        )
        self.assertEqual(
            scheduler.target_train_batch_groups(
                configured=2,
                pending_groups=0,
                train_queue_pressure=0.0,
                policy_step=2,
            ),
            1,
        )
        self.assertEqual(
            scheduler.max_policy_lag(
                configured=1,
                train_queue_pressure=0.0,
                policy_step=2,
            ),
            0,
        )

    def test_policy_lag_feedback_overrides_unaccepted_arm_protection(self):
        scheduler = ObjectiveScheduler(
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
        )
        stale = Trajectory(
            scenario_id="lag",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "lag|token",
                "scheduler/active_max_policy_lag": 3,
            },
        )

        scheduler.observe_stale_batch(
            groups=[TrajectoryGroup(scenario_id="lag", trajectories=(stale,))],
            policy_step=4,
            reason="lag_feedback",
        )
        lag = scheduler.max_policy_lag(
            configured=3,
            train_queue_pressure=0.0,
            policy_step=4,
        )
        metrics = scheduler.metrics()

        self.assertNotEqual(lag, 3)
        self.assertEqual(metrics["scheduler/arm/lag_token/accepted"], 0.0)
        self.assertEqual(metrics["scheduler/control/policy_lag_3/stale_updates"], 1.0)
        self.assertLess(metrics["scheduler/control/policy_lag_3/objective_ema"], 0.0)
        self.assertEqual(
            metrics[f"scheduler/control/policy_lag_{lag}/decisions"],
            1.0,
        )

    def test_stale_penalty_uses_estimated_lost_objective_when_available(self):
        scheduler = ObjectiveScheduler(
            ema_alpha=1.0,
            exploration_bonus=0.0,
        )
        trajectory = Trajectory(
            scenario_id="valuable",
            policy_step=0,
            messages=[],
            actions=[],
            reward=4.0,
            metrics={"cost/dollar_seconds": 2.0},
            metadata={
                "scheduler/arm_id": "valuable|token",
                "scheduler/active_target_train_batch_groups": 2,
                "scheduler/active_max_policy_lag": 1,
            },
        )
        group = TrajectoryGroup(
            scenario_id="valuable",
            trajectories=(trajectory,),
        )

        scheduler.observe_rollout(
            trajectory,
            accepted=True,
            dollar_seconds=1.0,
        )
        scheduler.observe_stale_batch(
            groups=[group],
            policy_step=3,
            reason="lagged",
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/stale_experience"], 1.0)
        self.assertEqual(metrics["scheduler/stale_sample_dollar_seconds"], 2.0)
        self.assertAlmostEqual(
            metrics["scheduler/stale_last_lost_reward_improving_experience"],
            8.0,
        )
        self.assertAlmostEqual(
            metrics["scheduler/stale_last_penalty_objective"],
            -4.0,
        )
        self.assertLess(
            metrics["scheduler/control/policy_lag_1/objective_ema"],
            metrics["scheduler/arm/valuable_token/marginal_objective_ema"],
        )

    def test_train_objective_scales_by_useful_experience_count(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        useful = tuple(
            Trajectory(
                scenario_id="batch",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "batch|token"},
            )
            for _ in range(3)
        )
        unsafe = Trajectory(
            scenario_id="batch",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "batch|token",
                "action/safe": False,
                "action/quality": 0.0,
            },
        )
        group = TrajectoryGroup(
            scenario_id="batch",
            trajectories=(*useful, unsafe),
        )

        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=2.0,
            policy_step=0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/train_last_reward_improvement"], 1.0)
        self.assertEqual(metrics["scheduler/train_last_experience_count"], 3.0)
        self.assertEqual(
            metrics["scheduler/train_last_reward_improving_experience"],
            3.0,
        )
        self.assertEqual(metrics["scheduler/train_last_objective"], 1.5)

    def test_train_objective_prefers_promotion_score_when_present(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        group = TrajectoryGroup(
            scenario_id="candidate",
            trajectories=(
                Trajectory(
                    scenario_id="candidate",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=1.0,
                    metadata={"scheduler/arm_id": "candidate|token"},
                ),
            ),
        )

        scheduler.observe_train(
            groups=[group],
            result=TrainResult(
                metrics={
                    "train/reward": 10.0,
                    "promotion/score": 0.0,
                    "promotion/promoted": 0.0,
                }
            ),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        metrics = scheduler.metrics()
        self.assertEqual(metrics["scheduler/train_last_reward_improvement"], 0.0)
        self.assertEqual(metrics["scheduler/train_last_objective"], 0.0)
        self.assertEqual(
            metrics["scheduler/arm/candidate_token/policy_improvement_objective_ema"],
            0.0,
        )

    def test_train_group_scoring_prefers_high_objective_arms(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        high = Trajectory(
            scenario_id="easy",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "easy|chunk(chunk_size=2)"},
        )
        low = Trajectory(
            scenario_id="hard",
            policy_step=0,
            messages=[],
            actions=[],
            reward=0.1,
            metadata={"scheduler/arm_id": "hard|token"},
        )

        scheduler.observe_rollout(high, accepted=True, dollar_seconds=1.0)
        scheduler.observe_rollout(low, accepted=True, dollar_seconds=1.0)

        high_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="easy", trajectories=(high,))],
            policy_step=0,
        )
        low_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="hard", trajectories=(low,))],
            policy_step=0,
        )

        self.assertGreater(high_score, low_score)

    def test_train_group_scoring_normalizes_queued_batch_by_sample_cost(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        history = Trajectory(
            scenario_id="costed",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "costed|token"},
        )
        cheap = Trajectory(
            scenario_id="costed",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metrics={"cost/dollar_seconds": 1.0},
            metadata={"scheduler/arm_id": "costed|token"},
        )
        expensive = Trajectory(
            scenario_id="costed",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metrics={"cost/dollar_seconds": 20.0},
            metadata={"scheduler/arm_id": "costed|token"},
        )

        scheduler.observe_rollout(history, accepted=True, dollar_seconds=1.0)
        cheap_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="costed", trajectories=(cheap,))],
            policy_step=0,
        )
        cheap_metrics = scheduler.metrics()
        expensive_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="costed", trajectories=(expensive,))],
            policy_step=0,
        )
        expensive_metrics = scheduler.metrics()

        self.assertGreater(cheap_score, expensive_score)
        self.assertEqual(
            cheap_metrics["scheduler/last_train_batch_sample_dollar_seconds"],
            1.0,
        )
        self.assertEqual(
            cheap_metrics["scheduler/last_train_batch_cost_normalized_priority"],
            cheap_score,
        )
        self.assertEqual(
            expensive_metrics["scheduler/last_train_batch_sample_dollar_seconds"],
            20.0,
        )
        self.assertAlmostEqual(
            expensive_metrics[
                "scheduler/last_train_batch_cost_normalized_priority"
            ],
            expensive_score,
        )
        self.assertAlmostEqual(expensive_score, cheap_score / 20.0)

    def test_train_group_scoring_boosts_useful_near_stale_batches(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            staleness_priority_weight=1.0,
        )
        fresh = Trajectory(
            scenario_id="fresh",
            policy_step=8,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "fresh|token",
                "scheduler/active_max_policy_lag": 4,
            },
        )
        near_stale = Trajectory(
            scenario_id="near-stale",
            policy_step=6,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "near-stale|token",
                "scheduler/active_max_policy_lag": 4,
            },
        )

        fresh_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="fresh", trajectories=(fresh,))],
            policy_step=10,
        )
        near_stale_score = scheduler.score_train_groups(
            [
                TrajectoryGroup(
                    scenario_id="near-stale",
                    trajectories=(near_stale,),
                )
            ],
            policy_step=10,
        )
        metrics = scheduler.metrics()

        self.assertGreater(near_stale_score, fresh_score)
        self.assertEqual(metrics["scheduler/last_train_batch_policy_lag"], 4.0)
        self.assertEqual(metrics["scheduler/last_train_batch_lag_limit"], 4.0)
        self.assertEqual(
            metrics["scheduler/last_train_batch_staleness_urgency"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/last_train_batch_staleness_bonus"],
            1.0,
        )

    def test_train_group_scoring_penalizes_off_policy_action_drift(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            off_policy_priority_weight=1.0,
            staleness_priority_weight=0.0,
        )
        low_drift = Trajectory(
            scenario_id="drift",
            policy_step=0,
            messages=[],
            actions=[
                ActionUnit(
                    kind="chunk",
                    payload=("alpha",),
                    token_count=1,
                    old_logprob=-2.0,
                    new_logprob=-1.9,
                )
            ],
            reward=1.0,
            metadata={"scheduler/arm_id": "drift|chunk(chunk_size=1)"},
        )
        high_drift = Trajectory(
            scenario_id="drift",
            policy_step=0,
            messages=[],
            actions=[
                ActionUnit(
                    kind="chunk",
                    payload=("alpha",),
                    token_count=1,
                    old_logprob=-2.0,
                    new_logprob=1.0,
                )
            ],
            reward=1.0,
            metadata={"scheduler/arm_id": "drift|chunk(chunk_size=1)"},
        )
        unaccounted = Trajectory(
            scenario_id="drift",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "drift|token"},
        )

        low_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="drift", trajectories=(low_drift,))],
            policy_step=0,
        )
        high_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="drift", trajectories=(high_drift,))],
            policy_step=0,
        )
        high_metrics = scheduler.metrics()
        unaccounted_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="drift", trajectories=(unaccounted,))],
            policy_step=0,
        )
        unaccounted_metrics = scheduler.metrics()

        self.assertGreater(low_score, high_score)
        self.assertAlmostEqual(
            high_metrics["scheduler/last_train_batch_old_new_logprob_coverage"],
            1.0,
        )
        self.assertAlmostEqual(
            high_metrics["scheduler/last_train_batch_off_policy_drift"],
            3.0,
        )
        self.assertAlmostEqual(
            high_metrics["scheduler/last_train_batch_off_policy_penalty"],
            3.0,
        )
        self.assertAlmostEqual(
            high_metrics[
                "scheduler/last_train_batch_priority_before_off_policy"
            ],
            1.0,
        )
        self.assertAlmostEqual(high_score, 0.25)
        self.assertAlmostEqual(unaccounted_score, 1.0)
        self.assertEqual(
            unaccounted_metrics[
                "scheduler/last_train_batch_off_policy_penalty"
            ],
            0.0,
        )

    def test_off_policy_action_drift_tightens_policy_lag(self):
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            exploration_bonus=0.0,
            off_policy_priority_weight=1.0,
            off_policy_cadence_tightening_threshold=0.5,
            off_policy_lag_tightening_threshold=0.5,
            staleness_priority_weight=0.0,
        )
        low_drift = Trajectory(
            scenario_id="lag",
            policy_step=0,
            messages=[],
            actions=[
                ActionUnit(
                    kind="chunk",
                    payload=("alpha",),
                    token_count=1,
                    old_logprob=-1.0,
                    new_logprob=-0.75,
                )
            ],
            reward=1.0,
        )
        high_drift = Trajectory(
            scenario_id="lag",
            policy_step=0,
            messages=[],
            actions=[
                ActionUnit(
                    kind="chunk",
                    payload=("alpha",),
                    token_count=1,
                    old_logprob=-2.0,
                    new_logprob=0.0,
                )
            ],
            reward=1.0,
        )

        scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="lag", trajectories=(low_drift,))],
            policy_step=0,
        )
        loose_cadence = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        loose_lag = scheduler.max_policy_lag(
            configured=3,
            train_queue_pressure=0.0,
            policy_step=0,
        )
        scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="lag", trajectories=(high_drift,))],
            policy_step=1,
        )
        tight_cadence = scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        tight_lag = scheduler.max_policy_lag(
            configured=3,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        metrics = scheduler.metrics()

        self.assertEqual(loose_cadence, 3)
        self.assertEqual(loose_lag, 3)
        self.assertEqual(tight_cadence, 1)
        self.assertEqual(tight_lag, 0)
        self.assertEqual(
            metrics["scheduler/cadence/last_off_policy_penalty"],
            2.0,
        )
        self.assertEqual(
            metrics["scheduler/cadence/off_policy_tightened"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/cadence/off_policy_tightening_threshold"],
            0.5,
        )
        self.assertEqual(
            metrics["scheduler/policy_lag/last_off_policy_penalty"],
            2.0,
        )
        self.assertEqual(
            metrics["scheduler/policy_lag/off_policy_tightened"],
            1.0,
        )
        self.assertEqual(
            metrics["scheduler/policy_lag/off_policy_tightening_threshold"],
            0.5,
        )

    def test_unsafe_high_reward_action_granularity_is_penalized(self):
        scenarios = [Scenario(id="task")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(exploration_bonus=0.0, unsafe_penalty=10.0)

        token_decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="task",
                policy_step=0,
                messages=[],
                actions=[],
                reward=0.4,
                metadata={"scheduler/arm_id": token_decision.arm_id},
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        chunk_decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="task",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={
                    "scheduler/arm_id": chunk_decision.arm_id,
                    "action/safe": False,
                    "action/quality": 0.0,
                },
            ),
            accepted=True,
            dollar_seconds=1.0,
        )

        next_decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )

        self.assertEqual(token_decision.arm_id, "task|token")
        self.assertEqual(chunk_decision.arm_id, "task|chunk(chunk_size=2)")
        self.assertEqual(next_decision.arm_id, "task|token")
        self.assertEqual(
            scheduler.metrics()["scheduler/arm/task_chunk_chunk_size_2/unsafe"],
            1.0,
        )

    def test_train_group_scoring_penalizes_unsafe_batches(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0, unsafe_penalty=10.0)
        safe = Trajectory(
            scenario_id="task",
            policy_step=0,
            messages=[],
            actions=[],
            reward=0.4,
            metadata={"scheduler/arm_id": "task|token"},
        )
        unsafe = Trajectory(
            scenario_id="task",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": "task|chunk(chunk_size=2)",
                "verifier/passed": False,
                "reconstruction/accuracy": 0.0,
            },
        )

        scheduler.observe_rollout(safe, accepted=True, dollar_seconds=1.0)
        scheduler.observe_rollout(unsafe, accepted=True, dollar_seconds=1.0)

        safe_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="task", trajectories=(safe,))],
            policy_step=0,
        )
        unsafe_score = scheduler.score_train_groups(
            [TrajectoryGroup(scenario_id="task", trajectories=(unsafe,))],
            policy_step=0,
        )

        self.assertGreater(safe_score, unsafe_score)

    def test_train_group_scoring_penalizes_current_unsafe_batch_from_good_arm(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0, unsafe_penalty=5.0)
        good_history = Trajectory(
            scenario_id="task",
            policy_step=0,
            messages=[],
            actions=[],
            reward=2.0,
            metadata={"scheduler/arm_id": "task|token"},
        )
        safe_current = Trajectory(
            scenario_id="task",
            policy_step=0,
            messages=[],
            actions=[],
            reward=2.0,
            metadata={"scheduler/arm_id": "task|token"},
        )
        unsafe_current = Trajectory(
            scenario_id="task",
            policy_step=0,
            messages=[],
            actions=[],
            reward=2.0,
            metadata={
                "scheduler/arm_id": "task|token",
                "verifier/passed": False,
            },
        )

        scheduler.observe_rollout(
            good_history,
            accepted=True,
            dollar_seconds=1.0,
        )
        safe_score = scheduler.score_train_groups(
            [
                TrajectoryGroup(
                    scenario_id="task",
                    trajectories=(safe_current,),
                )
            ],
            policy_step=0,
        )
        unsafe_score = scheduler.score_train_groups(
            [
                TrajectoryGroup(
                    scenario_id="task",
                    trajectories=(unsafe_current,),
                )
            ],
            policy_step=0,
        )

        self.assertGreater(safe_score, 0.0)
        self.assertLess(unsafe_score, 0.0)

    def test_train_policy_improvement_credit_can_override_rollout_reward(self):
        scenarios = [Scenario(id="sample"), Scenario(id="train")]
        codecs = [TokenActionCodec()]
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        sample_rich = Trajectory(
            scenario_id="sample",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "sample|token"},
        )
        train_useful = Trajectory(
            scenario_id="train",
            policy_step=0,
            messages=[],
            actions=[],
            reward=0.1,
            metadata={"scheduler/arm_id": "train|token"},
        )

        scheduler.observe_rollout(sample_rich, accepted=True, dollar_seconds=1.0)
        scheduler.observe_rollout(train_useful, accepted=True, dollar_seconds=1.0)

        before_train_credit = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )
        self.assertEqual(before_train_credit.arm_id, "sample|token")

        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="train",
                    trajectories=(train_useful,),
                )
            ],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        after_train_credit = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=1,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=2,
            configured_max_policy_lag=2,
        )

        self.assertEqual(after_train_credit.arm_id, "train|token")
        self.assertGreater(
            scheduler.metrics()[
                "scheduler/arm/train_token/policy_improvement_objective_ema"
            ],
            0.0,
        )
        self.assertEqual(
            scheduler.metrics()["scheduler/costs/train_dollar_seconds"],
            1.0,
        )

    def test_rollout_coverage_floor_forces_undercovered_arm(self):
        scenarios = [Scenario(id="task")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_rollout_coverage_fraction=0.25,
        )

        rewards = {
            "task|token": 1.0,
            "task|chunk(chunk_size=2)": 0.1,
        }
        decisions = []
        for _ in range(6):
            decision = scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=codecs,
                actor_id=0,
                policy_step=0,
                trajectory_queue_pressure=0.0,
                train_queue_pressure=0.0,
                configured_train_batch_groups=2,
                configured_max_policy_lag=2,
            )
            decisions.append(decision)
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id=decision.scenario.id,
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=rewards[decision.arm_id],
                    metadata={"scheduler/arm_id": decision.arm_id},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )

        metrics = scheduler.metrics()

        self.assertEqual(
            [decision.arm_id for decision in decisions],
            [
                "task|token",
                "task|chunk(chunk_size=2)",
                "task|token",
                "task|token",
                "task|token",
                "task|chunk(chunk_size=2)",
            ],
        )
        self.assertTrue(decisions[-1].metadata["coverage_forced"])
        self.assertEqual(metrics["scheduler/coverage/min_fraction"], 0.25)
        self.assertEqual(metrics["scheduler/coverage/forced_decisions"], 1.0)
        self.assertEqual(
            metrics["scheduler/coverage/last_target"],
            0.25,
        )
        self.assertLess(
            metrics[
                "scheduler/arm/task_chunk_chunk_size_2/decision_share"
            ],
            metrics["scheduler/arm/task_token/decision_share"],
        )

    def test_rollout_coverage_floor_respects_cost_cap(self):
        scenarios = [Scenario(id="task")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_rollout_coverage_fraction=0.25,
            max_rollout_coverage_cost_fraction=0.5,
        )

        rewards = {
            "task|token": 1.0,
            "task|chunk(chunk_size=2)": 0.1,
        }
        costs = {
            "task|token": 1.0,
            "task|chunk(chunk_size=2)": 10.0,
        }
        decisions = []
        for _ in range(6):
            decision = scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=codecs,
                actor_id=0,
                policy_step=0,
                trajectory_queue_pressure=0.0,
                train_queue_pressure=0.0,
                configured_train_batch_groups=2,
                configured_max_policy_lag=2,
            )
            decisions.append(decision)
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id=decision.scenario.id,
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=rewards[decision.arm_id],
                    metadata={"scheduler/arm_id": decision.arm_id},
                ),
                accepted=True,
                dollar_seconds=costs[decision.arm_id],
            )

        metrics = scheduler.metrics()

        self.assertEqual(decisions[-1].arm_id, "task|token")
        self.assertFalse(decisions[-1].metadata["coverage_forced"])
        self.assertTrue(decisions[-1].metadata["coverage_cost_limited"])
        self.assertEqual(decisions[-1].metadata["coverage_cost_limit"], 0.5)
        self.assertEqual(metrics["scheduler/coverage/forced_decisions"], 0.0)
        self.assertEqual(metrics["scheduler/coverage/max_cost_fraction"], 0.5)
        self.assertEqual(metrics["scheduler/coverage/last_cost_limited"], 1.0)
        self.assertGreater(
            metrics["scheduler/arm/task_chunk_chunk_size_2/sample_dollar_share"],
            0.5,
        )

    def test_rollout_coverage_floor_is_capped_by_arm_count(self):
        scenarios = [Scenario(id="left"), Scenario(id="right")]
        codecs = [TokenActionCodec()]
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_rollout_coverage_fraction=0.8,
        )

        decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )

        self.assertEqual(decision.metadata["coverage_target"], 0.5)

    def test_rollout_coverage_floor_preserves_new_arm_exploration(self):
        scenarios = [Scenario(id="task")]
        codecs = [TokenActionCodec(), ChunkActionCodec(chunk_size=2)]
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_rollout_coverage_fraction=0.25,
        )

        for _ in range(2):
            decision = scheduler.select_rollout(
                scenarios=scenarios,
                action_codecs=codecs,
                actor_id=0,
                policy_step=0,
                trajectory_queue_pressure=0.0,
                train_queue_pressure=0.0,
                configured_train_batch_groups=1,
                configured_max_policy_lag=1,
            )
            scheduler.observe_rollout(
                Trajectory(
                    scenario_id=decision.scenario.id,
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=1.0,
                    metadata={"scheduler/arm_id": decision.arm_id},
                ),
                accepted=True,
                dollar_seconds=1.0,
            )

        new_arm_decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=[
                TokenActionCodec(),
                ChunkActionCodec(chunk_size=2),
                ChunkActionCodec(chunk_size=4),
            ],
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )

        self.assertEqual(
            new_arm_decision.arm_id,
            "task|chunk(chunk_size=4)",
        )
        self.assertFalse(new_arm_decision.metadata["coverage_forced"])

    def test_train_credit_uses_arm_local_reward_baselines(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0, ema_alpha=1.0)
        high = Trajectory(
            scenario_id="high",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "high|token"},
        )
        low = Trajectory(
            scenario_id="low",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "low|token"},
        )

        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="high", trajectories=(high,))],
            result=TrainResult(metrics={"train/reward": 10.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )
        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="low", trajectories=(low,))],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=1,
        )
        low_metrics = scheduler.metrics()

        self.assertEqual(low_metrics["scheduler/train_last_objective"], 1.0)
        self.assertEqual(
            low_metrics["scheduler/arm/low_token/last_train_reward_improvement"],
            1.0,
        )
        self.assertEqual(
            low_metrics[
                "scheduler/arm/low_token/policy_improvement_objective_ema"
            ],
            1.0,
        )

        scheduler.observe_train(
            groups=[TrajectoryGroup(scenario_id="high", trajectories=(high,))],
            result=TrainResult(metrics={"train/reward": 9.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=2,
        )
        high_metrics = scheduler.metrics()

        self.assertEqual(high_metrics["scheduler/train_last_objective"], 0.0)
        self.assertEqual(
            high_metrics["scheduler/arm/high_token/last_train_reward"],
            9.0,
        )
        self.assertEqual(
            high_metrics["scheduler/arm/high_token/last_train_reward_improvement"],
            0.0,
        )
        self.assertEqual(
            high_metrics[
                "scheduler/arm/high_token/policy_improvement_objective_ema"
            ],
            0.0,
        )
        self.assertEqual(
            high_metrics[
                "scheduler/arm/high_token/total_reward_improving_experience"
            ],
            10.0,
        )

    def test_train_policy_improvement_credit_ignores_unsafe_actions(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        unsafe = Trajectory(
            scenario_id="unsafe",
            policy_step=0,
            messages=[],
            actions=[],
            reward=10.0,
            metadata={
                "scheduler/arm_id": "unsafe|chunk",
                "action/safe": False,
                "action/quality": 0.0,
            },
        )

        scheduler.observe_rollout(unsafe, accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[
                TrajectoryGroup(
                    scenario_id="unsafe",
                    trajectories=(unsafe,),
                )
            ],
            result=TrainResult(metrics={"train/reward": 10.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        metrics = scheduler.metrics()
        self.assertEqual(
            metrics["scheduler/arm/unsafe_chunk/policy_improvement_objective_ema"],
            0.0,
        )
        self.assertEqual(metrics["scheduler/arm/unsafe_chunk/train_updates"], 1.0)

    def test_roi_patience_stops_after_repeated_low_objective_train_steps(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_train_steps=1,
            roi_patience=2,
            min_train_objective=0.0,
        )
        group = TrajectoryGroup(
            scenario_id="flat",
            trajectories=(
                Trajectory(
                    scenario_id="flat",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=0.0,
                    metadata={"scheduler/arm_id": "flat|token"},
                ),
            ),
        )

        scheduler.observe_rollout(group.trajectories[0], accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 0.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )
        self.assertTrue(
            scheduler.should_continue_training(
                policy_step=1,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )

        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 0.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=1,
        )

        self.assertFalse(
            scheduler.should_continue_training(
                policy_step=2,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )
        self.assertEqual(scheduler.metrics()["scheduler/stop_recommended"], 1.0)

    def test_accounted_continuation_roi_counts_sample_cost(self):
        train_only = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_train_steps=1,
            roi_patience=1,
            min_train_objective=0.5,
        )
        accounted = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_train_steps=1,
            roi_patience=1,
            min_train_objective=0.5,
            continuation_objective="accounted",
        )

        for scheduler in (train_only, accounted):
            trajectory = Trajectory(
                scenario_id="expensive-sample",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "expensive-sample|token"},
            )
            group = TrajectoryGroup(
                scenario_id="expensive-sample",
                trajectories=(trajectory,),
            )
            scheduler.observe_rollout(
                trajectory,
                accepted=True,
                dollar_seconds=99.0,
            )
            scheduler.observe_train(
                groups=[group],
                result=TrainResult(metrics={"train/reward": 1.0}),
                duration_s=1.0,
                dollar_seconds=1.0,
                policy_step=0,
            )

        self.assertTrue(
            train_only.should_continue_training(
                policy_step=1,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )
        self.assertFalse(
            accounted.should_continue_training(
                policy_step=1,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )
        metrics = accounted.metrics()

        self.assertEqual(metrics["scheduler/train_last_objective"], 1.0)
        self.assertEqual(metrics["scheduler/accounted_last_objective"], 0.01)
        self.assertEqual(metrics["scheduler/continuation_last_objective"], 0.01)
        self.assertEqual(metrics["scheduler/accounted_last_dollar_seconds"], 100.0)
        self.assertEqual(metrics["scheduler/continuation/objective_accounted"], 1.0)
        self.assertEqual(metrics["scheduler/stop_recommended"], 1.0)

    def test_accounted_budget_stops_after_dollar_second_limit(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            max_accounted_dollar_seconds=3.0,
        )
        trajectory = Trajectory(
            scenario_id="budgeted",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "budgeted|token"},
        )
        group = TrajectoryGroup(
            scenario_id="budgeted",
            trajectories=(trajectory,),
        )

        scheduler.observe_rollout(
            trajectory,
            accepted=True,
            dollar_seconds=2.0,
        )
        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.5,
            policy_step=0,
        )

        self.assertFalse(
            scheduler.should_continue_training(
                policy_step=1,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/budget/max_accounted_dollar_seconds"], 3.0)
        self.assertEqual(metrics["scheduler/budget/accounted_dollar_seconds"], 3.5)
        self.assertEqual(
            metrics["scheduler/budget/remaining_accounted_dollar_seconds"],
            0.0,
        )
        self.assertGreater(metrics["scheduler/budget/accounted_fraction"], 1.0)
        self.assertEqual(metrics["scheduler/budget/accounted_exhausted"], 1.0)
        self.assertEqual(metrics["scheduler/stop_recommended"], 1.0)

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        restored_metrics = restored.metrics()

        self.assertEqual(restored.max_accounted_dollar_seconds, 3.0)
        self.assertEqual(
            restored_metrics["scheduler/budget/accounted_exhausted"],
            1.0,
        )
        self.assertFalse(
            restored.should_continue_training(
                policy_step=1,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )

    def test_accounted_budget_counts_inflight_rollout_reservations(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            max_accounted_dollar_seconds=3.0,
        )
        observed = Trajectory(
            scenario_id="budgeted",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={"scheduler/arm_id": "budgeted|token"},
        )
        scheduler.observe_rollout(
            observed,
            accepted=True,
            dollar_seconds=2.0,
        )

        decision = scheduler.select_rollout(
            scenarios=[Scenario(id="budgeted")],
            action_codecs=[TokenActionCodec()],
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )

        self.assertEqual(decision.arm_id, "budgeted|token")
        self.assertEqual(
            decision.metadata["reserved_rollout_dollar_seconds"],
            2.0,
        )
        self.assertFalse(
            scheduler.should_continue_training(
                policy_step=0,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/budget/accounted_dollar_seconds"], 2.0)
        self.assertEqual(
            metrics["scheduler/budget/reserved_inflight_rollout_dollar_seconds"],
            2.0,
        )
        self.assertEqual(
            metrics["scheduler/budget/projected_accounted_dollar_seconds"],
            4.0,
        )
        self.assertEqual(metrics["scheduler/budget/accounted_exhausted"], 1.0)
        self.assertEqual(
            metrics["scheduler/arm/budgeted_token/reserved_rollout_dollar_seconds"],
            2.0,
        )
        self.assertEqual(
            scheduler.state_dict()["arms"]["budgeted|token"][
                "reserved_rollout_dollar_seconds"
            ],
            0.0,
        )

        completed = Trajectory(
            scenario_id="budgeted",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metadata={
                "scheduler/arm_id": decision.arm_id,
                "scheduler/decision/reserved_rollout_dollar_seconds": 2.0,
            },
        )
        scheduler.observe_rollout(
            completed,
            accepted=True,
            dollar_seconds=2.0,
        )
        metrics = scheduler.metrics()

        self.assertEqual(
            metrics["scheduler/budget/reserved_inflight_rollout_dollar_seconds"],
            0.0,
        )
        self.assertEqual(
            metrics["scheduler/arm/budgeted_token/reserved_rollout_dollar_seconds"],
            0.0,
        )
        self.assertEqual(metrics["scheduler/budget/accounted_dollar_seconds"], 4.0)
        self.assertEqual(
            metrics["scheduler/budget/projected_accounted_dollar_seconds"],
            4.0,
        )

    def test_cancel_rollout_decision_releases_unspent_reservation(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            max_accounted_dollar_seconds=3.0,
        )
        scheduler.observe_rollout(
            Trajectory(
                scenario_id="budgeted",
                policy_step=0,
                messages=[],
                actions=[],
                reward=1.0,
                metadata={"scheduler/arm_id": "budgeted|token"},
            ),
            accepted=True,
            dollar_seconds=2.0,
        )
        decision = scheduler.select_rollout(
            scenarios=[Scenario(id="budgeted")],
            action_codecs=[TokenActionCodec()],
            actor_id=3,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )

        scheduler.cancel_rollout_decision(decision)
        metrics = scheduler.metrics()

        self.assertEqual(metrics["scheduler/total_rollout_decisions"], 0.0)
        self.assertEqual(metrics["scheduler/total_inflight_rollouts"], 0.0)
        self.assertEqual(
            metrics["scheduler/budget/reserved_inflight_rollout_dollar_seconds"],
            0.0,
        )
        self.assertEqual(
            metrics["scheduler/budget/projected_accounted_dollar_seconds"],
            2.0,
        )
        self.assertEqual(metrics["scheduler/actor/actor_3/decisions"], 0.0)
        self.assertEqual(metrics["scheduler/actor/actor_3/inflight"], 0.0)
        self.assertEqual(
            metrics["scheduler/arm/budgeted_token/reserved_rollout_dollar_seconds"],
            0.0,
        )
        self.assertEqual(metrics["scheduler/arm/budgeted_token/decisions"], 0.0)

    def test_positive_train_objective_resets_roi_patience(self):
        scheduler = ObjectiveScheduler(
            exploration_bonus=0.0,
            min_train_steps=1,
            roi_patience=1,
            min_train_objective=0.0,
        )
        group = TrajectoryGroup(
            scenario_id="improving",
            trajectories=(
                Trajectory(
                    scenario_id="improving",
                    policy_step=0,
                    messages=[],
                    actions=[],
                    reward=1.0,
                    metadata={"scheduler/arm_id": "improving|token"},
                ),
            ),
        )

        scheduler.observe_rollout(group.trajectories[0], accepted=True, dollar_seconds=1.0)
        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 1.0}),
            duration_s=1.0,
            dollar_seconds=1.0,
            policy_step=0,
        )

        self.assertTrue(
            scheduler.should_continue_training(
                policy_step=1,
                max_train_steps=10,
                pending_train_batches=0,
                train_queue_pressure=0.0,
            )
        )
        self.assertEqual(scheduler.metrics()["scheduler/low_roi_train_steps"], 0.0)

    def test_scheduler_state_round_trips_objective_and_control_memory(self):
        scenarios = [Scenario(id="cheap"), Scenario(id="expensive")]
        codecs = [TokenActionCodec()]
        scheduler = ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=4,
            min_policy_lag=0,
            max_policy_lag=3,
            min_actor_count=1,
            max_actor_count=4,
            ema_alpha=0.5,
            exploration_bonus=0.0,
            reward_efficiency_weight=0.25,
            staleness_priority_weight=0.5,
            off_policy_priority_weight=0.75,
            off_policy_cadence_tightening_threshold=0.2,
            off_policy_lag_tightening_threshold=0.2,
            confidence_penalty_weight=0.25,
            control_exploration_bonus=0.15,
            rollout_cadence_lag_control_weight=0.2,
            max_control_candidate_values=5,
            min_rollout_coverage_fraction=0.2,
            max_rollout_coverage_cost_fraction=0.4,
            roi_patience=3,
            min_train_objective=0.01,
            continuation_objective="accounted",
            max_accounted_dollar_seconds=100.0,
        )
        cheap = Trajectory(
            scenario_id="cheap",
            policy_step=0,
            messages=[],
            actions=[
                ActionUnit(
                    kind="token",
                    payload="cheap",
                    token_count=1,
                    old_logprob=-2.0,
                    new_logprob=-1.5,
                )
            ],
            reward=2.0,
            metadata={
                "actor_id": 7,
                "scheduler/arm_id": "cheap|token",
                "scheduler/active_target_train_batch_groups": 1,
                "scheduler/active_max_policy_lag": 0,
                "action/quality": 0.8,
            },
        )
        expensive = Trajectory(
            scenario_id="expensive",
            policy_step=0,
            messages=[],
            actions=[],
            reward=0.3,
            metadata={"scheduler/arm_id": "expensive|token"},
        )
        group = TrajectoryGroup(scenario_id="cheap", trajectories=(cheap,))

        scheduler.observe_rollout(cheap, accepted=True, dollar_seconds=1.0)
        scheduler.observe_rollout(expensive, accepted=True, dollar_seconds=5.0)
        scheduler.score_train_groups([group], policy_step=1)
        scheduler.target_train_batch_groups(
            configured=3,
            pending_groups=0,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        scheduler.max_policy_lag(
            configured=2,
            train_queue_pressure=0.0,
            policy_step=1,
        )
        scheduler.observe_train(
            groups=[group],
            result=TrainResult(metrics={"train/reward": 3.0}),
            duration_s=1.0,
            dollar_seconds=2.0,
            policy_step=1,
        )
        decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=7,
            policy_step=2,
            trajectory_queue_pressure=0.25,
            train_queue_pressure=0.0,
            configured_train_batch_groups=3,
            configured_max_policy_lag=2,
        )

        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        before_metrics = scheduler.metrics()
        restored_metrics = restored.metrics()

        self.assertEqual(decision.arm_id, "cheap|token")
        self.assertEqual(restored.reward_efficiency_weight, 0.25)
        self.assertEqual(restored.staleness_priority_weight, 0.5)
        self.assertEqual(restored.off_policy_priority_weight, 0.75)
        self.assertEqual(restored.off_policy_cadence_tightening_threshold, 0.2)
        self.assertEqual(restored.off_policy_lag_tightening_threshold, 0.2)
        self.assertEqual(restored.confidence_penalty_weight, 0.25)
        self.assertEqual(restored.control_exploration_bonus, 0.15)
        self.assertEqual(restored.rollout_cadence_lag_control_weight, 0.2)
        self.assertEqual(restored.max_control_candidate_values, 5)
        self.assertEqual(restored.min_rollout_coverage_fraction, 0.2)
        self.assertEqual(restored.max_rollout_coverage_cost_fraction, 0.4)
        self.assertEqual(restored.min_actor_count, 1)
        self.assertEqual(restored.max_actor_count_limit, 4)
        self.assertEqual(restored.continuation_objective, "accounted")
        self.assertEqual(restored.control_train_objective, "accounted")
        self.assertEqual(restored.max_accounted_dollar_seconds, 100.0)
        self.assertEqual(restored.roi_patience, 3)
        for key in (
            "scheduler/arm/cheap_token/pulls",
            "scheduler/arm/cheap_token/policy_improvement_objective_ema",
            "scheduler/arm/cheap_token/objective_observations",
            "scheduler/arm/cheap_token/objective_mean",
            "scheduler/arm/cheap_token/objective_stddev",
            "scheduler/arm/cheap_token/confidence_penalty",
            "scheduler/arm/cheap_token/last_train_reward",
            "scheduler/arm/cheap_token/total_reward_improving_experience",
            "scheduler/actor/actor_7/pulls",
            "scheduler/actor/actor_7/train_updates",
            "scheduler/actor/actor_7/rollout_objective_ema",
            "scheduler/actor/actor_7/train_objective_ema",
            "scheduler/actor/actor_7/total_objective",
            "scheduler/control/cadence_1/train_updates",
            "scheduler/control/cadence_1/score",
            "scheduler/control/cadence_1/exploration_score",
            "scheduler/control/policy_lag_0/train_updates",
            "scheduler/control/policy_lag_0/score",
            "scheduler/control/policy_lag_0/exploration_score",
            "scheduler/costs/rollout_dollar_seconds",
            "scheduler/costs/train_dollar_seconds",
            "scheduler/accounted_objective_ema",
            "scheduler/accounted_last_objective",
            "scheduler/accounted_last_reward_improving_experience",
            "scheduler/accounted_last_dollar_seconds",
            "scheduler/continuation_last_objective",
            "scheduler/continuation/objective_accounted",
            "scheduler/budget/max_accounted_dollar_seconds",
            "scheduler/budget/accounted_dollar_seconds",
            "scheduler/control/train_objective_accounted",
            "scheduler/train_last_experience_count",
            "scheduler/train_last_reward_improving_experience",
            "scheduler/last_train_batch_policy_lag",
            "scheduler/last_train_batch_lag_limit",
            "scheduler/last_train_batch_staleness_urgency",
            "scheduler/last_train_batch_staleness_bonus",
            "scheduler/last_train_batch_old_new_logprob_coverage",
            "scheduler/last_train_batch_off_policy_drift",
            "scheduler/last_train_batch_off_policy_penalty",
            "scheduler/last_train_batch_priority_before_off_policy",
            "scheduler/cadence/last_off_policy_penalty",
            "scheduler/cadence/off_policy_tightened",
            "scheduler/cadence/off_policy_tightening_threshold",
            "scheduler/policy_lag/last_off_policy_penalty",
            "scheduler/policy_lag/off_policy_tightened",
            "scheduler/policy_lag/off_policy_tightening_threshold",
            "scheduler/last_train_batch_reward_improving_experience",
            "scheduler/last_train_batch_sample_dollar_seconds",
            "scheduler/last_train_batch_cost_normalized_priority",
            "scheduler/last_arm/cheap_token",
            "scheduler/last_target_train_batch_groups",
            "scheduler/last_max_policy_lag",
            "scheduler/last_rollout_estimated_dollar_seconds",
            "scheduler/last_rollout_unobserved_cost_penalty",
            "scheduler/last_rollout_unobserved_cost_estimated",
            "scheduler/arm/cheap_token/estimated_rollout_dollar_seconds",
            "scheduler/arm/cheap_token/unobserved_rollout_cost_penalty",
            "scheduler/weights/control_exploration",
            "scheduler/weights/rollout_cadence_lag_control",
            "scheduler/weights/off_policy_priority",
            "scheduler/coverage/min_fraction",
            "scheduler/coverage/max_cost_fraction",
            "scheduler/coverage/forced_decisions",
            "scheduler/coverage/last_target",
            "scheduler/coverage/last_share",
            "scheduler/coverage/last_deficit",
            "scheduler/coverage/last_cost_share",
            "scheduler/coverage/last_cost_limited",
            "scheduler/max_control_candidate_values",
        ):
            self.assertAlmostEqual(restored_metrics[key], before_metrics[key])
        self.assertGreater(
            before_metrics[
                "scheduler/budget/reserved_inflight_rollout_dollar_seconds"
            ],
            0.0,
        )
        self.assertEqual(
            restored_metrics[
                "scheduler/budget/reserved_inflight_rollout_dollar_seconds"
            ],
            0.0,
        )
        self.assertEqual(
            restored_metrics["scheduler/budget/projected_accounted_dollar_seconds"],
            restored_metrics["scheduler/budget/accounted_dollar_seconds"],
        )

        original_next = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=8,
            policy_step=3,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=3,
            configured_max_policy_lag=2,
        )
        restored_next = restored.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=8,
            policy_step=3,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=3,
            configured_max_policy_lag=2,
        )

        self.assertEqual(restored_next.arm_id, original_next.arm_id)
        self.assertEqual(
            restored_next.target_train_batch_groups,
            original_next.target_train_batch_groups,
        )
        self.assertEqual(restored_next.max_policy_lag, original_next.max_policy_lag)

    def test_scheduler_state_round_trips_stale_feedback(self):
        scheduler = ObjectiveScheduler(
            stale_penalty_weight=2.0,
            ema_alpha=1.0,
            exploration_bonus=0.0,
        )
        trajectory = Trajectory(
            scenario_id="stale",
            policy_step=0,
            messages=[],
            actions=[],
            reward=1.0,
            metrics={"cost/dollar_seconds": 4.0},
            metadata={
                "scheduler/arm_id": "stale|token",
                "scheduler/active_target_train_batch_groups": 2,
                "scheduler/active_max_policy_lag": 1,
            },
        )
        group = TrajectoryGroup(scenario_id="stale", trajectories=(trajectory,))

        scheduler.observe_stale_batch(
            groups=[group],
            policy_step=5,
            reason="state-test",
        )
        restored = ObjectiveScheduler()
        restored.load_state_dict(scheduler.state_dict())
        before_metrics = scheduler.metrics()
        restored_metrics = restored.metrics()

        self.assertEqual(restored.stale_penalty_weight, 2.0)
        for key in (
            "scheduler/stale_batches",
            "scheduler/stale_trajectories",
            "scheduler/stale_experience",
            "scheduler/stale_last_penalty_objective",
            "scheduler/stale_last_experience_count",
            "scheduler/stale_last_lost_reward_improving_experience",
            "scheduler/stale_last_sample_dollar_seconds",
            "scheduler/stale_last_policy_step",
            "scheduler/stale_lost_reward_improving_experience",
            "scheduler/stale_sample_dollar_seconds",
            "scheduler/arm/stale_token/stale_updates",
            "scheduler/arm/stale_token/stale_experience",
            "scheduler/control/cadence_2/stale_updates",
            "scheduler/control/cadence_2/objective_ema",
            "scheduler/control/policy_lag_1/stale_updates",
            "scheduler/control/policy_lag_1/objective_ema",
        ):
            self.assertAlmostEqual(restored_metrics[key], before_metrics[key])

    def test_scheduler_state_load_tolerates_missing_sections(self):
        scheduler = ObjectiveScheduler(exploration_bonus=0.0)
        scheduler.load_state_dict(
            {
                "arms": {
                    "task|token": {
                        "pulls": 2,
                        "accepted": 1,
                        "marginal_objective_ema": 0.4,
                        "action_quality_ema": 1.0,
                    }
                }
            }
        )

        metrics = scheduler.metrics()
        decision = scheduler.select_rollout(
            scenarios=[Scenario(id="task")],
            action_codecs=[TokenActionCodec()],
            actor_id=0,
            policy_step=0,
            trajectory_queue_pressure=0.0,
            train_queue_pressure=0.0,
            configured_train_batch_groups=1,
            configured_max_policy_lag=1,
        )

        self.assertEqual(metrics["scheduler/total_rollout_decisions"], 2.0)
        self.assertEqual(metrics["scheduler/arm/task_token/pulls"], 2.0)
        self.assertEqual(decision.arm_id, "task|token")


if __name__ == "__main__":
    unittest.main()
