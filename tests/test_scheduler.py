import unittest

from calm_puffer_art import (
    ChunkActionCodec,
    ObjectiveScheduler,
    Scenario,
    TokenActionCodec,
    Trajectory,
    TrajectoryGroup,
    TrainResult,
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
            ema_alpha=0.5,
            exploration_bonus=0.0,
            reward_efficiency_weight=0.25,
            roi_patience=3,
            min_train_objective=0.01,
        )
        cheap = Trajectory(
            scenario_id="cheap",
            policy_step=0,
            messages=[],
            actions=[],
            reward=2.0,
            metadata={
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
        self.assertEqual(restored.roi_patience, 3)
        for key in (
            "scheduler/arm/cheap_token/pulls",
            "scheduler/arm/cheap_token/policy_improvement_objective_ema",
            "scheduler/arm/cheap_token/last_train_reward",
            "scheduler/arm/cheap_token/total_reward_improving_experience",
            "scheduler/control/cadence_1/train_updates",
            "scheduler/control/policy_lag_0/train_updates",
            "scheduler/costs/rollout_dollar_seconds",
            "scheduler/costs/train_dollar_seconds",
            "scheduler/train_last_experience_count",
            "scheduler/train_last_reward_improving_experience",
            "scheduler/last_arm/cheap_token",
            "scheduler/last_target_train_batch_groups",
            "scheduler/last_max_policy_lag",
        ):
            self.assertAlmostEqual(restored_metrics[key], before_metrics[key])

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
            "scheduler/stale_last_policy_step",
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
