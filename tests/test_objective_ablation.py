import asyncio
from math import isfinite
import unittest

from calm_puffer_art.objective_ablation import (
    ART_ACCOUNTED_NORTH_STAR,
    ACCOUNTED_NORTH_STAR,
    BENCHMARK_ACCOUNTED_NORTH_STAR,
    NORTH_STAR,
    run_action_space_ablation,
    run_ablation,
    run_art_runtime_benchmark,
    run_art_bridge_ablation,
    run_closed_loop_ablation,
)


def assert_runtime_control_payoff_metrics(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
    prefixes: tuple[str, ...],
) -> None:
    for prefix in prefixes:
        test_case.assertTrue(
            isfinite(metrics[f"{prefix}/mean_objective_per_decision"])
        )
        test_case.assertTrue(
            isfinite(metrics[f"{prefix}/mean_objective_per_feedback_update"])
        )


def assert_any_runtime_control_payoff_metric(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
    prefixes: tuple[str, ...],
) -> None:
    observed_prefixes = [
        prefix
        for prefix in prefixes
        if f"{prefix}/mean_objective_per_decision" in metrics
    ]
    test_case.assertGreater(len(observed_prefixes), 0)
    assert_runtime_control_payoff_metrics(
        test_case,
        metrics,
        tuple(observed_prefixes),
    )


def assert_control_context_payoff_metrics(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
) -> None:
    test_case.assertGreater(metrics["scheduler/control_context/keys"], 0.0)
    test_case.assertGreater(metrics["scheduler/control_context/decisions"], 0.0)
    test_case.assertGreater(
        metrics["scheduler/control_context/feedback_updates"],
        0.0,
    )
    test_case.assertTrue(
        isfinite(metrics["scheduler/control_context/mean_objective_per_decision"])
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "scheduler/control_context/mean_objective_per_feedback_update"
            ]
        )
    )


def assert_action_space_payoff_means(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
) -> None:
    test_case.assertTrue(
        isfinite(
            metrics[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_decision"
            ]
        )
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_post_decision_observation"
            ]
        )
    )


def assert_train_selection_payoff_metrics(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
) -> None:
    test_case.assertGreater(metrics["scheduler/train_selection/keys"], 0.0)
    test_case.assertGreater(metrics["scheduler/train_selection/decisions"], 0.0)
    test_case.assertGreater(
        metrics["scheduler/train_selection/feedback_updates"],
        0.0,
    )
    test_case.assertGreater(
        metrics["scheduler/train_selection/positive_objective_keys"],
        0.0,
    )
    test_case.assertGreater(
        metrics["scheduler/train_selection/total_objective"],
        0.0,
    )
    test_case.assertGreater(
        metrics["scheduler/train_selection/mean_objective_per_decision"],
        0.0,
    )
    test_case.assertGreater(
        metrics["scheduler/train_selection/mean_objective_per_feedback_update"],
        0.0,
    )
    test_case.assertTrue(
        isfinite(metrics["scheduler/last_train_batch_train_selection_score"])
    )


def assert_continuation_payoff_metrics(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
) -> None:
    test_case.assertGreater(metrics["scheduler/continuation/keys"], 0.0)
    test_case.assertGreater(metrics["scheduler/continuation/decisions"], 0.0)
    test_case.assertGreater(
        metrics["scheduler/continuation/feedback_updates"],
        0.0,
    )
    test_case.assertGreater(
        metrics["scheduler/continuation/positive_objective_keys"],
        0.0,
    )
    test_case.assertTrue(
        isfinite(metrics["scheduler/continuation/mean_objective_per_decision"])
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "scheduler/continuation/mean_objective_per_feedback_update"
            ]
        )
    )


def assert_promotion_decision_payoff_metrics(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
) -> None:
    test_case.assertGreater(metrics["promotion/decision/keys"], 0.0)
    test_case.assertGreater(metrics["promotion/decision/decisions"], 0.0)
    test_case.assertGreater(metrics["promotion/decision/promoted"], 0.0)
    test_case.assertGreater(
        metrics["promotion/decision/positive_reward_improving_keys"],
        0.0,
    )
    test_case.assertGreater(
        metrics["promotion/decision/total_candidate_improvement"],
        0.0,
    )
    test_case.assertGreater(
        metrics["promotion/decision/total_published_policy_improvement"],
        0.0,
    )
    test_case.assertGreater(
        metrics["promotion/decision/realized_reward_improving_experience"],
        0.0,
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "promotion/decision/"
                "mean_realized_reward_improving_experience_per_decision"
            ]
        )
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "promotion/decision/"
                "realized_reward_improving_experience_per_dollar_second"
            ]
        )
    )


def assert_art_publication_decision_payoff_metrics(
    test_case: unittest.TestCase,
    metrics: dict[str, float],
) -> None:
    test_case.assertGreater(metrics["art_backend/publication/decision/keys"], 0.0)
    test_case.assertGreater(
        metrics["art_backend/publication/decision/decisions"],
        0.0,
    )
    test_case.assertGreater(
        metrics["art_backend/publication/decision/published"],
        0.0,
    )
    test_case.assertGreater(
        metrics[
            "art_backend/publication/decision/positive_reward_improving_keys"
        ],
        0.0,
    )
    test_case.assertGreater(
        metrics[
            "art_backend/publication/decision/"
            "total_candidate_improvement"
        ],
        0.0,
    )
    test_case.assertGreater(
        metrics[
            "art_backend/publication/decision/"
            "total_published_policy_improvement"
        ],
        0.0,
    )
    test_case.assertGreater(
        metrics[
            "art_backend/publication/decision/"
            "realized_reward_improving_experience"
        ],
        0.0,
    )
    test_case.assertGreater(
        metrics["art_backend/publication/decision/total_dollar_seconds"],
        0.0,
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "art_backend/publication/decision/"
                "mean_realized_reward_improving_experience_per_decision"
            ]
        )
    )
    test_case.assertTrue(
        isfinite(
            metrics[
                "art_backend/publication/decision/"
                "realized_reward_improving_experience_per_dollar_second"
            ]
        )
    )


class ObjectiveAblationTests(unittest.TestCase):
    def test_objective_scheduler_beats_static_baseline_on_north_star(self):
        result = asyncio.run(run_ablation())

        static = result["static"]
        objective = result["objective"]
        lift = result["lift"]

        self.assertGreater(static[NORTH_STAR], 0.0)
        self.assertGreater(static[ACCOUNTED_NORTH_STAR], 0.0)
        self.assertGreater(objective[NORTH_STAR], 0.0)
        self.assertGreater(objective[ACCOUNTED_NORTH_STAR], static[ACCOUNTED_NORTH_STAR])
        self.assertGreater(lift["accounted_north_star_absolute"], 0.0)
        self.assertGreater(lift["accounted_north_star_ratio"], 1.0)
        self.assertGreater(
            objective["actions/semantic_bandwidth_tokens_per_decision"],
            static["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertGreater(
            objective["scheduler/arm/bandwidth_chunk_chunk_size_2/pulls"],
            objective["scheduler/arm/bandwidth_token/pulls"],
        )
        self.assertEqual(
            objective[
                "scheduler/arm/bandwidth_chunk_chunk_size_2/mean_rollout_dollar_seconds"
            ],
            1.5,
        )
        self.assertEqual(
            objective["scheduler/arm/bandwidth_token/mean_rollout_dollar_seconds"],
            1.0,
        )
        self.assertGreater(
            objective[
                "scheduler/arm/bandwidth_chunk_chunk_size_2/total_improvement_per_dollar_second"
            ],
            objective[
                "scheduler/arm/bandwidth_token/total_improvement_per_dollar_second"
            ],
        )
        self.assertGreater(
            objective["scheduler/control/cadence_1/train_updates"],
            0.0,
        )
        self.assertGreater(
            objective.get("scheduler/control/policy_lag_1/train_updates", 0.0)
            + objective.get("scheduler/control/policy_lag_2/train_updates", 0.0),
            0.0,
        )
        self.assertGreater(
            objective["scheduler/control/actor_count_1/rollout_updates"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/control/actor_count_2/rollout_updates"],
            0.0,
        )
        self.assertTrue(
            isfinite(objective["scheduler/control/actor_count_1/score"])
        )
        self.assertTrue(
            isfinite(objective["scheduler/control/actor_count_2/score"])
        )
        assert_runtime_control_payoff_metrics(
            self,
            objective,
            (
                "scheduler/control/cadence_1",
                "scheduler/control/policy_lag_1",
                "scheduler/control/policy_lag_2",
                "scheduler/control/admission_delay_ms_0",
            ),
        )
        assert_any_runtime_control_payoff_metric(
            self,
            objective,
            (
                "scheduler/control/actor_count_1",
                "scheduler/control/actor_count_2",
            ),
        )
        assert_train_selection_payoff_metrics(self, objective)
        assert_continuation_payoff_metrics(self, objective)
        assert_promotion_decision_payoff_metrics(self, objective)

    def test_adaptive_action_space_ablation_beats_fixed_bandwidth(self):
        result = asyncio.run(run_action_space_ablation())

        fixed = result["fixed"]
        adaptive = result["adaptive"]
        lift = result["lift"]

        self.assertGreater(
            adaptive[ACCOUNTED_NORTH_STAR],
            fixed[ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(lift["accounted_north_star_absolute"], 0.0)
        self.assertGreater(lift["accounted_north_star_ratio"], 1.0)
        self.assertGreater(
            adaptive["actions/semantic_bandwidth_tokens_per_decision"],
            fixed["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertGreaterEqual(adaptive["action_space/promotions"], 1.0)
        self.assertGreater(
            adaptive["scheduler/arm/semantic_chunk_chunk_size_4/pulls"],
            0.0,
        )
        self.assertGreater(
            adaptive[
                "scheduler/arm/semantic_chunk_chunk_size_4/total_improvement_per_dollar_second"
            ],
            adaptive[
                "scheduler/arm/semantic_chunk_chunk_size_2/total_improvement_per_dollar_second"
            ],
        )
        assert_action_space_payoff_means(self, adaptive)
        assert_control_context_payoff_metrics(self, adaptive)
        assert_promotion_decision_payoff_metrics(self, adaptive)

    def test_closed_loop_ablation_accounts_joint_scheduler_payoff(self):
        result = asyncio.run(run_closed_loop_ablation())

        static = result["static"]
        objective = result["objective"]
        lift = result["lift"]

        self.assertGreater(
            objective[ACCOUNTED_NORTH_STAR],
            static[ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(lift["accounted_north_star_absolute"], 0.0)
        self.assertGreater(lift["accounted_north_star_ratio"], 1.0)
        self.assertGreater(
            objective["actions/semantic_bandwidth_tokens_per_decision"],
            static["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertGreaterEqual(objective["action_space/promotions"], 1.0)
        self.assertEqual(objective["action_space/codec/chunk_chunk_size_4/active"], 1.0)
        self.assertEqual(objective["action_space/max_chunk_size"], 4.0)
        self.assertGreater(
            objective["action_space/decision/post_decision_observations"],
            0.0,
        )
        self.assertGreater(
            objective["action_space/decision/realized_objective_payoff"],
            0.0,
        )
        self.assertGreater(
            objective[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_decision"
            ],
            0.0,
        )
        self.assertGreater(
            objective[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_post_decision_observation"
            ],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/arm/closed_loop_chunk_chunk_size_4/pulls"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/control/cadence_1/train_updates"],
            0.0,
        )
        self.assertGreater(
            objective.get("scheduler/control/policy_lag_1/train_updates", 0.0)
            + objective.get("scheduler/control/policy_lag_2/train_updates", 0.0),
            0.0,
        )
        self.assertGreater(
            objective["scheduler/control/actor_count_2/rollout_updates"],
            0.0,
        )
        assert_runtime_control_payoff_metrics(
            self,
            objective,
            (
                "scheduler/control/cadence_1",
                "scheduler/control/policy_lag_1",
                "scheduler/control/policy_lag_2",
                "scheduler/control/admission_delay_ms_0",
            ),
        )
        assert_any_runtime_control_payoff_metric(
            self,
            objective,
            (
                "scheduler/control/actor_count_1",
                "scheduler/control/actor_count_2",
            ),
        )
        assert_control_context_payoff_metrics(self, objective)
        assert_train_selection_payoff_metrics(self, objective)
        assert_continuation_payoff_metrics(self, objective)
        self.assertGreater(objective["scheduler/joint_action/tuples"], 0.0)
        self.assertGreater(
            objective["scheduler/joint_action/feedback_updates"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/joint_action/positive_objective_tuples"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/joint_action/mean_objective_per_decision"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/joint_action/mean_objective_per_feedback_update"],
            0.0,
        )
        self.assertTrue(
            isfinite(objective["scheduler/last_train_batch_joint_action_score"])
        )
        assert_promotion_decision_payoff_metrics(self, objective)

    def test_art_bridge_ablation_accounts_external_producer_payoff(self):
        result = asyncio.run(run_art_bridge_ablation())

        static = result["static"]
        objective = result["objective"]
        lift = result["lift"]

        self.assertGreater(
            objective[ART_ACCOUNTED_NORTH_STAR],
            static[ART_ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(lift["accounted_north_star_absolute"], 0.0)
        self.assertGreater(lift["accounted_north_star_ratio"], 1.0)
        self.assertGreater(
            objective["actions/semantic_bandwidth_tokens_per_decision"],
            static["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertGreaterEqual(objective["action_space/promotions"], 1.0)
        self.assertEqual(objective["action_space/codec/chunk_chunk_size_4/active"], 1.0)
        self.assertEqual(objective["action_space/max_chunk_size"], 4.0)
        self.assertGreater(
            objective["action_space/decision/post_decision_observations"],
            0.0,
        )
        self.assertGreater(
            objective["action_space/decision/realized_objective_payoff"],
            0.0,
        )
        self.assertGreater(
            objective[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_decision"
            ],
            0.0,
        )
        self.assertGreater(
            objective[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_post_decision_observation"
            ],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/arm/art_bridge_chunk_chunk_size_4/pulls"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/control/cadence_1/train_updates"],
            0.0,
        )
        self.assertGreater(
            objective.get("scheduler/control/policy_lag_1/train_updates", 0.0)
            + objective.get("scheduler/control/policy_lag_2/train_updates", 0.0),
            0.0,
        )
        self.assertGreater(
            objective["scheduler/control/actor_count_2/rollout_updates"],
            0.0,
        )
        assert_runtime_control_payoff_metrics(
            self,
            objective,
            (
                "scheduler/control/cadence_1",
                "scheduler/control/policy_lag_1",
                "scheduler/control/policy_lag_2",
                "scheduler/control/admission_delay_ms_0",
            ),
        )
        assert_any_runtime_control_payoff_metric(
            self,
            objective,
            (
                "scheduler/control/actor_count_1",
                "scheduler/control/actor_count_2",
            ),
        )
        assert_control_context_payoff_metrics(self, objective)
        assert_train_selection_payoff_metrics(self, objective)
        assert_continuation_payoff_metrics(self, objective)
        self.assertGreater(objective["scheduler/joint_action/tuples"], 0.0)
        self.assertGreater(
            objective["scheduler/joint_action/feedback_updates"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/joint_action/positive_objective_tuples"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/joint_action/mean_objective_per_decision"],
            0.0,
        )
        self.assertGreater(
            objective["scheduler/joint_action/mean_objective_per_feedback_update"],
            0.0,
        )
        self.assertGreater(objective["art_backend/submitted_groups"], 0.0)
        self.assertGreater(objective["art_backend/completed_batches"], 0.0)
        assert_art_publication_decision_payoff_metrics(self, objective)

    def test_art_runtime_benchmark_compares_stock_async_and_semantic_modes(self):
        result = asyncio.run(run_art_runtime_benchmark())

        stock = result["stock_art"]
        async_runtime = result["art_async"]
        async_semantic = result["art_async_semantic"]
        lift = result["lift"]

        self.assertGreater(stock[BENCHMARK_ACCOUNTED_NORTH_STAR], 0.0)
        self.assertGreater(async_runtime[BENCHMARK_ACCOUNTED_NORTH_STAR], 0.0)
        self.assertGreater(
            async_semantic[BENCHMARK_ACCOUNTED_NORTH_STAR],
            async_runtime[BENCHMARK_ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(
            async_semantic[BENCHMARK_ACCOUNTED_NORTH_STAR],
            stock[BENCHMARK_ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(
            async_semantic["actions/semantic_bandwidth_tokens_per_decision"],
            async_runtime["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertEqual(
            stock["actions/semantic_bandwidth_tokens_per_decision"],
            1.0,
        )
        self.assertGreater(async_runtime["benchmark/submitted_groups"], 0.0)
        self.assertGreater(async_semantic["benchmark/submitted_groups"], 0.0)
        self.assertGreater(async_runtime["scheduler/joint_action/decisions"], 0.0)
        self.assertGreaterEqual(async_semantic["action_space/promotions"], 1.0)
        assert_control_context_payoff_metrics(self, async_semantic)
        self.assertGreater(
            lift["async_semantic_vs_async_accounted_north_star_ratio"],
            1.0,
        )
        self.assertGreater(
            lift["async_semantic_vs_stock_accounted_north_star_ratio"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
