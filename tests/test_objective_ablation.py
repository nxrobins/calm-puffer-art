import asyncio
import unittest

from calm_puffer_art.objective_ablation import (
    ACCOUNTED_NORTH_STAR,
    NORTH_STAR,
    run_action_space_ablation,
    run_ablation,
)


class ObjectiveAblationTests(unittest.TestCase):
    def test_objective_scheduler_beats_static_baseline_on_north_star(self):
        result = asyncio.run(run_ablation())

        static = result["static"]
        objective = result["objective"]
        lift = result["lift"]

        self.assertEqual(static[NORTH_STAR], 0.0)
        self.assertEqual(static[ACCOUNTED_NORTH_STAR], 0.0)
        self.assertGreater(objective[NORTH_STAR], static[NORTH_STAR])
        self.assertGreater(objective[ACCOUNTED_NORTH_STAR], static[ACCOUNTED_NORTH_STAR])
        self.assertGreater(lift["north_star_absolute"], 0.0)
        self.assertIsNone(lift["north_star_ratio"])
        self.assertGreater(objective["reward/delta"], static["reward/delta"])
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
            objective["scheduler/control/policy_lag_2/train_updates"],
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
        self.assertGreater(
            objective["scheduler/control/actor_count_1/score"],
            objective["scheduler/control/actor_count_2/score"],
        )

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
        self.assertGreater(adaptive["reward/delta"], fixed["reward/delta"])
        self.assertGreater(
            adaptive["actions/semantic_bandwidth_tokens_per_decision"],
            fixed["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertEqual(
            adaptive["action_space/codec/chunk_chunk_size_4/active"],
            1.0,
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


if __name__ == "__main__":
    unittest.main()
