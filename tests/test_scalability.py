import math
import unittest

from calm_puffer_art import run_scheduler_scalability_profile


class SchedulerScalabilityProfileTests(unittest.TestCase):
    def test_profile_reports_joint_action_and_checkpoint_growth(self):
        profile = run_scheduler_scalability_profile(
            scenario_count=2,
            codec_count=2,
            cadence_values=(1, 2),
            lag_values=(1, 2),
            actor_counts=(1, 2),
            admission_delay_ms_values=(0, 10),
            action_space_count=2,
            selector_trials=4,
        )

        expected_joint_keys = 2 * 2 * 2 * 2 * 2 * 2 * 2

        self.assertEqual(profile["scalability/arms"], 4.0)
        self.assertEqual(
            profile["scalability/expected_joint_action_keys"],
            float(expected_joint_keys),
        )
        self.assertEqual(
            profile["scalability/joint_action_keys"],
            float(expected_joint_keys),
        )
        self.assertEqual(profile["scalability/runtime_control_contexts"], 2.0)
        self.assertEqual(profile["scalability/runtime_control_keys"], 16.0)
        self.assertEqual(profile["scalability/global_runtime_control_keys"], 8.0)
        self.assertEqual(
            profile["scalability/observations"],
            float(expected_joint_keys),
        )
        self.assertGreater(profile["scalability/metrics_count"], 0.0)
        self.assertGreater(profile["scalability/state_json_bytes"], 0.0)
        self.assertGreater(
            profile["scalability/state_bytes_per_joint_action_key"],
            0.0,
        )
        self.assertGreater(
            profile["scalability/select_decisions_per_second"],
            0.0,
        )
        for key, value in profile.items():
            self.assertTrue(math.isfinite(value), key)


if __name__ == "__main__":
    unittest.main()
