import json
import runpy
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "controlled_art_ablation.py"


def load_namespace():
    examples = str(ROOT / "examples")
    sys.path.insert(0, examples)
    try:
        return runpy.run_path(str(SCRIPT))
    finally:
        sys.path.remove(examples)


class ControlledArtAblationTests(unittest.TestCase):
    def test_preflight_reports_fixed_budget_and_calm_exclusion(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--preflight", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(
            payload["conditions"],
            ["base", "direct_art", "async_scheduler"],
        )
        self.assertIn("calm", payload["excluded_conditions"])
        self.assertEqual(payload["seeds"], [101, 202, 303])
        self.assertEqual(payload["train_tasks"], 12)
        self.assertEqual(payload["heldout_tasks"], 50)
        self.assertEqual(payload["training_updates"], 18)
        self.assertEqual(payload["minimum_inference_requests"], 1188)
        self.assertEqual(payload["maximum_inference_requests"], 1188)

    def test_manifest_is_deterministic_disjoint_and_balanced(self):
        namespace = load_namespace()
        build_manifest = namespace["build_manifest"]
        fingerprint = namespace["manifest_fingerprint"]
        train, heldout = build_manifest(seed=7, train_tasks=12, heldout_tasks=50)
        repeated_train, repeated_heldout = build_manifest(
            seed=7,
            train_tasks=12,
            heldout_tasks=50,
        )

        self.assertEqual(
            fingerprint(train, heldout),
            fingerprint(repeated_train, repeated_heldout),
        )
        self.assertEqual(len(train), 12)
        self.assertEqual(len(heldout), 50)
        self.assertTrue(
            {item.task.id for item in train}.isdisjoint(
                item.task.id for item in heldout
            )
        )
        self.assertEqual(
            {
                stratum: sum(item.stratum == stratum for item in train)
                for stratum in namespace["STRATA"]
            },
            {"easy": 3, "medium": 3, "hard": 3, "challenge": 3},
        )

    def test_sample_summary_uses_small_sample_t_interval(self):
        namespace = load_namespace()
        summary = namespace["summarize_samples"]([0.0, 0.1, 0.2])

        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 0.1)
        self.assertAlmostEqual(summary["stddev"], 0.1)
        self.assertLess(summary["ci95_low"], 0.0)
        self.assertGreater(summary["ci95_high"], 0.2)

    def test_rollout_seeds_are_reproducible_and_distinct(self):
        namespace = load_namespace()
        rollout_seed = namespace["_rollout_seed"]
        kwargs = {
            "experiment_seed": 101,
            "task_id": "train-easy-01",
            "policy_step": 0,
            "rollout_namespace": "direct-0-0",
        }

        seeds = [rollout_seed(**kwargs, rollout_index=index) for index in range(4)]

        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(
            seeds,
            [rollout_seed(**kwargs, rollout_index=index) for index in range(4)],
        )

    def test_managed_training_timeout_is_retryable(self):
        namespace = load_namespace()

        self.assertTrue(namespace["_retryable_train_error"](TimeoutError()))

    def test_completion_diagnostics_separate_format_from_reward(self):
        namespace = load_namespace()
        diagnostics = namespace["_completion_diagnostics"](
            [
                {"parsed_answer": 7, "reward": 0.5},
                {"parsed_answer": None, "reward": 0.0},
                {"parsed_answer": 9, "reward": 1.0},
            ]
        )

        self.assertEqual(diagnostics["parsed_count"], 2)
        self.assertAlmostEqual(diagnostics["parse_rate"], 2 / 3)
        self.assertAlmostEqual(diagnostics["mean_reward_given_parsed"], 0.75)


if __name__ == "__main__":
    unittest.main()
