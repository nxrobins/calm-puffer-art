import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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
        self.assertEqual(payload["minimum_inference_requests"], 1788)
        self.assertEqual(payload["maximum_inference_requests"], 1788)
        self.assertEqual(
            payload["checkpoint_evaluations_per_trained_condition"],
            2,
        )
        self.assertEqual(payload["reward_targets"], [0.2, 0.225, 0.25])
        self.assertEqual(payload["telemetry"]["schema_version"], 1)
        self.assertTrue(payload["telemetry"]["missing_price_is_null_not_zero"])

    def test_preflight_can_budget_a_targeted_recovery_condition(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--preflight",
                "--json",
                "--seeds",
                "202",
                "--conditions",
                "direct_art",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["conditions"], ["direct_art"])
        self.assertEqual(payload["training_updates"], 3)
        self.assertEqual(payload["minimum_inference_requests"], 248)
        self.assertEqual(payload["maximum_inference_requests"], 248)

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

    def test_retrying_backend_emits_training_lifecycle(self):
        namespace = load_namespace()
        observed = []

        class Delegate:
            async def _get_step(self, model):
                return 0

            async def train(self, model, groups, **kwargs):
                return SimpleNamespace(
                    step=1,
                    metrics={"grad_norm": 2.0},
                    artifact_name="entity/project/model:step1",
                )

        backend = namespace["RetryingServerlessBackend"](
            art=SimpleNamespace(),
            delegate=Delegate(),
            max_attempts=1,
            timeout_seconds=1.0,
            attempt_observer=observed.append,
        )

        result = namespace["asyncio"].run(
            backend.train(SimpleNamespace(), [SimpleNamespace()])
        )

        self.assertEqual(result.step, 1)
        self.assertEqual(
            [event["status"] for event in observed],
            ["started", "completed"],
        )
        self.assertEqual(observed[-1]["artifact_name"], "entity/project/model:step1")
        self.assertEqual(observed[-1]["metrics"]["grad_norm"], 2.0)

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

    def test_condition_observers_record_live_evidence(self):
        namespace = load_namespace()
        TelemetryLedger = namespace["TelemetryLedger"]
        ChecksumTask = namespace["ChecksumTask"]
        CompletionRecord = namespace["CompletionRecord"]
        configure = namespace["_condition_args_with_telemetry"]
        task = ChecksumTask("heldout-easy-01", 1, 2, 3, 4, 5, 6, 97)

        with tempfile.TemporaryDirectory() as directory:
            ledger = TelemetryLedger(
                run_id="observer-test",
                path=Path(directory) / "telemetry.jsonl",
            )
            observed_args = configure(
                namespace["argparse"].Namespace(request_seed=1),
                telemetry=ledger,
                condition="async_scheduler",
                seed=1,
                task_strata={task.id: "easy"},
            )
            for split, reward, exact, parsed_answer in (
                ("heldout_before", 0.0, False, None),
                ("heldout_after", 1.0, True, task.answer),
            ):
                observed_args.completion_observer(
                    CompletionRecord(
                        task=task,
                        split=split,
                        content="FINAL=1",
                        parsed_answer=parsed_answer,
                        reward=reward,
                        exact=exact,
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        elapsed_s=0.1,
                        estimated_api_usd=0.0,
                        attempts=1,
                        choice=None,
                    )
                )
            observed_args.training_attempt_observer(
                {
                    "event": "started",
                    "attempt": 1,
                    "status": "started",
                    "initial_step": 0,
                }
            )
            observed_args.training_attempt_observer(
                {
                    "attempt": 1,
                    "status": "completed",
                    "wall_s": 1.0,
                    "initial_step": 0,
                    "observed_step": 1,
                    "metrics": {"grad_norm": 2.0},
                }
            )
            observed_args.scheduler_decision_observer(
                {
                    "train_step": 0,
                    "group_index": 0,
                    "selected_stratum": "easy",
                    "task_id": task.id,
                    "metadata": {
                        "scheduler/decision/estimated_rollout_dollar_seconds": 0.2
                    },
                }
            )

            summary = ledger.summary(expected_inference_requests=2)

        condition = summary["conditions"]["async_scheduler"]
        self.assertTrue(summary["healthy"])
        self.assertEqual(summary["coverage"]["request_coverage"], 1.0)
        self.assertEqual(condition["training"]["successful_updates"], 1)
        self.assertEqual(condition["scheduler"]["allocation"], {"easy": 1})
        self.assertEqual(
            condition["performance"]["heldout_mean_reward_delta"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
