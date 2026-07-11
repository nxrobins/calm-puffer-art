import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from calm_puffer_art.telemetry import (
    PricingConfig,
    TelemetryLedger,
    load_telemetry_events,
    minimum_cost_to_target,
    pareto_frontier,
    summarize_telemetry,
)

ROOT = Path(__file__).resolve().parents[1]


class TelemetryTests(unittest.TestCase):
    def test_pricing_distinguishes_missing_zero_and_estimated_cost(self):
        missing = PricingConfig()
        free = PricingConfig(
            input_usd_per_million_tokens=0.0,
            output_usd_per_million_tokens=0.0,
        )
        priced = PricingConfig(
            input_usd_per_million_tokens=2.0,
            output_usd_per_million_tokens=8.0,
        )

        self.assertEqual(
            missing.inference_cost(prompt_tokens=100, completion_tokens=50),
            (None, "unavailable"),
        )
        self.assertEqual(
            free.inference_cost(prompt_tokens=100, completion_tokens=50),
            (0.0, "estimated_from_token_rates"),
        )
        self.assertEqual(
            priced.inference_cost(prompt_tokens=100, completion_tokens=50),
            (0.0006, "estimated_from_token_rates"),
        )

    def test_ledger_persists_versioned_events_and_resumes_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            first = TelemetryLedger(run_id="run-1", path=path)
            first.emit("run_started", attributes={"model": "test"})
            second = TelemetryLedger(run_id="run-1", path=path)
            second.emit("run_finished")
            events = load_telemetry_events(path)

        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual(
            [event["event"] for event in events],
            ["run_started", "run_finished"],
        )
        self.assertTrue(all(event["schema_version"] == 1 for event in events))

    def test_summary_preserves_performance_cost_and_pricing_coverage(self):
        ledger = TelemetryLedger(
            run_id="summary",
            pricing=PricingConfig(
                input_usd_per_million_tokens=1.0,
                output_usd_per_million_tokens=2.0,
                trainer_usd_per_hour=3.6,
            ),
        )
        for phase, reward, exact, parsed in (
            ("heldout_before", 0.1, False, False),
            ("heldout_after", 0.4, True, True),
        ):
            ledger.record_inference(
                condition="direct_art",
                seed=1,
                phase=phase,
                task_id=f"{phase}-1",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                latency_s=0.25,
                attempts=1,
                reward=reward,
                exact=exact,
                parsed=parsed,
            )
        ledger.record_training_attempt(
            condition="direct_art",
            seed=1,
            attempt=1,
            status="completed",
            duration_s=10.0,
            initial_step=0,
            observed_step=1,
        )
        ledger.record_external_cost(
            condition="direct_art",
            seed=1,
            phase="evaluation",
            category="evaluator",
            amount_usd=0.005,
            provenance="measured",
            proxy_value=2.0,
            proxy_unit="verifier_seconds",
        )
        ledger.record_training_attempt(
            condition="direct_art",
            seed=1,
            attempt=2,
            status="failed",
            duration_s=5.0,
            initial_step=1,
            observed_step=1,
        )
        ledger.record_condition_finished(
            condition="direct_art",
            seed=1,
            status="completed",
            wall_s=12.0,
        )

        summary = ledger.summary(expected_inference_requests=2)
        condition = summary["conditions"]["direct_art"]

        self.assertEqual(summary["coverage"]["request_coverage"], 1.0)
        self.assertEqual(summary["coverage"]["inference_pricing_coverage"], 1.0)
        self.assertEqual(summary["coverage"]["trainer_pricing_coverage"], 1.0)
        self.assertIn("1", condition["by_seed"])
        self.assertAlmostEqual(
            condition["by_seed"]["1"]["performance"][
                "heldout_mean_reward_delta"
            ],
            0.3,
        )
        self.assertAlmostEqual(
            condition["performance"]["heldout_mean_reward_delta"],
            0.3,
        )
        self.assertEqual(
            condition["performance"]["heldout_exact_accuracy_delta"],
            1.0,
        )
        self.assertEqual(condition["cost"]["total_tokens"], 300)
        self.assertAlmostEqual(condition["cost"]["inference_usd"], 0.0004)
        self.assertEqual(condition["training"]["failed_attempts"], 1)
        self.assertAlmostEqual(condition["cost"]["trainer_usd"], 0.015)
        self.assertAlmostEqual(condition["cost"]["external_usd"], 0.005)
        self.assertAlmostEqual(condition["cost"]["total_usd"], 0.0204)

    def test_summary_flags_proxy_cost_format_gain_and_concentration(self):
        ledger = TelemetryLedger(run_id="alerts")
        for phase, parsed in (("heldout_before", False), ("heldout_after", True)):
            ledger.record_inference(
                condition="async_scheduler",
                seed=1,
                phase=phase,
                task_id=phase,
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                latency_s=0.1,
                attempts=1,
                reward=0.2,
                exact=False,
                parsed=parsed,
            )
        for index in range(4):
            ledger.record_scheduler_decision(
                condition="async_scheduler",
                seed=1,
                train_step=0,
                group_index=index,
                selected_stratum="hard" if index < 3 else "easy",
                task_id=f"task-{index}",
            )
        ledger.record_external_cost(
            condition="async_scheduler",
            seed=1,
            phase="train",
            category="tool",
            amount_usd=None,
            provenance="unavailable",
            proxy_value=1.0,
            proxy_unit="tool_calls",
        )

        summary = ledger.summary(expected_inference_requests=2)
        codes = {alert["code"] for alert in summary["alerts"]}

        self.assertIn("inference_pricing_incomplete", codes)
        self.assertIn("format_gain_without_exact_gain", codes)
        self.assertIn("scheduler_allocation_concentration", codes)
        self.assertIn("external_cost_pricing_incomplete", codes)
        self.assertIn("cost_to_target_unobservable", codes)
        self.assertEqual(summary["cost_performance"]["basis"], "token_proxy_millions")

    def test_summary_can_apply_rates_after_the_run(self):
        ledger = TelemetryLedger(run_id="reprice")
        ledger.record_inference(
            condition="base",
            seed=1,
            phase="heldout_before",
            task_id="task",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_s=0.1,
            attempts=1,
            reward=0.0,
            exact=False,
            parsed=False,
        )
        self.assertIsNone(ledger.events[0]["metrics"]["api_usd"])

        summary = summarize_telemetry(
            ledger.events,
            pricing=PricingConfig(
                input_usd_per_million_tokens=2.0,
                output_usd_per_million_tokens=8.0,
            ),
        )

        self.assertAlmostEqual(
            summary["conditions"]["base"]["cost"]["inference_usd"],
            0.0006,
        )
        self.assertIsNone(ledger.events[0]["metrics"]["api_usd"])

    def test_unfinished_lifecycles_are_errors(self):
        ledger = TelemetryLedger(run_id="unfinished")
        ledger.emit(
            "condition_started",
            dimensions={"condition": "direct_art", "seed": 1},
        )
        ledger.emit(
            "training_attempt_started",
            dimensions={
                "condition": "direct_art",
                "seed": 1,
                "attempt": 1,
                "initial_step": 0,
            },
        )

        active = ledger.summary(stale_after_s=600.0)
        summary = ledger.summary(stale_after_s=0.0)
        active_codes = {alert["code"] for alert in active["alerts"]}
        codes = {alert["code"] for alert in summary["alerts"]}

        self.assertTrue(active["healthy"])
        self.assertIn("active_condition", active_codes)
        self.assertIn("active_training_attempt", active_codes)
        self.assertFalse(summary["healthy"])
        self.assertIn("unfinished_condition", codes)
        self.assertIn("unfinished_training_attempt", codes)

    def test_finished_run_contract_has_complete_coverage(self):
        ledger = TelemetryLedger(run_id="contract")
        ledger.emit(
            "run_started",
            attributes={
                "minimum_inference_requests": 2,
                "maximum_inference_requests": 2,
                "expected_condition_runs": 1,
                "expected_training_updates": 0,
                "expected_scheduler_decisions": 1,
            },
        )
        ledger.emit(
            "condition_started",
            dimensions={"condition": "base", "seed": 1},
        )
        for phase in ("heldout_before", "heldout_after"):
            ledger.record_inference(
                condition="base",
                seed=1,
                phase=phase,
                task_id=phase,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                latency_s=0.1,
                attempts=1,
                reward=0.0,
                exact=False,
                parsed=False,
            )
        ledger.record_scheduler_decision(
            condition="base",
            seed=1,
            train_step=0,
            group_index=0,
            selected_stratum="easy",
            task_id="task",
        )
        ledger.record_condition_finished(
            condition="base",
            seed=1,
            status="completed",
            wall_s=1.0,
        )
        ledger.emit("run_finished", metrics={"conditions_ok": True})

        summary = ledger.summary()

        self.assertTrue(summary["healthy"])
        self.assertEqual(summary["coverage"]["request_coverage"], 1.0)
        self.assertEqual(summary["coverage"]["condition_run_coverage"], 1.0)
        self.assertEqual(summary["coverage"]["training_update_coverage"], 1.0)
        self.assertEqual(summary["coverage"]["scheduler_decision_coverage"], 1.0)

    def test_stale_abandoned_run_fails_contract_coverage(self):
        ledger = TelemetryLedger(run_id="abandoned")
        ledger.emit(
            "run_started",
            attributes={
                "minimum_inference_requests": 2,
                "maximum_inference_requests": 2,
                "expected_condition_runs": 1,
                "expected_training_updates": 1,
                "expected_scheduler_decisions": 1,
            },
        )

        summary = ledger.summary(stale_after_s=0.0)
        codes = {alert["code"] for alert in summary["alerts"]}

        self.assertFalse(summary["healthy"])
        self.assertIn("inference_request_coverage_mismatch", codes)
        self.assertIn("condition_run_coverage_mismatch", codes)
        self.assertIn("training_update_coverage_mismatch", codes)
        self.assertIn("scheduler_decision_coverage_mismatch", codes)

    def test_pareto_frontier_and_cost_to_target(self):
        points = [
            {"condition": "base", "cost": 1.0, "performance": 0.0},
            {"condition": "direct", "cost": 3.0, "performance": 0.5},
            {"condition": "scheduler", "cost": 2.5, "performance": 0.6},
            {"condition": "expensive", "cost": 4.0, "performance": 0.8},
        ]

        frontier = pareto_frontier(points)

        self.assertEqual(
            [point["condition"] for point in frontier],
            ["base", "scheduler", "expensive"],
        )
        self.assertEqual(minimum_cost_to_target(points, target=0.55), 2.5)
        self.assertIsNone(minimum_cost_to_target(points, target=0.9))

    def test_summary_builds_learning_curve_and_cost_to_target(self):
        ledger = TelemetryLedger(run_id="learning-curve")
        ledger.emit(
            "condition_started",
            dimensions={"condition": "direct_art", "seed": 1},
        )

        def inference(phase, task_id, tokens, reward):
            ledger.record_inference(
                condition="direct_art",
                seed=1,
                phase=phase,
                task_id=task_id,
                prompt_tokens=tokens // 2,
                completion_tokens=tokens - tokens // 2,
                total_tokens=tokens,
                latency_s=0.1,
                attempts=1,
                reward=reward,
                exact=False,
                parsed=True,
            )

        inference("heldout_before", "before", 100, 0.1)
        inference("train", "train-1", 30, 0.2)
        ledger.record_training_attempt(
            condition="direct_art",
            seed=1,
            attempt=1,
            status="completed",
            duration_s=1.0,
            initial_step=0,
            observed_step=1,
        )
        inference("heldout_checkpoint_step_1", "checkpoint", 100, 0.23)
        inference("train", "train-2", 40, 0.3)
        ledger.record_training_attempt(
            condition="direct_art",
            seed=1,
            attempt=1,
            status="completed",
            duration_s=1.0,
            initial_step=1,
            observed_step=2,
        )
        inference("heldout_after", "after", 100, 0.3)
        ledger.record_condition_finished(
            condition="direct_art",
            seed=1,
            status="completed",
            wall_s=3.0,
        )

        summary = ledger.summary(performance_targets=[0.2, 0.28, 0.4])
        condition = summary["conditions"]["direct_art"]
        curve = condition["learning_curve"]
        targets = summary["cost_performance"]["cost_to_target"]

        self.assertEqual(len(curve), 3)
        self.assertEqual(curve[0]["mean_learning_total_tokens"], 0.0)
        self.assertEqual(curve[1]["mean_learning_total_tokens"], 30.0)
        self.assertEqual(curve[2]["mean_learning_total_tokens"], 70.0)
        self.assertEqual(
            summary["cost_performance"]["cost_to_target_basis"],
            "mean_learning_token_proxy_millions",
        )
        self.assertAlmostEqual(targets[0]["minimum_learning_cost"], 0.00003)
        self.assertAlmostEqual(targets[1]["minimum_learning_cost"], 0.00007)
        self.assertFalse(targets[2]["reached"])
        self.assertNotIn(
            "cost_to_target_unobservable",
            {alert["code"] for alert in summary["alerts"]},
        )

    def test_non_finite_metrics_are_rejected(self):
        ledger = TelemetryLedger(run_id="finite")

        with self.assertRaises(ValueError):
            ledger.emit("bad", metrics={"loss": float("nan")})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"schema_version": 99}) + "\n")
            with self.assertRaises(ValueError):
                load_telemetry_events(path)

    def test_summary_rejects_broken_event_sequence(self):
        ledger = TelemetryLedger(run_id="sequence")
        ledger.emit("run_started")
        ledger.emit("run_finished")
        events = [dict(event) for event in ledger.events]
        events[1]["sequence"] = 3

        summary = summarize_telemetry(events)

        self.assertFalse(summary["healthy"])
        self.assertIn(
            "telemetry_sequence_broken",
            {alert["code"] for alert in summary["alerts"]},
        )

    def test_report_cli_can_fail_on_monitoring_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            ledger = TelemetryLedger(run_id="cli", path=path)
            ledger.emit(
                "condition_started",
                dimensions={"condition": "direct_art", "seed": 1},
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/telemetry_report.py",
                    str(path),
                    "--json",
                    "--fail-on-error",
                    "--stale-after-seconds",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 1)
        self.assertFalse(payload["healthy"])
        self.assertIn(
            "unfinished_condition",
            {alert["code"] for alert in payload["alerts"]},
        )


if __name__ == "__main__":
    unittest.main()
