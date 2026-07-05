import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calm_puffer_art.foundry_codegen import (
    AzureFoundryCodegenConfig,
    run_azure_foundry_budget_race,
)
from calm_puffer_art.foundry_harness import (
    FOUNDRY_HARNESS_ARTIFACT_FILES,
    FOUNDRY_HARNESS_OBJECTIVE_METRIC,
    FoundryHarnessManifest,
    aggregate_foundry_harness_summaries,
    analyze_foundry_harness_runs,
    compare_foundry_harness_runs,
    extract_foundry_harness_failures,
    foundry_harness_child_args,
    foundry_harness_promotion_readiness,
    load_foundry_harness_manifest,
    _next_hypotheses_payload,
    pairwise_foundry_harness_summaries,
    rank_foundry_harness_summaries,
    summarize_foundry_harness_result,
)


ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    for name in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "COVENANT_AZURE_KEY",
        "COVENANT_AZURE_ENDPOINT",
        "COVENANT_AZURE_API_VERSION",
    ):
        env.pop(name, None)
    return env


class _FakeCompletions:
    async def create(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "def solve(value, low, high):\n"
                            "    if value < low:\n"
                            "        return low\n"
                            "    if value > high:\n"
                            "        return high\n"
                            "    return value\n"
                        ),
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=17,
                total_tokens=28,
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class FoundryHarnessTests(unittest.TestCase):
    def test_manifest_defaults_and_child_args_are_applied(self):
        manifest = FoundryHarnessManifest.from_mapping(
            {
                "name": "mini",
                "deployment": "dry-run-placeholder",
            }
        )

        self.assertEqual(manifest.primary_condition, "full_trinity")
        self.assertEqual(manifest.conditions, ("full_trinity",))
        self.assertEqual(manifest.task_split, "standard")
        self.assertEqual(manifest.prompt_context_policy, "repair_prompt_only")
        self.assertTrue(manifest.budget_race)
        self.assertTrue(manifest.promotion_eligible)
        self.assertIn("token", manifest.action_codecs)
        args = foundry_harness_child_args(
            manifest,
            telemetry_path=Path(".codex/foundry-runs/mini/telemetry.jsonl"),
        )
        self.assertIn("--budget-race", args)
        self.assertIn("--task-split", args)
        self.assertIn("--prompt-context-policy", args)
        self.assertIn("repair_prompt_only", args)
        self.assertIn("--conditions", args)
        self.assertIn("full_trinity", args)
        self.assertIn("--request-timeout-s", args)
        self.assertIn("--verify-timeout-s", args)

    def test_manifest_rejects_unknown_and_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            FoundryHarnessManifest.from_mapping(
                {
                    "name": "bad",
                    "deployment": "dry-run-placeholder",
                    "typo": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "primary_condition"):
            FoundryHarnessManifest.from_mapping(
                {
                    "name": "bad",
                    "deployment": "dry-run-placeholder",
                    "primary_condition": "other",
                }
            )
        with self.assertRaisesRegex(ValueError, "train_steps"):
            FoundryHarnessManifest.from_mapping(
                {
                    "name": "bad",
                    "deployment": "dry-run-placeholder",
                    "train_steps": 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "primary_condition_not_selected"):
            FoundryHarnessManifest.from_mapping(
                {
                    "name": "bad",
                    "deployment": "dry-run-placeholder",
                    "primary_condition": "full_trinity",
                    "conditions": ["static_art"],
                }
            )
        with self.assertRaisesRegex(ValueError, "prompt_context_policy"):
            FoundryHarnessManifest.from_mapping(
                {
                    "name": "bad",
                    "deployment": "dry-run-placeholder",
                    "prompt_context_policy": "missing",
                }
            )

    def test_public_candidate_manifests_parse(self):
        baseline = load_foundry_harness_manifest("baseline")
        full = load_foundry_harness_manifest("full_trinity")
        hard_baseline = load_foundry_harness_manifest("hard_baseline")
        hard_full = load_foundry_harness_manifest("hard_full_trinity")
        frontier_baseline = load_foundry_harness_manifest("frontier_baseline")
        frontier_scheduler = load_foundry_harness_manifest("frontier_scheduler_only")
        frontier_full = load_foundry_harness_manifest("frontier_full_trinity")
        frontier_tag_guardrails = load_foundry_harness_manifest(
            "frontier_failure_tag_guardrails"
        )
        frontier_guardrails = load_foundry_harness_manifest(
            "frontier_data_model_guardrails"
        )

        self.assertEqual(baseline.primary_condition, "static_art")
        self.assertEqual(full.primary_condition, "full_trinity")
        self.assertEqual(baseline.conditions, ("static_art",))
        self.assertEqual(full.conditions, ("full_trinity",))
        self.assertEqual(baseline.task_split, "standard_heldout")
        self.assertEqual(full.task_split, "standard_heldout")
        self.assertEqual(hard_baseline.task_split, "hard_heldout")
        self.assertEqual(hard_full.task_split, "hard_heldout")
        self.assertEqual(frontier_baseline.task_split, "frontier_hard")
        self.assertEqual(frontier_scheduler.task_split, "frontier_hard")
        self.assertEqual(frontier_full.task_split, "frontier_hard")
        self.assertEqual(frontier_tag_guardrails.task_split, "frontier_hard")
        self.assertEqual(frontier_guardrails.task_split, "frontier_hard")
        self.assertEqual(frontier_scheduler.primary_condition, "scheduler_only")
        self.assertEqual(frontier_scheduler.conditions, ("scheduler_only",))
        self.assertEqual(frontier_tag_guardrails.primary_condition, "full_trinity")
        self.assertEqual(frontier_tag_guardrails.conditions, ("full_trinity",))
        self.assertEqual(
            frontier_tag_guardrails.prompt_context_policy,
            "failure_tag_guardrails",
        )
        self.assertFalse(frontier_tag_guardrails.promotion_eligible)
        self.assertEqual(frontier_guardrails.primary_condition, "full_trinity")
        self.assertEqual(
            frontier_guardrails.prompt_context_policy,
            "data_model_guardrails",
        )
        self.assertFalse(frontier_guardrails.promotion_eligible)
        self.assertEqual(baseline.promotion_metric, FOUNDRY_HARNESS_OBJECTIVE_METRIC)
        self.assertEqual(full.promotion_metric, FOUNDRY_HARNESS_OBJECTIVE_METRIC)

    def test_summary_and_failure_taxonomy_for_missing_env_payload(self):
        manifest = FoundryHarnessManifest.from_mapping(
            {
                "name": "mini",
                "deployment": "dry-run-placeholder",
                "primary_condition": "static_art",
            }
        )
        result = {
            "ok": False,
            "error": "azure_foundry_env_missing_required_keys",
            "error_type": "RuntimeError",
        }

        summary = summarize_foundry_harness_result(
            manifest,
            result,
            output_dir=Path(".codex/foundry-runs/mini"),
            returncode=1,
        )
        failures = extract_foundry_harness_failures(
            manifest,
            result,
            returncode=1,
            stderr="",
        )

        self.assertFalse(summary["ok"])
        self.assertEqual(failures["counts"]["setup_failure"], 1)
        self.assertEqual(failures["events"][0]["category"], "setup_failure")

    def test_successful_heartbeat_stderr_is_not_a_setup_failure(self):
        manifest = FoundryHarnessManifest.from_mapping(
            {
                "name": "mini",
                "deployment": "dry-run-placeholder",
                "primary_condition": "static_art",
            }
        )
        result = {
            "ok": True,
            "conditions": {
                "static_art": {
                    FOUNDRY_HARNESS_OBJECTIVE_METRIC: 0.1,
                    "costs/accounted_dollar_seconds": 1.0,
                }
            },
        }
        stderr = (
            '{"event":"run_started","deployment":"dry-run-placeholder"}\n'
            '{"event":"run_completed","ok":true}\n'
        )

        failures = extract_foundry_harness_failures(
            manifest,
            result,
            returncode=0,
            stderr=stderr,
        )

        self.assertTrue(failures["ok"])
        self.assertEqual(failures["counts"]["setup_failure"], 0)

    def test_comparison_ranks_static_scheduler_and_full_trinity_fixtures(self):
        lower = _summary("baseline", "static_art", 0.1, 10.0)
        middle = _summary("scheduler", "scheduler_only", 0.2, 12.0)
        higher = _summary("full_trinity", "full_trinity", 0.3, 9.0)
        ranked = rank_foundry_harness_summaries([middle, higher, lower])

        self.assertEqual(
            [item["candidate"] for item in ranked],
            ["full_trinity", "scheduler", "baseline"],
        )

    def test_comparison_prefers_heldout_score_when_available(self):
        primary_high = _summary("primary_high", "full_trinity", 10.0, 5.0)
        heldout_high = _summary("heldout_high", "full_trinity", 1.0, 5.0)
        heldout_high["heldout_score"] = 2.0
        heldout_high["ranking_score"] = 2.0
        heldout_high["ranking_score_source"] = "heldout"
        ranked = rank_foundry_harness_summaries([primary_high, heldout_high])

        self.assertEqual(ranked[0]["candidate"], "primary_high")
        primary_high["heldout_score"] = 0.5
        primary_high["ranking_score"] = 0.5
        primary_high["ranking_score_source"] = "heldout"
        ranked = rank_foundry_harness_summaries([primary_high, heldout_high])
        self.assertEqual(ranked[0]["candidate"], "heldout_high")

    def test_aggregate_comparison_groups_replicates_by_candidate(self):
        summaries = [
            _summary("baseline", "static_art", 0.1, 10.0),
            _summary("baseline", "static_art", 0.3, 14.0),
            _summary("full_trinity", "full_trinity", 0.25, 8.0),
            {
                **_summary("full_trinity", "full_trinity", 0.4, 9.0),
                "ok": False,
            },
        ]

        aggregates = aggregate_foundry_harness_summaries(summaries)

        self.assertEqual(aggregates[0]["candidate"], "baseline")
        self.assertEqual(aggregates[0]["runs"], 2)
        self.assertEqual(aggregates[0]["ok_runs"], 2)
        self.assertAlmostEqual(aggregates[0]["ranking_score_median"], 0.2)
        self.assertEqual(aggregates[1]["candidate"], "full_trinity")
        self.assertEqual(aggregates[1]["failed_runs"], 1)
        self.assertAlmostEqual(aggregates[1]["failure_rate"], 0.5)

    def test_pairwise_comparison_reports_win_rates_and_deltas(self):
        summaries = [
            _summary("baseline", "static_art", 0.1, 10.0),
            _summary("baseline", "static_art", 0.4, 12.0),
            _summary("full_trinity", "full_trinity", 0.2, 13.0),
            _summary("full_trinity", "full_trinity", 0.5, 14.0),
            _summary("scheduler", "scheduler_only", 0.3, 9.0),
            {
                **_summary("scheduler", "scheduler_only", 0.9, 9.0),
                "ok": False,
            },
        ]

        pairwise = pairwise_foundry_harness_summaries(summaries)
        payload = {
            (item["left_candidate"], item["right_candidate"]): item
            for item in pairwise
        }

        full_vs_scheduler = payload[("full_trinity", "scheduler")]
        self.assertEqual(full_vs_scheduler["pair_count"], 2)
        self.assertEqual(full_vs_scheduler["left_wins"], 1)
        self.assertEqual(full_vs_scheduler["right_wins"], 1)
        self.assertEqual(full_vs_scheduler["leader_candidate"], "full_trinity")
        self.assertAlmostEqual(
            full_vs_scheduler["mean_score_delta_left_minus_right"],
            0.05,
        )

        baseline_vs_full = payload[("baseline", "full_trinity")]
        self.assertEqual(baseline_vs_full["pair_count"], 4)
        self.assertEqual(baseline_vs_full["left_wins"], 1)
        self.assertEqual(baseline_vs_full["right_wins"], 3)
        self.assertAlmostEqual(baseline_vs_full["right_win_rate"], 0.75)
        self.assertAlmostEqual(
            baseline_vs_full[
                "mean_accounted_dollar_seconds_delta_left_minus_right"
            ],
            -2.5,
        )

    def test_promotion_readiness_promotes_only_replicated_candidate_wins(self):
        summaries = [
            _summary("baseline", "static_art", 0.1, 10.0),
            _summary("baseline", "static_art", 0.1, 10.5),
            _summary("baseline", "static_art", 0.1, 11.0),
            _summary("full_trinity", "full_trinity", 0.2, 12.0),
            _summary("full_trinity", "full_trinity", 0.25, 12.5),
            _summary("full_trinity", "full_trinity", 0.3, 13.0),
        ]
        comparison = {
            "candidate_aggregates": aggregate_foundry_harness_summaries(summaries),
            "candidate_pairwise": pairwise_foundry_harness_summaries(summaries),
        }

        readiness = foundry_harness_promotion_readiness(comparison)

        self.assertEqual(readiness["status"], "promote")
        self.assertEqual(readiness["baseline_candidate"], "baseline")
        decision = readiness["decisions"][0]
        self.assertEqual(decision["candidate"], "full_trinity")
        self.assertEqual(decision["status"], "promote")
        self.assertEqual(decision["pairwise_win_rate_vs_baseline"], 1.0)
        self.assertGreater(decision["median_score_delta_vs_baseline"], 0.0)

    def test_promotion_readiness_requires_replicates_and_holds_losing_candidates(self):
        summaries = [
            _summary("baseline", "static_art", 0.3, 10.0),
            _summary("baseline", "static_art", 0.2, 10.5),
            _summary("full_trinity", "full_trinity", 0.25, 12.0),
            _summary("full_trinity", "full_trinity", 0.1, 12.5),
        ]
        comparison = {
            "candidate_aggregates": aggregate_foundry_harness_summaries(summaries),
            "candidate_pairwise": pairwise_foundry_harness_summaries(summaries),
        }

        readiness = foundry_harness_promotion_readiness(comparison)
        self.assertEqual(readiness["status"], "needs_more_evidence")
        self.assertEqual(
            readiness["recommended_next_runs"][0],
            {
                "candidate": "baseline",
                "additional_successful_runs": 1,
            },
        )
        decision = readiness["decisions"][0]
        self.assertEqual(decision["status"], "needs_more_evidence")
        self.assertEqual(decision["additional_successful_runs"], 1)
        self.assertEqual(decision["baseline_additional_successful_runs"], 1)

        readiness = foundry_harness_promotion_readiness(
            comparison,
            min_successful_runs=1,
        )
        decision = readiness["decisions"][0]
        self.assertEqual(readiness["status"], "hold")
        self.assertEqual(decision["status"], "hold")
        self.assertIn("median_score_not_above_baseline", decision["reason"])

    def test_promotion_readiness_excludes_ineligible_probe_candidates(self):
        summaries = [
            _summary("baseline", "static_art", 0.3, 10.0),
            _summary("baseline", "static_art", 0.3, 10.0),
            _summary("baseline", "static_art", 0.3, 10.0),
            {
                **_summary("full_trinity", "full_trinity", 0.2, 12.0),
                "promotion_eligible": False,
            },
        ]
        comparison = {
            "candidate_aggregates": aggregate_foundry_harness_summaries(summaries),
            "candidate_pairwise": pairwise_foundry_harness_summaries(summaries),
        }

        readiness = foundry_harness_promotion_readiness(comparison)

        self.assertEqual(readiness["status"], "hold")
        self.assertEqual(readiness["recommended_next_runs"], [])
        self.assertEqual(readiness["decisions"], [])
        self.assertEqual(
            readiness["excluded_candidates"],
            [
                {
                    "candidate": "full_trinity",
                    "reason": "promotion_eligible_false",
                }
            ],
        )

    def test_compare_cli_reads_summary_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            for name, score in (("baseline", 0.1), ("full_trinity", 0.4)):
                run_dir = runs_dir / name
                run_dir.mkdir(parents=True)
                (run_dir / FOUNDRY_HARNESS_ARTIFACT_FILES["summary"]).write_text(
                    json.dumps(_summary(name, "full_trinity", score, 10.0)),
                    encoding="utf-8",
                )

            payload = compare_foundry_harness_runs(runs_dir)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["runs"], 2)
            self.assertEqual(payload["ok_runs"], 2)
            self.assertEqual(payload["ranking"][0]["candidate"], "full_trinity")
            self.assertEqual(
                payload["candidate_aggregates"][0]["candidate"],
                "full_trinity",
            )
            self.assertEqual(
                payload["candidate_pairwise"][0]["leader_candidate"],
                "full_trinity",
            )
            prefixed = compare_foundry_harness_runs(
                runs_dir,
                run_prefix="full",
            )
            self.assertEqual(prefixed["runs"], 1)
            self.assertEqual(prefixed["ranking"][0]["candidate"], "full_trinity")

            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/foundry_harness_compare.py",
                    "--runs",
                    str(runs_dir),
                    "--run-prefix",
                    "full",
                    "--json",
                ],
                cwd=ROOT,
                env=_subprocess_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            cli_payload = json.loads(completed.stdout)
            self.assertEqual(cli_payload["ranking"][0]["candidate"], "full_trinity")
            self.assertEqual(cli_payload["runs"], 1)
            self.assertIn("candidate_aggregates", cli_payload)
            self.assertIn("candidate_pairwise", cli_payload)

    def test_analyze_reads_run_artifacts_and_reports_diagnostics(self):
        metric = FOUNDRY_HARNESS_OBJECTIVE_METRIC

        def write_run(
            path: Path,
            *,
            candidate: str,
            condition: str,
            score: float,
            pass_rate: float,
            exhausted: float,
            task_results: list[dict[str, object]] | None = None,
        ) -> None:
            path.mkdir(parents=True)
            summary = {
                **_summary(candidate, condition, score, 10.0),
                "output_dir": str(path),
                "task_split": "frontier_hard",
                "prompt_context_policy": "data_model_guardrails",
                "ranking_score_source": "heldout",
                "heldout_score": score,
                "ranking_score": score,
            }
            result = {
                "conditions": {
                    condition: {
                        metric: score,
                        "foundry/learned_solutions": pass_rate * 32.0,
                        "foundry/model_calls": 32.0,
                        "foundry/observed_rollouts": 32.0,
                        "costs/accounted_dollar_seconds": 10.0,
                        "scheduler/budget/accounted_exhausted": exhausted,
                    }
                },
                "heldout": {
                    "conditions": {
                        condition: {
                            metric: score,
                            "heldout/pass_rate": pass_rate,
                            "heldout/by_family": {
                                "data_model": {
                                    "pass_rate": pass_rate,
                                    "passed": pass_rate * 5.0,
                                    "tasks": 5.0,
                                }
                            },
                            "heldout/by_difficulty": {
                                "4": {
                                    "pass_rate": pass_rate,
                                    "passed": pass_rate * 8.0,
                                    "tasks": 8.0,
                                }
                            },
                            "heldout/by_failure_tag": {
                                "none_sentinel": {
                                    "pass_rate": pass_rate,
                                    "passed": pass_rate * 2.0,
                                    "tasks": 2.0,
                                }
                            },
                            "heldout/task_results": task_results or [],
                        }
                    }
                },
                "non_saturation": {
                    "conditions": {
                        condition: {
                            "learned_fraction": pass_rate,
                            "heldout_pass_fraction": pass_rate,
                            "saturated": False,
                        }
                    }
                },
                "task_coverage": {"train": {"tasks": 32}},
            }
            failures = {
                "counts": {
                    category: 0
                    for category in (
                        "setup_failure",
                        "run_timeout",
                        "model_request_failure",
                        "verifier_timeout",
                        "verifier_crash",
                        "output_parse_failure",
                        "wrong_answer",
                        "cost_budget_exhausted",
                        "scheduler_stale_batch",
                        "scheduler_control_failure",
                    )
                }
            }
            (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (path / "failures.json").write_text(json.dumps(failures), encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory)
            write_run(
                runs_dir / "frontier-baseline",
                candidate="frontier_baseline",
                condition="static_art",
                score=0.3,
                pass_rate=0.9,
                exhausted=0.0,
                task_results=[
                    {
                        "task_id": "repair_none_sentinel",
                        "family": "real_bug_pattern",
                        "difficulty": "2",
                        "failure_tags": ["none_sentinel"],
                        "passed": True,
                        "failure_mode": "passed",
                    },
                    {
                        "task_id": "repair_schema_errors",
                        "passed": False,
                        "failure_mode": "unit_test_failed",
                    },
                ],
            )
            write_run(
                runs_dir / "frontier-full",
                candidate="frontier_full_trinity",
                condition="full_trinity",
                score=0.2,
                pass_rate=0.6,
                exhausted=1.0,
                task_results=[
                    {
                        "task_id": "repair_none_sentinel",
                        "family": "real_bug_pattern",
                        "difficulty": "2",
                        "failure_tags": ["none_sentinel"],
                        "passed": False,
                        "failure_mode": "unit_test_failed",
                    },
                    {
                        "task_id": "repair_schema_errors",
                        "family": "data_model",
                        "difficulty": "4",
                        "passed": True,
                        "failure_mode": "passed",
                    }
                ],
            )

            payload = analyze_foundry_harness_runs(runs_dir, run_prefix="frontier")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["winner"]["candidate"], "frontier_baseline")
        self.assertEqual(len(payload["deltas_vs_baseline"]), 1)
        delta = payload["deltas_vs_baseline"][0]
        self.assertEqual(delta["candidate"], "frontier_full_trinity")
        self.assertLess(delta["heldout_pass_rate_delta"], 0.0)
        full_diag = next(
            run
            for run in payload["runs"]
            if run["candidate"] == "frontier_full_trinity"
        )
        self.assertTrue(full_diag["budget_exhausted"])
        self.assertEqual(full_diag["prompt_context_policy"], "data_model_guardrails")
        self.assertEqual(
            full_diag["heldout_task_failures"][0]["task_id"],
            "repair_none_sentinel",
        )
        self.assertEqual(
            full_diag["heldout_task_failures"][0]["failure_tags"],
            ["none_sentinel"],
        )
        self.assertEqual(full_diag["weakest_failure_tags"][0]["name"], "none_sentinel")
        pockets = payload["failure_pockets"]
        self.assertEqual(pockets["baseline_candidate"], "frontier_baseline")
        full_pocket = next(
            item
            for item in pockets["by_candidate"]
            if item["candidate"] == "frontier_full_trinity"
        )
        self.assertEqual(full_pocket["task_observations"], 2)
        self.assertEqual(full_pocket["by_failure_tag"][0]["name"], "none_sentinel")
        full_delta = pockets["deltas_vs_baseline"][0]
        self.assertEqual(full_delta["candidate"], "frontier_full_trinity")
        self.assertEqual(
            full_delta["tasks_worse_than_baseline"][0]["task_id"],
            "repair_none_sentinel",
        )
        self.assertAlmostEqual(
            full_delta["tasks_worse_than_baseline"][0]["pass_rate_delta"],
            -1.0,
        )
        self.assertEqual(
            full_delta["tasks_better_than_baseline"][0]["task_id"],
            "repair_schema_errors",
        )
        self.assertEqual(
            full_delta["tasks_better_than_baseline"][0]["failure_tags"],
            ["none_sentinel", "ordering"],
        )
        self.assertAlmostEqual(
            full_delta["tasks_better_than_baseline"][0]["pass_rate_delta"],
            1.0,
        )
        next_hypotheses = payload["next_hypotheses"]
        self.assertEqual(next_hypotheses["status"], "needs_more_evidence")
        self.assertEqual(
            next_hypotheses["actions"][0]["action"],
            "run_additional_replicates",
        )

    def test_next_hypotheses_reports_unstable_lift_and_shared_pockets(self):
        comparison = {
            "candidate_aggregates": [
                {
                    "candidate": "baseline",
                    "primary_condition": "static_art",
                    "promotion_eligible": True,
                    "ok_runs": 3,
                    "failure_rate": 0.0,
                    "ranking_score_median": 0.2,
                },
                {
                    "candidate": "full",
                    "primary_condition": "full_trinity",
                    "promotion_eligible": True,
                    "ok_runs": 3,
                    "failure_rate": 0.0,
                    "ranking_score_median": 0.25,
                },
                {
                    "candidate": "probe",
                    "primary_condition": "full_trinity",
                    "promotion_eligible": False,
                    "ok_runs": 1,
                    "failure_rate": 0.0,
                    "ranking_score_median": 0.1,
                },
            ],
            "candidate_pairwise": [
                {
                    "left_candidate": "baseline",
                    "right_candidate": "full",
                    "pair_count": 9,
                    "left_win_rate": 0.5555555555555556,
                    "right_win_rate": 0.4444444444444444,
                    "leader_candidate": "baseline",
                }
            ],
        }
        readiness = foundry_harness_promotion_readiness(comparison)
        failure_pockets = {
            "baseline_candidate": "baseline",
            "by_candidate": [
                {
                    "candidate": "baseline",
                    "by_task": [
                        {
                            "task_id": "repair_nested_defaults",
                            "family": "data_model",
                            "difficulty": "4",
                            "failure_tags": ["mutation", "none_sentinel"],
                            "observations": 3,
                            "passed": 1,
                            "failed": 2,
                            "pass_rate": 1 / 3,
                        }
                    ],
                    "by_family": [
                        {
                            "name": "data_model",
                            "observations": 3,
                            "passed": 1,
                            "failed": 2,
                            "pass_rate": 1 / 3,
                        }
                    ],
                    "by_failure_tag": [
                        {
                            "name": "none_sentinel",
                            "observations": 3,
                            "passed": 1,
                            "failed": 2,
                            "pass_rate": 1 / 3,
                        }
                    ],
                },
                {
                    "candidate": "full",
                    "by_task": [
                        {
                            "task_id": "repair_nested_defaults",
                            "family": "data_model",
                            "difficulty": "4",
                            "failure_tags": ["mutation", "none_sentinel"],
                            "observations": 3,
                            "passed": 2,
                            "failed": 1,
                            "pass_rate": 2 / 3,
                        }
                    ],
                    "by_family": [
                        {
                            "name": "data_model",
                            "observations": 3,
                            "passed": 2,
                            "failed": 1,
                            "pass_rate": 2 / 3,
                        }
                    ],
                    "by_failure_tag": [
                        {
                            "name": "none_sentinel",
                            "observations": 3,
                            "passed": 2,
                            "failed": 1,
                            "pass_rate": 2 / 3,
                        }
                    ],
                },
            ],
            "deltas_vs_baseline": [
                {
                    "candidate": "full",
                    "tasks_better_than_baseline": [
                        {
                            "task_id": "repair_nested_defaults",
                            "pass_rate_delta": 1 / 3,
                        }
                    ],
                    "tasks_worse_than_baseline": [],
                },
                {
                    "candidate": "probe",
                    "tasks_better_than_baseline": [],
                    "tasks_worse_than_baseline": [
                        {
                            "task_id": "repair_query_params",
                            "pass_rate_delta": -1.0,
                        }
                    ],
                },
            ],
        }

        payload = _next_hypotheses_payload(
            comparison,
            failure_pockets,
            readiness,
        )

        actions = {action["action"]: action for action in payload["actions"]}
        self.assertEqual(payload["status"], "hold")
        self.assertIn("study_unstable_lift", actions)
        self.assertEqual(actions["study_unstable_lift"]["candidate"], "full")
        self.assertIn("treat_as_experimental_probe", actions)
        self.assertEqual(actions["treat_as_experimental_probe"]["candidate"], "probe")
        self.assertIn("design_targeted_candidate", actions)
        self.assertEqual(
            payload["shared_failure_pockets"]["tasks"][0]["task_id"],
            "repair_nested_defaults",
        )

    def test_batch_cli_missing_env_writes_replicate_artifacts_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir = root / "candidates"
            candidate_dir.mkdir()
            override_env_path = root / "override.env"
            override_deployment = "override-deployment"
            (candidate_dir / "missing.json").write_text(
                json.dumps(
                    {
                        "name": "missing",
                        "deployment": "dry-run-placeholder",
                        "env_path": str(root / ".env"),
                        "task_limit": 1,
                        "train_steps": 1,
                        "model_call_budget": 0,
                        "run_timeout_s": 5.0,
                        "heartbeat_interval_s": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            runs_dir = root / "runs"

            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/foundry_harness_batch.py",
                    "--candidates",
                    "missing",
                    "--candidate-dir",
                    str(candidate_dir),
                    "--runs-dir",
                    str(runs_dir),
                    "--env-path",
                    str(override_env_path),
                    "--deployment",
                    override_deployment,
                    "--run-prefix",
                    "batch",
                    "--replicates",
                    "2",
                    "--json",
                ],
                cwd=ROOT,
                env=_subprocess_env(),
                check=False,
                text=True,
                capture_output=True,
            )

            payload = json.loads(completed.stdout)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(payload["ok"])
            self.assertEqual(len(payload["runs"]), 2)
            self.assertEqual(
                payload["runs"][0]["first_failure_category"],
                "setup_failure",
            )
            self.assertEqual(
                payload["runs"][0]["failure_counts"]["setup_failure"],
                1,
            )
            self.assertTrue((runs_dir / "batch-batch-summary.json").exists())
            self.assertEqual(
                payload["comparison"]["candidate_aggregates"][0]["failed_runs"],
                2,
            )
            for replicate in (1, 2):
                output_dir = runs_dir / f"batch-missing-r{replicate:02d}"
                self.assertTrue((output_dir / "failures.json").exists())
                manifest = json.loads(
                    (output_dir / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["env_path"], str(override_env_path))
                self.assertEqual(manifest["deployment"], override_deployment)

    def test_run_cli_missing_env_writes_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "candidate.json"
            output_dir = root / "run"
            override_env_path = root / "override.env"
            override_deployment = "override-deployment"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "missing-env",
                        "deployment": "dry-run-placeholder",
                        "env_path": str(root / ".env"),
                        "task_limit": 1,
                        "train_steps": 1,
                        "model_call_budget": 0,
                        "run_timeout_s": 5.0,
                        "heartbeat_interval_s": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/foundry_harness_run.py",
                    "--candidate",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--env-path",
                    str(override_env_path),
                    "--deployment",
                    override_deployment,
                    "--json",
                ],
                cwd=ROOT,
                env=_subprocess_env(),
                check=False,
                text=True,
                capture_output=True,
            )

            payload = json.loads(completed.stdout)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(payload["ok"])
            for filename in FOUNDRY_HARNESS_ARTIFACT_FILES.values():
                self.assertTrue((output_dir / filename).exists(), filename)
            child_result = json.loads(
                (output_dir / "result.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            failures = json.loads(
                (output_dir / "failures.json").read_text(encoding="utf-8")
            )
            telemetry = [
                json.loads(line)
                for line in (output_dir / "telemetry.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertFalse(child_result["ok"])
            self.assertEqual(manifest["env_path"], str(override_env_path))
            self.assertEqual(manifest["deployment"], override_deployment)
            self.assertEqual(
                child_result["error"],
                "azure_foundry_env_missing_required_keys",
            )
            self.assertEqual(failures["counts"]["setup_failure"], 1)
            self.assertEqual(
                [event["event"] for event in telemetry],
                ["run_started", "run_failed"],
            )

    def test_fake_foundry_budget_race_can_feed_harness_summary(self):
        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            del name, config
            return _FakeClient()

        result = asyncio.run(
            run_azure_foundry_budget_race(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=2,
                    task_limit=1,
                    model_call_budget=2,
                    max_completion_tokens=64,
                ),
                budget_dollar_seconds=25.0,
                client_factory=client_factory,
            )
        )
        manifest = FoundryHarnessManifest.from_mapping(
            {
                "name": "full-trinity",
                "deployment": "dry-run-placeholder",
                "primary_condition": "full_trinity",
            }
        )

        summary = summarize_foundry_harness_result(
            manifest,
            result,
            output_dir=Path(".codex/foundry-runs/fake"),
            returncode=0,
        )

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["measurement"], "azure_foundry_budget_race")
        self.assertIn("full_trinity", summary["conditions"])
        self.assertIsNotNone(summary["primary_score"])


def _summary(
    candidate: str,
    primary_condition: str,
    score: float,
    spend: float,
) -> dict[str, object]:
    return {
        "ok": True,
        "candidate": candidate,
        "primary_condition": primary_condition,
        "primary_score": score,
        "ranking_score": score,
        "ranking_score_source": "primary",
        "primary_accounted_dollar_seconds": spend,
    }


if __name__ == "__main__":
    unittest.main()
