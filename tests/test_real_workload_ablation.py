import asyncio
import importlib.util
import json
import math
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

from calm_puffer_art.objective_ablation import ACCOUNTED_NORTH_STAR


ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


class RealWorkloadAblationTests(unittest.TestCase):
    def test_core_import_does_not_import_heavy_optional_integrations(self):
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "before=set(sys.modules); "
                "import calm_puffer_art; "
                "loaded=set(sys.modules)-before; "
                "forbidden={'torch','art','vllm','transformers','datasets'}; "
                "print(json.dumps(sorted("
                "name for name in loaded if name.split('.')[0] in forbidden"
                ")))"
            ),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_subprocess_env(),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_calm_optional_extra_declares_runtime_dependencies(self):
        pyproject = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = pyproject["project"]["optional-dependencies"]["calm"]

        self.assertIn("numpy>=1.26", dependencies)
        self.assertIn("torch>=2", dependencies)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is not installed",
    )
    def test_real_ablation_scheduler_beats_static_trainable_workload(self):
        from calm_puffer_art.objective_ablation import run_real_ablation

        result = asyncio.run(run_real_ablation())
        static = result["static"]
        objective = result["objective"]
        lift = result["lift"]

        self.assertGreater(objective[ACCOUNTED_NORTH_STAR], static[ACCOUNTED_NORTH_STAR])
        self.assertGreater(lift["accounted_north_star_absolute"], 0.0)
        self.assertGreater(lift["accounted_north_star_ratio"], 1.0)
        self.assertGreater(objective["promotion/latest_score"], 0.0)
        self.assertGreaterEqual(
            objective["promotion/latest_published_policy_score"],
            objective["promotion/latest_baseline_score"],
        )
        self.assertGreater(objective["scheduler/arm/easy_math_token/pulls"], 0.0)
        self.assertGreater(objective["scheduler/arm/hard_math_token/pulls"], 0.0)
        self.assertGreater(
            objective["scheduler/joint_action/feedback_updates"],
            0.0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is not installed",
    )
    def test_real_closed_loop_scheduler_beats_static_trainable_workload(self):
        from calm_puffer_art.objective_ablation import run_real_closed_loop_ablation

        result = asyncio.run(run_real_closed_loop_ablation())
        static = result["static"]
        objective = result["objective"]

        self.assertGreater(
            objective[ACCOUNTED_NORTH_STAR],
            static[ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(result["lift"]["accounted_north_star_absolute"], 0.0)
        self.assertGreater(result["lift"]["accounted_north_star_ratio"], 1.0)
        self.assertGreater(
            objective["actions/semantic_bandwidth_tokens_per_decision"],
            static["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertGreater(
            objective["scheduler/arm/easy_math_chunk_chunk_size_2/pulls"],
            0.0,
        )
        self.assertGreater(
            objective[
                "scheduler/arm/easy_math_chunk_chunk_size_2/"
                "semantic_bandwidth_tokens_per_decision"
            ],
            objective[
                "scheduler/arm/easy_math_token/"
                "semantic_bandwidth_tokens_per_decision"
            ],
        )
        self.assertLess(
            objective[
                "scheduler/arm/easy_math_chunk_chunk_size_2/"
                "mean_rollout_dollar_seconds"
            ],
            objective["scheduler/arm/easy_math_token/mean_rollout_dollar_seconds"],
        )
        self.assertGreater(
            objective[
                "scheduler/arm/easy_math_chunk_chunk_size_2/"
                "total_improvement_per_dollar_second"
            ],
            objective[
                "scheduler/arm/easy_math_token/"
                "total_improvement_per_dollar_second"
            ],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is not installed",
    )
    def test_real_semantic_budget_sweep_reports_finite_rows(self):
        from calm_puffer_art.objective_ablation import run_real_semantic_budget_sweep

        result = asyncio.run(
            run_real_semantic_budget_sweep(train_steps=(2, 4), repeats=1)
        )
        rows = result["rows"]

        self.assertEqual([row["max_train_steps"] for row in rows], [2, 4])
        self.assertIn("semantic_break_even_train_steps", result)
        for row in rows:
            self.assertTrue(math.isfinite(row["token_accounted_north_star"]))
            self.assertTrue(math.isfinite(row["semantic_accounted_north_star"]))
            self.assertTrue(math.isfinite(row["semantic_over_token_ratio"]))
            self.assertTrue(math.isfinite(row["semantic_over_token_absolute"]))
            self.assertIn("chunk2_improvement_per_dollar", row)
            self.assertIn("chunk4_improvement_per_dollar", row)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is not installed",
    )
    def test_real_chunk_length_sweep_reports_chunk_metrics(self):
        from calm_puffer_art.objective_ablation import run_real_chunk_length_sweep

        result = asyncio.run(
            run_real_chunk_length_sweep(response_multipliers=(1, 2), repeats=1)
        )
        rows = result["rows"]

        self.assertEqual([row["response_multiplier"] for row in rows], [1, 2])
        self.assertLess(rows[0]["response_tokens"], rows[1]["response_tokens"])
        self.assertIn("chunk4_recovers_at_response_tokens", result)
        for row in rows:
            self.assertTrue(math.isfinite(row["chunk2_improvement_per_dollar"]))
            self.assertTrue(math.isfinite(row["chunk4_improvement_per_dollar"]))
            self.assertIn("chunk4_active", row)
            self.assertIn("chunk4_disabled", row)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is not installed",
    )
    def test_real_workload_ablation_example_json_contract(self):
        command = [
            sys.executable,
            str(ROOT / "examples" / "real_workload_ablation.py"),
            "--json",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_subprocess_env(),
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["proof_scope"], "tiny_torch_verifiable_math")
        for section in ("scheduler_control", "closed_loop_control"):
            result = payload[section]
            self.assertGreater(
                result["objective"][ACCOUNTED_NORTH_STAR],
                result["static"][ACCOUNTED_NORTH_STAR],
            )
            self.assertGreater(result["lift"]["accounted_north_star_ratio"], 1.0)
        closed_loop = payload["closed_loop_control"]
        self.assertGreater(
            closed_loop["objective"][
                "actions/semantic_bandwidth_tokens_per_decision"
            ],
            closed_loop["static"]["actions/semantic_bandwidth_tokens_per_decision"],
        )
        self.assertGreater(
            closed_loop["objective"][
                "scheduler/arm/easy_math_chunk_chunk_size_2/pulls"
            ],
            0.0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is not installed",
    )
    def test_real_workload_ablation_example_sweeps_json_contract(self):
        command = [
            sys.executable,
            str(ROOT / "examples" / "real_workload_ablation.py"),
            "--json",
            "--include-sweeps",
            "--budget-train-steps",
            "2",
            "--response-multipliers",
            "1",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_subprocess_env(),
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)

        self.assertIn("semantic_sweeps", payload)
        self.assertIn("semantic_break_even_train_steps", payload)
        self.assertIn("chunk4_recovers_at_response_tokens", payload)
        sweeps = payload["semantic_sweeps"]
        self.assertEqual(len(sweeps["budget_sweep"]["rows"]), 1)
        self.assertEqual(len(sweeps["chunk_length_sweep"]["rows"]), 1)
        self.assertTrue(
            math.isfinite(
                sweeps["budget_sweep"]["rows"][0]["semantic_over_token_ratio"]
            )
        )


if __name__ == "__main__":
    unittest.main()
