import asyncio
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from calm_puffer_art.codegen_ablation import (
    CODEGEN_ACCOUNTED_NORTH_STAR,
    run_python_codegen_showcase,
)


ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


class PythonCodegenShowcaseTests(unittest.TestCase):
    def test_python_codegen_showcase_reports_full_trinity_lift(self):
        result = asyncio.run(
            run_python_codegen_showcase(max_train_steps=4, response_style=1)
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["proof_scope"], "tiny_unit_test_codegen_showcase")
        self.assertEqual(
            result["workload"],
            "embedded_python_function_synthesis_unit_tests",
        )
        self.assertEqual(
            result["winning_condition_by_accounted_north_star"],
            "full_trinity",
        )
        self.assertIn(
            result["winning_codec_by_improvement_per_dollar"],
            {"chunk2", "chunk4"},
        )
        self.assertGreater(
            result["lift"][
                "scheduler_over_static_accounted_north_star_ratio"
            ],
            1.0,
        )
        self.assertGreater(
            result["lift"][
                "full_trinity_over_scheduler_accounted_north_star_ratio"
            ],
            1.0,
        )
        self.assertGreater(
            result["lift"]["full_trinity_semantic_bandwidth_over_scheduler_ratio"],
            1.0,
        )

        conditions = result["conditions"]
        for name in ("static_art", "scheduler_only", "full_trinity"):
            self.assertIn(CODEGEN_ACCOUNTED_NORTH_STAR, conditions[name])
            self.assertTrue(math.isfinite(conditions[name][CODEGEN_ACCOUNTED_NORTH_STAR]))
            self.assertGreater(conditions[name]["costs/accounted_dollar_seconds"], 0.0)

        self.assertGreater(conditions["scheduler_only"]["token_pulls"], 0.0)
        self.assertGreater(conditions["full_trinity"]["chunk2_pulls"], 0.0)
        self.assertGreater(
            conditions["full_trinity"]["actions/semantic_bandwidth_tokens_per_decision"],
            conditions["scheduler_only"]["actions/semantic_bandwidth_tokens_per_decision"],
        )

    def test_python_codegen_showcase_example_json_contract(self):
        command = [
            sys.executable,
            str(ROOT / "examples" / "python_codegen_showcase.py"),
            "--json",
            "--train-steps",
            "4",
            "--response-style",
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

        self.assertEqual(payload["measurement"], "python_codegen_showcase")
        self.assertEqual(
            sorted(payload["conditions"]),
            ["full_trinity", "scheduler_only", "static_art"],
        )
        self.assertIn("lift", payload)
        self.assertEqual(
            payload["winning_condition_by_accounted_north_star"],
            "full_trinity",
        )



if __name__ == "__main__":
    unittest.main()
