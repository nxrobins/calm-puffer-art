import asyncio
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from calm_puffer_art.codegen_ablation import (
    run_codegen_semantic_sweep,
)


ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


class CodegenSemanticSweepTests(unittest.TestCase):
    def test_codegen_semantic_sweep_reports_fixed_codec_rows(self):
        result = asyncio.run(
            run_codegen_semantic_sweep(response_styles=(1,), max_train_steps=4)
        )
        rows = result["rows"]

        self.assertEqual(result["proof_scope"], "tiny_unit_test_codegen_fixed_codecs")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("chunk3_recovers_at_response_tokens", result)
        self.assertIn("chunk4_recovers_at_response_tokens", result)
        self.assertTrue(math.isfinite(row["accounted_north_star"]))
        for label in ("token", "chunk2", "chunk3", "chunk4"):
            self.assertIn(f"{label}_pulls", row)
            self.assertIn(f"{label}_improvement_per_dollar", row)
            self.assertIn(f"{label}_mean_rollout_dollar_seconds", row)
        self.assertGreater(row["mean_response_tokens"], 0.0)

    def test_codegen_semantic_sweep_response_styles_are_ordered(self):
        result = asyncio.run(
            run_codegen_semantic_sweep(
                response_styles=(1, 2),
                max_train_steps=4,
            )
        )
        rows = result["rows"]

        self.assertEqual([row["response_style"] for row in rows], [1, 2])
        self.assertLess(rows[0]["mean_response_tokens"], rows[1]["mean_response_tokens"])

    def test_codegen_semantic_sweep_example_json_contract(self):
        command = [
            sys.executable,
            str(ROOT / "examples" / "codegen_semantic_sweep.py"),
            "--json",
            "--response-styles",
            "1",
            "--train-steps",
            "4",
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

        self.assertEqual(payload["proof_scope"], "tiny_unit_test_codegen_fixed_codecs")
        self.assertEqual(len(payload["rows"]), 1)
        self.assertIn("chunk3_recovers_at_response_tokens", payload)
        self.assertIn("chunk4_recovers_at_response_tokens", payload)
        self.assertIn("winning_codec_by_improvement_per_dollar", payload["rows"][0])


if __name__ == "__main__":
    unittest.main()
