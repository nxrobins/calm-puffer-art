import asyncio
import importlib.util
import json
import os
import subprocess
import sys
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

    def test_calm_optional_extra_declares_torch(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('calm = ["torch>=2"]', pyproject)

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

        self.assertGreater(
            result["objective"][ACCOUNTED_NORTH_STAR],
            result["static"][ACCOUNTED_NORTH_STAR],
        )
        self.assertGreater(result["lift"]["accounted_north_star_absolute"], 0.0)
        self.assertGreater(result["lift"]["accounted_north_star_ratio"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
