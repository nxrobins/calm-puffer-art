import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LiveArtBridgeSmokeTests(unittest.TestCase):
    def test_core_import_is_lazy_without_art(self):
        before = set(sys.modules)
        import calm_puffer_art  # noqa: F401

        loaded = set(sys.modules) - before
        self.assertNotIn("art", loaded)

    def test_art_optional_extra_is_declared(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('art = ["openpipe-art>=0.4.9"]', pyproject)

    @unittest.skipUnless(
        importlib.util.find_spec("art") is not None,
        "openpipe-art is not installed",
    )
    def test_structural_mode_uses_real_art_objects_when_installed(self):
        command = [
            sys.executable,
            str(ROOT / "examples" / "live_art_bridge_smoke.py"),
            "--backend",
            "structural",
            "--groups",
            "2",
            "--rollouts-per-group",
            "2",
            "--json",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        metrics = payload["metrics"]

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["used_real_art_package"])
        self.assertFalse(payload["used_real_art_backend"])
        self.assertGreaterEqual(payload["submitted_groups"], 1)
        self.assertGreaterEqual(payload["completed_batches"], 1)
        self.assertGreaterEqual(payload["published_policy_updates"], 1)
        self.assertTrue(payload["raw_art_group_preserved"])
        self.assertTrue(payload["raw_art_trajectory_preserved"])
        self.assertTrue(payload["published_scheduler_state"])
        self.assertTrue(payload["published_art_backend_state"])
        for key in (
            "art_backend/sample_dollar_seconds",
            "art_backend/trainer_dollar_seconds",
            "art_backend/accounted_dollar_seconds",
            "scheduler/joint_action/tuples",
        ):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)


if __name__ == "__main__":
    unittest.main()
