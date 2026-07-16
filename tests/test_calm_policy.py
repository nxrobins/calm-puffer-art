import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ART = importlib.util.find_spec("art") is not None


@unittest.skipUnless(HAS_TORCH and HAS_ART, "torch and ART are not installed")
class CalmPolicyArtLossTests(unittest.TestCase):
    def test_state_conditioned_policy_executes_art_loss_and_updates(self):
        from calm_puffer_art.calm_domain import run_code_domain_codec_proof

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            domain_report = run_code_domain_codec_proof(
                output_dir=output_dir,
                chunk_sizes=(2,),
            )
            checkpoint = output_dir / "code-repair-chunk-2.pt"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "state_conditioned_chunk_policy.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(completed.stdout)

        self.assertTrue(domain_report["ok"])
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["state_conditioned"])
        self.assertEqual(payload["state_source"], "deterministic_context_features")
        self.assertFalse(payload["serving_model_hidden_states"])
        self.assertTrue(payload["art_loss_executed"])
        self.assertEqual(payload["art_loss_module"], "art.loss.loss_fn")
        self.assertFalse(payload["art_serverless_custom_action_supported"])
        self.assertEqual(payload["actions"], 48)
        self.assertEqual(payload["reconstruction_exact_rate"], 1.0)
        self.assertEqual(payload["old_logprob_coverage"], 1.0)
        self.assertEqual(payload["new_logprob_coverage"], 1.0)
        self.assertEqual(payload["reference_logprob_coverage"], 1.0)
        self.assertGreater(payload["gradient_norm"], 0.0)
        self.assertTrue(payload["policy_state_changed"])


if __name__ == "__main__":
    unittest.main()
