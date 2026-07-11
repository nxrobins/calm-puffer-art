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


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class CalmDomainProofTests(unittest.TestCase):
    def test_code_domain_proof_selects_only_exact_chunk_candidates(self):
        from calm_puffer_art.calm_domain import run_code_domain_codec_proof

        with tempfile.TemporaryDirectory() as directory:
            report = run_code_domain_codec_proof(output_dir=Path(directory))

        self.assertTrue(report["ok"])
        self.assertEqual(report["proof_scope"], "offline_domain_reconstruction_only")
        self.assertFalse(report["native_policy_logprobs"])
        self.assertFalse(report["art_loss_connected"])
        self.assertEqual(report["eligible_chunk_sizes"], [2])
        self.assertFalse(report["all_candidates_eligible"])

        rows = {row["chunk_size"]: row for row in report["rows"]}
        self.assertTrue(rows[2]["roundtrip_identity_preserved"])
        self.assertTrue(rows[2]["eligible_for_live_bridge"])
        self.assertEqual(rows[2]["holdout"]["exact_reconstruction_rate"], 1.0)
        self.assertEqual(
            rows[2]["holdout"]["semantic_bandwidth_tokens_per_decision"],
            2.0,
        )
        self.assertFalse(rows[4]["eligible_for_live_bridge"])
        self.assertLess(rows[4]["holdout"]["exact_reconstruction_rate"], 1.0)
        self.assertGreater(rows[4]["holdout"]["fallbacks"], 0)
        self.assertEqual(
            rows[4]["holdout"]["failure_modes"],
            {"reconstruction_drift": rows[4]["holdout"]["fallbacks"]},
        )
        self.assertTrue(rows[4]["unknown_token_fallback"]["fallback"])

    def test_code_domain_cli_persists_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "code_domain_chunk_codec.py"),
                    "--chunk-sizes",
                    "2",
                    "--output-dir",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(completed.stdout)
            persisted = json.loads(
                (output_dir / "report.json").read_text(encoding="utf-8")
            )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload, persisted)
        self.assertEqual(payload["eligible_chunk_sizes"], [2])


if __name__ == "__main__":
    unittest.main()
