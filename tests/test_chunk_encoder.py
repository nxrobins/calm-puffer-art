import ast
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

from calm_puffer_art.actions import action_codec_key
from calm_puffer_art.types import ActionUnit


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class ChunkEncoderDependencyTests(unittest.TestCase):
    def test_core_import_does_not_import_torch(self):
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "before=set(sys.modules); "
                "import calm_puffer_art; "
                "loaded=sorted(set(sys.modules)-before); "
                "print(json.dumps([name for name in loaded "
                "if name == 'torch' or name.startswith('torch.')]))"
            ),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertEqual(json.loads(completed.stdout), [])

    def test_calm_optional_extra_is_declared(self):
        pyproject = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = pyproject["project"]["optional-dependencies"]["calm"]

        self.assertIn("numpy>=1.26", dependencies)
        self.assertIn("torch>=2", dependencies)

    def test_chunk_encoder_does_not_import_art_or_vllm(self):
        source = ROOT / "src" / "calm_puffer_art" / "chunk_encoder.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        self.assertFalse({"art", "vllm"} & imported)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class LearnedChunkEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from calm_puffer_art.chunk_encoder import (
            LearnedChunkActionCodec,
            LearnedChunkEncoderConfig,
            train_smoke_chunk_encoder,
        )

        cls.config = LearnedChunkEncoderConfig()
        cls.bundle = train_smoke_chunk_encoder(cls.config)
        cls.codec = LearnedChunkActionCodec(cls.bundle)

    def test_smoke_training_reaches_exact_train_and_holdout_reconstruction(self):
        report = self.bundle.training_report

        self.assertEqual(report.proof_scope, "smoke_only")
        self.assertEqual(report.train_examples, 8)
        self.assertEqual(report.holdout_examples, 2)
        self.assertEqual(report.train_reconstruction_accuracy, 1.0)
        self.assertEqual(report.holdout_reconstruction_accuracy, 1.0)
        self.assertGreaterEqual(report.nll_improvement, 1e-6)

    def test_learned_actions_have_full_logprob_coverage_and_identity(self):
        from calm_puffer_art.chunk_encoder import validate_learned_chunk_actions

        encode_report = self.codec.encode_with_report("alpha beta gamma delta")
        stats = validate_learned_chunk_actions(encode_report.actions)
        improved = [
            action
            for action in encode_report.actions
            if action.new_logprob is not None
            and action.old_logprob is not None
            and action.new_logprob > action.old_logprob
        ]
        key = action_codec_key(self.codec)

        self.assertFalse(encode_report.fallback)
        self.assertEqual(encode_report.decoded_text, "alpha beta gamma delta")
        self.assertEqual(encode_report.reconstruction_accuracy, 1.0)
        self.assertEqual(stats.old_logprob_coverage, 1.0)
        self.assertEqual(stats.new_logprob_coverage, 1.0)
        self.assertEqual(stats.reference_logprob_coverage, 1.0)
        self.assertGreaterEqual(len(improved), 1)
        for fragment in (
            "vocab_hash=",
            "chunk_size=2",
            "latent_dim=16",
            "reconstruction_threshold=1.0",
            "reference_scorer_state_id=",
            "old_scorer_state_id=",
            "new_scorer_state_id=",
        ):
            self.assertIn(fragment, key)

    def test_unknown_token_fails_closed_to_token_fallback(self):
        from calm_puffer_art.chunk_encoder import validate_learned_chunk_actions

        encode_report = self.codec.encode_with_report("alpha omega")

        self.assertTrue(encode_report.fallback)
        self.assertFalse(encode_report.passed_reconstruction_threshold)
        self.assertEqual(encode_report.metadata["action/fallback"], True)
        self.assertEqual(encode_report.metadata["reconstruction/safe"], False)
        self.assertEqual(encode_report.metadata["failure/mode"], "unknown_token")
        self.assertTrue(encode_report.actions)
        self.assertTrue(
            all(
                action.metadata["action/fallback"] is True
                and action.metadata["failure/mode"] == "unknown_token"
                for action in encode_report.actions
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "fallback_actions_in_learned_chunk_metrics",
        ):
            validate_learned_chunk_actions(encode_report.actions)

    def test_missing_logprobs_and_bad_targets_fail_fast(self):
        from calm_puffer_art.chunk_encoder import (
            compute_reconstruction_accuracy,
            validate_learned_chunk_actions,
        )

        with self.assertRaisesRegex(ValueError, "missing_or_detached_chunk_logprobs"):
            validate_learned_chunk_actions(
                [
                    ActionUnit(
                        kind="learned_chunk",
                        payload=(0.0,),
                        token_count=1,
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "invalid_reconstruction_target"):
            compute_reconstruction_accuracy([0, 1], [0, 1])
        self.assertLess(compute_reconstruction_accuracy([1, 2], [1, 2, 3]), 1.0)

    def test_checkpoint_manifest_and_unsupported_modes_are_explicit(self):
        from calm_puffer_art.chunk_encoder import (
            LearnedChunkEncoderConfig,
            validate_checkpoint_manifest,
        )

        validate_checkpoint_manifest(self.bundle.checkpoint_manifest())
        with self.assertRaisesRegex(
            NotImplementedError,
            "learned_chunk_checkpoint_incomplete",
        ):
            validate_checkpoint_manifest({"encoder": "only"})
        with self.assertRaisesRegex(
            NotImplementedError,
            "unsupported_chunk_encoder_mode",
        ):
            LearnedChunkEncoderConfig(streaming=True).validate()
        with self.assertRaisesRegex(
            ValueError,
            "chunk_encoder_input_limit_exceeded",
        ):
            LearnedChunkEncoderConfig(reconstruction_threshold=0.99).validate()

    def test_chunk_encoder_smoke_example_json_contract(self):
        command = [
            sys.executable,
            str(ROOT / "examples" / "chunk_encoder_smoke.py"),
            "--json",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["proof_scope"], "smoke_only")
        self.assertTrue(payload["used_torch"])
        self.assertEqual(payload["chunk_size"], 2)
        self.assertEqual(payload["latent_dim"], 16)
        self.assertEqual(payload["train_reconstruction_accuracy"], 1.0)
        self.assertEqual(payload["holdout_reconstruction_accuracy"], 1.0)
        self.assertTrue(payload["passed_reconstruction_threshold"])
        self.assertGreaterEqual(payload["actions"], 1)
        self.assertGreater(payload["semantic_bandwidth"], 1.0)
        self.assertEqual(payload["old_logprob_coverage"], 1.0)
        self.assertEqual(payload["new_logprob_coverage"], 1.0)
        self.assertEqual(payload["reference_logprob_coverage"], 1.0)
        self.assertGreaterEqual(payload["new_logprob_improved_chunks"], 1)
        self.assertIn("mean_old_reference_logprob_abs_delta", payload)


if __name__ == "__main__":
    unittest.main()
