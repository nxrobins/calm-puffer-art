import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from calm_puffer_art.sigil_integration import (
    SigilCorpus,
    load_sigil_corpus,
    verify_sigil_code,
)


class SigilIntegrationTests(unittest.TestCase):
    def test_load_sigil_corpus_extracts_prompts_and_training_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            idiom = root / "idiom.jsonl"
            implementation = root / "implementation.jsonl"
            idiom.write_text(
                "\n".join(
                    [
                        json.dumps({"intent": "Make a function", "output": "fn f() {}"}),
                        json.dumps({"intent": "", "output": "ignored"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            implementation.write_text(
                json.dumps({"output": "module m; pub fn f() -> i64 { return 1; }"})
                + "\n",
                encoding="utf-8",
            )

            corpus = load_sigil_corpus(
                idiom_path=idiom,
                implementation_path=implementation,
            )

        self.assertIsInstance(corpus, SigilCorpus)
        self.assertEqual(corpus.prompts, ("Make a function",))
        self.assertEqual(
            corpus.training_outputs,
            (
                "module m; pub fn f() -> i64 { return 1; }",
                "module demo;\nfn f() {}\n",
                "module demo;\nignored\n",
            ),
        )
        self.assertEqual(corpus.prompt_count, 1)
        self.assertEqual(corpus.training_output_count, 3)

    def test_verify_sigil_code_reads_json_status(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"status": "ok"}),
            stderr="",
        )
        with patch("calm_puffer_art.sigil_integration.Path.exists", return_value=True):
            with patch("calm_puffer_art.sigil_integration.subprocess.run", return_value=completed):
                self.assertTrue(verify_sigil_code("module m;"))

    def test_verify_sigil_code_fails_closed_on_error_or_bad_json(self):
        error = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=json.dumps({"status": "error"}),
            stderr="",
        )
        bad_json = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="not json",
            stderr="",
        )
        with patch("calm_puffer_art.sigil_integration.Path.exists", return_value=True):
            with patch("calm_puffer_art.sigil_integration.subprocess.run", return_value=error):
                self.assertFalse(verify_sigil_code("module m;"))
            with patch("calm_puffer_art.sigil_integration.subprocess.run", return_value=bad_json):
                self.assertFalse(verify_sigil_code("module m;"))


if __name__ == "__main__":
    unittest.main()
