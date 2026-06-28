import asyncio
import importlib.util
import unittest
from unittest.mock import patch

from calm_puffer_art.sigil_ablation import (
    SIGIL_BUCKET_EASY,
    SIGIL_BUCKET_HARD,
    SIGIL_BUCKET_MEDIUM,
    SIGIL_WORKLOAD_PROOF_SCOPE,
    SigilAblationTask,
    build_sigil_ablation_tasks,
    run_sigil_workload_ablation,
    sigil_bucket_counts,
    sigil_difficulty_bucket,
)
from calm_puffer_art.sigil_integration import SigilCorpus


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _fixture_corpus() -> SigilCorpus:
    return SigilCorpus(
        prompts=("Return one", "Return two"),
        training_outputs=(
            "module train_a; pub fn a() -> i64 { return 1; }",
            "module train_b; pub fn b() -> i64 { return 2; }",
        ),
        idiom_rows=(
            {
                "id": "idiom_one",
                "intent": "Return one from a Sigil function.",
                "output": "pub fn one() -> i64 { return 1; }",
            },
            {
                "id": "idiom_two",
                "intent": "Return two from a Sigil function.",
                "output": "pub fn two() -> i64 { return 2; }",
            },
        ),
        implementation_rows=(),
    )


class SigilTaskConstructionTests(unittest.TestCase):
    def test_build_sigil_tasks_uses_intents_and_filters_by_verifier(self):
        with patch("calm_puffer_art.sigil_ablation.verify_sigil_code", return_value=True):
            tasks = build_sigil_ablation_tasks(_fixture_corpus())

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].prompt, "Return one from a Sigil function.")
        self.assertIn("module demo;", tasks[0].target_code)
        self.assertEqual(tasks[0].target_candidate, 0)
        self.assertEqual(tasks[1].target_candidate, 1)

    def test_sigil_tasks_are_bucketed_by_target_length_without_rewriting_ids(self):
        easy = SigilAblationTask(
            id="sigil_000",
            prompt="short",
            candidates=("module easy;",),
            rollout_dollar_seconds=0.1,
            target_candidate=0,
            source_id="easy",
        )
        medium = SigilAblationTask(
            id="sigil_001",
            prompt="medium",
            candidates=(" ".join(f"tok{i}" for i in range(30)),),
            rollout_dollar_seconds=0.1,
            target_candidate=0,
            source_id="medium",
        )
        hard = SigilAblationTask(
            id="sigil_002",
            prompt="hard",
            candidates=(" ".join(f"tok{i}" for i in range(81)),),
            rollout_dollar_seconds=0.1,
            target_candidate=0,
            source_id="hard",
        )

        self.assertEqual(sigil_difficulty_bucket(easy), SIGIL_BUCKET_EASY)
        self.assertEqual(sigil_difficulty_bucket(medium), SIGIL_BUCKET_MEDIUM)
        self.assertEqual(sigil_difficulty_bucket(hard), SIGIL_BUCKET_HARD)
        self.assertEqual(easy.id, "sigil_000")
        self.assertEqual(
            sigil_bucket_counts((easy, medium, hard)),
            {
                SIGIL_BUCKET_EASY: 1,
                SIGIL_BUCKET_MEDIUM: 1,
                SIGIL_BUCKET_HARD: 1,
            },
        )


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class SigilAblationTests(unittest.TestCase):
    def test_sigil_workload_ablation_reports_three_conditions(self):
        from calm_puffer_art.sigil_encoder import (
            SigilEncoderTrainingConfig,
            train_sigil_chunk_encoder,
        )

        corpus = _fixture_corpus()
        bundle = train_sigil_chunk_encoder(
            corpus.training_outputs,
            SigilEncoderTrainingConfig(
                train_steps=1,
                scorer_train_steps=1,
                max_chunks=2,
                timeout_s=10.0,
            ),
        )
        tasks = (
            SigilAblationTask(
                id="sigil_000",
                prompt="Return one from a Sigil function.",
                candidates=(
                    "modul invalid;",
                    "module valid_one; pub fn one() -> i64 { return 1; }",
                    "module invalid_two; retun 2;",
                ),
                rollout_dollar_seconds=0.1,
                target_candidate=1,
                source_id="idiom_one",
            ),
        )

        def verifier(code: str) -> bool:
            return "module valid_one;" in code

        with patch("calm_puffer_art.sigil_ablation.verify_sigil_code", verifier):
            result = asyncio.run(
                run_sigil_workload_ablation(
                    corpus=corpus,
                    tasks=tasks,
                    learned_bundle=bundle,
                    max_train_steps=1,
                )
            )

        self.assertEqual(result["proof_scope"], SIGIL_WORKLOAD_PROOF_SCOPE)
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["task_buckets"], {SIGIL_BUCKET_EASY: 1})
        self.assertEqual(
            set(result["conditions"]),
            {"static_art", "scheduler_only", "full_trinity"},
        )
        for summary in result["conditions"].values():
            self.assertIn("accounted_north_star", summary)
            self.assertEqual(summary["sigil/distinct_task_count"], 1.0)
            self.assertEqual(summary["sigil/distinct_bucket_count"], 1.0)
        self.assertEqual(
            result["conditions"]["scheduler_only"]["scheduler/expected_rollout_arms"],
            1.0,
        )
        self.assertEqual(
            result["conditions"]["full_trinity"]["scheduler/expected_rollout_arms"],
            5.0,
        )
        self.assertIn(
            "sigil/bucket/sigil_easy/codec/learned/pulls",
            result["conditions"]["full_trinity"],
        )
        self.assertIn("full_trinity_vs_scheduler_accounted_north_star_ratio", result["comparison"])
        self.assertEqual(result["encoder"]["proof_scope"], "sigil_corpus_v0")


if __name__ == "__main__":
    unittest.main()
