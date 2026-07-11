import asyncio
import importlib.util
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "real_art_weight_update.py"


class RealArtWeightUpdateTests(unittest.TestCase):
    def test_preflight_is_offline_and_keeps_train_and_heldout_disjoint(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--preflight", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dataset_valid"])
        self.assertEqual(payload["train_tasks"], 4)
        self.assertEqual(payload["heldout_tasks"], 4)
        self.assertEqual(payload["minimum_planned_inference_requests"], 24)
        self.assertEqual(payload["maximum_planned_inference_requests"], 40)
        self.assertEqual(payload["credential_name"], "WANDB_API_KEY")
        self.assertNotIn("credential_value", payload)

    def test_verifier_uses_last_final_answer_and_gives_exact_full_reward(self):
        namespace = runpy.run_path(str(SCRIPT))
        parse_final = namespace["_parse_final"]
        reward = namespace["_reward"]
        task = namespace["TRAIN_TASKS"][0]

        self.assertEqual(parse_final("scratch FINAL=2\nFINAL=-17"), -17)
        self.assertIsNone(parse_final("17"))
        self.assertEqual(reward(task, task.answer), 1.0)
        self.assertLess(reward(task, task.answer + 1), 1.0)
        self.assertEqual(reward(task, None), 0.0)

    def test_preflight_loads_ignored_env_file_without_exposing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("WANDB_API_KEY=test-secret\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--preflight",
                    "--env-path",
                    str(env_path),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        payload = json.loads(completed.stdout)

        self.assertTrue(payload["credential_ready"])
        self.assertNotIn("test-secret", completed.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("art") is not None,
        "openpipe-art is not installed",
    )
    def test_training_sampler_builds_current_art_groups_with_reward_variance(self):
        import art
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        namespace = runpy.run_path(str(SCRIPT))
        task = namespace["TRAIN_TASKS"][0]
        args = namespace["parse_args"](
            [
                "--train-task-limit",
                "1",
                "--heldout-task-limit",
                "1",
                "--rollouts-per-group",
                "2",
                "--max-rollouts-per-group",
                "2",
            ]
        )
        observed_completions = []
        args.completion_observer = observed_completions.append

        class FakeCompletions:
            def __init__(self):
                self.contents = [
                    f"FINAL={task.answer}",
                    f"FINAL={task.answer + 1}",
                ]

            async def create(self, **kwargs):
                content = self.contents.pop(0)
                return ChatCompletion(
                    id="proof-test",
                    choices=[
                        Choice(
                            index=0,
                            finish_reason="stop",
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=content,
                            ),
                        )
                    ],
                    created=0,
                    model="proof-test",
                    object="chat.completion",
                    usage=CompletionUsage(
                        prompt_tokens=20,
                        completion_tokens=4,
                        total_tokens=24,
                    ),
                )

        completions = FakeCompletions()
        client = type(
            "FakeClient",
            (),
            {"chat": type("FakeChat", (), {"completions": completions})()},
        )()

        groups, report = asyncio.run(
            namespace["_sample_training_groups"](
                art=art,
                client=client,
                inference_name="proof-test",
                policy_step=0,
                tasks=[task],
                args=args,
                semaphore=asyncio.Semaphore(2),
            )
        )

        self.assertEqual(len(groups), 1)
        self.assertIsInstance(groups[0], art.TrajectoryGroup)
        self.assertEqual(report["nonzero_advantage_group_count"], 1)
        self.assertEqual(report["excluded_uniform_reward_groups"], [])
        self.assertEqual(report["requests"], 2)
        self.assertEqual(len(observed_completions), 2)
        self.assertEqual(
            [record.total_tokens for record in observed_completions],
            [24, 24],
        )


if __name__ == "__main__":
    unittest.main()
