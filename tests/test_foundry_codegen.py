import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calm_puffer_art.foundry_codegen import (
    AzureFoundryCodegenConfig,
    PythonRepairTask,
    extract_python_solution,
    load_env_file,
    run_azure_foundry_budget_race,
    run_azure_foundry_codegen_ablation,
    verify_python_solution,
)


ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = """
```python
def solve(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value
```
"""
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=17,
                total_tokens=28,
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class FoundryCodegenTests(unittest.TestCase):
    def test_foundry_optional_extra_is_declared(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('foundry = ["openai>=1.0"]', pyproject)

    def test_core_import_does_not_import_openai_or_heavy_model_packages(self):
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "before=set(sys.modules); "
                "import calm_puffer_art; "
                "loaded=set(sys.modules)-before; "
                "forbidden={'openai','torch','art','vllm','transformers','datasets'}; "
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

    def test_load_env_file_sets_missing_keys_without_returning_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "COVENANT_AZURE_KEY=secret\n"
                "COVENANT_AZURE_ENDPOINT=https://example.invalid\n"
                "# ignored\n",
                encoding="utf-8",
            )
            old_key = os.environ.pop("COVENANT_AZURE_KEY", None)
            old_endpoint = os.environ.pop("COVENANT_AZURE_ENDPOINT", None)
            try:
                loaded = load_env_file(env_path)
                self.assertEqual(
                    loaded,
                    ("COVENANT_AZURE_KEY", "COVENANT_AZURE_ENDPOINT"),
                )
                self.assertEqual(os.environ["COVENANT_AZURE_KEY"], "secret")
            finally:
                if old_key is not None:
                    os.environ["COVENANT_AZURE_KEY"] = old_key
                else:
                    os.environ.pop("COVENANT_AZURE_KEY", None)
                if old_endpoint is not None:
                    os.environ["COVENANT_AZURE_ENDPOINT"] = old_endpoint
                else:
                    os.environ.pop("COVENANT_AZURE_ENDPOINT", None)

    def test_extract_python_solution_accepts_fenced_or_plain_code(self):
        fenced = "Here:\n```python\ndef solve(x):\n    return x\n```"
        plain = "notes\n\ndef solve(x):\n    return x + 1\n"

        self.assertEqual(extract_python_solution(fenced), "def solve(x):\n    return x")
        self.assertEqual(
            extract_python_solution(plain),
            "def solve(x):\n    return x + 1",
        )

    def test_verify_python_solution_passes_fails_and_times_out(self):
        task = PythonRepairTask(
            id="mini",
            prompt="Return x plus one.",
            signature="def solve(x)",
            buggy_code="def solve(x):\n    return x\n",
            tests=(((1,), 2), ((4,), 5)),
        )

        passed = verify_python_solution(
            task,
            "def solve(x):\n    return x + 1\n",
        )
        failed = verify_python_solution(
            task,
            "def solve(x):\n    return x\n",
        )
        timed_out = verify_python_solution(
            task,
            "def solve(x):\n    while True:\n        pass\n",
            timeout_s=0.2,
        )

        self.assertTrue(passed.passed)
        self.assertEqual(passed.failure_mode, "passed")
        self.assertFalse(failed.passed)
        self.assertEqual(failed.failure_mode, "unit_test_failed")
        self.assertFalse(timed_out.passed)
        self.assertEqual(timed_out.failure_mode, "timeout")

    def test_fake_foundry_ablation_reports_live_contract_shape(self):
        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            return _FakeClient()

        result = asyncio.run(
            run_azure_foundry_codegen_ablation(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=1,
                    task_limit=1,
                    model_call_budget=1,
                    max_completion_tokens=64,
                ),
                client_factory=client_factory,
            )
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["used_azure_foundry"])
        self.assertEqual(result["proof_scope"], "live_azure_foundry_python_repair")
        self.assertEqual(result["measurement"], "azure_foundry_codegen_ablation")
        self.assertEqual(
            sorted(result["conditions"]),
            ["full_trinity", "scheduler_only", "static_art"],
        )
        self.assertIn("winning_condition_by_accounted_north_star", result)
        for condition in result["conditions"].values():
            self.assertIn(
                "north_star/accounted_published_policy_reward_improving_experience_per_dollar_second",
                condition,
            )
            self.assertGreaterEqual(condition["foundry/model_calls"], 0.0)
            self.assertIn("foundry/codec/token/pulls", condition)
            self.assertIn("foundry/codec/chunk2/pulls", condition)
            self.assertIn("foundry/codec/chunk4/pulls", condition)

    def test_fake_foundry_budget_race_reports_performance_and_cost_contract(self):
        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            return _FakeClient()

        result = asyncio.run(
            run_azure_foundry_budget_race(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=2,
                    task_limit=1,
                    model_call_budget=2,
                    max_completion_tokens=64,
                ),
                budget_dollar_seconds=25.0,
                client_factory=client_factory,
            )
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["used_azure_foundry"])
        self.assertEqual(result["measurement"], "azure_foundry_budget_race")
        self.assertEqual(result["budget_dollar_seconds"], 25.0)
        self.assertIn("race", result)
        self.assertIn("performance_winner_by_learned_solutions", result["race"])
        self.assertIn("cost_winner_by_accounted_dollar_seconds", result["race"])
        self.assertIn(
            "efficiency_winner_by_learned_solutions_per_dollar_second",
            result["race"],
        )
        for condition in result["conditions"].values():
            self.assertIn("scheduler/budget/max_accounted_dollar_seconds", condition)
            self.assertIn("foundry/learned_solutions", condition)


if __name__ == "__main__":
    unittest.main()
