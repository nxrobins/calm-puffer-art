import asyncio
import importlib.util
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
    DEFAULT_FOUNDRY_MODEL_CALL_BUDGET,
    DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY,
    DEFAULT_FOUNDRY_TASK_ORDER_POLICY,
    FOUNDRY_COVERAGE_GAP_TASK_IDS,
    FOUNDRY_PROMPT_CONTEXT_POLICIES,
    FOUNDRY_TASK_ORDER_POLICIES,
    PythonRepairTask,
    _repair_prompt,
    _selected_foundry_task_selection,
    _validate_foundry_task_bank,
    available_foundry_corpora,
    available_foundry_task_splits,
    _env_first,
    extract_python_solution,
    load_env_file,
    run_azure_foundry_budget_race,
    run_azure_foundry_codegen_ablation,
    verify_python_solution,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_foundry_cli_module():
    spec = importlib.util.spec_from_file_location(
        "azure_foundry_codegen_ablation_cli",
        ROOT / "examples" / "azure_foundry_codegen_ablation.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load Foundry CLI module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


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
                "AZURE_OPENAI_API_KEY=secret\n"
                "AZURE_OPENAI_ENDPOINT=https://example.invalid\n"
                "# ignored\n",
                encoding="utf-8",
            )
            old_key = os.environ.pop("AZURE_OPENAI_API_KEY", None)
            old_endpoint = os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            try:
                loaded = load_env_file(env_path)
                self.assertEqual(
                    loaded,
                    ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
                )
                self.assertEqual(os.environ["AZURE_OPENAI_API_KEY"], "secret")
            finally:
                if old_key is not None:
                    os.environ["AZURE_OPENAI_API_KEY"] = old_key
                else:
                    os.environ.pop("AZURE_OPENAI_API_KEY", None)
                if old_endpoint is not None:
                    os.environ["AZURE_OPENAI_ENDPOINT"] = old_endpoint
                else:
                    os.environ.pop("AZURE_OPENAI_ENDPOINT", None)

    def test_azure_env_aliases_remain_backward_compatible(self):
        names = ("AZURE_OPENAI_API_KEY", "COVENANT_AZURE_KEY")
        old_values = {name: os.environ.pop(name, None) for name in names}
        try:
            os.environ["COVENANT_AZURE_KEY"] = "legacy"
            self.assertEqual(_env_first(*names), "legacy")

            os.environ["AZURE_OPENAI_API_KEY"] = "standard"
            self.assertEqual(_env_first(*names), "standard")
        finally:
            for name, value in old_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

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

    def test_verify_python_solution_enforces_memory_limit(self):
        task = PythonRepairTask(
            id="mini",
            prompt="Return x plus one.",
            signature="def solve(x)",
            buggy_code="def solve(x):\n    return x\n",
            tests=(((1,), 2),),
        )

        limited = verify_python_solution(
            task,
            "def solve(x):\n    values = [0] * 20_000_000\n    return x + 1\n",
            timeout_s=3.0,
            memory_limit_bytes=96 * 1024 * 1024,
        )

        self.assertFalse(limited.passed)
        self.assertIn(
            limited.failure_mode,
            {
                "exec_error",
                "resource_limit_exceeded",
                "resource_limit_unavailable",
                "unit_test_exception",
                "verifier_crashed",
            },
        )

    def test_foundry_cli_json_reports_missing_env_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.jsonl"
            env = _subprocess_env()
            for name in (
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
                "COVENANT_AZURE_KEY",
                "COVENANT_AZURE_ENDPOINT",
                "COVENANT_AZURE_API_VERSION",
            ):
                env.pop(name, None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/azure_foundry_codegen_ablation.py",
                    "--json",
                    "--env-path",
                    str(Path(directory) / ".env"),
                    "--task-limit",
                    "1",
                    "--train-steps",
                    "1",
                    "--model-call-budget",
                    "0",
                    "--deployment",
                    "dry-run-placeholder",
                    "--telemetry-path",
                    str(telemetry_path),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = json.loads(completed.stdout)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(payload["ok"])
            self.assertEqual(
                payload["error"],
                "azure_foundry_env_missing_required_keys",
            )
            self.assertNotIn("Traceback", completed.stderr)
            telemetry = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in telemetry],
                ["run_started", "run_failed"],
            )
            self.assertEqual(
                telemetry[-1]["error"],
                "azure_foundry_env_missing_required_keys",
            )

    def test_foundry_cli_watchdog_times_out_and_records_telemetry(self):
        module = _load_foundry_cli_module()

        async def slow_run():
            await asyncio.sleep(1.0)
            return {"ok": True, "measurement": "slow"}

        with tempfile.TemporaryDirectory() as directory:
            telemetry = module._TelemetrySink(
                path=Path(directory) / "telemetry.jsonl",
                echo_to_stderr=False,
            )
            with self.assertRaises(module.RunTimeoutError):
                asyncio.run(
                    module._run_with_watchdog(
                        slow_run,
                        telemetry=telemetry,
                        run_timeout_s=0.01,
                        heartbeat_interval_s=0.0,
                        run_metadata={"deployment": "test"},
                    )
                )
            events = [
                json.loads(line)
                for line in telemetry.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [event["event"] for event in events],
            ["run_started", "run_timeout"],
        )
        self.assertEqual(events[-1]["timeout_s"], 0.01)

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

    def test_fake_foundry_budget_race_supports_single_condition_heldout_split(self):
        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            return _FakeClient()

        result = asyncio.run(
            run_azure_foundry_budget_race(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=1,
                    task_limit=1,
                    task_split="standard_heldout",
                    model_call_budget=1,
                    max_completion_tokens=64,
                ),
                budget_dollar_seconds=25.0,
                conditions=("static_art",),
                client_factory=client_factory,
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["task_split"], "standard_heldout")
        self.assertEqual(result["selected_conditions"], ["static_art"])
        self.assertEqual(sorted(result["conditions"]), ["static_art"])
        self.assertEqual(result["heldout_tasks"], 1)
        self.assertEqual(result["heldout"]["task_ids"], ["repair_clamp"])
        heldout_static = result["heldout"]["conditions"]["static_art"]
        self.assertEqual(heldout_static["heldout/passed"], 1.0)
        self.assertGreater(
            heldout_static[
                "north_star/accounted_published_policy_reward_improving_experience_per_dollar_second"
            ],
            0.0,
        )

    def test_frontier_task_splits_have_expected_sizes_and_metadata(self):
        self.assertIn("frontier_ladder_v1", available_foundry_corpora())
        self.assertIn("frontier_full", available_foundry_task_splits())

        expected_sizes = {
            "frontier_smoke": 8,
            "frontier_balanced": 24,
            "frontier_hard": 32,
            "frontier_full": 40,
        }
        for split, expected_size in expected_sizes.items():
            selection = _selected_foundry_task_selection(999, split)
            self.assertEqual(len(selection.train), expected_size)
            self.assertEqual(len(selection.heldout), expected_size)
            self.assertEqual(
                {task.id for task in selection.train},
                {task.id for task in selection.heldout},
            )
            self.assertTrue(all(task.family != "general" for task in selection.train))
            self.assertTrue(all(1 <= task.difficulty <= 5 for task in selection.train))
            self.assertTrue(all(task.failure_tags for task in selection.train))

        smoke = _selected_foundry_task_selection(999, "frontier_smoke")
        self.assertEqual(
            {task.family for task in smoke.train},
            {
                "sequence",
                "string_parse",
                "interval",
                "state_machine",
                "graph",
                "data_model",
                "numeric",
                "real_bug_pattern",
            },
        )

    def test_task_order_policy_reorders_coverage_gaps_without_changing_split(self):
        self.assertEqual(DEFAULT_FOUNDRY_TASK_ORDER_POLICY, "split_order")
        self.assertIn("coverage_gap_first", FOUNDRY_TASK_ORDER_POLICIES)

        default_selection = _selected_foundry_task_selection(
            999,
            "frontier_hard",
        )
        reordered_selection = _selected_foundry_task_selection(
            999,
            "frontier_hard",
            "coverage_gap_first",
        )
        target_ids = [
            task_id
            for task_id in FOUNDRY_COVERAGE_GAP_TASK_IDS
            if task_id in {task.id for task in default_selection.train}
        ]

        self.assertEqual(
            {task.id for task in reordered_selection.train},
            {task.id for task in default_selection.train},
        )
        self.assertEqual(
            [task.id for task in reordered_selection.train[: len(target_ids)]],
            target_ids,
        )
        self.assertNotEqual(
            [task.id for task in default_selection.train[: len(target_ids)]],
            target_ids,
        )
        self.assertEqual(
            {task.id for task in reordered_selection.heldout},
            {task.id for task in reordered_selection.train},
        )

    def test_frontier_invalid_split_and_duplicate_task_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "task_split"):
            AzureFoundryCodegenConfig(task_split="missing").validate()
        with self.assertRaisesRegex(ValueError, "prompt_context_policy"):
            AzureFoundryCodegenConfig(prompt_context_policy="missing").validate()
        with self.assertRaisesRegex(ValueError, "task_order_policy"):
            AzureFoundryCodegenConfig(task_order_policy="missing").validate()
        with self.assertRaisesRegex(ValueError, "task_order_policy"):
            _selected_foundry_task_selection(1, "frontier_hard", "missing")

        task = PythonRepairTask(
            id="duplicate",
            prompt="Return x.",
            signature="def solve(x)",
            buggy_code="def solve(x):\n    return None\n",
            tests=(((1,), 1),),
            family="sequence",
            difficulty=1,
            failure_tags=("ordering",),
        )
        with self.assertRaisesRegex(ValueError, "duplicate_train_ids"):
            _validate_foundry_task_bank(
                "bad",
                train_tasks=(task, task),
                heldout_tasks=(),
            )

    def test_prompt_context_policy_adds_metadata_and_data_model_guardrails(self):
        task = PythonRepairTask(
            id="repair_nested_defaults",
            prompt="Deep-fill missing dictionary keys from defaults.",
            signature="def solve(data, defaults)",
            buggy_code="def solve(data, defaults):\n    return data\n",
            tests=((({}, {"a": 1}), {"a": 1}),),
            family="data_model",
            difficulty=4,
            failure_tags=("mutation", "aliasing"),
        )

        base_prompt = _repair_prompt(task)
        metadata_prompt = _repair_prompt(
            task,
            prompt_context_policy="task_metadata",
        )
        guardrail_prompt = _repair_prompt(
            task,
            prompt_context_policy="data_model_guardrails",
        )
        tag_guardrail_prompt = _repair_prompt(
            task,
            prompt_context_policy="failure_tag_guardrails",
        )
        sequence_prompt = _repair_prompt(
            PythonRepairTask(
                id="repair_rotate_left",
                prompt="Rotate a list left.",
                signature="def solve(items, n)",
                buggy_code="def solve(items, n):\n    return items\n",
                tests=((([1, 2, 3], 1), [2, 3, 1]),),
                family="sequence",
                difficulty=2,
                failure_tags=("off_by_one",),
            ),
            prompt_context_policy="data_model_guardrails",
        )

        self.assertEqual(DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY, "repair_prompt_only")
        self.assertIn("data_model_guardrails", FOUNDRY_PROMPT_CONTEXT_POLICIES)
        self.assertIn("failure_tag_guardrails", FOUNDRY_PROMPT_CONTEXT_POLICIES)
        self.assertNotIn("Context:", base_prompt)
        self.assertIn("- family: data_model", metadata_prompt)
        self.assertIn("- failure tags: mutation, aliasing", metadata_prompt)
        self.assertIn("deep-copy nested defaults", guardrail_prompt)
        self.assertIn("deterministic input/schema order", guardrail_prompt)
        self.assertIn("- failure-tag guardrails:", tag_guardrail_prompt)
        self.assertIn("mutation: do not mutate caller-owned inputs", tag_guardrail_prompt)
        self.assertIn("aliasing: avoid sharing mutable nested containers", tag_guardrail_prompt)
        self.assertNotIn("deep-copy nested defaults", tag_guardrail_prompt)
        self.assertNotIn("Context:", sequence_prompt)

    def test_fake_foundry_result_records_prompt_context_policy(self):
        fake_clients: list[_FakeClient] = []

        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            del name
            self.assertEqual(config.prompt_context_policy, "data_model_guardrails")
            client = _FakeClient()
            fake_clients.append(client)
            return client

        result = asyncio.run(
            run_azure_foundry_budget_race(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=1,
                    task_limit=1,
                    task_split="frontier_hard",
                    model_call_budget=1,
                    max_completion_tokens=64,
                    prompt_context_policy="data_model_guardrails",
                ),
                budget_dollar_seconds=40.0,
                conditions=("static_art",),
                client_factory=client_factory,
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["prompt_context_policy"], "data_model_guardrails")
        prompt = fake_clients[0].completions.calls[0]["messages"][1]["content"]
        self.assertIn("Repair this Python function.", prompt)

    def test_fake_foundry_result_records_task_order_policy(self):
        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            del name
            self.assertEqual(config.task_order_policy, "coverage_gap_first")
            return _FakeClient()

        result = asyncio.run(
            run_azure_foundry_budget_race(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=1,
                    task_limit=4,
                    task_split="frontier_hard",
                    task_order_policy="coverage_gap_first",
                    model_call_budget=4,
                    max_completion_tokens=64,
                ),
                budget_dollar_seconds=40.0,
                conditions=("static_art",),
                client_factory=client_factory,
            )
        )

        target_ids = [
            task_id
            for task_id in FOUNDRY_COVERAGE_GAP_TASK_IDS
            if task_id in result["train_task_ids"]
        ]
        self.assertEqual(result["task_order_policy"], "coverage_gap_first")
        self.assertEqual(result["train_task_ids"][: len(target_ids)], target_ids)

    def test_fake_foundry_frontier_smoke_reports_coverage_and_non_saturation(self):
        def client_factory(name: str, config: AzureFoundryCodegenConfig):
            return _FakeClient()

        result = asyncio.run(
            run_azure_foundry_budget_race(
                config=AzureFoundryCodegenConfig(
                    max_train_steps=2,
                    task_limit=8,
                    task_split="frontier_smoke",
                    model_call_budget=8,
                    max_completion_tokens=64,
                ),
                budget_dollar_seconds=80.0,
                conditions=("static_art",),
                client_factory=client_factory,
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["task_split"], "frontier_smoke")
        self.assertEqual(result["task_coverage"]["train"]["tasks"], 8)
        self.assertEqual(
            len(result["task_coverage"]["train"]["families"]),
            8,
        )
        saturation = result["non_saturation"]["conditions"]["static_art"]
        self.assertLess(saturation["learned_fraction"], 1.0)
        self.assertFalse(saturation["saturated"])
        heldout_static = result["heldout"]["conditions"]["static_art"]
        self.assertIn("heldout/by_family", heldout_static)
        self.assertIn("heldout/by_difficulty", heldout_static)

    def test_frontier_hard_is_larger_than_default_model_call_budget(self):
        selection = _selected_foundry_task_selection(999, "frontier_hard")

        self.assertGreater(len(selection.train), DEFAULT_FOUNDRY_MODEL_CALL_BUDGET)
        self.assertEqual(len(selection.train), 32)


if __name__ == "__main__":
    unittest.main()
