from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

from .actions import (
    ActionCodec,
    AdaptiveActionSpace,
    ChunkActionCodec,
    TokenActionCodec,
    action_codec_key,
    safe_metric_key,
)
from .codegen_ablation import CODEGEN_ACCOUNTED_NORTH_STAR, CODEGEN_NORTH_STAR
from .runtime import ControlPlane, ControlPlaneConfig, RolloutContext
from .scheduler import (
    ActorStats,
    ObjectiveScheduler,
    SchedulerDecision,
    _decision_to_state,
)
from .types import (
    Message,
    PolicySnapshot,
    RunSummary,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


DEFAULT_FOUNDRY_ENV_PATH = Path(".env")
DEFAULT_FOUNDRY_DEPLOYMENT = "gpt-5.5"
DEFAULT_FOUNDRY_API_VERSION = "2025-04-01-preview"
DEFAULT_FOUNDRY_TRAIN_STEPS = 2
DEFAULT_FOUNDRY_TASK_LIMIT = 2
DEFAULT_FOUNDRY_MODEL_CALL_BUDGET = 4
DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS = 700
DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S = 120.0
DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S = 3.0
DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS = 2.0
DEFAULT_FOUNDRY_MEMORY_DOLLAR_SECONDS = 0.05
DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS = 0.04
DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS = 150.0
DEFAULT_FOUNDRY_TASK_SPLIT = "standard"
DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY = "repair_prompt_only"
DEFAULT_FOUNDRY_TASK_ORDER_POLICY = "split_order"
FOUNDRY_PROMPT_CONTEXT_POLICIES = (
    "repair_prompt_only",
    "task_metadata",
    "data_model_guardrails",
    "failure_tag_guardrails",
)
FOUNDRY_TASK_ORDER_POLICIES = (
    "split_order",
    "coverage_gap_first",
    "lift_pocket_first",
)
FOUNDRY_COVERAGE_GAP_TASK_IDS = (
    "repair_nested_defaults",
    "repair_schema_errors",
    "repair_schedule_conflicts",
    "repair_query_params",
    "repair_topological_layers",
    "repair_lru_cache_trace",
)
FOUNDRY_LIFT_POCKET_TASK_IDS = (
    "repair_schedule_conflicts",
    "repair_query_params",
)
DEFAULT_FOUNDRY_CONDITIONS = ("static_art", "scheduler_only", "full_trinity")
FOUNDRY_CONDITIONS = (
    "static_art",
    "scheduler_only",
    "chunk2_only",
    "chunk4_only",
    "full_trinity",
    "full_trinity_patient_demote",
    "full_trinity_no_demote",
)
FOUNDRY_TASK_FAMILIES = (
    "sequence",
    "string_parse",
    "interval",
    "state_machine",
    "graph",
    "data_model",
    "numeric",
    "real_bug_pattern",
    "general",
)
FOUNDRY_FAILURE_TAG_GUARDRAILS = {
    "aliasing": "avoid sharing mutable nested containers between inputs and outputs",
    "boundary": "check empty, first, last, inclusive, and exclusive boundary cases",
    "dedupe": "preserve the required representative and order when removing duplicates",
    "dependency_cycle": "separate acyclic traversal from cycle detection and reporting",
    "edge_empty": "handle empty inputs before indexing, popping, or taking extrema",
    "modulo": "normalize rotations and wraparound with modulo after empty checks",
    "mutation": "do not mutate caller-owned inputs unless the task explicitly requires it",
    "none_sentinel": "distinguish missing values, None, and other falsy values",
    "off_by_one": "check loop ranges, page boundaries, and inclusive/exclusive endpoints",
    "ordering": "preserve deterministic input order unless the task requires sorting",
    "parser_escape": "parse quoted or escaped separators before splitting on delimiters",
    "parser_nesting": "track nesting depth before treating delimiters as structural",
    "rounding": "round only at the required final precision boundary",
    "stable_sort": "preserve original tie order when sort keys compare equal",
    "state_eviction": "update recency or eviction state on every read and write transition",
    "state_transition": "make each state transition explicit before updating stored state",
    "truthiness": "test for None or missing values explicitly instead of generic truthiness",
    "unicode": "normalize text boundaries without dropping non-ASCII semantic content",
}


@dataclass(frozen=True)
class PythonRepairTask:
    id: str
    prompt: str
    signature: str
    buggy_code: str
    tests: tuple[tuple[tuple[Any, ...], Any], ...]
    family: str = "general"
    difficulty: int = 1
    failure_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FoundryTaskSelection:
    split: str
    train: tuple[PythonRepairTask, ...]
    heldout: tuple[PythonRepairTask, ...]


@dataclass(frozen=True)
class FoundryCorpusSpec:
    name: str
    description: str
    source: str
    train_tasks: tuple[PythonRepairTask, ...]
    heldout_tasks: tuple[PythonRepairTask, ...] = ()
    external_adapter: bool = False


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    failure_mode: str
    tests_passed: int = 0
    tests_total: int = 0


@dataclass(frozen=True)
class FoundryGeneration:
    code: str
    raw_text: str
    source: str
    model_called: bool
    duration_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    failure_mode: str | None = None


@dataclass(frozen=True)
class AzureFoundryCodegenConfig:
    env_path: Path = DEFAULT_FOUNDRY_ENV_PATH
    deployment: str = DEFAULT_FOUNDRY_DEPLOYMENT
    api_version: str = DEFAULT_FOUNDRY_API_VERSION
    max_train_steps: int = DEFAULT_FOUNDRY_TRAIN_STEPS
    task_limit: int = DEFAULT_FOUNDRY_TASK_LIMIT
    model_call_budget: int = DEFAULT_FOUNDRY_MODEL_CALL_BUDGET
    max_completion_tokens: int = DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS
    request_timeout_s: float = DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S
    verify_timeout_s: float = DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S
    verify_memory_limit_bytes: int = DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES
    request_dollar_seconds: float = DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS
    memory_dollar_seconds: float = DEFAULT_FOUNDRY_MEMORY_DOLLAR_SECONDS
    action_unit_dollar_seconds: float = DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS
    trainer_dollar_seconds: float = 0.35
    task_split: str = DEFAULT_FOUNDRY_TASK_SPLIT
    prompt_context_policy: str = DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY
    task_order_policy: str = DEFAULT_FOUNDRY_TASK_ORDER_POLICY

    def validate(self) -> None:
        if self.max_train_steps < 1:
            raise ValueError("foundry_max_train_steps_must_be_positive")
        if self.task_limit < 1:
            raise ValueError("foundry_task_limit_must_be_positive")
        if self.model_call_budget < 0:
            raise ValueError("foundry_model_call_budget_must_be_non_negative")
        if self.max_completion_tokens < 1:
            raise ValueError("foundry_max_completion_tokens_must_be_positive")
        if self.verify_memory_limit_bytes < 1:
            raise ValueError("foundry_verify_memory_limit_bytes_must_be_positive")
        for name in (
            "request_timeout_s",
            "verify_timeout_s",
            "request_dollar_seconds",
            "memory_dollar_seconds",
            "action_unit_dollar_seconds",
            "trainer_dollar_seconds",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name}_must_be_finite_non_negative")
        if self.task_split not in _FOUNDRY_TASK_SPLITS:
            raise ValueError("foundry_task_split_unknown")
        if self.prompt_context_policy not in FOUNDRY_PROMPT_CONTEXT_POLICIES:
            raise ValueError("foundry_prompt_context_policy_unknown")
        if self.task_order_policy not in FOUNDRY_TASK_ORDER_POLICIES:
            raise ValueError("foundry_task_order_policy_unknown")


class AzureFoundryCodegenPolicy:
    def __init__(
        self,
        *,
        client: Any,
        config: AzureFoundryCodegenConfig,
        learned_solutions: Mapping[str, str] | None = None,
        model_calls: int = 0,
    ) -> None:
        self.client = client
        self.config = config
        self.learned_solutions = dict(learned_solutions or {})
        self.model_calls = int(model_calls)

    def clone(self) -> "AzureFoundryCodegenPolicy":
        return AzureFoundryCodegenPolicy(
            client=self.client,
            config=self.config,
            learned_solutions=self.learned_solutions,
            model_calls=self.model_calls,
        )

    async def generate(self, task: PythonRepairTask) -> FoundryGeneration:
        if task.id in self.learned_solutions:
            code = self.learned_solutions[task.id]
            return FoundryGeneration(
                code=code,
                raw_text=code,
                source="policy_memory",
                model_called=False,
                duration_s=0.0,
            )
        if self.model_calls >= self.config.model_call_budget:
            code = _fallback_solution(task)
            return FoundryGeneration(
                code=code,
                raw_text=code,
                source="model_call_budget_exhausted",
                model_called=False,
                duration_s=0.0,
                failure_mode="model_call_budget_exhausted",
            )

        self.model_calls += 1
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**self._request_kwargs(task)),
                timeout=self.config.request_timeout_s,
            )
        except Exception as exc:
            duration_s = time.perf_counter() - started
            code = _fallback_solution(task)
            return FoundryGeneration(
                code=code,
                raw_text="",
                source="azure_foundry_error",
                model_called=True,
                duration_s=duration_s,
                failure_mode=f"{type(exc).__name__}: {exc}",
            )
        duration_s = time.perf_counter() - started
        raw_text = _response_text(response)
        usage = _response_usage(response)
        return FoundryGeneration(
            code=extract_python_solution(raw_text),
            raw_text=raw_text,
            source="azure_foundry",
            model_called=True,
            duration_s=duration_s,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _request_kwargs(self, task: PythonRepairTask) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You repair small Python functions. Return only Python "
                        "code defining solve with the requested signature. Do not "
                        "return markdown, prose, imports, file IO, networking, "
                        "classes, or top-level test code."
                    ),
                },
                {
                    "role": "user",
                    "content": _repair_prompt(
                        task,
                        prompt_context_policy=self.config.prompt_context_policy,
                    ),
                },
            ],
        }
        if self.config.deployment.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = self.config.max_completion_tokens
        else:
            kwargs["max_tokens"] = self.config.max_completion_tokens
            kwargs["temperature"] = 0.2
        return kwargs


class AzureFoundryCodegenTrainer:
    def __init__(
        self,
        *,
        tasks: Sequence[PythonRepairTask],
        trainer_dollar_seconds: float,
    ) -> None:
        self.tasks = tuple(tasks)
        self.trainer_dollar_seconds = trainer_dollar_seconds

    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        current_policy = current.policy
        if not isinstance(current_policy, AzureFoundryCodegenPolicy):
            raise TypeError("azure_foundry_codegen_policy_required")
        next_policy = current_policy.clone()
        trajectories = [
            trajectory
            for group in groups
            for trajectory in group.trajectories
        ]
        for trajectory in trajectories:
            if trajectory.reward <= 0.0:
                continue
            task_id = str(trajectory.metadata.get("foundry/task_id", ""))
            code = str(trajectory.metadata.get("foundry/code", ""))
            if task_id and code:
                next_policy.learned_solutions[task_id] = code
        rewards = [trajectory.reward for trajectory in trajectories]
        memory_coverage = len(next_policy.learned_solutions) / max(1, len(self.tasks))
        model_calls = sum(
            1.0
            for trajectory in trajectories
            if trajectory.metadata.get("foundry/model_called") is True
        )
        verifier_passed = sum(1.0 for trajectory in trajectories if trajectory.reward > 0.0)
        prompt_tokens = sum(
            _metadata_float(trajectory.metadata, "foundry/prompt_tokens")
            for trajectory in trajectories
        )
        completion_tokens = sum(
            _metadata_float(trajectory.metadata, "foundry/completion_tokens")
            for trajectory in trajectories
        )
        source_model = sum(
            1.0
            for trajectory in trajectories
            if trajectory.metadata.get("foundry/source") == "azure_foundry"
        )
        source_memory = sum(
            1.0
            for trajectory in trajectories
            if trajectory.metadata.get("foundry/source") == "policy_memory"
        )
        return TrainResult(
            policy=next_policy,
            checkpoint_id=f"azure-foundry-codegen-step-{current.step + 1}",
            metrics={
                "train/reward": memory_coverage,
                "train/batch_reward": fmean(rewards) if rewards else 0.0,
                "train/dollar_seconds": self.trainer_dollar_seconds,
                "foundry/train_examples": float(len(trajectories)),
                "foundry/learned_solutions": float(len(next_policy.learned_solutions)),
                "foundry/model_calls_used": float(next_policy.model_calls),
                "foundry/batch_model_calls": model_calls,
                "foundry/batch_verifier_passed_rollouts": verifier_passed,
                "foundry/batch_rollouts": float(len(trajectories)),
                "foundry/batch_prompt_tokens": prompt_tokens,
                "foundry/batch_completion_tokens": completion_tokens,
                "foundry/batch_source_model_rollouts": source_model,
                "foundry/batch_source_memory_rollouts": source_memory,
            },
            metadata={
                "foundry/learned_solution_ids": sorted(next_policy.learned_solutions),
                "foundry/learned_solutions": dict(next_policy.learned_solutions),
            },
        )


async def azure_foundry_codegen_rollout(
    policy: AzureFoundryCodegenPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    task = _foundry_task_for_scenario(scenario)
    generation = await policy.generate(task)
    verification = await asyncio.to_thread(
        verify_python_solution,
        task,
        generation.code,
        timeout_s=policy.config.verify_timeout_s,
        memory_limit_bytes=policy.config.verify_memory_limit_bytes,
    )
    actions = context.action_codec.encode(generation.code)
    for action in actions:
        action.metadata.setdefault("foundry/task_id", task.id)
        action.metadata.setdefault("foundry/source", generation.source)
    rollout_cost = _foundry_rollout_dollar_seconds(
        config=policy.config,
        generation=generation,
        action_units=len(actions),
    )
    metadata = {
        **dict(context.decision_metadata),
        "scenario_id": scenario.id,
        "foundry/workload": "azure_foundry_python_repair_unit_tests",
        "foundry/task_id": task.id,
        "foundry/signature": task.signature,
        "foundry/source": generation.source,
        "foundry/model_called": generation.model_called,
        "foundry/model_calls_used": policy.model_calls,
        "foundry/prompt_tokens": generation.prompt_tokens,
        "foundry/completion_tokens": generation.completion_tokens,
        "foundry/total_tokens": generation.total_tokens,
        "foundry/code": generation.code,
        "foundry/raw_text_bytes": len(generation.raw_text.encode("utf-8")),
        "verifier/passed": verification.passed,
        "verifier/tests_passed": verification.tests_passed,
        "verifier/tests_total": verification.tests_total,
        "action/safe": True,
        "reconstruction/accuracy": 1.0,
        "reconstruction/safe": True,
    }
    if not verification.passed:
        metadata["verifier/failure_mode"] = verification.failure_mode
    if generation.failure_mode:
        metadata["foundry/failure_mode"] = generation.failure_mode
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=[
            Message(
                role="user",
                content=_repair_prompt(
                    task,
                    prompt_context_policy=policy.config.prompt_context_policy,
                ),
            )
        ],
        actions=actions,
        reward=1.0 if verification.passed else 0.0,
        metrics={
            "rollout/dollar_seconds": rollout_cost,
            "foundry/request_duration_s": generation.duration_s,
        },
        metadata=metadata,
    )


async def run_azure_foundry_codegen_ablation(
    *,
    config: AzureFoundryCodegenConfig | None = None,
    conditions: Sequence[str] | None = None,
    client_factory: Callable[[str, AzureFoundryCodegenConfig], Any] | None = None,
) -> dict[str, Any]:
    resolved = config or AzureFoundryCodegenConfig()
    resolved.validate()
    selection = _selected_foundry_task_selection(
        resolved.task_limit,
        resolved.task_split,
        resolved.task_order_policy,
    )
    selected_conditions = _normalize_foundry_conditions(conditions)
    summaries: dict[str, RunSummary] = {}
    for name in selected_conditions:
        summaries[name] = await _run_foundry_named_condition(
            name=name,
            config=resolved,
            tasks=selection.train,
            budget_dollar_seconds=None,
            client_factory=client_factory,
        )

    condition_metrics = {
        name: _foundry_summary_metrics(summary, tasks=selection.train)
        for name, summary in summaries.items()
    }
    payload: dict[str, Any] = {
        "ok": True,
        "proof_scope": "live_azure_foundry_python_repair",
        "measurement": "azure_foundry_codegen_ablation",
        "used_azure_foundry": client_factory is None,
        "deployment": resolved.deployment,
        "api_version": resolved.api_version,
        "env_path": str(resolved.env_path),
        "task_split": selection.split,
        "prompt_context_policy": resolved.prompt_context_policy,
        "task_order_policy": resolved.task_order_policy,
        "tasks": len(selection.train),
        "heldout_tasks": len(selection.heldout),
        "train_task_ids": [task.id for task in selection.train],
        "heldout_task_ids": [task.id for task in selection.heldout],
        "task_coverage": _foundry_task_coverage_payload(selection),
        "selected_conditions": list(selected_conditions),
        "max_train_steps": resolved.max_train_steps,
        "model_call_budget_per_condition": resolved.model_call_budget,
        "action_unit_dollar_seconds": resolved.action_unit_dollar_seconds,
        "request_dollar_seconds": resolved.request_dollar_seconds,
        "conditions": condition_metrics,
        "lift": _foundry_lift_metrics(condition_metrics),
        "winning_condition_by_accounted_north_star": max(
            condition_metrics,
            key=lambda name: condition_metrics[name].get(
                CODEGEN_ACCOUNTED_NORTH_STAR,
                0.0,
            ),
        ),
    }
    heldout = _foundry_heldout_payload(
        summaries,
        heldout_tasks=selection.heldout,
        config=resolved,
    )
    if heldout is not None:
        payload["heldout"] = heldout
    payload["non_saturation"] = _foundry_non_saturation_payload(
        condition_metrics,
        heldout,
        task_count=len(selection.train),
        heldout_task_count=len(selection.heldout),
    )
    return payload


async def run_azure_foundry_budget_race(
    *,
    config: AzureFoundryCodegenConfig | None = None,
    budget_dollar_seconds: float = DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS,
    conditions: Sequence[str] | None = None,
    client_factory: Callable[[str, AzureFoundryCodegenConfig], Any] | None = None,
) -> dict[str, Any]:
    resolved = config or AzureFoundryCodegenConfig()
    resolved.validate()
    if not isfinite(budget_dollar_seconds) or budget_dollar_seconds <= 0.0:
        raise ValueError("foundry_budget_dollar_seconds_must_be_positive")
    selection = _selected_foundry_task_selection(
        resolved.task_limit,
        resolved.task_split,
        resolved.task_order_policy,
    )
    selected_conditions = _normalize_foundry_conditions(conditions)
    summaries: dict[str, RunSummary] = {}
    for name in selected_conditions:
        summaries[name] = await _run_foundry_named_condition(
            name=name,
            config=resolved,
            tasks=selection.train,
            budget_dollar_seconds=budget_dollar_seconds,
            client_factory=client_factory,
        )

    condition_metrics = {
        name: _foundry_summary_metrics(summary, tasks=selection.train)
        for name, summary in summaries.items()
    }
    race = _foundry_budget_race_metrics(condition_metrics)
    payload: dict[str, Any] = {
        "ok": True,
        "proof_scope": "live_azure_foundry_python_repair",
        "measurement": "azure_foundry_budget_race",
        "used_azure_foundry": client_factory is None,
        "deployment": resolved.deployment,
        "api_version": resolved.api_version,
        "env_path": str(resolved.env_path),
        "task_split": selection.split,
        "prompt_context_policy": resolved.prompt_context_policy,
        "task_order_policy": resolved.task_order_policy,
        "tasks": len(selection.train),
        "heldout_tasks": len(selection.heldout),
        "train_task_ids": [task.id for task in selection.train],
        "heldout_task_ids": [task.id for task in selection.heldout],
        "task_coverage": _foundry_task_coverage_payload(selection),
        "selected_conditions": list(selected_conditions),
        "max_train_steps": resolved.max_train_steps,
        "budget_dollar_seconds": budget_dollar_seconds,
        "model_call_budget_per_condition": resolved.model_call_budget,
        "action_unit_dollar_seconds": resolved.action_unit_dollar_seconds,
        "request_dollar_seconds": resolved.request_dollar_seconds,
        "conditions": condition_metrics,
        "race": race,
        "winning_condition_by_accounted_north_star": max(
            condition_metrics,
            key=lambda name: condition_metrics[name].get(
                CODEGEN_ACCOUNTED_NORTH_STAR,
                0.0,
            ),
        ),
    }
    heldout = _foundry_heldout_payload(
        summaries,
        heldout_tasks=selection.heldout,
        config=resolved,
    )
    if heldout is not None:
        payload["heldout"] = heldout
    payload["non_saturation"] = _foundry_non_saturation_payload(
        condition_metrics,
        heldout,
        task_count=len(selection.train),
        heldout_task_count=len(selection.heldout),
    )
    return payload


def create_azure_foundry_client(config: AzureFoundryCodegenConfig) -> Any:
    load_env_file(config.env_path)
    key = _env_first("AZURE_OPENAI_API_KEY", "COVENANT_AZURE_KEY")
    endpoint = _env_first("AZURE_OPENAI_ENDPOINT", "COVENANT_AZURE_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError("azure_foundry_env_missing_required_keys")
    try:
        from openai import AsyncAzureOpenAI
    except ImportError as exc:
        raise RuntimeError("azure_foundry_openai_extra_not_installed") from exc
    return AsyncAzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=_env_first(
            "AZURE_OPENAI_API_VERSION",
            "COVENANT_AZURE_API_VERSION",
        )
        or config.api_version,
        timeout=config.request_timeout_s,
    )


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_env_file(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        loaded.append(key)
    return tuple(loaded)


def extract_python_solution(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for index in range(1, len(parts), 2):
            block = parts[index].strip()
            if block.startswith("python"):
                block = block[len("python") :].strip()
            if "def solve" in block:
                return _trim_to_solve(block)
    if "def solve" in stripped:
        return _trim_to_solve(stripped)
    return stripped


def verify_python_solution(
    task: PythonRepairTask,
    code: str,
    *,
    timeout_s: float = DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S,
    memory_limit_bytes: int = DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES,
) -> VerificationResult:
    payload = {
        "code": code,
        "memory_limit_bytes": memory_limit_bytes,
        "tests": [
            {"args": list(args), "expected": expected}
            for args, expected in task.tests
        ],
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _VERIFY_SCRIPT],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env=_verifier_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            passed=False,
            failure_mode="timeout",
            tests_total=len(task.tests),
        )
    if completed.returncode != 0:
        failure_mode = (
            "resource_limit_exceeded"
            if memory_limit_bytes > 0 and not completed.stderr.strip()
            else "verifier_crashed"
        )
        return VerificationResult(
            passed=False,
            failure_mode=failure_mode,
            tests_total=len(task.tests),
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return VerificationResult(
            passed=False,
            failure_mode="invalid_verifier_output",
            tests_total=len(task.tests),
        )
    return VerificationResult(
        passed=bool(result.get("passed")),
        failure_mode=str(result.get("failure_mode", "unknown")),
        tests_passed=int(result.get("tests_passed", 0)),
        tests_total=int(result.get("tests_total", len(task.tests))),
    )


def _verifier_subprocess_env() -> dict[str, str]:
    allowed_names = (
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    )
    child_env = {
        name: value
        for name in allowed_names
        if (value := os.environ.get(name)) is not None
    }
    child_env["PYTHONHASHSEED"] = "0"
    child_env["PYTHONIOENCODING"] = "utf-8"
    return child_env


async def _run_foundry_condition(
    *,
    name: str,
    config: AzureFoundryCodegenConfig,
    tasks: Sequence[PythonRepairTask],
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec],
    client_factory: Callable[[str, AzureFoundryCodegenConfig], Any] | None,
) -> RunSummary:
    client = (
        client_factory(name, config)
        if client_factory is not None
        else create_azure_foundry_client(config)
    )
    policy = AzureFoundryCodegenPolicy(
        client=client,
        config=config,
    )
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=1,
            group_size=1,
            train_batch_groups=1,
            max_train_steps=config.max_train_steps,
            queue_max_trajectories=2,
            train_queue_capacity=1,
            max_policy_lag=1,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=tuple(
            Scenario(id=task.id, payload={"task": task})
            for task in tasks
        ),
        initial_policy=policy,
        trainer=AzureFoundryCodegenTrainer(
            tasks=tasks,
            trainer_dollar_seconds=config.trainer_dollar_seconds,
        ),
        workflow=azure_foundry_codegen_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


async def _run_foundry_named_condition(
    *,
    name: str,
    config: AzureFoundryCodegenConfig,
    tasks: Sequence[PythonRepairTask],
    budget_dollar_seconds: float | None,
    client_factory: Callable[[str, AzureFoundryCodegenConfig], Any] | None,
) -> RunSummary:
    if name == "static_art":
        scheduler = (
            _foundry_static_budget_scheduler(budget_dollar_seconds)
            if budget_dollar_seconds is not None
            else None
        )
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=scheduler,
            action_space=None,
            action_codecs=[TokenActionCodec()],
            client_factory=client_factory,
        )
    if name == "scheduler_only":
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=_foundry_scheduler(budget_dollar_seconds),
            action_space=None,
            action_codecs=[TokenActionCodec()],
            client_factory=client_factory,
        )
    if name == "chunk2_only":
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=_foundry_scheduler(budget_dollar_seconds),
            action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=2),
            action_codecs=[
                TokenActionCodec(),
                ChunkActionCodec(chunk_size=2),
            ],
            client_factory=client_factory,
        )
    if name == "chunk4_only":
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=_foundry_scheduler(budget_dollar_seconds),
            action_space=AdaptiveActionSpace(min_chunk_size=4, max_chunk_size=4),
            action_codecs=[
                TokenActionCodec(),
                ChunkActionCodec(chunk_size=4),
            ],
            client_factory=client_factory,
        )
    if name == "full_trinity":
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=_foundry_scheduler(budget_dollar_seconds),
            action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
            action_codecs=[
                TokenActionCodec(),
                ChunkActionCodec(chunk_size=2),
                ChunkActionCodec(chunk_size=4),
            ],
            client_factory=client_factory,
        )
    if name == "full_trinity_patient_demote":
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=_foundry_scheduler(budget_dollar_seconds),
            action_space=AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
                demotion_min_pulls=4,
            ),
            action_codecs=[
                TokenActionCodec(),
                ChunkActionCodec(chunk_size=2),
                ChunkActionCodec(chunk_size=4),
            ],
            client_factory=client_factory,
        )
    if name == "full_trinity_no_demote":
        return await _run_foundry_condition(
            name=name,
            config=config,
            tasks=tasks,
            scheduler=_foundry_scheduler(budget_dollar_seconds),
            action_space=AdaptiveActionSpace(
                min_chunk_size=2,
                max_chunk_size=4,
                demotion_min_pulls=1_000_000,
            ),
            action_codecs=[
                TokenActionCodec(),
                ChunkActionCodec(chunk_size=2),
                ChunkActionCodec(chunk_size=4),
            ],
            client_factory=client_factory,
        )
    raise ValueError(f"unknown_foundry_condition: {name}")


def _foundry_summary_metrics(
    summary: RunSummary,
    *,
    tasks: Sequence[PythonRepairTask],
) -> dict[str, float]:
    metrics = dict(summary.metrics)
    keys = [
        CODEGEN_NORTH_STAR,
        CODEGEN_ACCOUNTED_NORTH_STAR,
        "reward/delta",
        "data/groups_trained",
        "data/train_steps",
        "actions/semantic_bandwidth_tokens_per_decision",
        "costs/rollout_dollar_seconds",
        "costs/trainer_dollar_seconds",
        "costs/accounted_dollar_seconds",
        "promotion/latest_score",
        "promotion/latest_baseline_score",
        "promotion/latest_improvement",
        "promotion/latest_published_policy_score",
        "action_space/active_codecs",
        "action_space/promotions",
        "action_space/demotions",
        "action_space/demotion_min_pulls",
        "action_space/codec/chunk_chunk_size_2/active",
        "action_space/codec/chunk_chunk_size_4/active",
        "action_space/codec/chunk_chunk_size_4/disabled",
        "scheduler/budget/max_accounted_dollar_seconds",
        "scheduler/budget/accounted_dollar_seconds",
        "scheduler/budget/projected_accounted_dollar_seconds",
        "scheduler/budget/remaining_accounted_dollar_seconds",
        "scheduler/budget/accounted_fraction",
        "scheduler/budget/accounted_exhausted",
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
    ]
    values = {
        key: float(metrics[key])
        for key in keys
        if key in metrics
    }
    values.update(_foundry_rollout_totals(summary))
    values.update(_foundry_codec_metrics(metrics, tasks))
    return values


def _foundry_rollout_totals(summary: RunSummary) -> dict[str, float]:
    model_calls = 0.0
    passed = 0.0
    total = 0.0
    prompt_tokens = 0.0
    completion_tokens = 0.0
    source_model = 0.0
    source_memory = 0.0
    learned_solutions = 0.0
    for checkpoint in summary.checkpoints:
        metrics = checkpoint.metrics
        model_calls = max(
            model_calls,
            float(metrics.get("foundry/model_calls_used", 0.0)),
        )
        learned_solutions = max(
            learned_solutions,
            float(metrics.get("foundry/learned_solutions", 0.0)),
        )
        passed += float(metrics.get("foundry/batch_verifier_passed_rollouts", 0.0))
        total += float(metrics.get("foundry/batch_rollouts", 0.0))
        prompt_tokens += float(metrics.get("foundry/batch_prompt_tokens", 0.0))
        completion_tokens += float(
            metrics.get("foundry/batch_completion_tokens", 0.0)
        )
        source_model += float(metrics.get("foundry/batch_source_model_rollouts", 0.0))
        source_memory += float(
            metrics.get("foundry/batch_source_memory_rollouts", 0.0)
        )
    return {
        "foundry/model_calls": model_calls,
        "foundry/learned_solutions": learned_solutions,
        "foundry/verifier_passed_rollouts": passed,
        "foundry/observed_rollouts": total,
        "foundry/prompt_tokens": prompt_tokens,
        "foundry/completion_tokens": completion_tokens,
        "foundry/source_model_rollouts": source_model,
        "foundry/source_memory_rollouts": source_memory,
    }


def _foundry_codec_metrics(
    metrics: Mapping[str, float],
    tasks: Sequence[PythonRepairTask],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for label, codec in _foundry_codec_labels():
        codec_key = action_codec_key(codec)
        values[f"foundry/codec/{label}/pulls"] = _foundry_codec_sum(
            metrics,
            tasks,
            codec_key,
            "pulls",
        )
        for metric_name in (
            "total_improvement_per_dollar_second",
            "mean_rollout_dollar_seconds",
            "semantic_bandwidth_tokens_per_decision",
            "source_tokens_per_dollar_second",
            "failure_rate",
        ):
            values[f"foundry/codec/{label}/{metric_name}"] = (
                _foundry_codec_weighted(metrics, tasks, codec_key, metric_name)
            )
    return values


def _foundry_codec_labels() -> tuple[tuple[str, ActionCodec], ...]:
    return (
        ("token", TokenActionCodec()),
        ("chunk2", ChunkActionCodec(chunk_size=2)),
        ("chunk3", ChunkActionCodec(chunk_size=3)),
        ("chunk4", ChunkActionCodec(chunk_size=4)),
    )


def _foundry_codec_sum(
    metrics: Mapping[str, float],
    tasks: Sequence[PythonRepairTask],
    codec_key: str,
    metric_name: str,
) -> float:
    return sum(
        _mapping_float(
            metrics,
            f"scheduler/arm/{safe_metric_key(f'{task.id}|{codec_key}')}/"
            f"{metric_name}",
        )
        for task in tasks
    )


def _foundry_codec_weighted(
    metrics: Mapping[str, float],
    tasks: Sequence[PythonRepairTask],
    codec_key: str,
    metric_name: str,
) -> float:
    weighted_total = 0.0
    total_pulls = 0.0
    for task in tasks:
        arm = f"scheduler/arm/{safe_metric_key(f'{task.id}|{codec_key}')}"
        pulls = _mapping_float(metrics, f"{arm}/pulls")
        weighted_total += pulls * _mapping_float(metrics, f"{arm}/{metric_name}")
        total_pulls += pulls
    return weighted_total / total_pulls if total_pulls else 0.0


def _foundry_budget_race_metrics(
    conditions: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    learned = {
        name: float(metrics.get("foundry/learned_solutions", 0.0))
        for name, metrics in conditions.items()
    }
    costs = {
        name: float(metrics.get("costs/accounted_dollar_seconds", 0.0))
        for name, metrics in conditions.items()
    }
    efficiency = {
        name: (
            learned[name] / costs[name]
            if costs[name] > 0.0
            else 0.0
        )
        for name in conditions
    }
    full_learned = learned.get("full_trinity", 0.0)
    scheduler_learned = learned.get("scheduler_only", 0.0)
    static_learned = learned.get("static_art", 0.0)
    full_cost = costs.get("full_trinity", 0.0)
    scheduler_cost = costs.get("scheduler_only", 0.0)
    static_cost = costs.get("static_art", 0.0)
    return {
        "performance_winner_by_learned_solutions": max(
            learned,
            key=lambda name: (learned[name], -costs[name]),
        ),
        "cost_winner_by_accounted_dollar_seconds": min(
            costs,
            key=lambda name: (costs[name], -learned[name]),
        ),
        "efficiency_winner_by_learned_solutions_per_dollar_second": max(
            efficiency,
            key=efficiency.get,
        ),
        "learned_solutions": learned,
        "accounted_dollar_seconds": costs,
        "learned_solutions_per_dollar_second": efficiency,
        "full_trinity_over_scheduler_learned_solution_delta": (
            full_learned - scheduler_learned
        ),
        "full_trinity_over_static_learned_solution_delta": (
            full_learned - static_learned
        ),
        "full_trinity_cost_delta_vs_scheduler": full_cost - scheduler_cost,
        "full_trinity_cost_delta_vs_static": full_cost - static_cost,
        "full_trinity_wins_performance_and_cost_vs_scheduler": (
            full_learned > scheduler_learned and full_cost <= scheduler_cost
        ),
        "full_trinity_wins_performance_and_cost_vs_static": (
            full_learned > static_learned and full_cost <= static_cost
        ),
    }


def _normalize_foundry_conditions(
    conditions: Sequence[str] | None,
) -> tuple[str, ...]:
    if conditions is None:
        return DEFAULT_FOUNDRY_CONDITIONS
    normalized: list[str] = []
    for condition in conditions:
        name = str(condition)
        if name not in FOUNDRY_CONDITIONS:
            raise ValueError(f"unknown_foundry_condition: {name}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("at_least_one_foundry_condition_required")
    return tuple(normalized)


def _foundry_lift_metrics(
    conditions: Mapping[str, Mapping[str, float]],
) -> dict[str, float | None]:
    static_score = _condition_accounted_score(conditions, "static_art")
    scheduler_score = _condition_accounted_score(conditions, "scheduler_only")
    full_score = _condition_accounted_score(conditions, "full_trinity")
    values: dict[str, float | None] = {}
    if static_score is not None and scheduler_score is not None:
        values["scheduler_over_static_accounted_north_star_ratio"] = _finite_ratio(
            scheduler_score,
            static_score,
        )
        values["scheduler_over_static_accounted_north_star_absolute"] = (
            scheduler_score - static_score
        )
    if static_score is not None and full_score is not None:
        values["full_trinity_over_static_accounted_north_star_ratio"] = _finite_ratio(
            full_score,
            static_score,
        )
        values["full_trinity_over_static_accounted_north_star_absolute"] = (
            full_score - static_score
        )
    if scheduler_score is not None and full_score is not None:
        values["full_trinity_over_scheduler_accounted_north_star_ratio"] = (
            _finite_ratio(full_score, scheduler_score)
        )
        values["full_trinity_over_scheduler_accounted_north_star_absolute"] = (
            full_score - scheduler_score
        )
    return values


def _condition_accounted_score(
    conditions: Mapping[str, Mapping[str, float]],
    name: str,
) -> float | None:
    metrics = conditions.get(name)
    if metrics is None:
        return None
    return float(metrics.get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0))


def _foundry_heldout_payload(
    summaries: Mapping[str, RunSummary],
    *,
    heldout_tasks: Sequence[PythonRepairTask],
    config: AzureFoundryCodegenConfig,
) -> dict[str, Any] | None:
    if not heldout_tasks:
        return None
    condition_metrics = {
        name: _foundry_heldout_condition_metrics(
            summary,
            heldout_tasks=heldout_tasks,
            config=config,
        )
        for name, summary in summaries.items()
    }
    if not condition_metrics:
        return None
    winner = max(
        condition_metrics,
        key=lambda name: condition_metrics[name].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0),
    )
    return {
        "task_count": len(heldout_tasks),
        "task_ids": [task.id for task in heldout_tasks],
        "conditions": condition_metrics,
        "winning_condition_by_accounted_north_star": winner,
        "score": condition_metrics[winner].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0),
    }


def _foundry_heldout_condition_metrics(
    summary: RunSummary,
    *,
    heldout_tasks: Sequence[PythonRepairTask],
    config: AzureFoundryCodegenConfig,
) -> dict[str, Any]:
    learned_solutions = _latest_learned_solutions(summary)
    passed = 0.0
    missing = 0.0
    verifier_failures = 0.0
    tests_passed = 0.0
    tests_total = 0.0
    task_results: list[dict[str, Any]] = []
    for task in heldout_tasks:
        code = learned_solutions.get(task.id)
        if code is None:
            missing += 1.0
            tests_total += float(len(task.tests))
            task_results.append(
                _heldout_task_result(
                    task,
                    passed=False,
                    failure_mode="missing_learned_solution",
                    tests_passed=0.0,
                    tests_total=float(len(task.tests)),
                )
            )
            continue
        verification = verify_python_solution(
            task,
            code,
            timeout_s=config.verify_timeout_s,
            memory_limit_bytes=config.verify_memory_limit_bytes,
        )
        tests_passed += float(verification.tests_passed)
        tests_total += float(verification.tests_total)
        if verification.passed:
            passed += 1.0
        else:
            verifier_failures += 1.0
        task_results.append(
            _heldout_task_result(
                task,
                passed=verification.passed,
                failure_mode=verification.failure_mode,
                tests_passed=float(verification.tests_passed),
                tests_total=float(verification.tests_total),
            )
        )
    task_count = float(len(heldout_tasks))
    accounted = float(summary.metrics.get("costs/accounted_dollar_seconds", 0.0))
    heldout_pass_rate = passed / task_count if task_count else 0.0
    tests_pass_rate = tests_passed / tests_total if tests_total else 0.0
    metrics: dict[str, Any] = {
        CODEGEN_NORTH_STAR: heldout_pass_rate,
        CODEGEN_ACCOUNTED_NORTH_STAR: (
            heldout_pass_rate / accounted if accounted > 0.0 else 0.0
        ),
        "heldout/tasks": task_count,
        "heldout/passed": passed,
        "heldout/missing": missing,
        "heldout/verifier_failures": verifier_failures,
        "heldout/pass_rate": heldout_pass_rate,
        "heldout/tests_passed": tests_passed,
        "heldout/tests_total": tests_total,
        "heldout/tests_pass_rate": tests_pass_rate,
        "costs/accounted_dollar_seconds": accounted,
    }
    metrics.update(_heldout_breakdown_metrics(task_results))
    metrics["heldout/task_results"] = task_results
    return metrics


def _latest_learned_solutions(summary: RunSummary) -> dict[str, str]:
    for checkpoint in reversed(summary.checkpoints):
        learned = checkpoint.metadata.get("foundry/learned_solutions")
        if isinstance(learned, Mapping):
            return {
                str(task_id): str(code)
                for task_id, code in learned.items()
                if isinstance(task_id, str) and isinstance(code, str)
            }
    return {}


def _foundry_task_coverage_payload(
    selection: FoundryTaskSelection,
) -> dict[str, Any]:
    return {
        "split": selection.split,
        "train": _foundry_task_metadata_summary(selection.train),
        "heldout": _foundry_task_metadata_summary(selection.heldout),
    }


def _foundry_task_metadata_summary(
    tasks: Sequence[PythonRepairTask],
) -> dict[str, Any]:
    return {
        "tasks": len(tasks),
        "families": _count_by_string(task.family for task in tasks),
        "difficulties": _count_by_string(str(task.difficulty) for task in tasks),
        "failure_tags": _count_by_string(
            tag
            for task in tasks
            for tag in task.failure_tags
        ),
    }


def _foundry_non_saturation_payload(
    conditions: Mapping[str, Mapping[str, float]],
    heldout: Mapping[str, Any] | None,
    *,
    task_count: int,
    heldout_task_count: int,
) -> dict[str, Any]:
    condition_payload: dict[str, dict[str, Any]] = {}
    heldout_conditions = {}
    if isinstance(heldout, Mapping) and isinstance(heldout.get("conditions"), Mapping):
        heldout_conditions = heldout["conditions"]
    for name, metrics in conditions.items():
        learned = float(metrics.get("foundry/learned_solutions", 0.0))
        learned_fraction = learned / task_count if task_count else 0.0
        heldout_metrics = (
            heldout_conditions.get(name, {})
            if isinstance(heldout_conditions, Mapping)
            else {}
        )
        heldout_pass_rate = (
            _mapping_float(heldout_metrics, "heldout/pass_rate")
            if isinstance(heldout_metrics, Mapping)
            else 0.0
        )
        saturated = learned_fraction >= 1.0 and (
            heldout_task_count == 0 or heldout_pass_rate >= 1.0
        )
        condition_payload[name] = {
            "learned_fraction": learned_fraction,
            "heldout_pass_fraction": heldout_pass_rate,
            "saturated": saturated,
        }
    return {
        "task_count": task_count,
        "heldout_task_count": heldout_task_count,
        "conditions": condition_payload,
        "any_saturated": any(
            bool(values["saturated"])
            for values in condition_payload.values()
        ),
        "all_saturated": bool(condition_payload)
        and all(bool(values["saturated"]) for values in condition_payload.values()),
    }


def _heldout_task_result(
    task: PythonRepairTask,
    *,
    passed: bool,
    failure_mode: str,
    tests_passed: float,
    tests_total: float,
) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "family": task.family,
        "difficulty": str(task.difficulty),
        "failure_tags": list(task.failure_tags),
        "passed": passed,
        "failure_mode": failure_mode,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
    }


def _heldout_breakdown_metrics(
    task_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "heldout/by_family": _heldout_breakdown(task_results, "family"),
        "heldout/by_difficulty": _heldout_breakdown(task_results, "difficulty"),
        "heldout/by_failure_tag": _heldout_failure_tag_breakdown(task_results),
    }


def _heldout_breakdown(
    task_results: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for result in task_results:
        bucket = str(result.get(key, "unknown")) or "unknown"
        values = buckets.setdefault(
            bucket,
            {
                "tasks": 0.0,
                "passed": 0.0,
                "tests_passed": 0.0,
                "tests_total": 0.0,
            },
        )
        values["tasks"] += 1.0
        values["passed"] += 1.0 if result.get("passed") is True else 0.0
        values["tests_passed"] += _mapping_float(result, "tests_passed")
        values["tests_total"] += _mapping_float(result, "tests_total")
    for values in buckets.values():
        tasks = values["tasks"]
        tests_total = values["tests_total"]
        values["pass_rate"] = values["passed"] / tasks if tasks else 0.0
        values["tests_pass_rate"] = (
            values["tests_passed"] / tests_total if tests_total else 0.0
        )
    return buckets


def _heldout_failure_tag_breakdown(
    task_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for result in task_results:
        tags = result.get("failure_tags")
        if not isinstance(tags, list | tuple) or not tags:
            tags = ("unknown",)
        for tag in tags:
            bucket = str(tag) or "unknown"
            values = buckets.setdefault(
                bucket,
                {
                    "tasks": 0.0,
                    "passed": 0.0,
                    "tests_passed": 0.0,
                    "tests_total": 0.0,
                },
            )
            values["tasks"] += 1.0
            values["passed"] += 1.0 if result.get("passed") is True else 0.0
            values["tests_passed"] += _mapping_float(result, "tests_passed")
            values["tests_total"] += _mapping_float(result, "tests_total")
    for values in buckets.values():
        tasks = values["tasks"]
        tests_total = values["tests_total"]
        values["pass_rate"] = values["passed"] / tasks if tasks else 0.0
        values["tests_pass_rate"] = (
            values["tests_passed"] / tests_total if tests_total else 0.0
        )
    return buckets


def _count_by_string(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _foundry_scheduler(
    max_accounted_dollar_seconds: float | None = None,
) -> ObjectiveScheduler:
    return ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=1,
        min_policy_lag=1,
        max_policy_lag=1,
        min_actor_count=1,
        max_actor_count=1,
        exploration_bonus=0.0,
        max_accounted_dollar_seconds=max_accounted_dollar_seconds,
    )


def _foundry_static_budget_scheduler(
    max_accounted_dollar_seconds: float,
) -> "_StaticRoundRobinBudgetScheduler":
    return _StaticRoundRobinBudgetScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=1,
        min_policy_lag=1,
        max_policy_lag=1,
        min_actor_count=1,
        max_actor_count=1,
        exploration_bonus=0.0,
        max_accounted_dollar_seconds=max_accounted_dollar_seconds,
    )


class _StaticRoundRobinBudgetScheduler(ObjectiveScheduler):
    """Fixed arm order with ObjectiveScheduler accounting and budget stops."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._static_rollout_index = 0

    def select_rollout(
        self,
        *,
        scenarios: Sequence[Scenario],
        action_codecs: Sequence[ActionCodec],
        actor_id: int,
        policy_step: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        configured_train_batch_groups: int,
        configured_max_policy_lag: int,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> SchedulerDecision:
        if not scenarios:
            raise ValueError("at_least_one_scenario_is_required")
        if not action_codecs:
            raise ValueError("at_least_one_action_codec_is_required")
        target_train_batch_groups = self.target_train_batch_groups(
            configured=configured_train_batch_groups,
            pending_groups=0,
            train_queue_pressure=train_queue_pressure,
            policy_step=policy_step,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        max_policy_lag = self.max_policy_lag(
            configured=configured_max_policy_lag,
            train_queue_pressure=train_queue_pressure,
            policy_step=policy_step,
            target_train_batch_groups=target_train_batch_groups,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        arms = self._arm_candidates(scenarios, action_codecs)
        arm_id, scenario, codec = arms[self._static_rollout_index % len(arms)]
        self._static_rollout_index += 1
        decision_score = self._score_arm(
            arm_id,
            scenario,
            codec,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        selected_stats = self._arms[arm_id]
        estimated_rollout_dollar_seconds = self._estimated_rollout_dollar_seconds(
            arm_id,
            scenario,
            codec,
        )
        reserved_rollout_dollar_seconds = max(0.0, estimated_rollout_dollar_seconds)
        self._record_arm_decision(
            selected_stats,
            reserved_rollout_dollar_seconds=reserved_rollout_dollar_seconds,
        )
        actor_stats = self._actors.setdefault(actor_id, ActorStats())
        actor_stats.decisions += 1
        actor_stats.inflight += 1
        joint_action_key = self._candidate_joint_action_key(
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        metadata: dict[str, Any] = {
            "actor_id": actor_id,
            "policy_step": policy_step,
            "trajectory_queue_pressure": trajectory_queue_pressure,
            "train_queue_pressure": train_queue_pressure,
            "score": decision_score,
            "objective_score": self._arm_value(selected_stats),
            "exploration_score": self._exploration_value(selected_stats),
            "joint_action_score": self._joint_action_score(
                arm_id=arm_id,
                target_train_batch_groups=target_train_batch_groups,
                max_policy_lag=max_policy_lag,
                active_actor_count=active_actor_count,
                rollout_admission_delay_ms=rollout_admission_delay_ms,
                action_space_key=action_space_key,
            ),
            "joint_action_score_weight": self.joint_action_objective_weight,
            "inflight_rollouts": selected_stats.inflight,
            "coverage_forced": False,
            "expected_rollout_dollar_seconds": estimated_rollout_dollar_seconds,
            "estimated_rollout_dollar_seconds": estimated_rollout_dollar_seconds,
            "reserved_rollout_dollar_seconds": reserved_rollout_dollar_seconds,
            "unobserved_rollout_cost_penalty": self._unobserved_rollout_cost_penalty(
                arm_id,
                scenario,
                codec,
            ),
        }
        if action_space_key is not None:
            metadata["action_space_key"] = action_space_key
        if joint_action_key is not None:
            metadata["joint_action_key"] = joint_action_key
            self._record_joint_action_decision(joint_action_key)
        decision = SchedulerDecision(
            scenario=scenario,
            action_codec=codec,
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            metadata=metadata,
        )
        self._last_decision = decision
        self._last_decision_snapshot = _decision_to_state(decision)
        return decision


def _foundry_rollout_dollar_seconds(
    *,
    config: AzureFoundryCodegenConfig,
    generation: FoundryGeneration,
    action_units: int,
) -> float:
    request_cost = (
        config.request_dollar_seconds
        if generation.model_called
        else config.memory_dollar_seconds
    )
    return request_cost + config.action_unit_dollar_seconds * max(1, action_units)


def _selected_foundry_tasks(
    task_limit: int,
    task_split: str = DEFAULT_FOUNDRY_TASK_SPLIT,
    task_order_policy: str = DEFAULT_FOUNDRY_TASK_ORDER_POLICY,
) -> tuple[PythonRepairTask, ...]:
    return _selected_foundry_task_selection(
        task_limit,
        task_split,
        task_order_policy,
    ).train


def _selected_foundry_task_selection(
    task_limit: int,
    task_split: str = DEFAULT_FOUNDRY_TASK_SPLIT,
    task_order_policy: str = DEFAULT_FOUNDRY_TASK_ORDER_POLICY,
) -> FoundryTaskSelection:
    if task_split not in _FOUNDRY_TASK_SPLITS:
        raise ValueError("foundry_task_split_unknown")
    if task_order_policy not in FOUNDRY_TASK_ORDER_POLICIES:
        raise ValueError("foundry_task_order_policy_unknown")
    train_bank, heldout_bank = _FOUNDRY_TASK_SPLITS[task_split]
    ordered_train_bank = _ordered_foundry_tasks(train_bank, task_order_policy)
    train = _limit_foundry_tasks(ordered_train_bank, task_limit)
    heldout = _heldout_for_train(heldout_bank, train)
    return FoundryTaskSelection(split=task_split, train=train, heldout=heldout)


def _ordered_foundry_tasks(
    tasks: Sequence[PythonRepairTask],
    task_order_policy: str,
) -> tuple[PythonRepairTask, ...]:
    if task_order_policy == "split_order":
        return tuple(tasks)
    if task_order_policy == "coverage_gap_first":
        return _priority_ordered_foundry_tasks(tasks, FOUNDRY_COVERAGE_GAP_TASK_IDS)
    if task_order_policy == "lift_pocket_first":
        return _priority_ordered_foundry_tasks(tasks, FOUNDRY_LIFT_POCKET_TASK_IDS)
    raise ValueError("foundry_task_order_policy_unknown")


def _priority_ordered_foundry_tasks(
    tasks: Sequence[PythonRepairTask],
    priority_task_ids: Sequence[str],
) -> tuple[PythonRepairTask, ...]:
    target_rank = {
        task_id: index
        for index, task_id in enumerate(priority_task_ids)
    }
    fallback_rank = len(target_rank)
    return tuple(
        task
        for _, task in sorted(
            enumerate(tasks),
            key=lambda item: (
                target_rank.get(item[1].id, fallback_rank),
                item[0],
            ),
        )
    )


def _limit_foundry_tasks(
    tasks: Sequence[PythonRepairTask],
    task_limit: int,
) -> tuple[PythonRepairTask, ...]:
    return tuple(tasks[: max(1, min(task_limit, len(tasks)))])


def _heldout_for_train(
    heldout_tasks: Sequence[PythonRepairTask],
    train_tasks: Sequence[PythonRepairTask],
) -> tuple[PythonRepairTask, ...]:
    train_ids = {task.id for task in train_tasks}
    return tuple(task for task in heldout_tasks if task.id in train_ids)


def _foundry_task_for_scenario(scenario: Scenario) -> PythonRepairTask:
    task = scenario.payload.get("task")
    if isinstance(task, PythonRepairTask):
        return task
    for candidate in _ALL_FOUNDRY_TASKS_BY_ID.values():
        if candidate.id == scenario.id:
            return candidate
    raise ValueError(f"unknown_foundry_codegen_scenario: {scenario.id}")


def _repair_prompt(
    task: PythonRepairTask,
    *,
    prompt_context_policy: str = DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY,
) -> str:
    tests = "\n".join(
        f"- solve{args!r} == {expected!r}"
        for args, expected in task.tests
    )
    context = _repair_prompt_context(task, prompt_context_policy)
    context_block = f"\n\nContext:\n{context}" if context else ""
    return textwrap.dedent(
        f"""
        Repair this Python function.

        Requirements:
        - Define exactly one function with signature: {task.signature}
        - Preserve the function name solve.
        - Return only code, no markdown.
        - Do not import modules or perform IO.

        Task:
        {task.prompt}{context_block}

        Buggy code:
        {task.buggy_code.strip()}

        Unit tests:
        {tests}
        """
    ).strip()


def _repair_prompt_context(
    task: PythonRepairTask,
    prompt_context_policy: str,
) -> str:
    if prompt_context_policy == "repair_prompt_only":
        return ""
    lines = [
        f"- family: {task.family}",
        f"- difficulty: {task.difficulty}",
    ]
    if task.failure_tags:
        lines.append("- failure tags: " + ", ".join(task.failure_tags))
    if prompt_context_policy == "task_metadata":
        return "\n".join(lines)
    if prompt_context_policy == "failure_tag_guardrails":
        guardrails = _failure_tag_guardrail_lines(task.failure_tags)
        if guardrails:
            lines.append("- failure-tag guardrails:")
            lines.extend(guardrails)
        return "\n".join(lines)
    if prompt_context_policy == "data_model_guardrails":
        if task.family != "data_model":
            return ""
        lines.extend(
            [
                "- preserve input objects unless the task explicitly asks to mutate",
                "- deep-copy nested defaults before filling missing dictionary keys",
                "- distinguish missing keys, None values, and other falsy values",
                "- report schema errors in deterministic input/schema order",
            ]
        )
        return "\n".join(lines)
    raise ValueError("foundry_prompt_context_policy_unknown")


def _failure_tag_guardrail_lines(failure_tags: Sequence[str]) -> list[str]:
    return [
        f"  - {tag}: {FOUNDRY_FAILURE_TAG_GUARDRAILS[tag]}"
        for tag in failure_tags
        if tag in FOUNDRY_FAILURE_TAG_GUARDRAILS
    ]


def _fallback_solution(task: PythonRepairTask) -> str:
    return f"{task.signature}:\n    return None\n"


def _trim_to_solve(text: str) -> str:
    start = text.find("def solve")
    if start < 0:
        return text.strip()
    return text[start:].strip()


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    return content or ""


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, 0)
        try:
            values[key] = max(0, int(value))
        except (TypeError, ValueError):
            values[key] = 0
    return values


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata.get(key, 0.0)
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if isfinite(parsed) else 0.0


def _mapping_float(values: Mapping[str, Any], key: str) -> float:
    return _metadata_float(values, key)


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


_VERIFY_SCRIPT = r"""
import ast
import ctypes
import json
import os
import sys

payload = json.loads(sys.stdin.read())
code = payload.get("code", "")
tests = payload.get("tests", [])
memory_limit_bytes = int(payload.get("memory_limit_bytes") or 0)
safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

def emit(passed, failure_mode, tests_passed=0):
    print(json.dumps({
        "passed": passed,
        "failure_mode": failure_mode,
        "tests_passed": tests_passed,
        "tests_total": len(tests),
    }))

def apply_windows_memory_limit(limit_bytes):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JobObjectBasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_process_memory = 0x100
    job_object_limit_kill_on_job_close = 0x2000

    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = JobObjectExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = (
        job_object_limit_process_memory | job_object_limit_kill_on_job_close
    )
    info.ProcessMemoryLimit = limit_bytes
    if not kernel32.SetInformationJobObject(
        job,
        job_object_extended_limit_information,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(job)
        return False
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        kernel32.CloseHandle(job)
        return False
    globals()["_VERIFIER_WINDOWS_JOB_HANDLE"] = job
    return True

def apply_posix_memory_limit(limit_bytes):
    try:
        import resource
    except ImportError:
        return False
    applied = False
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit = getattr(resource, limit_name, None)
        if limit is None:
            continue
        try:
            soft, hard = resource.getrlimit(limit)
            capped = (
                limit_bytes
                if hard == resource.RLIM_INFINITY
                else min(limit_bytes, hard)
            )
            resource.setrlimit(limit, (capped, capped))
        except (OSError, ValueError):
            continue
        applied = True
    return applied

def apply_memory_limit(limit_bytes):
    if limit_bytes <= 0:
        return True
    if os.name == "nt":
        return apply_windows_memory_limit(limit_bytes)
    if sys.platform == "darwin":
        return True
    return apply_posix_memory_limit(limit_bytes)

def darwin_memory_limit_exceeded(limit_bytes):
    if sys.platform != "darwin" or limit_bytes <= 0:
        return False
    try:
        import resource
    except ImportError:
        return True
    peak_resident_bytes = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    return peak_resident_bytes > limit_bytes

if not apply_memory_limit(memory_limit_bytes):
    emit(False, "resource_limit_unavailable")
    raise SystemExit(0)

try:
    tree = ast.parse(code)
except SyntaxError:
    emit(False, "syntax_error")
    raise SystemExit(0)

for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.With, ast.AsyncWith)):
        emit(False, "forbidden_syntax")
        raise SystemExit(0)
    if isinstance(node, ast.Attribute) and "__" in node.attr:
        emit(False, "forbidden_name")
        raise SystemExit(0)
    if isinstance(node, ast.Name) and "__" in node.id:
        emit(False, "forbidden_name")
        raise SystemExit(0)
    if isinstance(node, ast.arg) and "__" in node.arg:
        emit(False, "forbidden_name")
        raise SystemExit(0)

namespace = {}
try:
    exec(compile(tree, "<azure_foundry_codegen>", "exec"), {"__builtins__": safe_builtins}, namespace)
except Exception:
    emit(False, "exec_error")
    raise SystemExit(0)
if darwin_memory_limit_exceeded(memory_limit_bytes):
    emit(False, "resource_limit_exceeded")
    raise SystemExit(0)

solve = namespace.get("solve")
if not callable(solve):
    emit(False, "missing_solve")
    raise SystemExit(0)

passed = 0
for item in tests:
    args = item.get("args", [])
    expected = item.get("expected")
    try:
        actual = solve(*args)
    except Exception:
        emit(False, "unit_test_exception", passed)
        raise SystemExit(0)
    if darwin_memory_limit_exceeded(memory_limit_bytes):
        emit(False, "resource_limit_exceeded", passed)
        raise SystemExit(0)
    if actual != expected:
        emit(False, "unit_test_failed", passed)
        raise SystemExit(0)
    passed += 1

emit(True, "passed", passed)
"""


_FOUNDRY_TASKS = (
    PythonRepairTask(
        id="repair_clamp",
        prompt="Clamp value into the inclusive interval [low, high].",
        signature="def solve(value, low, high)",
        buggy_code="""
def solve(value, low, high):
    if value < low:
        return high
    if value > high:
        return low
    return value
""",
        tests=(
            ((5, 1, 9), 5),
            ((-3, 0, 4), 0),
            ((12, 0, 4), 4),
            ((4, 4, 9), 4),
        ),
    ),
    PythonRepairTask(
        id="repair_dedupe_order",
        prompt="Remove duplicate values while preserving first-seen order.",
        signature="def solve(items)",
        buggy_code="""
def solve(items):
    return sorted(set(items))
""",
        tests=(
            (([1, 2, 1, 3, 2],), [1, 2, 3]),
            ((["b", "a", "b"],), ["b", "a"]),
            (([],), []),
            (([4, 4, 4],), [4]),
        ),
    ),
    PythonRepairTask(
        id="repair_balanced_brackets",
        prompt="Return True when (), [], and {} brackets are balanced.",
        signature="def solve(text)",
        buggy_code="""
def solve(text):
    return text.count("(") == text.count(")")
""",
        tests=(
            (("([]){}",), True),
            (("([)]",), False),
            (("",), True),
            (("{[()]}",), True),
            (("{[(])}",), False),
        ),
    ),
    PythonRepairTask(
        id="repair_top_k_frequent",
        prompt="Return the k most frequent integers sorted by frequency descending, then value ascending.",
        signature="def solve(values, k)",
        buggy_code="""
def solve(values, k):
    return sorted(set(values))[:k]
""",
        tests=(
            (([3, 1, 3, 2, 2, 3], 2), [3, 2]),
            (([5, 4, 4, 5], 2), [4, 5]),
            (([9], 1), [9]),
        ),
    ),
    PythonRepairTask(
        id="repair_flatten_once",
        prompt="Flatten one level of nested lists while preserving item order.",
        signature="def solve(items)",
        buggy_code="""
def solve(items):
    return items
""",
        tests=(
            (([[1, 2], [3], [], [4, 5]],), [1, 2, 3, 4, 5]),
            (((["a"], ["b", "c"]),), ["a", "b", "c"]),
            (([],), []),
        ),
    ),
    PythonRepairTask(
        id="repair_rotate_left",
        prompt="Rotate a list left by n positions, wrapping around and preserving empty lists.",
        signature="def solve(items, n)",
        buggy_code="""
def solve(items, n):
    return items[n:]
""",
        tests=(
            (([1, 2, 3, 4], 1), [2, 3, 4, 1]),
            (([1, 2, 3, 4], 6), [3, 4, 1, 2]),
            (([], 3), []),
            ((["a"], 99), ["a"]),
        ),
    ),
    PythonRepairTask(
        id="repair_longest_unique_run",
        prompt="Return the length of the longest contiguous substring with no repeated characters.",
        signature="def solve(text)",
        buggy_code="""
def solve(text):
    return len(set(text))
""",
        tests=(
            (("abcabcbb",), 3),
            (("bbbbb",), 1),
            (("pwwkew",), 3),
            (("",), 0),
            (("abba",), 2),
        ),
    ),
    PythonRepairTask(
        id="repair_merge_intervals",
        prompt="Merge overlapping inclusive intervals and return them sorted by start.",
        signature="def solve(intervals)",
        buggy_code="""
def solve(intervals):
    return intervals
""",
        tests=(
            (([[1, 3], [2, 6], [8, 10], [9, 12]],), [[1, 6], [8, 12]]),
            (([[5, 7], [1, 2], [2, 4]],), [[1, 4], [5, 7]]),
            (([],), []),
            (([[1, 1]],), [[1, 1]]),
        ),
    ),
    PythonRepairTask(
        id="repair_palindrome_normalized",
        prompt="Return True when text is a palindrome after ignoring case and non-alphanumeric characters.",
        signature="def solve(text)",
        buggy_code="""
def solve(text):
    return text == text[::-1]
""",
        tests=(
            (("A man, a plan, a canal: Panama!",), True),
            (("race a car",), False),
            (("",), True),
            (("No lemon, no melon",), True),
        ),
    ),
    PythonRepairTask(
        id="repair_chunk_list",
        prompt="Split items into consecutive chunks of the requested positive size.",
        signature="def solve(items, size)",
        buggy_code="""
def solve(items, size):
    return [items]
""",
        tests=(
            (([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]),
            (([1, 2, 3], 5), [[1, 2, 3]]),
            (([], 3), []),
        ),
    ),
    PythonRepairTask(
        id="repair_transpose_matrix",
        prompt="Transpose a rectangular matrix represented as a list of rows.",
        signature="def solve(matrix)",
        buggy_code="""
def solve(matrix):
    return matrix
""",
        tests=(
            (([[1, 2, 3], [4, 5, 6]],), [[1, 4], [2, 5], [3, 6]]),
            (([[1], [2], [3]],), [[1, 2, 3]]),
            (([],), []),
        ),
    ),
    PythonRepairTask(
        id="repair_word_counts",
        prompt="Return a dictionary counting how many times each word appears.",
        signature="def solve(words)",
        buggy_code="""
def solve(words):
    return {word: 1 for word in words}
""",
        tests=(
            ((["red", "blue", "red"],), {"red": 2, "blue": 1}),
            (([],), {}),
            ((["x", "x", "x"],), {"x": 3}),
        ),
    ),
    PythonRepairTask(
        id="repair_second_largest_unique",
        prompt="Return the second largest distinct value, or None when fewer than two distinct values exist.",
        signature="def solve(values)",
        buggy_code="""
def solve(values):
    return sorted(values)[-2]
""",
        tests=(
            (([5, 1, 5, 3],), 3),
            (([2, 2],), None),
            (([-1, -3, -2],), -2),
            (([],), None),
        ),
    ),
    PythonRepairTask(
        id="repair_common_prefix",
        prompt="Return the longest common prefix shared by all strings.",
        signature="def solve(words)",
        buggy_code="""
def solve(words):
    return words[0]
""",
        tests=(
            ((["flower", "flow", "flight"],), "fl"),
            ((["dog", "racecar", "car"],), ""),
            (([],), ""),
            ((["solo"],), "solo"),
        ),
    ),
    PythonRepairTask(
        id="repair_pair_sum_indices",
        prompt="Return the first pair of indices whose values sum to target, or an empty list.",
        signature="def solve(values, target)",
        buggy_code="""
def solve(values, target):
    return []
""",
        tests=(
            (([2, 7, 11, 15], 9), [0, 1]),
            (([3, 2, 4], 6), [1, 2]),
            (([1, 2, 3], 99), []),
            (([3, 3], 6), [0, 1]),
        ),
    ),
    PythonRepairTask(
        id="repair_run_length_encode",
        prompt="Compress consecutive equal values as [value, count] pairs.",
        signature="def solve(items)",
        buggy_code="""
def solve(items):
    return [[item, 1] for item in items]
""",
        tests=(
            ((["a", "a", "b", "b", "b", "a"],), [["a", 2], ["b", 3], ["a", 1]]),
            (([],), []),
            (([1, 1, 1],), [[1, 3]]),
        ),
    ),
    PythonRepairTask(
        id="repair_normalize_path",
        prompt="Normalize a slash-separated path by removing empty and dot segments and applying double-dot parent segments.",
        signature="def solve(path)",
        buggy_code="""
def solve(path):
    return path
""",
        tests=(
            (("/a//b/./c/../d",), "/a/b/d"),
            (("a/b/../../c",), "/c"),
            (("/",), "/"),
            (("/../x",), "/x"),
        ),
    ),
)


_FOUNDRY_STANDARD_HELDOUT_TASKS = (
    PythonRepairTask(
        id="repair_clamp",
        prompt="Clamp value into the inclusive interval [low, high].",
        signature="def solve(value, low, high)",
        buggy_code="",
        tests=(
            ((-10, -5, -1), -5),
            ((3, -5, -1), -1),
            ((0, 0, 0), 0),
        ),
    ),
    PythonRepairTask(
        id="repair_dedupe_order",
        prompt="Remove duplicate values while preserving first-seen order.",
        signature="def solve(items)",
        buggy_code="",
        tests=(
            ((["a", "a", "b", "a", "c"],), ["a", "b", "c"]),
            (((2, 1, 2, 3, 1),), [2, 1, 3]),
        ),
    ),
    PythonRepairTask(
        id="repair_balanced_brackets",
        prompt="Return True when (), [], and {} brackets are balanced.",
        signature="def solve(text)",
        buggy_code="",
        tests=(
            (("if (x[0] == {y: 1})",), True),
            (("[(])",), False),
            (("abc)",), False),
        ),
    ),
    PythonRepairTask(
        id="repair_top_k_frequent",
        prompt="Return the k most frequent integers sorted by frequency descending, then value ascending.",
        signature="def solve(values, k)",
        buggy_code="",
        tests=(
            (([1, 1, 2, 2, 3, 3], 2), [1, 2]),
            (([-1, -1, -2, 3, 3, 3], 2), [3, -1]),
        ),
    ),
    PythonRepairTask(
        id="repair_flatten_once",
        prompt="Flatten one level of nested lists while preserving item order.",
        signature="def solve(items)",
        buggy_code="",
        tests=(
            ((([], [1], [2, 3], []),), [1, 2, 3]),
            (([["x", "y"], ["z"]],), ["x", "y", "z"]),
        ),
    ),
    PythonRepairTask(
        id="repair_rotate_left",
        prompt="Rotate a list left by n positions, wrapping around and preserving empty lists.",
        signature="def solve(items, n)",
        buggy_code="",
        tests=(
            (([1, 2, 3], 0), [1, 2, 3]),
            (([1, 2, 3], -1), [3, 1, 2]),
            (([1, 2, 3], 7), [2, 3, 1]),
        ),
    ),
    PythonRepairTask(
        id="repair_longest_unique_run",
        prompt="Return the length of the longest contiguous substring with no repeated characters.",
        signature="def solve(text)",
        buggy_code="",
        tests=(
            (("dvdf",), 3),
            (("tmmzuxt",), 5),
        ),
    ),
    PythonRepairTask(
        id="repair_merge_intervals",
        prompt="Merge overlapping inclusive intervals and return them sorted by start.",
        signature="def solve(intervals)",
        buggy_code="",
        tests=(
            (([[3, 5], [1, 10], [12, 13]],), [[1, 10], [12, 13]]),
            (([[-4, -1], [-2, 2], [3, 3]],), [[-4, 2], [3, 3]]),
        ),
    ),
)


_FOUNDRY_HARD_TASKS = (
    PythonRepairTask(
        id="repair_canonical_ranges",
        prompt=(
            "Given integers in any order, remove duplicates, sort them, and compress "
            "consecutive runs as strings like '1-3'. Singleton runs are just '5'."
        ),
        signature="def solve(values)",
        buggy_code="""
def solve(values):
    return [str(value) for value in values]
""",
        tests=(
            (([3, 1, 2, 7, 8, 10],), ["1-3", "7-8", "10"]),
            (([5, 5, 4, 2],), ["2", "4-5"]),
            (([],), []),
        ),
    ),
    PythonRepairTask(
        id="repair_spiral_matrix",
        prompt="Return all values of a rectangular matrix in clockwise spiral order.",
        signature="def solve(matrix)",
        buggy_code="""
def solve(matrix):
    return [item for row in matrix for item in row]
""",
        tests=(
            (([[1, 2, 3], [4, 5, 6]],), [1, 2, 3, 6, 5, 4]),
            (([[1], [2], [3]],), [1, 2, 3]),
            (([],), []),
        ),
    ),
    PythonRepairTask(
        id="repair_topological_layers",
        prompt=(
            "Given nodes and directed edges [before, after], return sorted layers of "
            "nodes whose prerequisites have all appeared in earlier layers."
        ),
        signature="def solve(nodes, edges)",
        buggy_code="""
def solve(nodes, edges):
    return [sorted(nodes)]
""",
        tests=(
            ((["a", "b", "c"], [["a", "b"], ["a", "c"]]), [["a"], ["b", "c"]]),
            (
                (["cook", "eat", "shop", "wash"], [["shop", "cook"], ["cook", "eat"]]),
                [["shop", "wash"], ["cook"], ["eat"]],
            ),
        ),
    ),
    PythonRepairTask(
        id="repair_lru_cache_trace",
        prompt=(
            "Simulate an LRU cache of the given capacity over operations. Each "
            "operation is ['get', key] or ['put', key, value]. Return get results, "
            "using -1 for misses."
        ),
        signature="def solve(capacity, operations)",
        buggy_code="""
def solve(capacity, operations):
    return []
""",
        tests=(
            (
                (
                    2,
                    [
                        ["put", "a", 1],
                        ["put", "b", 2],
                        ["get", "a"],
                        ["put", "c", 3],
                        ["get", "b"],
                    ],
                ),
                [1, -1],
            ),
            ((1, [["put", "x", 7], ["put", "y", 8], ["get", "x"], ["get", "y"]]), [-1, 8]),
        ),
    ),
    PythonRepairTask(
        id="repair_sliding_window_max",
        prompt="Return the maximum value for each consecutive window of size k.",
        signature="def solve(values, k)",
        buggy_code="""
def solve(values, k):
    return values
""",
        tests=(
            (([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7]),
            (([4, 2, 12], 1), [4, 2, 12]),
            (([], 3), []),
        ),
    ),
)


_FOUNDRY_HARD_HELDOUT_TASKS = (
    PythonRepairTask(
        id="repair_canonical_ranges",
        prompt="Compress sorted consecutive runs after deduping and sorting.",
        signature="def solve(values)",
        buggy_code="",
        tests=(
            (([-2, -1, 0, 2, 4, 5, 6],), ["-2-0", "2", "4-6"]),
            (([9, 7, 8, 8, 10, 12],), ["7-10", "12"]),
        ),
    ),
    PythonRepairTask(
        id="repair_spiral_matrix",
        prompt="Return all values of a rectangular matrix in clockwise spiral order.",
        signature="def solve(matrix)",
        buggy_code="",
        tests=(
            (([[1, 2], [3, 4], [5, 6]],), [1, 2, 4, 6, 5, 3]),
            (([[1, 2, 3]],), [1, 2, 3]),
        ),
    ),
    PythonRepairTask(
        id="repair_topological_layers",
        prompt="Return sorted prerequisite layers for a directed acyclic graph.",
        signature="def solve(nodes, edges)",
        buggy_code="",
        tests=(
            (
                (
                    ["a", "b", "c", "d"],
                    [["a", "c"], ["b", "c"], ["c", "d"]],
                ),
                [["a", "b"], ["c"], ["d"]],
            ),
            ((["z"], []), [["z"]]),
        ),
    ),
    PythonRepairTask(
        id="repair_lru_cache_trace",
        prompt="Simulate LRU get and put operations and return get results.",
        signature="def solve(capacity, operations)",
        buggy_code="",
        tests=(
            (
                (
                    2,
                    [
                        ["put", "a", 1],
                        ["put", "b", 2],
                        ["get", "a"],
                        ["put", "c", 3],
                        ["get", "a"],
                        ["get", "b"],
                        ["get", "c"],
                    ],
                ),
                [1, 1, -1, 3],
            ),
        ),
    ),
    PythonRepairTask(
        id="repair_sliding_window_max",
        prompt="Return the maximum value for each consecutive window of size k.",
        signature="def solve(values, k)",
        buggy_code="",
        tests=(
            (([9, 8, 7, 6], 2), [9, 8, 7]),
            (([2, 2, 2], 5), []),
        ),
    ),
)


def _frontier_existing_task(
    task_id: str,
    *,
    family: str,
    difficulty: int,
    failure_tags: Sequence[str],
) -> PythonRepairTask:
    return _annotated_task(
        _EXISTING_FOUNDRY_TRAIN_TASKS_BY_ID[task_id],
        family=family,
        difficulty=difficulty,
        failure_tags=failure_tags,
    )


def _frontier_existing_heldout_task(
    task_id: str,
    *,
    family: str,
    difficulty: int,
    failure_tags: Sequence[str],
) -> PythonRepairTask:
    return _annotated_task(
        _EXISTING_FOUNDRY_HELDOUT_TASKS_BY_ID[task_id],
        family=family,
        difficulty=difficulty,
        failure_tags=failure_tags,
    )


def _frontier_task(
    *,
    task_id: str,
    prompt: str,
    signature: str,
    buggy_code: str,
    tests: tuple[tuple[tuple[Any, ...], Any], ...],
    family: str,
    difficulty: int,
    failure_tags: Sequence[str],
) -> PythonRepairTask:
    return PythonRepairTask(
        id=task_id,
        prompt=prompt,
        signature=signature,
        buggy_code=buggy_code,
        tests=tests,
        family=family,
        difficulty=difficulty,
        failure_tags=tuple(failure_tags),
    )


def _frontier_heldout_task(
    *,
    task_id: str,
    prompt: str,
    signature: str,
    tests: tuple[tuple[tuple[Any, ...], Any], ...],
    family: str,
    difficulty: int,
    failure_tags: Sequence[str],
) -> PythonRepairTask:
    return _frontier_task(
        task_id=task_id,
        prompt=prompt,
        signature=signature,
        buggy_code="",
        tests=tests,
        family=family,
        difficulty=difficulty,
        failure_tags=failure_tags,
    )


def _annotated_task(
    task: PythonRepairTask,
    *,
    family: str,
    difficulty: int,
    failure_tags: Sequence[str],
) -> PythonRepairTask:
    return PythonRepairTask(
        id=task.id,
        prompt=task.prompt,
        signature=task.signature,
        buggy_code=task.buggy_code,
        tests=task.tests,
        family=family,
        difficulty=difficulty,
        failure_tags=tuple(failure_tags),
    )


_EXISTING_FOUNDRY_TRAIN_TASKS_BY_ID = {
    task.id: task
    for task in _FOUNDRY_TASKS + _FOUNDRY_HARD_TASKS
}
_EXISTING_FOUNDRY_HELDOUT_TASKS_BY_ID = {
    task.id: task
    for task in _FOUNDRY_STANDARD_HELDOUT_TASKS + _FOUNDRY_HARD_HELDOUT_TASKS
}


_FRONTIER_TASKS = (
    _frontier_existing_task(
        "repair_clamp",
        family="sequence",
        difficulty=1,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_existing_task(
        "repair_rotate_left",
        family="sequence",
        difficulty=2,
        failure_tags=("modulo", "edge_empty"),
    ),
    _frontier_existing_task(
        "repair_dedupe_order",
        family="sequence",
        difficulty=1,
        failure_tags=("ordering", "mutation"),
    ),
    _frontier_existing_task(
        "repair_chunk_list",
        family="sequence",
        difficulty=1,
        failure_tags=("boundary", "edge_empty"),
    ),
    _frontier_existing_task(
        "repair_run_length_encode",
        family="sequence",
        difficulty=2,
        failure_tags=("state_transition", "edge_empty"),
    ),
    _frontier_existing_task(
        "repair_balanced_brackets",
        family="string_parse",
        difficulty=2,
        failure_tags=("parser_nesting", "ordering"),
    ),
    _frontier_existing_task(
        "repair_normalize_path",
        family="string_parse",
        difficulty=3,
        failure_tags=("parser_escape", "edge_empty"),
    ),
    _frontier_existing_task(
        "repair_merge_intervals",
        family="interval",
        difficulty=2,
        failure_tags=("ordering", "boundary"),
    ),
    _frontier_task(
        task_id="repair_csv_row_split",
        prompt=(
            "Split one CSV row into fields. Commas inside double quotes do not "
            "split fields, and doubled quotes inside a quoted field become one "
            "quote character."
        ),
        signature="def solve(row)",
        buggy_code="""
def solve(row):
    return row.split(",")
""",
        tests=(
            (("a,b,c",), ["a", "b", "c"]),
            (('"a,b",c',), ["a,b", "c"]),
            (('"a""b",c',), ['a"b', "c"]),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("parser_escape", "ordering"),
    ),
    _frontier_task(
        task_id="repair_query_params",
        prompt=(
            "Parse a URL query string into a dictionary from key to list of "
            "values. A plus sign represents a space; repeated keys preserve "
            "input order."
        ),
        signature="def solve(query)",
        buggy_code="""
def solve(query):
    return {}
""",
        tests=(
            (("a=1&b=2&a=3",), {"a": ["1", "3"], "b": ["2"]}),
            (("q=red+blue&empty=",), {"q": ["red blue"], "empty": [""]}),
            (("",), {}),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("parser_escape", "edge_empty"),
    ),
    _frontier_task(
        task_id="repair_version_compare",
        prompt=(
            "Compare dotted numeric versions. Return -1 if left is lower, 1 if "
            "left is higher, and 0 when they are equal after trimming trailing "
            "zero components."
        ),
        signature="def solve(left, right)",
        buggy_code="""
def solve(left, right):
    return 0 if left == right else 1
""",
        tests=(
            (("1.2.0", "1.2"), 0),
            (("1.10", "1.2"), 1),
            (("2.0", "2.0.1"), -1),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("ordering", "edge_empty"),
    ),
    _frontier_existing_task(
        "repair_canonical_ranges",
        family="interval",
        difficulty=3,
        failure_tags=("ordering", "dedupe"),
    ),
    _frontier_task(
        task_id="repair_calendar_gaps",
        prompt=(
            "Given a start day, end day, and busy inclusive intervals, return "
            "free inclusive intervals within the range after merging busy spans."
        ),
        signature="def solve(start, end, busy)",
        buggy_code="""
def solve(start, end, busy):
    return [[start, end]]
""",
        tests=(
            ((1, 10, [[2, 3], [5, 5], [8, 12]]), [[1, 1], [4, 4], [6, 7]]),
            ((1, 3, []), [[1, 3]]),
            ((1, 3, [[1, 3]]), []),
        ),
        family="interval",
        difficulty=3,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_task(
        task_id="repair_rate_limit_windows",
        prompt=(
            "Given sorted event timestamps and a window size, return the maximum "
            "number of events in any inclusive window [t, t + window]."
        ),
        signature="def solve(timestamps, window)",
        buggy_code="""
def solve(timestamps, window):
    return len(timestamps)
""",
        tests=(
            (([1, 2, 4, 8, 9], 3), 3),
            (([5, 5, 5], 0), 3),
            (([], 10), 0),
        ),
        family="interval",
        difficulty=4,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_task(
        task_id="repair_schedule_conflicts",
        prompt=(
            "Given intervals [start, end), return the sorted intervals that "
            "overlap any other interval. Adjacent endpoints do not conflict."
        ),
        signature="def solve(intervals)",
        buggy_code="""
def solve(intervals):
    return []
""",
        tests=(
            (([[1, 3], [3, 5], [4, 6], [7, 8]],), [[3, 5], [4, 6]]),
            (([[1, 4], [2, 3], [3, 6]],), [[1, 4], [2, 3], [3, 6]]),
            (([],), []),
        ),
        family="interval",
        difficulty=3,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_existing_task(
        "repair_lru_cache_trace",
        family="state_machine",
        difficulty=4,
        failure_tags=("state_eviction", "ordering"),
    ),
    _frontier_task(
        task_id="repair_undo_redo_stack",
        prompt=(
            "Apply editor operations. ['type', text] appends text, ['undo'] "
            "reverts the last type, and ['redo'] reapplies the most recently "
            "undone type unless a new type clears redo history."
        ),
        signature="def solve(operations)",
        buggy_code="""
def solve(operations):
    return ""
""",
        tests=(
            (([["type", "a"], ["type", "b"], ["undo"], ["redo"]],), "ab"),
            (([["type", "a"], ["undo"], ["type", "c"], ["redo"]],), "c"),
        ),
        family="state_machine",
        difficulty=3,
        failure_tags=("state_transition", "state_eviction"),
    ),
    _frontier_task(
        task_id="repair_inventory_reconcile",
        prompt=(
            "Apply inventory deltas to an initial stock dictionary. Drop items "
            "whose final count is zero and return sorted [item, count] pairs."
        ),
        signature="def solve(initial, deltas)",
        buggy_code="""
def solve(initial, deltas):
    return sorted(initial.items())
""",
        tests=(
            (({"a": 2, "b": 1}, [["a", -1], ["c", 5], ["b", -1]]), [["a", 1], ["c", 5]]),
            (({}, [["x", 1], ["x", -1]]), []),
        ),
        family="state_machine",
        difficulty=3,
        failure_tags=("mutation", "none_sentinel"),
    ),
    _frontier_task(
        task_id="repair_streaming_sessionize",
        prompt=(
            "Group sorted event timestamps into sessions. A new session starts "
            "when the gap from the previous timestamp is greater than max_gap. "
            "Return [start, end, count] for each session."
        ),
        signature="def solve(timestamps, max_gap)",
        buggy_code="""
def solve(timestamps, max_gap):
    return []
""",
        tests=(
            (([1, 2, 6, 7, 20], 3), [[1, 2, 2], [6, 7, 2], [20, 20, 1]]),
            (([], 5), []),
        ),
        family="state_machine",
        difficulty=3,
        failure_tags=("boundary", "state_transition"),
    ),
    _frontier_task(
        task_id="repair_debounce_events",
        prompt=(
            "Given [time, key, value] events sorted by time, keep only the last "
            "event for each key in a burst where the next same-key event occurs "
            "within wait time. Return kept [time, key, value] events in input "
            "order."
        ),
        signature="def solve(events, wait)",
        buggy_code="""
def solve(events, wait):
    return events
""",
        tests=(
            (
                ([[1, "a", "x"], [3, "a", "y"], [10, "a", "z"]], 3),
                [[3, "a", "y"], [10, "a", "z"]],
            ),
            (([[1, "a", "x"], [2, "b", "y"], [3, "a", "z"]], 1), [[1, "a", "x"], [2, "b", "y"], [3, "a", "z"]]),
        ),
        family="state_machine",
        difficulty=4,
        failure_tags=("state_eviction", "ordering"),
    ),
    _frontier_existing_task(
        "repair_topological_layers",
        family="graph",
        difficulty=4,
        failure_tags=("dependency_cycle", "ordering"),
    ),
    _frontier_task(
        task_id="repair_dependency_closure",
        prompt=(
            "Given a dependency map from node to direct dependencies, return all "
            "transitive dependencies of target sorted alphabetically."
        ),
        signature="def solve(dependencies, target)",
        buggy_code="""
def solve(dependencies, target):
    return dependencies.get(target, [])
""",
        tests=(
            (({"app": ["db", "api"], "api": ["auth"], "db": ["auth"]}, "app"), ["api", "auth", "db"]),
            (({"a": ["b"], "b": ["a"]}, "a"), ["b"]),
        ),
        family="graph",
        difficulty=3,
        failure_tags=("dependency_cycle", "dedupe"),
    ),
    _frontier_task(
        task_id="repair_shortest_grid_path",
        prompt=(
            "Return the shortest 4-neighbor path length from start to goal in a "
            "grid of strings where '#' is blocked. Return -1 when unreachable."
        ),
        signature="def solve(grid, start, goal)",
        buggy_code="""
def solve(grid, start, goal):
    return -1
""",
        tests=(
            ((["..", ".."], [0, 0], [1, 1]), 2),
            (([".#", "#."], [0, 0], [1, 1]), -1),
        ),
        family="graph",
        difficulty=4,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_task(
        task_id="repair_permission_inheritance",
        prompt=(
            "Given role parents and direct grants, return sorted effective "
            "permissions for a role including inherited parent permissions."
        ),
        signature="def solve(role, parents, grants)",
        buggy_code="""
def solve(role, parents, grants):
    return sorted(grants.get(role, []))
""",
        tests=(
            (("admin", {"admin": "editor", "editor": "viewer"}, {"viewer": ["read"], "editor": ["write"], "admin": ["delete"]}), ["delete", "read", "write"]),
            (("guest", {}, {}), []),
        ),
        family="graph",
        difficulty=4,
        failure_tags=("dependency_cycle", "dedupe"),
    ),
    _frontier_task(
        task_id="repair_cycle_explain",
        prompt=(
            "Given directed edges, return sorted nodes that belong to at least "
            "one directed cycle. Return [] when the graph is acyclic."
        ),
        signature="def solve(edges)",
        buggy_code="""
def solve(edges):
    return []
""",
        tests=(
            (([["a", "b"], ["b", "c"], ["c", "a"], ["c", "d"]],), ["a", "b", "c"]),
            (([["a", "b"], ["b", "c"]],), []),
        ),
        family="graph",
        difficulty=5,
        failure_tags=("dependency_cycle", "ordering"),
    ),
    _frontier_task(
        task_id="repair_json_pointer_get",
        prompt=(
            "Resolve a JSON Pointer path against nested dictionaries and lists. "
            "Support ~1 for slash and ~0 for tilde. Return None when missing."
        ),
        signature="def solve(document, pointer)",
        buggy_code="""
def solve(document, pointer):
    return document.get(pointer)
""",
        tests=(
            (({"a": [{"b": 3}]}, "/a/0/b"), 3),
            (({"a/b": {"~": 7}}, "/a~1b/~0"), 7),
            (({"a": []}, "/a/0"), None),
        ),
        family="data_model",
        difficulty=3,
        failure_tags=("parser_escape", "none_sentinel"),
    ),
    _frontier_task(
        task_id="repair_patch_apply",
        prompt=(
            "Apply top-level patches to a dictionary. Operations are "
            "['set', key, value] and ['delete', key]. Return the patched copy."
        ),
        signature="def solve(document, patches)",
        buggy_code="""
def solve(document, patches):
    return document
""",
        tests=(
            (({"a": 1}, [["set", "b", 2], ["delete", "a"]]), {"b": 2}),
            (({"x": 1}, [["delete", "missing"]]), {"x": 1}),
        ),
        family="data_model",
        difficulty=4,
        failure_tags=("mutation", "none_sentinel"),
    ),
    _frontier_task(
        task_id="repair_group_anagrams",
        prompt=(
            "Group words by anagram signature. Return groups sorted by their "
            "first word's first appearance, and preserve word order inside each "
            "group."
        ),
        signature="def solve(words)",
        buggy_code="""
def solve(words):
    return [[word] for word in words]
""",
        tests=(
            ((["eat", "tea", "tan", "ate", "nat"],), [["eat", "tea", "ate"], ["tan", "nat"]]),
            (([],), []),
        ),
        family="data_model",
        difficulty=2,
        failure_tags=("ordering", "dedupe"),
    ),
    _frontier_task(
        task_id="repair_nested_defaults",
        prompt=(
            "Deep-fill missing dictionary keys from defaults without replacing "
            "existing non-dictionary values. Return a new merged dictionary."
        ),
        signature="def solve(value, defaults)",
        buggy_code="""
def solve(value, defaults):
    result = dict(defaults)
    result.update(value)
    return result
""",
        tests=(
            (({"a": {"x": 1}}, {"a": {"x": 0, "y": 2}, "b": 3}), {"a": {"x": 1, "y": 2}, "b": 3}),
            (({"a": 5}, {"a": {"x": 1}}), {"a": 5}),
        ),
        family="data_model",
        difficulty=4,
        failure_tags=("mutation", "aliasing"),
    ),
    _frontier_task(
        task_id="repair_schema_errors",
        prompt=(
            "Validate a record against a schema mapping field to type name "
            "('int', 'str', 'bool', or 'list'). Return sorted missing or "
            "wrong-type field names."
        ),
        signature="def solve(record, schema)",
        buggy_code="""
def solve(record, schema):
    return []
""",
        tests=(
            (({"a": 1, "b": "x"}, {"a": "int", "b": "str"}), []),
            (({"a": True}, {"a": "int", "b": "list"}), ["a", "b"]),
        ),
        family="data_model",
        difficulty=3,
        failure_tags=("none_sentinel", "ordering"),
    ),
    _frontier_existing_task(
        "repair_sliding_window_max",
        family="numeric",
        difficulty=4,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_task(
        task_id="repair_running_median_small",
        prompt=(
            "Return the median after each prefix of the input list. For even "
            "prefix lengths, use the average of the two middle values."
        ),
        signature="def solve(values)",
        buggy_code="""
def solve(values):
    return values
""",
        tests=(
            (([3, 1, 5, 2],), [3, 2.0, 3, 2.5]),
            (([],), []),
        ),
        family="numeric",
        difficulty=4,
        failure_tags=("ordering", "edge_empty"),
    ),
    _frontier_task(
        task_id="repair_percentile_bucket",
        prompt=(
            "Assign each value to the first threshold it is less than or equal "
            "to. Return bucket indices, where values above all thresholds use "
            "len(thresholds)."
        ),
        signature="def solve(values, thresholds)",
        buggy_code="""
def solve(values, thresholds):
    return [0 for value in values]
""",
        tests=(
            (([1, 5, 10, 11], [3, 10]), [0, 1, 1, 2]),
            (([], [1]), []),
        ),
        family="numeric",
        difficulty=3,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_task(
        task_id="repair_currency_rounding",
        prompt=(
            "Given integer cents and integer basis points, add the percentage "
            "fee using half-up rounding to the nearest cent."
        ),
        signature="def solve(cents, basis_points)",
        buggy_code="""
def solve(cents, basis_points):
    return cents + cents * basis_points // 10000
""",
        tests=(
            ((1000, 250), 1025),
            ((999, 333), 1032),
            ((1, 5000), 2),
        ),
        family="numeric",
        difficulty=3,
        failure_tags=("rounding", "boundary"),
    ),
    _frontier_task(
        task_id="repair_histogram_bins",
        prompt=(
            "Count values into half-open bins [start, end), except the final bin "
            "includes its right endpoint. Return counts for each bin."
        ),
        signature="def solve(values, bins)",
        buggy_code="""
def solve(values, bins):
    return [0 for item in bins]
""",
        tests=(
            (([0, 1, 2, 3], [[0, 2], [2, 3]]), [2, 2]),
            (([5], [[0, 5]]), [1]),
        ),
        family="numeric",
        difficulty=3,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_task(
        task_id="repair_mutation_aliasing",
        prompt=(
            "Return a matrix with value added to each cell without mutating the "
            "input matrix or reusing row objects in the result."
        ),
        signature="def solve(matrix, value)",
        buggy_code="""
def solve(matrix, value):
    for row in matrix:
        for index in range(len(row)):
            row[index] += value
    return matrix
""",
        tests=(
            (([[1, 2], [3]], 10), [[11, 12], [13]]),
            (([], 5), []),
        ),
        family="real_bug_pattern",
        difficulty=4,
        failure_tags=("mutation", "aliasing"),
    ),
    _frontier_task(
        task_id="repair_none_sentinel",
        prompt=(
            "Return default only when value is None. Other falsy values such as "
            "0, empty string, False, and [] must be returned unchanged."
        ),
        signature="def solve(value, default)",
        buggy_code="""
def solve(value, default):
    return value or default
""",
        tests=(
            ((None, "x"), "x"),
            ((0, 7), 0),
            ((False, True), False),
        ),
        family="real_bug_pattern",
        difficulty=2,
        failure_tags=("none_sentinel", "truthiness"),
    ),
    _frontier_task(
        task_id="repair_stable_sort_tie",
        prompt=(
            "Sort records by score descending while preserving original order "
            "for equal scores. Return record names."
        ),
        signature="def solve(records)",
        buggy_code="""
def solve(records):
    return [record["name"] for record in sorted(records)]
""",
        tests=(
            (([{"name": "a", "score": 2}, {"name": "b", "score": 2}, {"name": "c", "score": 3}],), ["c", "a", "b"]),
            (([],), []),
        ),
        family="real_bug_pattern",
        difficulty=2,
        failure_tags=("ordering", "stable_sort"),
    ),
    _frontier_task(
        task_id="repair_unicode_slug",
        prompt=(
            "Create a lowercase slug by keeping Unicode alphanumeric "
            "characters, replacing runs of other characters with one hyphen, "
            "and trimming leading or trailing hyphens."
        ),
        signature="def solve(text)",
        buggy_code="""
def solve(text):
    return text.lower().replace(" ", "-")
""",
        tests=(
            (("Hello, World!",), "hello-world"),
            ((" Caf\u00e9  Mundo ",), "caf\u00e9-mundo"),
            (("---",), ""),
        ),
        family="real_bug_pattern",
        difficulty=3,
        failure_tags=("unicode", "parser_escape"),
    ),
    _frontier_task(
        task_id="repair_boundary_pagination",
        prompt=(
            "Return items for a 1-based page number and positive page size. "
            "Pages before 1 return the first page; pages beyond the end return "
            "an empty list."
        ),
        signature="def solve(items, page, size)",
        buggy_code="""
def solve(items, page, size):
    start = page * size
    return items[start:start + size]
""",
        tests=(
            (([1, 2, 3, 4, 5], 1, 2), [1, 2]),
            (([1, 2, 3, 4, 5], 3, 2), [5]),
            (([1, 2, 3], 0, 2), [1, 2]),
        ),
        family="real_bug_pattern",
        difficulty=2,
        failure_tags=("off_by_one", "boundary"),
    ),
)


_FRONTIER_HELDOUT_TASKS = (
    _frontier_existing_heldout_task(
        "repair_clamp",
        family="sequence",
        difficulty=1,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_existing_heldout_task(
        "repair_rotate_left",
        family="sequence",
        difficulty=2,
        failure_tags=("modulo", "edge_empty"),
    ),
    _frontier_existing_heldout_task(
        "repair_dedupe_order",
        family="sequence",
        difficulty=1,
        failure_tags=("ordering", "mutation"),
    ),
    _frontier_heldout_task(
        task_id="repair_chunk_list",
        prompt="Split items into consecutive chunks of the requested positive size.",
        signature="def solve(items, size)",
        tests=(
            (([1, 2, 3, 4], 2), [[1, 2], [3, 4]]),
            (([1], 3), [[1]]),
        ),
        family="sequence",
        difficulty=1,
        failure_tags=("boundary", "edge_empty"),
    ),
    _frontier_heldout_task(
        task_id="repair_run_length_encode",
        prompt="Compress consecutive equal values as [value, count] pairs.",
        signature="def solve(items)",
        tests=(
            (([1, 2, 2, 1],), [[1, 1], [2, 2], [1, 1]]),
            ((["x"],), [["x", 1]]),
        ),
        family="sequence",
        difficulty=2,
        failure_tags=("state_transition", "edge_empty"),
    ),
    _frontier_existing_heldout_task(
        "repair_balanced_brackets",
        family="string_parse",
        difficulty=2,
        failure_tags=("parser_nesting", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_normalize_path",
        prompt="Normalize slash-separated paths with dot and parent segments.",
        signature="def solve(path)",
        tests=(
            (("/x/y/../../z",), "/z"),
            (("a/./b//c",), "/a/b/c"),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("parser_escape", "edge_empty"),
    ),
    _frontier_existing_heldout_task(
        "repair_merge_intervals",
        family="interval",
        difficulty=2,
        failure_tags=("ordering", "boundary"),
    ),
    _frontier_heldout_task(
        task_id="repair_csv_row_split",
        prompt="Split one CSV row into fields, including quoted commas and quotes.",
        signature="def solve(row)",
        tests=(
            (('"x","y,z","q""r"',), ["x", "y,z", 'q"r']),
            (("",), [""]),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("parser_escape", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_query_params",
        prompt="Parse query string keys into ordered value lists.",
        signature="def solve(query)",
        tests=(
            (("a=1&a=&b=two+words",), {"a": ["1", ""], "b": ["two words"]}),
            (("flag",), {"flag": [""]}),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("parser_escape", "edge_empty"),
    ),
    _frontier_heldout_task(
        task_id="repair_version_compare",
        prompt="Compare dotted numeric versions with trailing-zero normalization.",
        signature="def solve(left, right)",
        tests=(
            (("0.9.9", "1.0"), -1),
            (("3.0.0", "3"), 0),
        ),
        family="string_parse",
        difficulty=3,
        failure_tags=("ordering", "edge_empty"),
    ),
    _frontier_existing_heldout_task(
        "repair_canonical_ranges",
        family="interval",
        difficulty=3,
        failure_tags=("ordering", "dedupe"),
    ),
    _frontier_heldout_task(
        task_id="repair_calendar_gaps",
        prompt="Return free inclusive intervals after merging busy spans.",
        signature="def solve(start, end, busy)",
        tests=(
            ((0, 5, [[-2, 1], [3, 4]]), [[2, 2], [5, 5]]),
            ((3, 3, []), [[3, 3]]),
        ),
        family="interval",
        difficulty=3,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_rate_limit_windows",
        prompt="Return the maximum event count in any inclusive time window.",
        signature="def solve(timestamps, window)",
        tests=(
            (([1, 4, 4, 5, 9], 1), 3),
            (([1, 10], 8), 1),
        ),
        family="interval",
        difficulty=4,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_heldout_task(
        task_id="repair_schedule_conflicts",
        prompt="Return half-open intervals that overlap at least one other interval.",
        signature="def solve(intervals)",
        tests=(
            (([[0, 1], [1, 2], [1, 3]],), [[1, 2], [1, 3]]),
            (([[5, 6]],), []),
        ),
        family="interval",
        difficulty=3,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_existing_heldout_task(
        "repair_lru_cache_trace",
        family="state_machine",
        difficulty=4,
        failure_tags=("state_eviction", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_undo_redo_stack",
        prompt="Apply type, undo, and redo operations to text.",
        signature="def solve(operations)",
        tests=(
            (([["type", "ab"], ["undo"], ["redo"], ["undo"]],), ""),
            (([["undo"], ["type", "x"], ["redo"]],), "x"),
        ),
        family="state_machine",
        difficulty=3,
        failure_tags=("state_transition", "state_eviction"),
    ),
    _frontier_heldout_task(
        task_id="repair_inventory_reconcile",
        prompt="Apply inventory deltas and drop zero-count items.",
        signature="def solve(initial, deltas)",
        tests=(
            (({"b": 2}, [["a", 1], ["b", -2], ["a", 2]]), [["a", 3]]),
            (({"x": -1}, [["x", 1]]), []),
        ),
        family="state_machine",
        difficulty=3,
        failure_tags=("mutation", "none_sentinel"),
    ),
    _frontier_heldout_task(
        task_id="repair_streaming_sessionize",
        prompt="Group sorted timestamps into sessions using a max gap.",
        signature="def solve(timestamps, max_gap)",
        tests=(
            (([5, 5, 6, 12], 1), [[5, 6, 3], [12, 12, 1]]),
            (([1], 0), [[1, 1, 1]]),
        ),
        family="state_machine",
        difficulty=3,
        failure_tags=("boundary", "state_transition"),
    ),
    _frontier_heldout_task(
        task_id="repair_debounce_events",
        prompt="Keep only the last same-key event inside each debounce burst.",
        signature="def solve(events, wait)",
        tests=(
            (([[1, "a", 1], [2, "a", 2], [4, "a", 3]], 2), [[4, "a", 3]]),
            (([], 3), []),
        ),
        family="state_machine",
        difficulty=4,
        failure_tags=("state_eviction", "ordering"),
    ),
    _frontier_existing_heldout_task(
        "repair_topological_layers",
        family="graph",
        difficulty=4,
        failure_tags=("dependency_cycle", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_dependency_closure",
        prompt="Return sorted transitive dependencies for a target.",
        signature="def solve(dependencies, target)",
        tests=(
            (({"root": ["a"], "a": ["b"], "b": ["c"]}, "root"), ["a", "b", "c"]),
            (({}, "x"), []),
        ),
        family="graph",
        difficulty=3,
        failure_tags=("dependency_cycle", "dedupe"),
    ),
    _frontier_heldout_task(
        task_id="repair_shortest_grid_path",
        prompt="Return shortest grid path length with walls.",
        signature="def solve(grid, start, goal)",
        tests=(
            ((["...", ".#.", "..."], [0, 0], [2, 2]), 4),
            ((["#"], [0, 0], [0, 0]), -1),
        ),
        family="graph",
        difficulty=4,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_permission_inheritance",
        prompt="Return inherited and direct permissions for a role.",
        signature="def solve(role, parents, grants)",
        tests=(
            (("child", {"child": "base"}, {"base": ["read"], "child": ["read", "write"]}), ["read", "write"]),
            (("base", {"x": "base"}, {"base": ["read"]}), ["read"]),
        ),
        family="graph",
        difficulty=4,
        failure_tags=("dependency_cycle", "dedupe"),
    ),
    _frontier_heldout_task(
        task_id="repair_cycle_explain",
        prompt="Return sorted directed-cycle nodes.",
        signature="def solve(edges)",
        tests=(
            (([["a", "b"], ["b", "a"], ["c", "d"], ["d", "c"]],), ["a", "b", "c", "d"]),
            (([],), []),
        ),
        family="graph",
        difficulty=5,
        failure_tags=("dependency_cycle", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_json_pointer_get",
        prompt="Resolve JSON Pointer paths with ~ escapes.",
        signature="def solve(document, pointer)",
        tests=(
            ((["zero", {"a": 1}], "/1/a"), 1),
            (({"": 5}, ""), {"": 5}),
        ),
        family="data_model",
        difficulty=3,
        failure_tags=("parser_escape", "none_sentinel"),
    ),
    _frontier_heldout_task(
        task_id="repair_patch_apply",
        prompt="Apply top-level set and delete patches to a dictionary copy.",
        signature="def solve(document, patches)",
        tests=(
            (({"a": 1}, [["set", "a", 2]]), {"a": 2}),
            (({}, [["delete", "x"], ["set", "x", 1]]), {"x": 1}),
        ),
        family="data_model",
        difficulty=4,
        failure_tags=("mutation", "none_sentinel"),
    ),
    _frontier_heldout_task(
        task_id="repair_group_anagrams",
        prompt="Group anagrams preserving first group and word order.",
        signature="def solve(words)",
        tests=(
            ((["bob", "obb", "cat", "act", "dog"],), [["bob", "obb"], ["cat", "act"], ["dog"]]),
            ((["aa", "aa"],), [["aa", "aa"]]),
        ),
        family="data_model",
        difficulty=2,
        failure_tags=("ordering", "dedupe"),
    ),
    _frontier_heldout_task(
        task_id="repair_nested_defaults",
        prompt="Deep-fill missing dictionary keys from defaults.",
        signature="def solve(value, defaults)",
        tests=(
            (({}, {"a": {"b": 1}}), {"a": {"b": 1}}),
            (({"a": {"b": 2, "c": 3}}, {"a": {"b": 1}}), {"a": {"b": 2, "c": 3}}),
        ),
        family="data_model",
        difficulty=4,
        failure_tags=("mutation", "aliasing"),
    ),
    _frontier_heldout_task(
        task_id="repair_schema_errors",
        prompt="Return missing or wrong-type schema fields.",
        signature="def solve(record, schema)",
        tests=(
            (({"items": []}, {"items": "list", "ok": "bool"}), ["ok"]),
            (({"flag": 1}, {"flag": "bool"}), ["flag"]),
        ),
        family="data_model",
        difficulty=3,
        failure_tags=("none_sentinel", "ordering"),
    ),
    _frontier_existing_heldout_task(
        "repair_sliding_window_max",
        family="numeric",
        difficulty=4,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_heldout_task(
        task_id="repair_running_median_small",
        prompt="Return prefix medians.",
        signature="def solve(values)",
        tests=(
            (([1, 2, 3, 4],), [1, 1.5, 2, 2.5]),
            (([5, -1],), [5, 2.0]),
        ),
        family="numeric",
        difficulty=4,
        failure_tags=("ordering", "edge_empty"),
    ),
    _frontier_heldout_task(
        task_id="repair_percentile_bucket",
        prompt="Assign values to ordered threshold buckets.",
        signature="def solve(values, thresholds)",
        tests=(
            (([-1, 0, 5], [0, 4]), [0, 0, 2]),
            (([10], []), [0]),
        ),
        family="numeric",
        difficulty=3,
        failure_tags=("boundary", "ordering"),
    ),
    _frontier_heldout_task(
        task_id="repair_currency_rounding",
        prompt="Add basis-point fee with half-up cent rounding.",
        signature="def solve(cents, basis_points)",
        tests=(
            ((250, 125), 253),
            ((333, 333), 344),
        ),
        family="numeric",
        difficulty=3,
        failure_tags=("rounding", "boundary"),
    ),
    _frontier_heldout_task(
        task_id="repair_histogram_bins",
        prompt="Count values into half-open bins with final inclusive right edge.",
        signature="def solve(values, bins)",
        tests=(
            (([1, 2, 2, 3], [[1, 2], [2, 3]]), [1, 3]),
            (([], [[0, 1]]), [0]),
        ),
        family="numeric",
        difficulty=3,
        failure_tags=("boundary", "off_by_one"),
    ),
    _frontier_heldout_task(
        task_id="repair_mutation_aliasing",
        prompt="Add to matrix values without mutating or aliasing rows.",
        signature="def solve(matrix, value)",
        tests=(
            (([[0], [1, 2]], -1), [[-1], [0, 1]]),
            (([[], [2]], 3), [[], [5]]),
        ),
        family="real_bug_pattern",
        difficulty=4,
        failure_tags=("mutation", "aliasing"),
    ),
    _frontier_heldout_task(
        task_id="repair_none_sentinel",
        prompt="Use the default only for None, not other falsy values.",
        signature="def solve(value, default)",
        tests=(
            (("", "fallback"), ""),
            (([], [1]), []),
        ),
        family="real_bug_pattern",
        difficulty=2,
        failure_tags=("none_sentinel", "truthiness"),
    ),
    _frontier_heldout_task(
        task_id="repair_stable_sort_tie",
        prompt="Sort by score descending while preserving equal-score order.",
        signature="def solve(records)",
        tests=(
            (([{"name": "x", "score": 1}, {"name": "y", "score": 1}],), ["x", "y"]),
            (([{"name": "z", "score": -1}, {"name": "a", "score": 0}],), ["a", "z"]),
        ),
        family="real_bug_pattern",
        difficulty=2,
        failure_tags=("ordering", "stable_sort"),
    ),
    _frontier_heldout_task(
        task_id="repair_unicode_slug",
        prompt="Create a lowercase Unicode-aware slug.",
        signature="def solve(text)",
        tests=(
            (("na\u00efve test",), "na\u00efve-test"),
            (("A__B",), "a-b"),
        ),
        family="real_bug_pattern",
        difficulty=3,
        failure_tags=("unicode", "parser_escape"),
    ),
    _frontier_heldout_task(
        task_id="repair_boundary_pagination",
        prompt="Return 1-based page slices with lower-bound page clamping.",
        signature="def solve(items, page, size)",
        tests=(
            (([1, 2, 3], 2, 5), []),
            (([1, 2, 3, 4], -3, 3), [1, 2, 3]),
        ),
        family="real_bug_pattern",
        difficulty=2,
        failure_tags=("off_by_one", "boundary"),
    ),
)


_FRONTIER_TASKS_BY_ID = {task.id: task for task in _FRONTIER_TASKS}


def _frontier_tasks_by_id(task_ids: Sequence[str]) -> tuple[PythonRepairTask, ...]:
    return tuple(_FRONTIER_TASKS_BY_ID[task_id] for task_id in task_ids)


_FRONTIER_SMOKE_TASKS = _frontier_tasks_by_id(
    (
        "repair_clamp",
        "repair_balanced_brackets",
        "repair_merge_intervals",
        "repair_undo_redo_stack",
        "repair_dependency_closure",
        "repair_json_pointer_get",
        "repair_percentile_bucket",
        "repair_none_sentinel",
    )
)
_FRONTIER_BALANCED_TASKS = _frontier_tasks_by_id(
    (
        "repair_clamp",
        "repair_rotate_left",
        "repair_run_length_encode",
        "repair_balanced_brackets",
        "repair_csv_row_split",
        "repair_version_compare",
        "repair_merge_intervals",
        "repair_canonical_ranges",
        "repair_calendar_gaps",
        "repair_lru_cache_trace",
        "repair_undo_redo_stack",
        "repair_streaming_sessionize",
        "repair_topological_layers",
        "repair_dependency_closure",
        "repair_shortest_grid_path",
        "repair_json_pointer_get",
        "repair_group_anagrams",
        "repair_schema_errors",
        "repair_sliding_window_max",
        "repair_percentile_bucket",
        "repair_currency_rounding",
        "repair_none_sentinel",
        "repair_stable_sort_tie",
        "repair_boundary_pagination",
    )
)
_FRONTIER_HARD_TASKS = _FRONTIER_TASKS[8:]
_FRONTIER_FULL_TASKS = _FRONTIER_TASKS


_FOUNDRY_CORPUS_REGISTRY = {
    "embedded_standard": FoundryCorpusSpec(
        name="embedded_standard",
        description="Original embedded Python repair corpus.",
        source="embedded",
        train_tasks=_FOUNDRY_TASKS,
        heldout_tasks=_FOUNDRY_STANDARD_HELDOUT_TASKS,
    ),
    "embedded_hard": FoundryCorpusSpec(
        name="embedded_hard",
        description="Initial hard Python repair corpus.",
        source="embedded",
        train_tasks=_FOUNDRY_HARD_TASKS,
        heldout_tasks=_FOUNDRY_HARD_HELDOUT_TASKS,
    ),
    "frontier_ladder_v1": FoundryCorpusSpec(
        name="frontier_ladder_v1",
        description="Hand-rolled non-saturating frontier repair ladder.",
        source="embedded",
        train_tasks=_FRONTIER_FULL_TASKS,
        heldout_tasks=_FRONTIER_HELDOUT_TASKS,
    ),
}
FOUNDRY_OPTIONAL_EXTERNAL_CORPORA = (
    "evalplus",
    "bigcodebench",
    "quixbugs",
    "bugsinpy",
    "swe_bench",
    "livecodebench",
)


_FOUNDRY_TASK_SPLITS = {
    "standard": (_FOUNDRY_TASKS, ()),
    "standard_heldout": (_FOUNDRY_TASKS, _FOUNDRY_STANDARD_HELDOUT_TASKS),
    "hard": (_FOUNDRY_HARD_TASKS, ()),
    "hard_heldout": (_FOUNDRY_HARD_TASKS, _FOUNDRY_HARD_HELDOUT_TASKS),
    "mixed_heldout": (
        _FOUNDRY_TASKS + _FOUNDRY_HARD_TASKS,
        _FOUNDRY_STANDARD_HELDOUT_TASKS + _FOUNDRY_HARD_HELDOUT_TASKS,
    ),
    "frontier_smoke": (
        _FRONTIER_SMOKE_TASKS,
        _heldout_for_train(_FRONTIER_HELDOUT_TASKS, _FRONTIER_SMOKE_TASKS),
    ),
    "frontier_balanced": (
        _FRONTIER_BALANCED_TASKS,
        _heldout_for_train(_FRONTIER_HELDOUT_TASKS, _FRONTIER_BALANCED_TASKS),
    ),
    "frontier_hard": (
        _FRONTIER_HARD_TASKS,
        _heldout_for_train(_FRONTIER_HELDOUT_TASKS, _FRONTIER_HARD_TASKS),
    ),
    "frontier_full": (
        _FRONTIER_FULL_TASKS,
        _heldout_for_train(_FRONTIER_HELDOUT_TASKS, _FRONTIER_FULL_TASKS),
    ),
}
_ALL_FOUNDRY_TASKS_BY_ID = {
    task.id: task
    for bank in (
        _FOUNDRY_TASKS,
        _FOUNDRY_STANDARD_HELDOUT_TASKS,
        _FOUNDRY_HARD_TASKS,
        _FOUNDRY_HARD_HELDOUT_TASKS,
        _FRONTIER_FULL_TASKS,
        _FRONTIER_HELDOUT_TASKS,
    )
    for task in bank
}


def available_foundry_task_splits() -> tuple[str, ...]:
    return tuple(sorted(_FOUNDRY_TASK_SPLITS))


def available_foundry_corpora() -> tuple[str, ...]:
    return tuple(sorted(_FOUNDRY_CORPUS_REGISTRY))


def foundry_task_metadata_index() -> dict[str, dict[str, Any]]:
    return {
        task_id: {
            "task_id": task.id,
            "family": task.family,
            "difficulty": str(task.difficulty),
            "failure_tags": list(task.failure_tags),
        }
        for task_id, task in _ALL_FOUNDRY_TASKS_BY_ID.items()
    }


def _validate_foundry_registries() -> bool:
    for name, (train_tasks, heldout_tasks) in _FOUNDRY_TASK_SPLITS.items():
        _validate_foundry_task_bank(
            name,
            train_tasks=train_tasks,
            heldout_tasks=heldout_tasks,
            require_heldout_subset=True,
        )
    for spec in _FOUNDRY_CORPUS_REGISTRY.values():
        _validate_foundry_corpus_spec(spec)
    return True


def _validate_foundry_corpus_spec(spec: FoundryCorpusSpec) -> None:
    if not spec.name:
        raise ValueError("foundry_corpus_name_required")
    if spec.external_adapter:
        raise ValueError("foundry_external_corpus_not_registered_by_default")
    _validate_foundry_task_bank(
        spec.name,
        train_tasks=spec.train_tasks,
        heldout_tasks=spec.heldout_tasks,
        require_heldout_subset=False,
    )


def _validate_foundry_task_bank(
    name: str,
    *,
    train_tasks: Sequence[PythonRepairTask],
    heldout_tasks: Sequence[PythonRepairTask],
    require_heldout_subset: bool = True,
) -> None:
    if not train_tasks:
        raise ValueError(f"foundry_task_bank_empty: {name}")
    _reject_duplicate_task_ids(name, train_tasks, label="train")
    _reject_duplicate_task_ids(name, heldout_tasks, label="heldout")
    train_ids = {task.id for task in train_tasks}
    if require_heldout_subset:
        unknown_heldout = sorted(task.id for task in heldout_tasks if task.id not in train_ids)
        if unknown_heldout:
            raise ValueError(
                f"foundry_task_bank_unknown_heldout_ids: {name}:"
                + ",".join(unknown_heldout)
            )
    for task in tuple(train_tasks) + tuple(heldout_tasks):
        _validate_foundry_task_metadata(name, task)


def _reject_duplicate_task_ids(
    bank_name: str,
    tasks: Sequence[PythonRepairTask],
    *,
    label: str,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for task in tasks:
        if task.id in seen:
            duplicates.add(task.id)
        seen.add(task.id)
    if duplicates:
        raise ValueError(
            f"foundry_task_bank_duplicate_{label}_ids: {bank_name}:"
            + ",".join(sorted(duplicates))
        )


def _validate_foundry_task_metadata(bank_name: str, task: PythonRepairTask) -> None:
    if not task.id or not task.signature.startswith("def solve"):
        raise ValueError(f"foundry_task_invalid_contract: {bank_name}:{task.id}")
    if task.family not in FOUNDRY_TASK_FAMILIES:
        raise ValueError(f"foundry_task_unknown_family: {bank_name}:{task.id}")
    if task.difficulty < 1 or task.difficulty > 5:
        raise ValueError(f"foundry_task_invalid_difficulty: {bank_name}:{task.id}")
    for tag in task.failure_tags:
        if not tag or not all(char.isalnum() or char == "_" for char in tag):
            raise ValueError(f"foundry_task_invalid_failure_tag: {bank_name}:{task.id}")


_FOUNDRY_TASK_REGISTRY_VALIDATED = _validate_foundry_registries()
