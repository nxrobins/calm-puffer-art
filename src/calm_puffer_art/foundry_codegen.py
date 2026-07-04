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
from typing import Any, Callable, Mapping, Sequence

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


@dataclass(frozen=True)
class PythonRepairTask:
    id: str
    prompt: str
    signature: str
    buggy_code: str
    tests: tuple[tuple[tuple[Any, ...], Any], ...]


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
                {"role": "user", "content": _repair_prompt(task)},
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
        messages=[Message(role="user", content=_repair_prompt(task))],
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
    client_factory: Callable[[str, AzureFoundryCodegenConfig], Any] | None = None,
) -> dict[str, Any]:
    resolved = config or AzureFoundryCodegenConfig()
    resolved.validate()
    tasks = _selected_foundry_tasks(resolved.task_limit)

    static = await _run_foundry_condition(
        name="static_art",
        config=resolved,
        tasks=tasks,
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
        client_factory=client_factory,
    )
    scheduler_only = await _run_foundry_condition(
        name="scheduler_only",
        config=resolved,
        tasks=tasks,
        scheduler=_foundry_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec()],
        client_factory=client_factory,
    )
    full_trinity = await _run_foundry_condition(
        name="full_trinity",
        config=resolved,
        tasks=tasks,
        scheduler=_foundry_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=[
            TokenActionCodec(),
            ChunkActionCodec(chunk_size=2),
            ChunkActionCodec(chunk_size=4),
        ],
        client_factory=client_factory,
    )

    conditions = {
        "static_art": _foundry_summary_metrics(static),
        "scheduler_only": _foundry_summary_metrics(scheduler_only),
        "full_trinity": _foundry_summary_metrics(full_trinity),
    }
    static_score = conditions["static_art"].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0)
    scheduler_score = conditions["scheduler_only"].get(
        CODEGEN_ACCOUNTED_NORTH_STAR,
        0.0,
    )
    full_score = conditions["full_trinity"].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0)
    return {
        "ok": True,
        "proof_scope": "live_azure_foundry_python_repair",
        "measurement": "azure_foundry_codegen_ablation",
        "used_azure_foundry": client_factory is None,
        "deployment": resolved.deployment,
        "api_version": resolved.api_version,
        "env_path": str(resolved.env_path),
        "tasks": len(tasks),
        "max_train_steps": resolved.max_train_steps,
        "model_call_budget_per_condition": resolved.model_call_budget,
        "action_unit_dollar_seconds": resolved.action_unit_dollar_seconds,
        "request_dollar_seconds": resolved.request_dollar_seconds,
        "conditions": conditions,
        "lift": {
            "scheduler_over_static_accounted_north_star_ratio": _finite_ratio(
                scheduler_score,
                static_score,
            ),
            "scheduler_over_static_accounted_north_star_absolute": (
                scheduler_score - static_score
            ),
            "full_trinity_over_static_accounted_north_star_ratio": _finite_ratio(
                full_score,
                static_score,
            ),
            "full_trinity_over_static_accounted_north_star_absolute": (
                full_score - static_score
            ),
            "full_trinity_over_scheduler_accounted_north_star_ratio": _finite_ratio(
                full_score,
                scheduler_score,
            ),
            "full_trinity_over_scheduler_accounted_north_star_absolute": (
                full_score - scheduler_score
            ),
        },
        "winning_condition_by_accounted_north_star": max(
            conditions,
            key=lambda name: conditions[name].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0),
        ),
    }


async def run_azure_foundry_budget_race(
    *,
    config: AzureFoundryCodegenConfig | None = None,
    budget_dollar_seconds: float = DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS,
    client_factory: Callable[[str, AzureFoundryCodegenConfig], Any] | None = None,
) -> dict[str, Any]:
    resolved = config or AzureFoundryCodegenConfig()
    resolved.validate()
    if not isfinite(budget_dollar_seconds) or budget_dollar_seconds <= 0.0:
        raise ValueError("foundry_budget_dollar_seconds_must_be_positive")
    tasks = _selected_foundry_tasks(resolved.task_limit)

    static = await _run_foundry_condition(
        name="static_art",
        config=resolved,
        tasks=tasks,
        scheduler=_foundry_static_budget_scheduler(budget_dollar_seconds),
        action_space=None,
        action_codecs=[TokenActionCodec()],
        client_factory=client_factory,
    )
    scheduler_only = await _run_foundry_condition(
        name="scheduler_only",
        config=resolved,
        tasks=tasks,
        scheduler=_foundry_scheduler(budget_dollar_seconds),
        action_space=None,
        action_codecs=[TokenActionCodec()],
        client_factory=client_factory,
    )
    full_trinity = await _run_foundry_condition(
        name="full_trinity",
        config=resolved,
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

    conditions = {
        "static_art": _foundry_summary_metrics(static),
        "scheduler_only": _foundry_summary_metrics(scheduler_only),
        "full_trinity": _foundry_summary_metrics(full_trinity),
    }
    race = _foundry_budget_race_metrics(conditions)
    return {
        "ok": True,
        "proof_scope": "live_azure_foundry_python_repair",
        "measurement": "azure_foundry_budget_race",
        "used_azure_foundry": client_factory is None,
        "deployment": resolved.deployment,
        "api_version": resolved.api_version,
        "env_path": str(resolved.env_path),
        "tasks": len(tasks),
        "max_train_steps": resolved.max_train_steps,
        "budget_dollar_seconds": budget_dollar_seconds,
        "model_call_budget_per_condition": resolved.model_call_budget,
        "action_unit_dollar_seconds": resolved.action_unit_dollar_seconds,
        "request_dollar_seconds": resolved.request_dollar_seconds,
        "conditions": conditions,
        "race": race,
    }


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


def _foundry_summary_metrics(summary: RunSummary) -> dict[str, float]:
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
    values.update(_foundry_codec_metrics(metrics, _FOUNDRY_TASKS))
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


def _selected_foundry_tasks(task_limit: int) -> tuple[PythonRepairTask, ...]:
    return _FOUNDRY_TASKS[: max(1, min(task_limit, len(_FOUNDRY_TASKS)))]


def _foundry_task_for_scenario(scenario: Scenario) -> PythonRepairTask:
    task = scenario.payload.get("task")
    if isinstance(task, PythonRepairTask):
        return task
    for candidate in _FOUNDRY_TASKS:
        if candidate.id == scenario.id:
            return candidate
    raise ValueError(f"unknown_foundry_codegen_scenario: {scenario.id}")


def _repair_prompt(task: PythonRepairTask) -> str:
    tests = "\n".join(
        f"- solve{args!r} == {expected!r}"
        for args, expected in task.tests
    )
    return textwrap.dedent(
        f"""
        Repair this Python function.

        Requirements:
        - Define exactly one function with signature: {task.signature}
        - Preserve the function name solve.
        - Return only code, no markdown.
        - Do not import modules or perform IO.

        Task:
        {task.prompt}

        Buggy code:
        {task.buggy_code.strip()}

        Unit tests:
        {tests}
        """
    ).strip()


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
            resource.setrlimit(limit, (capped, hard))
        except (OSError, ValueError):
            continue
        applied = True
    return applied

def apply_memory_limit(limit_bytes):
    if limit_bytes <= 0:
        return True
    if os.name == "nt":
        return apply_windows_memory_limit(limit_bytes)
    return apply_posix_memory_limit(limit_bytes)

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
