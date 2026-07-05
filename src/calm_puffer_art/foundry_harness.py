from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from .codegen_ablation import CODEGEN_ACCOUNTED_NORTH_STAR, CODEGEN_NORTH_STAR
from .foundry_codegen import (
    DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_DEPLOYMENT,
    DEFAULT_FOUNDRY_ENV_PATH,
    DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS,
    DEFAULT_FOUNDRY_MODEL_CALL_BUDGET,
    DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S,
    DEFAULT_FOUNDRY_TASK_LIMIT,
    DEFAULT_FOUNDRY_TASK_SPLIT,
    DEFAULT_FOUNDRY_TRAIN_STEPS,
    DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES,
    DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S,
)


FOUNDRY_HARNESS_OBJECTIVE_METRIC = CODEGEN_ACCOUNTED_NORTH_STAR
DEFAULT_FOUNDRY_HARNESS_DIR = Path("harnesses/foundry")
DEFAULT_FOUNDRY_RUNS_DIR = Path(".codex/foundry-runs")
FOUNDRY_HARNESS_ARTIFACT_FILES = {
    "manifest": "manifest.json",
    "result": "result.json",
    "telemetry": "telemetry.jsonl",
    "stdout": "stdout.txt",
    "stderr": "stderr.txt",
    "summary": "summary.json",
    "failures": "failures.json",
}

FOUNDRY_HARNESS_FAILURE_CATEGORIES = (
    "setup_failure",
    "run_timeout",
    "model_request_failure",
    "verifier_timeout",
    "verifier_crash",
    "output_parse_failure",
    "wrong_answer",
    "cost_budget_exhausted",
    "scheduler_stale_batch",
    "scheduler_control_failure",
)

_CONDITION_ORDER = ("static_art", "scheduler_only", "full_trinity")
_MANIFEST_KEYS = {
    "action_codecs",
    "action_unit_dollar_seconds",
    "budget_dollar_seconds",
    "budget_race",
    "conditions",
    "deployment",
    "description",
    "env_path",
    "heartbeat_interval_s",
    "max_completion_tokens",
    "model_call_budget",
    "name",
    "output_contract",
    "primary_condition",
    "promotion_metric",
    "prompt_context_policy",
    "request_dollar_seconds",
    "request_timeout_s",
    "retry_policy",
    "run_timeout_s",
    "scheduler_mode",
    "task_limit",
    "task_split",
    "telemetry_filename",
    "train_steps",
    "verifier",
    "verify_memory_limit_mib",
    "verify_timeout_s",
}
_SUMMARY_METRIC_KEYS = (
    CODEGEN_NORTH_STAR,
    CODEGEN_ACCOUNTED_NORTH_STAR,
    "foundry/learned_solutions",
    "foundry/model_calls",
    "foundry/verifier_passed_rollouts",
    "foundry/observed_rollouts",
    "foundry/source_model_rollouts",
    "foundry/source_memory_rollouts",
    "costs/accounted_dollar_seconds",
    "costs/rollout_dollar_seconds",
    "costs/trainer_dollar_seconds",
    "actions/semantic_bandwidth_tokens_per_decision",
    "scheduler/budget/max_accounted_dollar_seconds",
    "scheduler/budget/accounted_dollar_seconds",
    "scheduler/budget/accounted_fraction",
    "scheduler/budget/accounted_exhausted",
    "scheduler/joint_action/tuples",
    "scheduler/joint_action/decisions",
    "scheduler/joint_action/feedback_updates",
    "action_space/active_codecs",
    "action_space/promotions",
    "action_space/demotions",
    "foundry/codec/token/pulls",
    "foundry/codec/chunk2/pulls",
    "foundry/codec/chunk3/pulls",
    "foundry/codec/chunk4/pulls",
    "foundry/codec/token/failure_rate",
    "foundry/codec/chunk2/failure_rate",
    "foundry/codec/chunk3/failure_rate",
    "foundry/codec/chunk4/failure_rate",
)


@dataclass(frozen=True)
class FoundryHarnessManifest:
    name: str
    description: str = ""
    primary_condition: str = "full_trinity"
    budget_race: bool = True
    env_path: Path = DEFAULT_FOUNDRY_ENV_PATH
    deployment: str = DEFAULT_FOUNDRY_DEPLOYMENT
    train_steps: int = DEFAULT_FOUNDRY_TRAIN_STEPS
    task_limit: int = DEFAULT_FOUNDRY_TASK_LIMIT
    model_call_budget: int = DEFAULT_FOUNDRY_MODEL_CALL_BUDGET
    max_completion_tokens: int = DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS
    request_timeout_s: float = DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S
    verify_timeout_s: float = DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S
    request_dollar_seconds: float = DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS
    action_unit_dollar_seconds: float = DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS
    verify_memory_limit_mib: int = (
        DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES // (1024 * 1024)
    )
    budget_dollar_seconds: float = DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS
    run_timeout_s: float = 3600.0
    heartbeat_interval_s: float = 30.0
    telemetry_filename: str = "telemetry.jsonl"
    prompt_context_policy: str = "repair_prompt_only"
    output_contract: str = "python_solve_function_only"
    verifier: str = "isolated_subprocess_unit_tests"
    retry_policy: str = "none"
    task_split: str = DEFAULT_FOUNDRY_TASK_SPLIT
    scheduler_mode: str = "full_trinity"
    action_codecs: tuple[str, ...] = ("token", "chunk2", "chunk4")
    conditions: tuple[str, ...] = ("full_trinity",)
    promotion_metric: str = FOUNDRY_HARNESS_OBJECTIVE_METRIC

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FoundryHarnessManifest":
        unknown = sorted(set(values) - _MANIFEST_KEYS)
        if unknown:
            raise ValueError(
                "unknown_foundry_harness_manifest_keys: " + ",".join(unknown)
            )
        if "name" not in values:
            raise ValueError("foundry_harness_manifest_name_required")
        primary_condition = _string_value(
            values,
            "primary_condition",
            "full_trinity",
        )
        manifest = cls(
            name=_string_value(values, "name", ""),
            description=_string_value(values, "description", ""),
            primary_condition=primary_condition,
            budget_race=_bool_value(values, "budget_race", True),
            env_path=Path(_string_value(values, "env_path", str(DEFAULT_FOUNDRY_ENV_PATH))),
            deployment=_string_value(values, "deployment", DEFAULT_FOUNDRY_DEPLOYMENT),
            train_steps=_int_value(
                values,
                "train_steps",
                DEFAULT_FOUNDRY_TRAIN_STEPS,
            ),
            task_limit=_int_value(values, "task_limit", DEFAULT_FOUNDRY_TASK_LIMIT),
            model_call_budget=_int_value(
                values,
                "model_call_budget",
                DEFAULT_FOUNDRY_MODEL_CALL_BUDGET,
            ),
            max_completion_tokens=_int_value(
                values,
                "max_completion_tokens",
                DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS,
            ),
            request_timeout_s=_float_value(
                values,
                "request_timeout_s",
                DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S,
            ),
            verify_timeout_s=_float_value(
                values,
                "verify_timeout_s",
                DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S,
            ),
            request_dollar_seconds=_float_value(
                values,
                "request_dollar_seconds",
                DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS,
            ),
            action_unit_dollar_seconds=_float_value(
                values,
                "action_unit_dollar_seconds",
                DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS,
            ),
            verify_memory_limit_mib=_int_value(
                values,
                "verify_memory_limit_mib",
                DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES // (1024 * 1024),
            ),
            budget_dollar_seconds=_float_value(
                values,
                "budget_dollar_seconds",
                DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS,
            ),
            run_timeout_s=_float_value(values, "run_timeout_s", 3600.0),
            heartbeat_interval_s=_float_value(
                values,
                "heartbeat_interval_s",
                30.0,
            ),
            telemetry_filename=_string_value(
                values,
                "telemetry_filename",
                "telemetry.jsonl",
            ),
            prompt_context_policy=_string_value(
                values,
                "prompt_context_policy",
                "repair_prompt_only",
            ),
            output_contract=_string_value(
                values,
                "output_contract",
                "python_solve_function_only",
            ),
            verifier=_string_value(
                values,
                "verifier",
                "isolated_subprocess_unit_tests",
            ),
            retry_policy=_string_value(values, "retry_policy", "none"),
            task_split=_string_value(
                values,
                "task_split",
                DEFAULT_FOUNDRY_TASK_SPLIT,
            ),
            scheduler_mode=_string_value(values, "scheduler_mode", "full_trinity"),
            action_codecs=_string_tuple_value(
                values,
                "action_codecs",
                ("token", "chunk2", "chunk4"),
            ),
            conditions=_string_tuple_value(
                values,
                "conditions",
                (primary_condition,),
            ),
            promotion_metric=_string_value(
                values,
                "promotion_metric",
                FOUNDRY_HARNESS_OBJECTIVE_METRIC,
            ),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.name or not _is_safe_candidate_name(self.name):
            raise ValueError("foundry_harness_manifest_name_must_be_safe")
        if self.primary_condition not in _CONDITION_ORDER:
            raise ValueError("foundry_harness_primary_condition_unknown")
        if not self.conditions:
            raise ValueError("foundry_harness_conditions_required")
        for condition in self.conditions:
            if condition not in _CONDITION_ORDER:
                raise ValueError("foundry_harness_condition_unknown")
        if self.primary_condition not in self.conditions:
            raise ValueError("foundry_harness_primary_condition_not_selected")
        if self.train_steps < 1:
            raise ValueError("foundry_harness_train_steps_must_be_positive")
        if self.task_limit < 1:
            raise ValueError("foundry_harness_task_limit_must_be_positive")
        if self.model_call_budget < 0:
            raise ValueError("foundry_harness_model_call_budget_must_be_non_negative")
        if self.max_completion_tokens < 1:
            raise ValueError(
                "foundry_harness_max_completion_tokens_must_be_positive"
            )
        if self.verify_memory_limit_mib < 1:
            raise ValueError("foundry_harness_verify_memory_limit_mib_must_be_positive")
        if self.request_timeout_s <= 0.0:
            raise ValueError("foundry_harness_request_timeout_s_must_be_positive")
        if self.verify_timeout_s <= 0.0:
            raise ValueError("foundry_harness_verify_timeout_s_must_be_positive")
        if self.budget_dollar_seconds <= 0.0:
            raise ValueError("foundry_harness_budget_dollar_seconds_must_be_positive")
        for name in (
            "request_dollar_seconds",
            "action_unit_dollar_seconds",
            "run_timeout_s",
            "heartbeat_interval_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"foundry_harness_{name}_must_be_non_negative")
        if not self.deployment:
            raise ValueError("foundry_harness_deployment_required")
        if not self.telemetry_filename or Path(self.telemetry_filename).name != (
            self.telemetry_filename
        ):
            raise ValueError("foundry_harness_telemetry_filename_must_be_basename")
        if not self.action_codecs:
            raise ValueError("foundry_harness_action_codecs_required")
        for codec in self.action_codecs:
            if not codec or not _is_safe_candidate_name(codec):
                raise ValueError("foundry_harness_action_codecs_must_be_safe")
        for name in (
            "prompt_context_policy",
            "output_contract",
            "verifier",
            "retry_policy",
            "task_split",
            "scheduler_mode",
            "promotion_metric",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"foundry_harness_{name}_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "primary_condition": self.primary_condition,
            "budget_race": self.budget_race,
            "env_path": str(self.env_path),
            "deployment": self.deployment,
            "train_steps": self.train_steps,
            "task_limit": self.task_limit,
            "model_call_budget": self.model_call_budget,
            "max_completion_tokens": self.max_completion_tokens,
            "request_timeout_s": self.request_timeout_s,
            "verify_timeout_s": self.verify_timeout_s,
            "request_dollar_seconds": self.request_dollar_seconds,
            "action_unit_dollar_seconds": self.action_unit_dollar_seconds,
            "verify_memory_limit_mib": self.verify_memory_limit_mib,
            "budget_dollar_seconds": self.budget_dollar_seconds,
            "run_timeout_s": self.run_timeout_s,
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "telemetry_filename": self.telemetry_filename,
            "prompt_context_policy": self.prompt_context_policy,
            "output_contract": self.output_contract,
            "verifier": self.verifier,
            "retry_policy": self.retry_policy,
            "task_split": self.task_split,
            "scheduler_mode": self.scheduler_mode,
            "action_codecs": list(self.action_codecs),
            "conditions": list(self.conditions),
            "promotion_metric": self.promotion_metric,
        }


def load_foundry_harness_manifest(
    candidate: str | Path,
    *,
    candidate_dir: Path = DEFAULT_FOUNDRY_HARNESS_DIR,
) -> FoundryHarnessManifest:
    path = resolve_foundry_harness_manifest_path(candidate, candidate_dir=candidate_dir)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_foundry_harness_manifest_json: {path}") from exc
    if not isinstance(values, Mapping):
        raise ValueError("foundry_harness_manifest_must_be_json_object")
    return FoundryHarnessManifest.from_mapping(values)


def resolve_foundry_harness_manifest_path(
    candidate: str | Path,
    *,
    candidate_dir: Path = DEFAULT_FOUNDRY_HARNESS_DIR,
) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.suffix:
        return candidate_path
    return candidate_dir / f"{candidate_path.name}.json"


def foundry_harness_child_args(
    manifest: FoundryHarnessManifest,
    *,
    telemetry_path: Path,
) -> list[str]:
    args = [
        "--json",
        "--env-path",
        str(manifest.env_path),
        "--deployment",
        manifest.deployment,
        "--train-steps",
        str(manifest.train_steps),
        "--task-limit",
        str(manifest.task_limit),
        "--task-split",
        manifest.task_split,
        "--conditions",
        *manifest.conditions,
        "--model-call-budget",
        str(manifest.model_call_budget),
        "--max-completion-tokens",
        str(manifest.max_completion_tokens),
        "--request-timeout-s",
        str(manifest.request_timeout_s),
        "--verify-timeout-s",
        str(manifest.verify_timeout_s),
        "--request-dollar-seconds",
        str(manifest.request_dollar_seconds),
        "--action-unit-dollar-seconds",
        str(manifest.action_unit_dollar_seconds),
        "--verify-memory-limit-mib",
        str(manifest.verify_memory_limit_mib),
        "--run-timeout-s",
        str(manifest.run_timeout_s),
        "--heartbeat-interval-s",
        str(manifest.heartbeat_interval_s),
        "--telemetry-path",
        str(telemetry_path),
    ]
    if manifest.budget_race:
        args.extend(
            [
                "--budget-race",
                "--budget-dollar-seconds",
                str(manifest.budget_dollar_seconds),
            ]
        )
    return args


def summarize_foundry_harness_result(
    manifest: FoundryHarnessManifest,
    result: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    conditions = _condition_summaries(result.get("conditions"))
    primary = conditions.get(manifest.primary_condition, {})
    primary_score = _optional_float(primary.get(manifest.promotion_metric))
    if primary_score is None:
        primary_score = _optional_float(primary.get(FOUNDRY_HARNESS_OBJECTIVE_METRIC))
    heldout_score = _heldout_score(result, manifest)
    ranking_score = heldout_score if heldout_score is not None else primary_score
    winning_condition = result.get("winning_condition_by_accounted_north_star")
    if not winning_condition and conditions:
        winning_condition = max(
            conditions,
            key=lambda name: (
                _optional_float(
                    conditions[name].get(FOUNDRY_HARNESS_OBJECTIVE_METRIC)
                )
                or 0.0
            ),
        )

    return {
        "ok": bool(result.get("ok")) and (returncode in (None, 0)),
        "candidate": manifest.name,
        "description": manifest.description,
        "primary_condition": manifest.primary_condition,
        "conditions_selected": list(manifest.conditions),
        "task_split": manifest.task_split,
        "promotion_metric": manifest.promotion_metric,
        "objective_metric": FOUNDRY_HARNESS_OBJECTIVE_METRIC,
        "primary_score": primary_score,
        "heldout_score": heldout_score,
        "ranking_score": ranking_score,
        "ranking_score_source": (
            "heldout" if heldout_score is not None else "primary"
        ),
        "primary_accounted_dollar_seconds": _optional_float(
            primary.get("costs/accounted_dollar_seconds")
        ),
        "primary_learned_solutions": _optional_float(
            primary.get("foundry/learned_solutions")
        ),
        "primary_model_calls": _optional_float(primary.get("foundry/model_calls")),
        "measurement": _string_or_none(result.get("measurement")),
        "proof_scope": _string_or_none(result.get("proof_scope")),
        "budget_race": bool(manifest.budget_race),
        "returncode": returncode,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "winning_condition_by_accounted_north_star": winning_condition,
        "race": result.get("race", {}),
        "task_coverage": result.get("task_coverage", {}),
        "non_saturation": result.get("non_saturation", {}),
        "conditions": conditions,
    }


def extract_foundry_harness_failures(
    manifest: FoundryHarnessManifest,
    result: Mapping[str, Any],
    *,
    returncode: int | None = None,
    stderr: str = "",
) -> dict[str, Any]:
    del manifest
    counts = {category: 0 for category in FOUNDRY_HARNESS_FAILURE_CATEGORIES}
    events: list[dict[str, Any]] = []
    run_failed = returncode not in (None, 0) or not bool(result.get("ok"))

    error_text = " ".join(
        str(value)
        for value in (
            result.get("error_type"),
            result.get("error"),
            stderr if run_failed else "",
        )
        if value
    )
    if error_text:
        _record_failure(
            counts,
            events,
            _failure_category_for_text(error_text, fallback="setup_failure"),
            "run",
            error_text,
        )
    if returncode not in (None, 0) and not events:
        _record_failure(
            counts,
            events,
            "setup_failure",
            "run",
            f"child_process_returncode={returncode}",
        )

    for path, value in _walk_values(result):
        lowered_path = path.lower()
        if "failure_mode" in lowered_path:
            category = _failure_category_for_text(str(value), fallback=None)
            if category is not None:
                _record_failure(counts, events, category, path, str(value))
        if lowered_path.endswith("scheduler/budget/accounted_exhausted"):
            amount = _optional_float(value)
            if amount and amount > 0.0:
                _record_failure(
                    counts,
                    events,
                    "cost_budget_exhausted",
                    path,
                    str(value),
                )
        if "stale" in lowered_path:
            amount = _optional_float(value)
            if amount and amount > 0.0:
                _record_failure(
                    counts,
                    events,
                    "scheduler_stale_batch",
                    path,
                    str(value),
                )
        if "control" in lowered_path and "failure" in lowered_path:
            amount = _optional_float(value)
            if amount is None or amount > 0.0:
                _record_failure(
                    counts,
                    events,
                    "scheduler_control_failure",
                    path,
                    str(value),
                )

    return {
        "ok": bool(result.get("ok")) and not any(
            counts[category]
            for category in (
                "setup_failure",
                "run_timeout",
                "model_request_failure",
                "verifier_timeout",
                "verifier_crash",
                "output_parse_failure",
            )
        ),
        "counts": counts,
        "events": events,
    }


def read_foundry_harness_run_summaries(
    runs_dir: Path,
    *,
    run_prefix: str | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return summaries
    for path in sorted(runs_dir.iterdir()):
        if run_prefix is not None and not path.name.startswith(run_prefix):
            continue
        summary_path = path / FOUNDRY_HARNESS_ARTIFACT_FILES["summary"]
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(summary, dict):
            summaries.append(summary)
    return summaries


def rank_foundry_harness_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = [dict(summary) for summary in summaries]
    ranked.sort(key=_summary_rank_key)
    for index, summary in enumerate(ranked, start=1):
        summary["rank"] = index
    return ranked


def compare_foundry_harness_runs(
    runs_dir: Path,
    *,
    run_prefix: str | None = None,
) -> dict[str, Any]:
    summaries = read_foundry_harness_run_summaries(
        runs_dir,
        run_prefix=run_prefix,
    )
    ranked = rank_foundry_harness_summaries(summaries)
    aggregates = aggregate_foundry_harness_summaries(summaries)
    ok_runs = sum(1 for summary in ranked if bool(summary.get("ok")))
    return {
        "ok": bool(ranked),
        "runs": len(ranked),
        "ok_runs": ok_runs,
        "failed_runs": len(ranked) - ok_runs,
        "has_successful_runs": ok_runs > 0,
        "runs_dir": str(runs_dir),
        "run_prefix": run_prefix,
        "objective_metric": FOUNDRY_HARNESS_OBJECTIVE_METRIC,
        "ranking": ranked,
        "candidate_aggregates": aggregates,
    }


def analyze_foundry_harness_runs(
    runs_dir: Path,
    *,
    run_prefix: str | None = None,
) -> dict[str, Any]:
    comparison = compare_foundry_harness_runs(runs_dir, run_prefix=run_prefix)
    run_diagnostics = [
        _foundry_run_diagnostics(Path(summary["output_dir"]), summary)
        for summary in comparison["ranking"]
        if summary.get("output_dir")
    ]
    winner = run_diagnostics[0] if run_diagnostics else None
    baseline = next(
        (
            diagnostic
            for diagnostic in run_diagnostics
            if diagnostic.get("primary_condition") == "static_art"
        ),
        None,
    )
    return {
        "ok": bool(run_diagnostics),
        "runs_dir": str(runs_dir),
        "run_prefix": run_prefix,
        "objective_metric": FOUNDRY_HARNESS_OBJECTIVE_METRIC,
        "winner": winner,
        "baseline": baseline,
        "runs": run_diagnostics,
        "comparison": comparison,
        "deltas_vs_baseline": [
            _diagnostic_delta_vs_baseline(diagnostic, baseline)
            for diagnostic in run_diagnostics
            if baseline is not None and diagnostic is not baseline
        ],
    }


def aggregate_foundry_harness_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        candidate = str(summary.get("candidate", ""))
        if not candidate:
            continue
        by_candidate.setdefault(candidate, []).append(summary)

    aggregates = [
        _candidate_aggregate(candidate, candidate_summaries)
        for candidate, candidate_summaries in by_candidate.items()
    ]
    aggregates.sort(key=_aggregate_rank_key)
    for index, aggregate in enumerate(aggregates, start=1):
        aggregate["rank"] = index
    return aggregates


def _string_value(
    values: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"foundry_harness_{key}_must_be_string")
    return value


def _foundry_run_diagnostics(
    output_dir: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    result = _read_json_object(output_dir / FOUNDRY_HARNESS_ARTIFACT_FILES["result"])
    failures = _read_json_object(output_dir / FOUNDRY_HARNESS_ARTIFACT_FILES["failures"])
    primary_condition = str(summary.get("primary_condition", ""))
    condition = _mapping_child(result.get("conditions"), primary_condition)
    heldout = _mapping_child(
        _mapping_child(result.get("heldout"), "conditions"),
        primary_condition,
    )
    non_saturation = _mapping_child(
        _mapping_child(result.get("non_saturation"), "conditions"),
        primary_condition,
    )
    return {
        "candidate": summary.get("candidate"),
        "ok": bool(summary.get("ok")),
        "output_dir": str(output_dir),
        "primary_condition": primary_condition,
        "task_split": summary.get("task_split") or result.get("task_split"),
        "ranking_score": _optional_float(summary.get("ranking_score")),
        "ranking_score_source": summary.get("ranking_score_source"),
        "heldout_pass_rate": _optional_float(heldout.get("heldout/pass_rate")),
        "learned_fraction": _optional_float(non_saturation.get("learned_fraction")),
        "heldout_pass_fraction": _optional_float(
            non_saturation.get("heldout_pass_fraction")
        ),
        "saturated": bool(non_saturation.get("saturated")),
        "learned_solutions": _optional_float(condition.get("foundry/learned_solutions")),
        "model_calls": _optional_float(condition.get("foundry/model_calls")),
        "observed_rollouts": _optional_float(condition.get("foundry/observed_rollouts")),
        "accounted_dollar_seconds": _optional_float(
            condition.get("costs/accounted_dollar_seconds")
        ),
        "budget_exhausted": bool(
            (_optional_float(condition.get("scheduler/budget/accounted_exhausted")) or 0.0)
            > 0.0
        ),
        "failure_counts": failures.get("counts", {}),
        "task_coverage": result.get("task_coverage", {}),
        "heldout_by_family": heldout.get("heldout/by_family", {}),
        "heldout_by_difficulty": heldout.get("heldout/by_difficulty", {}),
        "heldout_task_failures": _heldout_task_failures(
            heldout.get("heldout/task_results")
        ),
        "weakest_families": _weakest_breakdowns(heldout.get("heldout/by_family")),
        "weakest_difficulties": _weakest_breakdowns(
            heldout.get("heldout/by_difficulty")
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _mapping_child(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    child = value.get(key)
    return dict(child) if isinstance(child, Mapping) else {}


def _heldout_task_failures(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    failures: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or item.get("passed") is True:
            continue
        failures.append(
            {
                "task_id": item.get("task_id"),
                "family": item.get("family"),
                "difficulty": item.get("difficulty"),
                "failure_mode": item.get("failure_mode"),
            }
        )
    return failures


def _weakest_breakdowns(value: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for name, metrics in value.items():
        if not isinstance(metrics, Mapping):
            continue
        rows.append(
            {
                "name": str(name),
                "pass_rate": _optional_float(metrics.get("pass_rate")),
                "passed": _optional_float(metrics.get("passed")),
                "tasks": _optional_float(metrics.get("tasks")),
            }
        )
    rows.sort(
        key=lambda row: (
            row["pass_rate"] if row["pass_rate"] is not None else 1.0,
            -(row["tasks"] or 0.0),
            row["name"],
        )
    )
    return rows[:limit]


def _diagnostic_delta_vs_baseline(
    diagnostic: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if baseline is None:
        return {}
    return {
        "candidate": diagnostic.get("candidate"),
        "baseline": baseline.get("candidate"),
        "ranking_score_delta": _optional_delta(
            diagnostic.get("ranking_score"),
            baseline.get("ranking_score"),
        ),
        "heldout_pass_rate_delta": _optional_delta(
            diagnostic.get("heldout_pass_rate"),
            baseline.get("heldout_pass_rate"),
        ),
        "learned_fraction_delta": _optional_delta(
            diagnostic.get("learned_fraction"),
            baseline.get("learned_fraction"),
        ),
        "accounted_dollar_seconds_delta": _optional_delta(
            diagnostic.get("accounted_dollar_seconds"),
            baseline.get("accounted_dollar_seconds"),
        ),
        "model_calls_delta": _optional_delta(
            diagnostic.get("model_calls"),
            baseline.get("model_calls"),
        ),
        "budget_exhausted": diagnostic.get("budget_exhausted"),
        "baseline_budget_exhausted": baseline.get("budget_exhausted"),
    }


def _optional_delta(left: Any, right: Any) -> float | None:
    left_float = _optional_float(left)
    right_float = _optional_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _bool_value(
    values: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"foundry_harness_{key}_must_be_boolean")
    return value


def _int_value(values: Mapping[str, Any], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"foundry_harness_{key}_must_be_integer")
    return value


def _float_value(values: Mapping[str, Any], key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"foundry_harness_{key}_must_be_number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"foundry_harness_{key}_must_be_finite")
    return parsed


def _string_tuple_value(
    values: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = values.get(key, default)
    if not isinstance(value, list | tuple):
        raise ValueError(f"foundry_harness_{key}_must_be_list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"foundry_harness_{key}_must_be_string_list")
    return tuple(value)


def _is_safe_candidate_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))


def _condition_summaries(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        return {}
    summaries: dict[str, dict[str, float]] = {}
    for name in _CONDITION_ORDER:
        metrics = value.get(name)
        if not isinstance(metrics, Mapping):
            continue
        summaries[name] = {
            key: float(metrics[key])
            for key in _SUMMARY_METRIC_KEYS
            if key in metrics and _optional_float(metrics[key]) is not None
        }
    return summaries


def _heldout_score(
    result: Mapping[str, Any],
    manifest: FoundryHarnessManifest,
) -> float | None:
    direct = _optional_float(result.get("heldout_score"))
    if direct is not None:
        return direct
    for key in ("heldout", "held_out"):
        heldout = result.get(key)
        if not isinstance(heldout, Mapping):
            continue
        direct = _optional_float(heldout.get("score"))
        if direct is not None:
            return direct
        conditions = heldout.get("conditions")
        if isinstance(conditions, Mapping):
            condition = conditions.get(manifest.primary_condition)
            if isinstance(condition, Mapping):
                score = _optional_float(condition.get(manifest.promotion_metric))
                if score is not None:
                    return score
                score = _optional_float(
                    condition.get(FOUNDRY_HARNESS_OBJECTIVE_METRIC)
                )
                if score is not None:
                    return score
    return None


def _failure_category_for_text(text: str, *, fallback: str | None) -> str | None:
    lowered = text.lower()
    if "run_timeout" in lowered or "foundry_run_timeout" in lowered:
        return "run_timeout"
    if "jsondecodeerror" in lowered or "invalid_json" in lowered:
        return "output_parse_failure"
    if "invalid_verifier_output" in lowered:
        return "output_parse_failure"
    if "timeout" in lowered and "verifier" in lowered:
        return "verifier_timeout"
    if lowered == "timeout":
        return "verifier_timeout"
    if "verifier_crashed" in lowered:
        return "verifier_crash"
    if "syntax_error" in lowered or "missing_solve" in lowered:
        return "output_parse_failure"
    if "unit_test_failed" in lowered or "wrong" in lowered:
        return "wrong_answer"
    if "model_call_budget_exhausted" in lowered or "budget_exhausted" in lowered:
        return "cost_budget_exhausted"
    if "azure_foundry_error" in lowered or "rate" in lowered:
        return "model_request_failure"
    if "env_missing" in lowered or "required_keys" in lowered:
        return "setup_failure"
    if "no module named" in lowered or "openai" in lowered:
        return "setup_failure"
    return fallback


def _record_failure(
    counts: dict[str, int],
    events: list[dict[str, Any]],
    category: str | None,
    source: str,
    message: str,
) -> None:
    if category is None:
        return
    counts[category] += 1
    events.append(
        {
            "category": category,
            "source": source,
            "message": message,
        }
    )


def _walk_values(value: Any, path: str = "$") -> Sequence[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_walk_values(item, f"{path}.{key}"))
        return found
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.extend(_walk_values(item, f"{path}[{index}]"))
        return found
    found.append((path, value))
    return found


def _summary_rank_key(summary: Mapping[str, Any]) -> tuple[int, float, float, str]:
    score = _summary_score(summary)
    if score is None:
        score = float("-inf")
    spend = _optional_float(summary.get("primary_accounted_dollar_seconds"))
    if spend is None:
        spend = float("inf")
    return (
        0 if bool(summary.get("ok")) else 1,
        -score,
        spend,
        str(summary.get("candidate", "")),
    )


def _candidate_aggregate(
    candidate: str,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ok_summaries = [summary for summary in summaries if bool(summary.get("ok"))]
    score_values = [
        value
        for value in (_summary_score(summary) for summary in ok_summaries)
        if value is not None
    ]
    spend_values = [
        value
        for value in (
            _optional_float(summary.get("primary_accounted_dollar_seconds"))
            for summary in ok_summaries
        )
        if value is not None
    ]
    best_summary = max(
        ok_summaries,
        key=lambda summary: _summary_score(summary) or float("-inf"),
        default=None,
    )
    first = summaries[0] if summaries else {}
    runs = len(summaries)
    ok_runs = len(ok_summaries)
    return {
        "candidate": candidate,
        "runs": runs,
        "ok_runs": ok_runs,
        "failed_runs": runs - ok_runs,
        "failure_rate": (runs - ok_runs) / runs if runs else None,
        "primary_condition": first.get("primary_condition"),
        "ranking_score_source": _aggregate_score_source(ok_summaries),
        "ranking_score_mean": fmean(score_values) if score_values else None,
        "ranking_score_median": median(score_values) if score_values else None,
        "ranking_score_best": max(score_values) if score_values else None,
        "ranking_score_worst": min(score_values) if score_values else None,
        "primary_accounted_dollar_seconds_mean": (
            fmean(spend_values) if spend_values else None
        ),
        "best_output_dir": (
            str(best_summary.get("output_dir"))
            if best_summary is not None and best_summary.get("output_dir")
            else None
        ),
    }


def _aggregate_rank_key(aggregate: Mapping[str, Any]) -> tuple[int, float, float, str]:
    score = _optional_float(aggregate.get("ranking_score_median"))
    if score is None:
        score = float("-inf")
    failure_rate = _optional_float(aggregate.get("failure_rate"))
    if failure_rate is None:
        failure_rate = 1.0
    spend = _optional_float(aggregate.get("primary_accounted_dollar_seconds_mean"))
    if spend is None:
        spend = float("inf")
    return (
        0 if int(aggregate.get("ok_runs", 0)) > 0 else 1,
        failure_rate,
        -score,
        spend,
        str(aggregate.get("candidate", "")),
    )


def _aggregate_score_source(summaries: Sequence[Mapping[str, Any]]) -> str | None:
    if not summaries:
        return None
    sources = {
        str(summary.get("ranking_score_source", "primary"))
        for summary in summaries
    }
    if sources == {"heldout"}:
        return "heldout"
    if sources == {"primary"}:
        return "primary"
    return "mixed"


def _summary_score(summary: Mapping[str, Any]) -> float | None:
    score = _optional_float(summary.get("heldout_score"))
    if score is None:
        score = _optional_float(summary.get("ranking_score"))
    if score is None:
        score = _optional_float(summary.get("primary_score"))
    return score


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
