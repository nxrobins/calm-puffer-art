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
    DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY,
    DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S,
    DEFAULT_FOUNDRY_TASK_LIMIT,
    DEFAULT_FOUNDRY_TASK_ORDER_POLICY,
    DEFAULT_FOUNDRY_TASK_SPLIT,
    DEFAULT_FOUNDRY_TRAIN_STEPS,
    DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES,
    DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S,
    FOUNDRY_PROMPT_CONTEXT_POLICIES,
    FOUNDRY_TASK_ORDER_POLICIES,
    foundry_task_metadata_index,
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
FOUNDRY_COVERAGE_GAP_CANDIDATE = "frontier_coverage_gap_first"
FOUNDRY_CHUNK2_ONLY_CANDIDATE = "frontier_chunk2_only"
FOUNDRY_LIFT_POCKET_CANDIDATE = "frontier_lift_pocket_first"

_CONDITION_ORDER = ("static_art", "scheduler_only", "chunk2_only", "full_trinity")
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
    "promotion_eligible",
    "promotion_metric",
    "prompt_context_policy",
    "request_dollar_seconds",
    "request_timeout_s",
    "retry_policy",
    "run_timeout_s",
    "scheduler_mode",
    "task_limit",
    "task_order_policy",
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
    prompt_context_policy: str = DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY
    output_contract: str = "python_solve_function_only"
    verifier: str = "isolated_subprocess_unit_tests"
    retry_policy: str = "none"
    task_split: str = DEFAULT_FOUNDRY_TASK_SPLIT
    task_order_policy: str = DEFAULT_FOUNDRY_TASK_ORDER_POLICY
    scheduler_mode: str = "full_trinity"
    action_codecs: tuple[str, ...] = ("token", "chunk2", "chunk4")
    conditions: tuple[str, ...] = ("full_trinity",)
    promotion_metric: str = FOUNDRY_HARNESS_OBJECTIVE_METRIC
    promotion_eligible: bool = True

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
                DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY,
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
            task_order_policy=_string_value(
                values,
                "task_order_policy",
                DEFAULT_FOUNDRY_TASK_ORDER_POLICY,
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
            promotion_eligible=_bool_value(values, "promotion_eligible", True),
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
            "task_order_policy",
            "scheduler_mode",
            "promotion_metric",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"foundry_harness_{name}_required")
        if self.prompt_context_policy not in FOUNDRY_PROMPT_CONTEXT_POLICIES:
            raise ValueError("foundry_harness_prompt_context_policy_unknown")
        if self.task_order_policy not in FOUNDRY_TASK_ORDER_POLICIES:
            raise ValueError("foundry_harness_task_order_policy_unknown")

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
            "task_order_policy": self.task_order_policy,
            "scheduler_mode": self.scheduler_mode,
            "action_codecs": list(self.action_codecs),
            "conditions": list(self.conditions),
            "promotion_eligible": self.promotion_eligible,
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
        "--task-order-policy",
        manifest.task_order_policy,
        "--prompt-context-policy",
        manifest.prompt_context_policy,
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
        "prompt_context_policy": manifest.prompt_context_policy,
        "task_order_policy": manifest.task_order_policy,
        "promotion_metric": manifest.promotion_metric,
        "promotion_eligible": manifest.promotion_eligible,
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
    pairwise = pairwise_foundry_harness_summaries(summaries)
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
        "candidate_pairwise": pairwise,
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
    failure_pockets = _failure_pocket_payload(run_diagnostics, baseline)
    promotion_readiness = foundry_harness_promotion_readiness(comparison)
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
        "failure_pockets": failure_pockets,
        "promotion_readiness": promotion_readiness,
        "next_hypotheses": _next_hypotheses_payload(
            comparison,
            failure_pockets,
            promotion_readiness,
        ),
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


def pairwise_foundry_harness_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        if not bool(summary.get("ok")):
            continue
        if _summary_score(summary) is None:
            continue
        candidate = str(summary.get("candidate", ""))
        if not candidate:
            continue
        by_candidate.setdefault(candidate, []).append(summary)

    candidates = sorted(by_candidate)
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            comparisons.append(
                _candidate_pairwise_comparison(
                    left,
                    by_candidate[left],
                    right,
                    by_candidate[right],
                )
            )
    comparisons.sort(
        key=lambda item: (
            -float(item.get("left_win_rate") or 0.0),
            str(item.get("left_candidate", "")),
            str(item.get("right_candidate", "")),
        )
    )
    return comparisons


def foundry_harness_promotion_readiness(
    comparison: Mapping[str, Any],
    *,
    baseline_condition: str = "static_art",
    min_successful_runs: int = 3,
    min_pairwise_win_rate: float = 0.6,
) -> dict[str, Any]:
    aggregates = [
        dict(item)
        for item in comparison.get("candidate_aggregates", [])
        if isinstance(item, Mapping)
    ]
    eligible_aggregates = [
        aggregate
        for aggregate in aggregates
        if bool(aggregate.get("promotion_eligible", True))
    ]
    excluded_candidates = [
        {
            "candidate": aggregate.get("candidate"),
            "reason": "promotion_eligible_false",
        }
        for aggregate in aggregates
        if not bool(aggregate.get("promotion_eligible", True))
    ]
    baseline = next(
        (
            aggregate
            for aggregate in eligible_aggregates
            if aggregate.get("primary_condition") == baseline_condition
        ),
        None,
    )
    decisions = [
        _promotion_decision(
            candidate=aggregate,
            baseline=baseline,
            pairwise=comparison.get("candidate_pairwise", []),
            min_successful_runs=min_successful_runs,
            min_pairwise_win_rate=min_pairwise_win_rate,
        )
        for aggregate in eligible_aggregates
        if baseline is None or aggregate.get("candidate") != baseline.get("candidate")
    ]
    promotable = [
        decision
        for decision in decisions
        if decision.get("status") == "promote"
    ]
    recommended_next_runs = [
        {
            "candidate": decision.get("candidate"),
            "additional_successful_runs": decision.get("additional_successful_runs"),
        }
        for decision in decisions
        if (decision.get("additional_successful_runs") or 0) > 0
    ]
    if baseline is not None:
        baseline_runs = int(baseline.get("ok_runs", 0) or 0)
        baseline_needs_runs = max(0, min_successful_runs - baseline_runs)
        if baseline_needs_runs:
            recommended_next_runs.insert(
                0,
                {
                    "candidate": baseline.get("candidate"),
                    "additional_successful_runs": baseline_needs_runs,
                },
            )
    status = "hold"
    if baseline is None:
        status = "needs_baseline"
    elif recommended_next_runs:
        status = "needs_more_evidence"
    elif promotable:
        status = "promote"
    return {
        "ok": baseline is not None,
        "status": status,
        "baseline_candidate": baseline.get("candidate") if baseline else None,
        "baseline_condition": baseline_condition,
        "min_successful_runs": min_successful_runs,
        "min_pairwise_win_rate": min_pairwise_win_rate,
        "decisions": decisions,
        "excluded_candidates": excluded_candidates,
        "recommended_next_runs": recommended_next_runs,
    }


def _promotion_decision(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    pairwise: Any,
    min_successful_runs: int,
    min_pairwise_win_rate: float,
) -> dict[str, Any]:
    candidate_name = str(candidate.get("candidate", ""))
    candidate_runs = int(candidate.get("ok_runs", 0) or 0)
    candidate_score = _optional_float(candidate.get("ranking_score_median"))
    candidate_failure_rate = _optional_float(candidate.get("failure_rate"))
    if candidate_failure_rate is None:
        candidate_failure_rate = 1.0
    if baseline is None:
        return {
            "candidate": candidate_name,
            "status": "needs_baseline",
            "reason": "no_static_baseline_candidate",
            "candidate_ok_runs": candidate_runs,
            "additional_successful_runs": max(0, min_successful_runs - candidate_runs),
        }

    baseline_name = str(baseline.get("candidate", ""))
    baseline_runs = int(baseline.get("ok_runs", 0) or 0)
    baseline_score = _optional_float(baseline.get("ranking_score_median"))
    baseline_failure_rate = _optional_float(baseline.get("failure_rate"))
    if baseline_failure_rate is None:
        baseline_failure_rate = 1.0
    pair = _pairwise_between(pairwise, candidate_name, baseline_name)
    pairwise_win_rate = _candidate_win_rate_from_pairwise(pair, candidate_name)
    pairwise_leader = pair.get("leader_candidate") if pair else None
    median_delta = _optional_delta(candidate_score, baseline_score)
    candidate_needs_runs = max(0, min_successful_runs - candidate_runs)
    baseline_needs_runs = max(0, min_successful_runs - baseline_runs)
    reasons: list[str] = []
    if candidate_needs_runs:
        reasons.append("candidate_needs_more_successful_runs")
    if baseline_needs_runs:
        reasons.append("baseline_needs_more_successful_runs")
    if candidate_failure_rate > baseline_failure_rate:
        reasons.append("candidate_failure_rate_exceeds_baseline")
    if median_delta is None or median_delta <= 0.0:
        reasons.append("median_score_not_above_baseline")
    if pairwise_win_rate is None:
        reasons.append("missing_pairwise_evidence")
    elif pairwise_win_rate < min_pairwise_win_rate:
        reasons.append("pairwise_win_rate_below_threshold")
    if pairwise_leader not in (candidate_name, None) and pairwise_win_rate is not None:
        reasons.append("pairwise_leader_is_not_candidate")

    if candidate_needs_runs or baseline_needs_runs:
        status = "needs_more_evidence"
    elif reasons:
        status = "hold"
    else:
        status = "promote"
        reasons.append("candidate_beats_baseline_gate")

    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "status": status,
        "reason": ";".join(reasons),
        "candidate_ok_runs": candidate_runs,
        "baseline_ok_runs": baseline_runs,
        "additional_successful_runs": candidate_needs_runs,
        "baseline_additional_successful_runs": baseline_needs_runs,
        "candidate_median_score": candidate_score,
        "baseline_median_score": baseline_score,
        "median_score_delta_vs_baseline": median_delta,
        "candidate_failure_rate": candidate_failure_rate,
        "baseline_failure_rate": baseline_failure_rate,
        "pairwise_win_rate_vs_baseline": pairwise_win_rate,
        "pairwise_leader": pairwise_leader,
        "pairwise_pair_count": pair.get("pair_count") if pair else 0,
    }


def _pairwise_between(
    pairwise: Any,
    left_candidate: str,
    right_candidate: str,
) -> dict[str, Any]:
    if not isinstance(pairwise, list | tuple):
        return {}
    for item in pairwise:
        if not isinstance(item, Mapping):
            continue
        left = str(item.get("left_candidate", ""))
        right = str(item.get("right_candidate", ""))
        if {left, right} == {left_candidate, right_candidate}:
            return dict(item)
    return {}


def _candidate_win_rate_from_pairwise(
    pair: Mapping[str, Any],
    candidate: str,
) -> float | None:
    if not pair:
        return None
    if pair.get("left_candidate") == candidate:
        return _optional_float(pair.get("left_win_rate"))
    if pair.get("right_candidate") == candidate:
        return _optional_float(pair.get("right_win_rate"))
    return None


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
    manifest = _read_json_object(output_dir / FOUNDRY_HARNESS_ARTIFACT_FILES["manifest"])
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
    task_results = _heldout_task_results(
        heldout.get("heldout/task_results"),
        metadata_index=foundry_task_metadata_index(),
    )
    return {
        "candidate": summary.get("candidate"),
        "ok": bool(summary.get("ok")),
        "output_dir": str(output_dir),
        "primary_condition": primary_condition,
        "task_split": summary.get("task_split") or result.get("task_split"),
        "prompt_context_policy": summary.get("prompt_context_policy")
        or result.get("prompt_context_policy")
        or manifest.get("prompt_context_policy"),
        "task_order_policy": summary.get("task_order_policy")
        or result.get("task_order_policy")
        or manifest.get("task_order_policy"),
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
        "heldout_by_failure_tag": heldout.get("heldout/by_failure_tag", {}),
        "heldout_task_results": task_results,
        "heldout_task_failures": _heldout_task_failures(task_results),
        "weakest_families": _weakest_breakdowns(heldout.get("heldout/by_family")),
        "weakest_difficulties": _weakest_breakdowns(
            heldout.get("heldout/by_difficulty")
        ),
        "weakest_failure_tags": _weakest_breakdowns(
            heldout.get("heldout/by_failure_tag")
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
    return [
        item
        for item in value
        if isinstance(item, Mapping)
        if item.get("passed") is not True
    ]


def _heldout_task_results(
    value: Any,
    *,
    metadata_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    results: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        task_id = item.get("task_id")
        metadata = (
            metadata_index.get(str(task_id), {})
            if task_id is not None and metadata_index is not None
            else {}
        )
        family = item.get("family") or metadata.get("family")
        difficulty = item.get("difficulty") or metadata.get("difficulty")
        failure_tags = _string_list(item.get("failure_tags")) or _string_list(
            metadata.get("failure_tags")
        )
        results.append(
            {
                "task_id": task_id,
                "family": str(family) if family is not None else None,
                "difficulty": (
                    str(difficulty) if difficulty is not None else None
                ),
                "failure_tags": failure_tags,
                "passed": item.get("passed") is True,
                "failure_mode": item.get("failure_mode"),
                "tests_passed": _optional_float(item.get("tests_passed")),
                "tests_total": _optional_float(item.get("tests_total")),
            }
        )
    return results


def _failure_pocket_payload(
    run_diagnostics: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    by_candidate = _candidate_failure_pockets(run_diagnostics)
    baseline_candidate = baseline.get("candidate") if baseline else None
    return {
        "baseline_candidate": baseline_candidate,
        "by_candidate": by_candidate,
        "deltas_vs_baseline": _failure_pocket_deltas_vs_baseline(
            by_candidate,
            str(baseline_candidate) if baseline_candidate else None,
        ),
    }


def _candidate_failure_pockets(
    run_diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    collectors: dict[str, dict[str, Any]] = {}
    for diagnostic in run_diagnostics:
        candidate = str(diagnostic.get("candidate", ""))
        task_results = diagnostic.get("heldout_task_results")
        if not candidate or not isinstance(task_results, list):
            continue
        collector = collectors.setdefault(candidate, _new_failure_pocket(candidate))
        collector["runs_with_task_results"] += 1
        for result in task_results:
            if not isinstance(result, Mapping):
                continue
            _record_failure_pocket_observation(collector, result)

    pockets = [
        _finalize_failure_pocket(collector)
        for collector in collectors.values()
    ]
    pockets.sort(key=lambda item: str(item.get("candidate", "")))
    return pockets


def _new_failure_pocket(candidate: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "runs_with_task_results": 0,
        "task_observations": 0,
        "passed_observations": 0,
        "failed_observations": 0,
        "failure_modes": {},
        "by_task": {},
        "by_family": {},
        "by_difficulty": {},
        "by_failure_tag": {},
        "missing_failure_tag_observations": 0,
    }


def _record_failure_pocket_observation(
    collector: dict[str, Any],
    result: Mapping[str, Any],
) -> None:
    task_id = str(result.get("task_id") or "unknown")
    family = str(result.get("family") or "unknown")
    difficulty = str(result.get("difficulty") or "unknown")
    failure_tags = tuple(_string_list(result.get("failure_tags")))
    passed = result.get("passed") is True
    failure_mode = str(
        result.get("failure_mode") or ("passed" if passed else "unknown")
    )
    collector["task_observations"] += 1
    if passed:
        collector["passed_observations"] += 1
    else:
        collector["failed_observations"] += 1
        _increment_count(collector["failure_modes"], failure_mode)

    _record_pocket_row(
        collector["by_task"],
        task_id,
        passed=passed,
        failure_mode=failure_mode,
        metadata={
            "family": family,
            "difficulty": difficulty,
            "failure_tags": list(failure_tags),
        },
    )
    _record_pocket_row(
        collector["by_family"],
        family,
        passed=passed,
        failure_mode=failure_mode,
    )
    _record_pocket_row(
        collector["by_difficulty"],
        difficulty,
        passed=passed,
        failure_mode=failure_mode,
    )
    if not failure_tags:
        collector["missing_failure_tag_observations"] += 1
    for tag in failure_tags:
        _record_pocket_row(
            collector["by_failure_tag"],
            tag,
            passed=passed,
            failure_mode=failure_mode,
        )


def _record_pocket_row(
    bucket: dict[str, dict[str, Any]],
    name: str,
    *,
    passed: bool,
    failure_mode: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    row = bucket.setdefault(
        name,
        {
            "name": name,
            "observations": 0,
            "passed": 0,
            "failed": 0,
            "failure_modes": {},
        },
    )
    row["observations"] += 1
    if metadata:
        for key, value in metadata.items():
            if key not in row or not row[key]:
                row[key] = value
    if passed:
        row["passed"] += 1
    else:
        row["failed"] += 1
        _increment_count(row["failure_modes"], failure_mode)


def _finalize_failure_pocket(collector: Mapping[str, Any]) -> dict[str, Any]:
    task_observations = int(collector.get("task_observations", 0) or 0)
    passed_observations = int(collector.get("passed_observations", 0) or 0)
    failed_observations = int(collector.get("failed_observations", 0) or 0)
    by_task = _finalize_pocket_rows(
        collector.get("by_task"),
        name_key="task_id",
    )
    by_family = _finalize_pocket_rows(collector.get("by_family"))
    by_difficulty = _finalize_pocket_rows(collector.get("by_difficulty"))
    by_failure_tag = _finalize_pocket_rows(collector.get("by_failure_tag"))
    return {
        "candidate": collector.get("candidate"),
        "runs_with_task_results": int(
            collector.get("runs_with_task_results", 0) or 0
        ),
        "task_observations": task_observations,
        "passed_observations": passed_observations,
        "failed_observations": failed_observations,
        "pass_rate": (
            passed_observations / task_observations
            if task_observations
            else None
        ),
        "top_failure_modes": _top_counts(collector.get("failure_modes")),
        "missing_failure_tag_observations": int(
            collector.get("missing_failure_tag_observations", 0) or 0
        ),
        "by_task": by_task,
        "weakest_tasks": by_task[:10],
        "strongest_tasks": _strongest_pocket_rows(by_task)[:10],
        "by_family": by_family,
        "by_difficulty": by_difficulty,
        "by_failure_tag": by_failure_tag,
    }


def _finalize_pocket_rows(value: Any, *, name_key: str = "name") -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for row in value.values():
        if not isinstance(row, Mapping):
            continue
        observations = int(row.get("observations", 0) or 0)
        passed = int(row.get("passed", 0) or 0)
        failed = int(row.get("failed", 0) or 0)
        name = str(row.get("name", ""))
        finalized = {
            name_key: name,
            "observations": observations,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / observations if observations else None,
            "failure_rate": failed / observations if observations else None,
            "top_failure_modes": _top_counts(row.get("failure_modes")),
        }
        for key in ("family", "difficulty", "failure_tags"):
            if key in row:
                finalized[key] = row[key]
        rows.append(finalized)
    rows.sort(key=_weakest_pocket_key)
    return rows


def _failure_pocket_deltas_vs_baseline(
    by_candidate: Sequence[Mapping[str, Any]],
    baseline_candidate: str | None,
) -> list[dict[str, Any]]:
    if baseline_candidate is None:
        return []
    pockets = {
        str(pocket.get("candidate", "")): pocket
        for pocket in by_candidate
        if isinstance(pocket, Mapping)
    }
    baseline = pockets.get(baseline_candidate)
    if baseline is None:
        return []
    deltas: list[dict[str, Any]] = []
    for candidate, pocket in sorted(pockets.items()):
        if candidate == baseline_candidate:
            continue
        task_deltas = _pocket_delta_rows(
            pocket.get("by_task"),
            baseline.get("by_task"),
            name_key="task_id",
        )
        deltas.append(
            {
                "candidate": candidate,
                "baseline": baseline_candidate,
                "tasks_worse_than_baseline": _lowest_delta_rows(task_deltas)[:10],
                "tasks_better_than_baseline": _highest_delta_rows(task_deltas)[:10],
                "by_family": _pocket_delta_rows(
                    pocket.get("by_family"),
                    baseline.get("by_family"),
                ),
                "by_difficulty": _pocket_delta_rows(
                    pocket.get("by_difficulty"),
                    baseline.get("by_difficulty"),
                ),
                "by_failure_tag": _pocket_delta_rows(
                    pocket.get("by_failure_tag"),
                    baseline.get("by_failure_tag"),
                ),
            }
        )
    return deltas


def _next_hypotheses_payload(
    comparison: Mapping[str, Any],
    failure_pockets: Mapping[str, Any],
    promotion_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    status = str(promotion_readiness.get("status") or "unknown")
    aggregates = [
        dict(item)
        for item in comparison.get("candidate_aggregates", [])
        if isinstance(item, Mapping)
    ]
    aggregate_by_candidate = {
        str(item.get("candidate", "")): item
        for item in aggregates
        if item.get("candidate")
    }

    if status == "promote":
        actions.extend(_promotion_actions(promotion_readiness))
    elif status == "needs_more_evidence":
        actions.extend(_replicate_actions(promotion_readiness))

    for decision in promotion_readiness.get("decisions", []):
        if not isinstance(decision, Mapping):
            continue
        candidate = str(decision.get("candidate", ""))
        reason = str(decision.get("reason", ""))
        median_delta = _optional_float(decision.get("median_score_delta_vs_baseline"))
        if (
            decision.get("status") == "hold"
            and "pairwise_win_rate" in reason
            and median_delta is not None
            and median_delta > 0.0
        ):
            positive_task_pockets = _positive_task_pockets(
                failure_pockets,
                candidate,
            )
            actions.append(
                {
                    "action": "study_unstable_lift",
                    "candidate": candidate,
                    "reason": reason,
                    "pairwise_win_rate_vs_baseline": _optional_float(
                        decision.get("pairwise_win_rate_vs_baseline")
                    ),
                    "median_score_delta_vs_baseline": _optional_float(
                        decision.get("median_score_delta_vs_baseline")
                    ),
                    "positive_task_pockets": positive_task_pockets,
                }
            )
            if candidate == "frontier_full_trinity":
                actions.append(
                    _stability_candidate_action(
                        aggregate_by_candidate,
                        str(promotion_readiness.get("baseline_candidate") or ""),
                    )
                )
                if positive_task_pockets:
                    actions.append(
                        _lift_pocket_candidate_action(
                            aggregate_by_candidate,
                            str(promotion_readiness.get("baseline_candidate") or ""),
                            positive_task_pockets,
                        )
                    )
        elif decision.get("status") == "hold":
            actions.append(
                {
                    "action": "do_not_promote",
                    "candidate": candidate,
                    "reason": reason,
                    "median_score_delta_vs_baseline": _optional_float(
                        decision.get("median_score_delta_vs_baseline")
                    ),
                }
            )

    for excluded in promotion_readiness.get("excluded_candidates", []):
        if not isinstance(excluded, Mapping):
            continue
        candidate = str(excluded.get("candidate", ""))
        aggregate = aggregate_by_candidate.get(candidate, {})
        actions.append(
            {
                "action": "treat_as_experimental_probe",
                "candidate": candidate,
                "reason": excluded.get("reason"),
                "ranking_score_median": _optional_float(
                    aggregate.get("ranking_score_median")
                ),
                "ok_runs": int(aggregate.get("ok_runs", 0) or 0),
                "negative_task_pockets": _negative_task_pockets(
                    failure_pockets,
                    candidate,
                ),
            }
        )

    baseline_candidate = promotion_readiness.get("baseline_candidate")
    eligible_candidates = [
        str(item.get("candidate", ""))
        for item in aggregates
        if bool(item.get("promotion_eligible", True)) and item.get("candidate")
    ]
    shared_failure_pockets = _shared_failure_pockets(
        failure_pockets,
        eligible_candidates,
    )
    if (
        status == "hold"
        and shared_failure_pockets.get("tasks")
        and not any(action.get("action") == "promote_candidate" for action in actions)
    ):
        suggested_lever = _suggested_candidate_lever(shared_failure_pockets)
        actions.append(
            _targeted_candidate_action(
                suggested_lever,
                shared_failure_pockets,
                aggregate_by_candidate,
                str(baseline_candidate or ""),
            )
        )

    if status == "hold" and not actions:
        actions.append(
            {
                "action": "hold",
                "reason": "no_promotable_candidate_and_no_specific_followup",
            }
        )

    return {
        "status": status,
        "baseline_candidate": baseline_candidate,
        "actions": actions,
        "shared_failure_pockets": shared_failure_pockets,
    }


def _targeted_candidate_action(
    suggested_lever: str,
    shared_failure_pockets: Mapping[str, Any],
    aggregate_by_candidate: Mapping[str, Mapping[str, Any]],
    baseline_candidate: str,
) -> dict[str, Any]:
    target_tasks = shared_failure_pockets.get("tasks", [])[:5]
    target_failure_tags = shared_failure_pockets.get("failure_tags", [])[:5]
    if suggested_lever != "task_allocation_or_budget_coverage":
        return {
            "action": "design_targeted_candidate",
            "reason": "eligible_candidates_share_unsolved_heldout_pockets",
            "candidate_scope": "new_probe_not_promotion_eligible_until_replicated",
            "suggested_lever": suggested_lever,
            "target_tasks": target_tasks,
            "target_failure_tags": target_failure_tags,
        }

    aggregate = aggregate_by_candidate.get(FOUNDRY_COVERAGE_GAP_CANDIDATE)
    if not aggregate:
        return {
            "action": "run_existing_targeted_candidate",
            "candidate": FOUNDRY_COVERAGE_GAP_CANDIDATE,
            "reason": "coverage_gap_probe_exists_without_successful_runs",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": suggested_lever,
            "recommended_successful_runs": 3,
            "command": (
                "python examples\\foundry_harness_batch.py --candidates "
                f"{FOUNDRY_COVERAGE_GAP_CANDIDATE} --replicates 3 --json"
            ),
            "target_tasks": target_tasks,
            "target_failure_tags": target_failure_tags,
        }

    ok_runs = int(aggregate.get("ok_runs", 0) or 0)
    if ok_runs < 3:
        return {
            "action": "replicate_existing_targeted_candidate",
            "candidate": FOUNDRY_COVERAGE_GAP_CANDIDATE,
            "reason": "coverage_gap_probe_needs_replicates",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": suggested_lever,
            "ok_runs": ok_runs,
            "additional_successful_runs": max(0, 3 - ok_runs),
            "command": (
                "python examples\\foundry_harness_batch.py --candidates "
                f"{FOUNDRY_COVERAGE_GAP_CANDIDATE} --replicates "
                f"{max(0, 3 - ok_runs)} --json"
            ),
            "target_tasks": target_tasks,
            "target_failure_tags": target_failure_tags,
        }

    candidate_median = _optional_float(aggregate.get("ranking_score_median"))
    baseline_aggregate = aggregate_by_candidate.get(baseline_candidate, {})
    baseline_median = _optional_float(
        baseline_aggregate.get("ranking_score_median")
    )
    if (
        candidate_median is not None
        and baseline_median is not None
        and candidate_median <= baseline_median
    ):
        return {
            "action": "reject_existing_targeted_candidate",
            "candidate": FOUNDRY_COVERAGE_GAP_CANDIDATE,
            "baseline": baseline_candidate,
            "reason": "coverage_gap_probe_underperformed_baseline",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": suggested_lever,
            "ok_runs": ok_runs,
            "ranking_score_median": candidate_median,
            "baseline_ranking_score_median": baseline_median,
            "median_score_delta_vs_baseline": candidate_median - baseline_median,
            "target_tasks": target_tasks,
            "target_failure_tags": target_failure_tags,
        }

    return {
        "action": "study_existing_targeted_candidate",
        "candidate": FOUNDRY_COVERAGE_GAP_CANDIDATE,
        "reason": "coverage_gap_probe_has_replicates",
        "candidate_scope": "experimental_not_promotion_eligible",
        "suggested_lever": suggested_lever,
        "ok_runs": ok_runs,
        "ranking_score_median": candidate_median,
        "baseline_ranking_score_median": baseline_median,
        "target_tasks": target_tasks,
        "target_failure_tags": target_failure_tags,
    }


def _stability_candidate_action(
    aggregate_by_candidate: Mapping[str, Mapping[str, Any]],
    baseline_candidate: str,
) -> dict[str, Any]:
    aggregate = aggregate_by_candidate.get(FOUNDRY_CHUNK2_ONLY_CANDIDATE)
    if not aggregate:
        return {
            "action": "run_existing_stability_candidate",
            "candidate": FOUNDRY_CHUNK2_ONLY_CANDIDATE,
            "reason": "full_trinity_unstable_lift_probe_without_successful_runs",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": "codec_action_space_stability",
            "recommended_successful_runs": 3,
            "command": (
                "python examples\\foundry_harness_batch.py --candidates "
                f"{FOUNDRY_CHUNK2_ONLY_CANDIDATE} --replicates 3 --json"
            ),
        }

    ok_runs = int(aggregate.get("ok_runs", 0) or 0)
    if ok_runs < 3:
        return {
            "action": "replicate_existing_stability_candidate",
            "candidate": FOUNDRY_CHUNK2_ONLY_CANDIDATE,
            "reason": "chunk2_only_probe_needs_replicates",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": "codec_action_space_stability",
            "ok_runs": ok_runs,
            "additional_successful_runs": max(0, 3 - ok_runs),
            "command": (
                "python examples\\foundry_harness_batch.py --candidates "
                f"{FOUNDRY_CHUNK2_ONLY_CANDIDATE} --replicates "
                f"{max(0, 3 - ok_runs)} --json"
            ),
        }

    candidate_median = _optional_float(aggregate.get("ranking_score_median"))
    baseline_aggregate = aggregate_by_candidate.get(baseline_candidate, {})
    baseline_median = _optional_float(
        baseline_aggregate.get("ranking_score_median")
    )
    if (
        candidate_median is not None
        and baseline_median is not None
        and candidate_median <= baseline_median
    ):
        return {
            "action": "reject_existing_stability_candidate",
            "candidate": FOUNDRY_CHUNK2_ONLY_CANDIDATE,
            "baseline": baseline_candidate,
            "reason": "chunk2_only_probe_underperformed_baseline",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": "codec_action_space_stability",
            "ok_runs": ok_runs,
            "ranking_score_median": candidate_median,
            "baseline_ranking_score_median": baseline_median,
            "median_score_delta_vs_baseline": candidate_median - baseline_median,
        }

    return {
        "action": "study_existing_stability_candidate",
        "candidate": FOUNDRY_CHUNK2_ONLY_CANDIDATE,
        "reason": "chunk2_only_probe_has_replicates",
        "candidate_scope": "experimental_not_promotion_eligible",
        "suggested_lever": "codec_action_space_stability",
        "ok_runs": ok_runs,
        "ranking_score_median": candidate_median,
        "baseline_ranking_score_median": baseline_median,
    }


def _lift_pocket_candidate_action(
    aggregate_by_candidate: Mapping[str, Mapping[str, Any]],
    baseline_candidate: str,
    positive_task_pockets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_tasks = [
        str(pocket.get("task_id"))
        for pocket in positive_task_pockets[:5]
        if pocket.get("task_id")
    ]
    aggregate = aggregate_by_candidate.get(FOUNDRY_LIFT_POCKET_CANDIDATE)
    if not aggregate:
        return {
            "action": "run_existing_lift_pocket_candidate",
            "candidate": FOUNDRY_LIFT_POCKET_CANDIDATE,
            "reason": "full_trinity_unstable_lift_has_positive_task_pockets",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": "task_allocation_lift_replay",
            "recommended_successful_runs": 3,
            "target_tasks": target_tasks,
            "command": (
                "python examples\\foundry_harness_batch.py --candidates "
                f"{FOUNDRY_LIFT_POCKET_CANDIDATE} --replicates 3 --json"
            ),
        }

    ok_runs = int(aggregate.get("ok_runs", 0) or 0)
    if ok_runs < 3:
        return {
            "action": "replicate_existing_lift_pocket_candidate",
            "candidate": FOUNDRY_LIFT_POCKET_CANDIDATE,
            "reason": "lift_pocket_probe_needs_replicates",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": "task_allocation_lift_replay",
            "ok_runs": ok_runs,
            "additional_successful_runs": max(0, 3 - ok_runs),
            "target_tasks": target_tasks,
            "command": (
                "python examples\\foundry_harness_batch.py --candidates "
                f"{FOUNDRY_LIFT_POCKET_CANDIDATE} --replicates "
                f"{max(0, 3 - ok_runs)} --json"
            ),
        }

    candidate_median = _optional_float(aggregate.get("ranking_score_median"))
    baseline_aggregate = aggregate_by_candidate.get(baseline_candidate, {})
    baseline_median = _optional_float(
        baseline_aggregate.get("ranking_score_median")
    )
    if (
        candidate_median is not None
        and baseline_median is not None
        and candidate_median <= baseline_median
    ):
        return {
            "action": "reject_existing_lift_pocket_candidate",
            "candidate": FOUNDRY_LIFT_POCKET_CANDIDATE,
            "baseline": baseline_candidate,
            "reason": "lift_pocket_probe_underperformed_baseline",
            "candidate_scope": "experimental_not_promotion_eligible",
            "suggested_lever": "task_allocation_lift_replay",
            "ok_runs": ok_runs,
            "ranking_score_median": candidate_median,
            "baseline_ranking_score_median": baseline_median,
            "median_score_delta_vs_baseline": candidate_median - baseline_median,
            "target_tasks": target_tasks,
        }

    return {
        "action": "study_existing_lift_pocket_candidate",
        "candidate": FOUNDRY_LIFT_POCKET_CANDIDATE,
        "reason": "lift_pocket_probe_has_replicates",
        "candidate_scope": "experimental_not_promotion_eligible",
        "suggested_lever": "task_allocation_lift_replay",
        "ok_runs": ok_runs,
        "ranking_score_median": candidate_median,
        "baseline_ranking_score_median": baseline_median,
        "target_tasks": target_tasks,
    }


def _promotion_actions(
    promotion_readiness: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for decision in promotion_readiness.get("decisions", []):
        if not isinstance(decision, Mapping):
            continue
        if decision.get("status") != "promote":
            continue
        actions.append(
            {
                "action": "promote_candidate",
                "candidate": decision.get("candidate"),
                "reason": decision.get("reason"),
                "pairwise_win_rate_vs_baseline": _optional_float(
                    decision.get("pairwise_win_rate_vs_baseline")
                ),
                "median_score_delta_vs_baseline": _optional_float(
                    decision.get("median_score_delta_vs_baseline")
                ),
            }
        )
    return actions


def _replicate_actions(
    promotion_readiness: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in promotion_readiness.get("recommended_next_runs", []):
        if not isinstance(item, Mapping):
            continue
        actions.append(
            {
                "action": "run_additional_replicates",
                "candidate": item.get("candidate"),
                "additional_successful_runs": int(
                    item.get("additional_successful_runs", 0) or 0
                ),
            }
        )
    return actions


def _positive_task_pockets(
    failure_pockets: Mapping[str, Any],
    candidate: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    delta = _failure_pocket_delta_for_candidate(failure_pockets, candidate)
    rows = delta.get("tasks_better_than_baseline")
    return [dict(row) for row in rows[:limit]] if isinstance(rows, list) else []


def _negative_task_pockets(
    failure_pockets: Mapping[str, Any],
    candidate: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    delta = _failure_pocket_delta_for_candidate(failure_pockets, candidate)
    rows = delta.get("tasks_worse_than_baseline")
    return [dict(row) for row in rows[:limit]] if isinstance(rows, list) else []


def _failure_pocket_delta_for_candidate(
    failure_pockets: Mapping[str, Any],
    candidate: str,
) -> dict[str, Any]:
    deltas = failure_pockets.get("deltas_vs_baseline")
    if not isinstance(deltas, list):
        return {}
    for item in deltas:
        if not isinstance(item, Mapping):
            continue
        if item.get("candidate") == candidate:
            return dict(item)
    return {}


def _shared_failure_pockets(
    failure_pockets: Mapping[str, Any],
    candidates: Sequence[str],
) -> dict[str, Any]:
    by_candidate = failure_pockets.get("by_candidate")
    if not isinstance(by_candidate, list):
        return {"candidates": [], "tasks": [], "families": [], "failure_tags": []}
    pocket_by_candidate = {
        str(item.get("candidate", "")): item
        for item in by_candidate
        if isinstance(item, Mapping) and item.get("candidate")
    }
    selected = [
        candidate
        for candidate in candidates
        if candidate in pocket_by_candidate
    ]
    return {
        "candidates": selected,
        "tasks": _shared_failure_rows(
            pocket_by_candidate,
            selected,
            section="by_task",
            name_key="task_id",
        ),
        "families": _shared_failure_rows(
            pocket_by_candidate,
            selected,
            section="by_family",
            name_key="name",
        ),
        "failure_tags": _shared_failure_rows(
            pocket_by_candidate,
            selected,
            section="by_failure_tag",
            name_key="name",
        ),
    }


def _shared_failure_rows(
    pocket_by_candidate: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    section: str,
    name_key: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    rows_by_candidate = {
        candidate: _rows_by_name(
            pocket_by_candidate[candidate].get(section),
            name_key=name_key,
        )
        for candidate in candidates
        if candidate in pocket_by_candidate
    }
    if not rows_by_candidate:
        return []
    shared_names = set.intersection(
        *(set(rows) for rows in rows_by_candidate.values())
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(shared_names):
        candidate_rows = {
            candidate: rows[name]
            for candidate, rows in rows_by_candidate.items()
            if name in rows
        }
        failed_total = sum(
            int(row.get("failed", 0) or 0)
            for row in candidate_rows.values()
        )
        if failed_total <= 0:
            continue
        pass_rates = [
            value
            for value in (
                _optional_float(row.get("pass_rate"))
                for row in candidate_rows.values()
            )
            if value is not None
        ]
        observations = sum(
            int(row.get("observations", 0) or 0)
            for row in candidate_rows.values()
        )
        row = {
            name_key: name,
            "candidate_pass_rates": {
                candidate: _optional_float(candidate_row.get("pass_rate"))
                for candidate, candidate_row in sorted(candidate_rows.items())
            },
            "mean_pass_rate": fmean(pass_rates) if pass_rates else None,
            "failed_total": failed_total,
            "observations": observations,
        }
        top_failure_modes = _combined_top_failure_modes(candidate_rows.values())
        if top_failure_modes:
            row["top_failure_modes"] = top_failure_modes
            row["dominant_failure_mode"] = top_failure_modes[0]["name"]
            row["failure_mode_class"] = _failure_mode_class(top_failure_modes)
        for metadata_key in ("family", "difficulty", "failure_tags"):
            metadata_value = next(
                (
                    candidate_row.get(metadata_key)
                    for candidate_row in candidate_rows.values()
                    if candidate_row.get(metadata_key)
                ),
                None,
            )
            if metadata_value is not None:
                row[metadata_key] = metadata_value
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["mean_pass_rate"] if row["mean_pass_rate"] is not None else 1.0,
            -int(row["failed_total"]),
            -int(row["observations"]),
            str(row.get(name_key, "")),
        )
    )
    return rows[:limit]


def _combined_top_failure_modes(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        top_modes = row.get("top_failure_modes")
        if not isinstance(top_modes, list):
            continue
        for mode in top_modes:
            if not isinstance(mode, Mapping):
                continue
            name = mode.get("name")
            count = mode.get("count")
            if not isinstance(name, str) or not isinstance(count, int | float):
                continue
            counts[name] = counts.get(name, 0) + int(count)
    return _top_counts(counts, limit=limit)


def _failure_mode_class(top_failure_modes: Sequence[Mapping[str, Any]]) -> str:
    names = {
        str(item.get("name", ""))
        for item in top_failure_modes
        if isinstance(item, Mapping)
    }
    if not names:
        return "unknown"
    if names == {"missing_learned_solution"}:
        return "coverage_gap"
    if any(name in names for name in ("unit_test_failed", "wrong_answer")):
        return "repair_quality_gap"
    if any("timeout" in name for name in names):
        return "runtime_or_verifier_gap"
    if any("crash" in name or "syntax" in name for name in names):
        return "output_contract_gap"
    return "mixed_gap"


def _suggested_candidate_lever(shared_failure_pockets: Mapping[str, Any]) -> str:
    tasks = shared_failure_pockets.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return "inspect_failure_pockets"
    classes = {
        str(task.get("failure_mode_class", "unknown"))
        for task in tasks[:5]
        if isinstance(task, Mapping)
    }
    if classes == {"coverage_gap"}:
        return "task_allocation_or_budget_coverage"
    if "repair_quality_gap" in classes:
        return "repair_prompt_or_verifier_feedback"
    if "runtime_or_verifier_gap" in classes:
        return "timeout_or_verifier_hardening"
    if "output_contract_gap" in classes:
        return "output_contract_hardening"
    return "mixed_failure_mode_probe"


def _pocket_delta_rows(
    candidate_rows: Any,
    baseline_rows: Any,
    *,
    name_key: str = "name",
) -> list[dict[str, Any]]:
    candidate_by_name = _rows_by_name(candidate_rows, name_key=name_key)
    baseline_by_name = _rows_by_name(baseline_rows, name_key=name_key)
    rows: list[dict[str, Any]] = []
    for name in sorted(set(candidate_by_name) | set(baseline_by_name)):
        candidate = candidate_by_name.get(name, {})
        baseline = baseline_by_name.get(name, {})
        candidate_rate = _optional_float(candidate.get("pass_rate"))
        baseline_rate = _optional_float(baseline.get("pass_rate"))
        row = {
            name_key: name,
            "candidate_pass_rate": candidate_rate,
            "baseline_pass_rate": baseline_rate,
            "pass_rate_delta": _optional_delta(candidate_rate, baseline_rate),
            "candidate_observations": int(candidate.get("observations", 0) or 0),
            "baseline_observations": int(baseline.get("observations", 0) or 0),
            "candidate_failed": int(candidate.get("failed", 0) or 0),
            "baseline_failed": int(baseline.get("failed", 0) or 0),
        }
        for key in ("family", "difficulty", "failure_tags"):
            if key in candidate:
                row[key] = candidate[key]
            elif key in baseline:
                row[key] = baseline[key]
        rows.append(row)
    rows.sort(key=_pocket_delta_key)
    return rows


def _rows_by_name(value: Any, *, name_key: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            continue
        name = row.get(name_key)
        if name is not None:
            rows[str(name)] = row
    return rows


def _weakest_pocket_key(row: Mapping[str, Any]) -> tuple[float, int, int, str]:
    pass_rate = _optional_float(row.get("pass_rate"))
    if pass_rate is None:
        pass_rate = 1.0
    return (
        pass_rate,
        -int(row.get("failed", 0) or 0),
        -int(row.get("observations", 0) or 0),
        str(row.get("task_id") or row.get("name") or ""),
    )


def _strongest_pocket_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -(_optional_float(row.get("pass_rate")) or 0.0),
            -int(row.get("observations", 0) or 0),
            str(row.get("task_id") or row.get("name") or ""),
        ),
    )


def _lowest_delta_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if _optional_float(row.get("pass_rate_delta")) is not None
        and (_optional_float(row.get("pass_rate_delta")) or 0.0) < 0.0
    ]


def _highest_delta_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            dict(row)
            for row in rows
            if _optional_float(row.get("pass_rate_delta")) is not None
            and (_optional_float(row.get("pass_rate_delta")) or 0.0) > 0.0
        ),
        key=lambda row: (
            -(_optional_float(row.get("pass_rate_delta")) or 0.0),
            -int(row.get("candidate_observations", 0) or 0),
            str(row.get("task_id") or row.get("name") or ""),
        ),
    )


def _pocket_delta_key(row: Mapping[str, Any]) -> tuple[float, int, str]:
    delta = _optional_float(row.get("pass_rate_delta"))
    if delta is None:
        delta = 0.0
    return (
        delta,
        -int(row.get("candidate_observations", 0) or 0),
        str(row.get("task_id") or row.get("name") or ""),
    )


def _top_counts(value: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    rows = [
        {"name": str(name), "count": int(count)}
        for name, count in value.items()
        if isinstance(count, int | float) and count > 0
    ]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["name"])))
    return rows[:limit]


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if isinstance(item, str)]


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
        "promotion_eligible": _summary_promotion_eligible(first),
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


def _candidate_pairwise_comparison(
    left: str,
    left_summaries: Sequence[Mapping[str, Any]],
    right: str,
    right_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_deltas: list[float] = []
    spend_deltas: list[float] = []
    left_wins = 0
    right_wins = 0
    ties = 0

    for left_summary in left_summaries:
        left_score = _summary_score(left_summary)
        if left_score is None:
            continue
        left_spend = _optional_float(
            left_summary.get("primary_accounted_dollar_seconds")
        )
        for right_summary in right_summaries:
            right_score = _summary_score(right_summary)
            if right_score is None:
                continue
            right_spend = _optional_float(
                right_summary.get("primary_accounted_dollar_seconds")
            )
            score_delta = left_score - right_score
            score_deltas.append(score_delta)
            if left_spend is not None and right_spend is not None:
                spend_deltas.append(left_spend - right_spend)
            if math.isclose(score_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                ties += 1
            elif score_delta > 0.0:
                left_wins += 1
            else:
                right_wins += 1

    pair_count = len(score_deltas)
    mean_score_delta = fmean(score_deltas) if score_deltas else None
    leader = _pairwise_leader(
        left,
        right,
        left_wins=left_wins,
        right_wins=right_wins,
        mean_score_delta=mean_score_delta,
    )
    return {
        "left_candidate": left,
        "right_candidate": right,
        "left_runs": len(left_summaries),
        "right_runs": len(right_summaries),
        "pair_count": pair_count,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "left_win_rate": left_wins / pair_count if pair_count else None,
        "right_win_rate": right_wins / pair_count if pair_count else None,
        "tie_rate": ties / pair_count if pair_count else None,
        "mean_score_delta_left_minus_right": mean_score_delta,
        "mean_accounted_dollar_seconds_delta_left_minus_right": (
            fmean(spend_deltas) if spend_deltas else None
        ),
        "leader_candidate": leader,
        "leader_win_rate": (
            max(left_wins, right_wins) / pair_count if pair_count else None
        ),
    }


def _pairwise_leader(
    left: str,
    right: str,
    *,
    left_wins: int,
    right_wins: int,
    mean_score_delta: float | None,
) -> str | None:
    if left_wins > right_wins:
        return left
    if right_wins > left_wins:
        return right
    if mean_score_delta is None:
        return None
    if math.isclose(mean_score_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return left if mean_score_delta > 0.0 else right


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


def _summary_promotion_eligible(summary: Mapping[str, Any]) -> bool:
    value = summary.get("promotion_eligible")
    if isinstance(value, bool):
        return value
    candidate = summary.get("candidate")
    if isinstance(candidate, str) and candidate:
        try:
            return load_foundry_harness_manifest(candidate).promotion_eligible
        except (OSError, ValueError):
            return True
    return True


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
