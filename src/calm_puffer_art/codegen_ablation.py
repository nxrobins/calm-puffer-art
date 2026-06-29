from __future__ import annotations

import ast
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from .actions import (
    ActionCodec,
    AdaptiveActionSpace,
    ChunkActionCodec,
    TokenActionCodec,
    action_codec_key,
    safe_metric_key,
)
from .runtime import ControlPlane, ControlPlaneConfig, RolloutContext
from .scheduler import ObjectiveScheduler
from .types import (
    Message,
    PolicySnapshot,
    RunSummary,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


CODEGEN_NORTH_STAR = (
    "north_star/published_policy_reward_improving_experience_per_dollar_second"
)
CODEGEN_ACCOUNTED_NORTH_STAR = (
    "north_star/accounted_published_policy_reward_improving_experience_per_dollar_second"
)
DEFAULT_CODEGEN_RESPONSE_STYLES = (1, 2, 4)
DEFAULT_CODEGEN_TRAIN_STEPS = 12
DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS = 0.08
CODEGEN_RECOVERY_MIN_PULLS = 2.0
DEFAULT_CODEGEN_SHOWCASE_TRAIN_STEPS = 8
DEFAULT_CODEGEN_SHOWCASE_RESPONSE_STYLE = 4


@dataclass(frozen=True)
class CodegenCandidate:
    compact: str
    structured: str
    expanded: str

    def render(self, response_style: int) -> str:
        if response_style <= 1:
            return self.compact.strip()
        if response_style <= 2:
            return self.structured.strip()
        return self.expanded.strip()


@dataclass(frozen=True)
class CodegenTask:
    id: str
    prompt: str
    candidates: tuple[CodegenCandidate, ...]
    tests: tuple[tuple[tuple[Any, ...], Any], ...]
    rollout_dollar_seconds: float
    target_candidate: int = 0


class CodegenPolicy:
    """Tiny template policy for deterministic codegen ablations."""

    def __init__(
        self,
        scores: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._scores = {
            task_id: tuple(float(value) for value in values)
            for task_id, values in (scores or {}).items()
        }

    def render(
        self,
        task: CodegenTask,
        *,
        response_style: int,
    ) -> tuple[int, str]:
        scores = self._scores.get(task.id)
        if scores is None:
            # Start from the verifier-passing template so this domain probe
            # isolates action-representation cost on structured code outputs.
            scores = tuple(0.0 for _ in task.candidates)
        candidate_index = max(
            range(len(task.candidates)),
            key=lambda index: (scores[index], -index),
        )
        return candidate_index, task.candidates[candidate_index].render(
            response_style
        )

    def update_from_trajectories(
        self,
        trajectories: Sequence[Trajectory],
    ) -> "CodegenPolicy":
        scores = {task_id: list(values) for task_id, values in self._scores.items()}
        for trajectory in trajectories:
            task_id = str(trajectory.metadata.get("codegen/task_id", ""))
            selected = int(trajectory.metadata.get("codegen/candidate_index", 0))
            target = int(trajectory.metadata.get("codegen/target_candidate", 0))
            candidate_count = int(trajectory.metadata.get("codegen/candidate_count", 0))
            if not task_id or candidate_count <= 0:
                continue
            values = scores.setdefault(
                task_id,
                [0.0 for _ in range(candidate_count)],
            )
            values[target] += 1.0
            if trajectory.reward <= 0.0 and 0 <= selected < len(values):
                values[selected] -= 0.25
        return CodegenPolicy(scores=scores)


class CodegenTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        trajectories = [
            trajectory
            for group in groups
            for trajectory in group.trajectories
        ]
        rewards = [trajectory.reward for trajectory in trajectories]
        current_policy = current.policy
        if not isinstance(current_policy, CodegenPolicy):
            raise TypeError("codegen_policy_required")
        next_policy = current_policy.update_from_trajectories(trajectories)
        return TrainResult(
            policy=next_policy,
            checkpoint_id=f"codegen-step-{current.step + 1}",
            metrics={
                "train/reward": fmean(rewards) if rewards else 0.0,
                "train/dollar_seconds": 0.35,
                "codegen/train_examples": float(len(trajectories)),
            },
        )


async def codegen_ablation_rollout(
    policy: CodegenPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    task = _codegen_task_for_scenario(scenario)
    response_style = int(scenario.payload.get("response_style", 1))
    candidate_index, code = policy.render(task, response_style=response_style)
    passed, failure_mode = _verify_codegen_solution(task, code)
    actions = context.action_codec.encode(code)
    for action in actions:
        action.metadata.setdefault("codegen/task_id", task.id)
        action.metadata.setdefault("codegen/candidate_index", candidate_index)
        action.metadata.setdefault("codegen/response_style", response_style)
    rollout_cost = _codegen_rollout_dollar_seconds(
        task=task,
        scenario=scenario,
        action_units=len(actions),
    )
    metadata = {
        **dict(context.decision_metadata),
        "scenario_id": scenario.id,
        "codegen/workload": "unit_test_templates",
        "codegen/task_id": task.id,
        "codegen/prompt": task.prompt,
        "codegen/code": code,
        "codegen/candidate_index": candidate_index,
        "codegen/candidate_count": len(task.candidates),
        "codegen/target_candidate": task.target_candidate,
        "codegen/response_style": response_style,
        "codegen/source_tokens": sum(action.token_count for action in actions),
        "action/safe": True,
        "reconstruction/accuracy": 1.0,
        "reconstruction/safe": True,
        "verifier/passed": passed,
    }
    if not passed:
        metadata["verifier/failure_mode"] = failure_mode
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=[Message(role="user", content=task.prompt)],
        actions=actions,
        reward=1.0 if passed else 0.0,
        metrics={"rollout/dollar_seconds": rollout_cost},
        metadata=metadata,
    )


async def run_codegen_semantic_sweep(
    *,
    response_styles: Sequence[int] = DEFAULT_CODEGEN_RESPONSE_STYLES,
    max_train_steps: int = DEFAULT_CODEGEN_TRAIN_STEPS,
    repeats: int = 1,
    action_unit_dollar_seconds: float = DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
) -> dict[str, Any]:
    styles = _positive_int_sequence(response_styles, "response_styles")
    train_steps = _positive_int(max_train_steps, "max_train_steps")
    repeat_count = _positive_int(repeats, "repeats")
    rows = []
    for response_style in styles:
        runs = [
            await _run_codegen_semantic_point(
                response_style=response_style,
                max_train_steps=train_steps,
                action_unit_dollar_seconds=action_unit_dollar_seconds,
            )
            for _ in range(repeat_count)
        ]
        row = _mean_numeric_rows(runs)
        row["response_style"] = response_style
        row["measurement_repeats"] = repeat_count
        rows.append(row)
    chunk3_recovers_at = _first_recovery_tokens(rows, "chunk3")
    chunk4_recovers_at = _first_recovery_tokens(rows, "chunk4")
    return {
        "proof_scope": "tiny_unit_test_codegen_fixed_codecs",
        "measurement": "codegen_semantic_bandwidth_sweep",
        "max_train_steps": train_steps,
        "action_unit_dollar_seconds": action_unit_dollar_seconds,
        "repeats": repeat_count,
        "rows": rows,
        "chunk3_recovers_at_response_tokens": chunk3_recovers_at,
        "chunk4_recovers_at_response_tokens": chunk4_recovers_at,
    }


async def run_python_codegen_showcase(
    *,
    max_train_steps: int = DEFAULT_CODEGEN_SHOWCASE_TRAIN_STEPS,
    response_style: int = DEFAULT_CODEGEN_SHOWCASE_RESPONSE_STYLE,
    action_unit_dollar_seconds: float = DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
) -> dict[str, Any]:
    """Compare ART-shaped codegen conditions on unit-test verified tasks."""

    train_steps = _positive_int(max_train_steps, "max_train_steps")
    style = _positive_int(response_style, "response_style")
    if action_unit_dollar_seconds < 0.0:
        raise ValueError("action_unit_dollar_seconds_must_be_non_negative")

    static = await _run_codegen_condition(
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
        response_style=style,
        max_train_steps=train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    scheduler_only = await _run_codegen_condition(
        scheduler=_codegen_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec()],
        response_style=style,
        max_train_steps=train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    full_trinity = await _run_codegen_condition(
        scheduler=_codegen_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
        response_style=style,
        max_train_steps=train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )

    conditions = {
        "static_art": _codegen_summary_metrics(static),
        "scheduler_only": _codegen_summary_metrics(scheduler_only),
        "full_trinity": _codegen_summary_metrics(full_trinity),
    }
    static_score = conditions["static_art"].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0)
    scheduler_score = conditions["scheduler_only"].get(
        CODEGEN_ACCOUNTED_NORTH_STAR,
        0.0,
    )
    full_score = conditions["full_trinity"].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0)
    winning_condition = max(
        conditions,
        key=lambda name: conditions[name].get(CODEGEN_ACCOUNTED_NORTH_STAR, 0.0),
    )
    full_codec_winner = _winning_codegen_codec(conditions["full_trinity"])

    return {
        "ok": True,
        "proof_scope": "tiny_unit_test_codegen_showcase",
        "measurement": "python_codegen_showcase",
        "workload": "embedded_python_function_synthesis_unit_tests",
        "max_train_steps": train_steps,
        "response_style": style,
        "mean_response_tokens": _mean_codegen_response_tokens(style),
        "action_unit_dollar_seconds": action_unit_dollar_seconds,
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
            "full_trinity_semantic_bandwidth_over_scheduler_ratio": _finite_ratio(
                conditions["full_trinity"].get(
                    "actions/semantic_bandwidth_tokens_per_decision",
                    0.0,
                ),
                conditions["scheduler_only"].get(
                    "actions/semantic_bandwidth_tokens_per_decision",
                    0.0,
                ),
            ),
        },
        "winning_condition_by_accounted_north_star": winning_condition,
        "winning_codec_by_improvement_per_dollar": full_codec_winner,
        "chunk4_promoted": (
            conditions["full_trinity"].get("action_space/promotions", 0.0) > 0.0
        ),
        "chunk4_active": (
            conditions["full_trinity"].get(
                "action_space/codec/chunk_chunk_size_4/active",
                0.0,
            )
            > 0.0
        ),
    }


async def _run_codegen_semantic_point(
    *,
    response_style: int,
    max_train_steps: int,
    action_unit_dollar_seconds: float,
) -> dict[str, Any]:
    summary = await _run_codegen_workload(
        response_style=response_style,
        max_train_steps=max_train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    metrics = dict(summary.metrics)
    row = {
        "response_style": response_style,
        "mean_response_tokens": _mean_codegen_response_tokens(response_style),
        "accounted_north_star": _metric(metrics, CODEGEN_ACCOUNTED_NORTH_STAR),
        "north_star": _metric(metrics, CODEGEN_NORTH_STAR),
        "accounted_dollar_seconds": _metric(
            metrics,
            "costs/accounted_dollar_seconds",
        ),
        "semantic_bandwidth_tokens_per_decision": _metric(
            metrics,
            "actions/semantic_bandwidth_tokens_per_decision",
        ),
        "train_steps": _metric(metrics, "data/train_steps"),
        "groups_trained": _metric(metrics, "data/groups_trained"),
    }
    row.update(_codegen_codec_metrics(metrics))
    row["winning_codec_by_improvement_per_dollar"] = (
        _winning_codegen_codec(row)
    )
    row["chunk3_recovers"] = 1.0 if _codegen_chunk_recovered(row, "chunk3") else 0.0
    row["chunk4_recovers"] = 1.0 if _codegen_chunk_recovered(row, "chunk4") else 0.0
    return row


async def _run_codegen_workload(
    *,
    response_style: int,
    max_train_steps: int,
    action_unit_dollar_seconds: float,
) -> RunSummary:
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        min_actor_count=1,
        max_actor_count=2,
        exploration_bonus=0.0,
    )
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=2,
            group_size=1,
            train_batch_groups=2,
            max_train_steps=max_train_steps,
            queue_max_trajectories=8,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=_codegen_scenarios(
            response_style=response_style,
            action_unit_dollar_seconds=action_unit_dollar_seconds,
        ),
        initial_policy=CodegenPolicy(),
        trainer=CodegenTrainer(),
        workflow=codegen_ablation_rollout,
        action_codecs=_codegen_action_codecs(),
        scheduler=scheduler,
    )


async def _run_codegen_condition(
    *,
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec],
    response_style: int,
    max_train_steps: int,
    action_unit_dollar_seconds: float,
) -> RunSummary:
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=2,
            group_size=1,
            train_batch_groups=2,
            max_train_steps=max_train_steps,
            queue_max_trajectories=8,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=_codegen_scenarios(
            response_style=response_style,
            action_unit_dollar_seconds=action_unit_dollar_seconds,
        ),
        initial_policy=_initial_wrong_codegen_policy(),
        trainer=CodegenTrainer(),
        workflow=codegen_ablation_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


def _codegen_scenarios(
    *,
    response_style: int,
    action_unit_dollar_seconds: float,
) -> tuple[Scenario, ...]:
    return tuple(
        Scenario(
            id=task.id,
            payload={
                "response_style": response_style,
                "rollout_dollar_seconds": task.rollout_dollar_seconds,
                "action_unit_dollar_seconds": action_unit_dollar_seconds,
            },
        )
        for task in _CODEGEN_TASKS
    )


def _codegen_action_codecs() -> list[ActionCodec]:
    return [
        TokenActionCodec(),
        ChunkActionCodec(chunk_size=2),
        ChunkActionCodec(chunk_size=3),
        ChunkActionCodec(chunk_size=4),
    ]


def _codegen_scheduler() -> ObjectiveScheduler:
    return ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        min_actor_count=1,
        max_actor_count=2,
        exploration_bonus=0.0,
    )


def _initial_wrong_codegen_policy() -> CodegenPolicy:
    scores = {}
    for task in _CODEGEN_TASKS:
        values = [0.0 for _ in task.candidates]
        wrong_index = 1 if len(values) > 1 else 0
        values[wrong_index] = 1.0
        scores[task.id] = values
    return CodegenPolicy(scores=scores)


def _codegen_task_for_scenario(scenario: Scenario) -> CodegenTask:
    for task in _CODEGEN_TASKS:
        if task.id == scenario.id:
            return task
    raise ValueError(f"unknown codegen scenario: {scenario.id}")


def _verify_codegen_solution(task: CodegenTask, code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                return False, "forbidden_syntax"
        namespace: dict[str, Any] = {}
        safe_builtins = {
            "len": len,
            "range": range,
            "set": set,
            "sorted": sorted,
        }
        exec(  # noqa: S102 - deterministic embedded candidate code only.
            compile(tree, "<codegen_ablation>", "exec"),
            {"__builtins__": safe_builtins},
            namespace,
        )
        solve = namespace.get("solve")
        if not callable(solve):
            return False, "missing_solve"
        for args, expected in task.tests:
            if solve(*args) != expected:
                return False, "unit_test_failed"
    except Exception:
        return False, "exception"
    return True, "passed"


def _codegen_rollout_dollar_seconds(
    *,
    task: CodegenTask,
    scenario: Scenario,
    action_units: int,
) -> float:
    action_unit_cost = float(scenario.payload.get("action_unit_dollar_seconds", 0.0))
    return task.rollout_dollar_seconds + action_unit_cost * max(1, action_units)


def _mean_codegen_response_tokens(response_style: int) -> float:
    token_counts = [
        len(TokenActionCodec().encode(task.candidates[task.target_candidate].render(response_style)))
        for task in _CODEGEN_TASKS
    ]
    return fmean(token_counts) if token_counts else 0.0


def _codegen_codec_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    row: dict[str, float] = {}
    for label, codec in _codegen_codec_labels():
        codec_key = safe_metric_key(action_codec_key(codec))
        row[f"{label}_pulls"] = _codegen_codec_sum(metrics, codec_key, "pulls")
        row[f"{label}_improvement_per_dollar"] = _codegen_codec_weighted(
            metrics,
            codec_key,
            "total_improvement_per_dollar_second",
        )
        row[f"{label}_mean_rollout_dollar_seconds"] = _codegen_codec_weighted(
            metrics,
            codec_key,
            "mean_rollout_dollar_seconds",
        )
        row[f"{label}_semantic_bandwidth_tokens_per_decision"] = (
            _codegen_codec_weighted(
                metrics,
                codec_key,
                "semantic_bandwidth_tokens_per_decision",
            )
        )
        row[f"{label}_source_tokens_per_dollar_second"] = _codegen_codec_weighted(
            metrics,
            codec_key,
            "source_tokens_per_dollar_second",
        )
    return row


def _codegen_codec_labels() -> tuple[tuple[str, ActionCodec], ...]:
    return (
        ("token", TokenActionCodec()),
        ("chunk2", ChunkActionCodec(chunk_size=2)),
        ("chunk3", ChunkActionCodec(chunk_size=3)),
        ("chunk4", ChunkActionCodec(chunk_size=4)),
    )


def _codegen_summary_metrics(summary: RunSummary) -> dict[str, Any]:
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
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
        "scheduler/joint_action/positive_objective_tuples",
        "scheduler/joint_action/mean_objective_per_decision",
        "scheduler/continuation/decisions",
        "scheduler/continuation/feedback_updates",
    ]
    summary_values = {
        key: float(metrics[key])
        for key in keys
        if key in metrics
    }
    summary_values.update(_codegen_codec_metrics(metrics))
    summary_values["winning_codec_by_improvement_per_dollar"] = (
        _winning_codegen_codec(summary_values)
    )
    return summary_values


def _codegen_codec_sum(
    metrics: Mapping[str, float],
    codec_key: str,
    metric_name: str,
) -> float:
    return sum(
        _metric(metrics, f"scheduler/arm/{task.id}_{codec_key}/{metric_name}")
        for task in _CODEGEN_TASKS
    )


def _codegen_codec_weighted(
    metrics: Mapping[str, float],
    codec_key: str,
    metric_name: str,
) -> float:
    weighted_total = 0.0
    total_pulls = 0.0
    for task in _CODEGEN_TASKS:
        arm = f"scheduler/arm/{task.id}_{codec_key}"
        pulls = _metric(metrics, f"{arm}/pulls")
        weighted_total += pulls * _metric(metrics, f"{arm}/{metric_name}")
        total_pulls += pulls
    return weighted_total / total_pulls if total_pulls else 0.0


def _winning_codegen_codec(row: Mapping[str, Any]) -> str:
    labels = ("token", "chunk2", "chunk3", "chunk4")
    scored = [
        (label, float(row.get(f"{label}_improvement_per_dollar", 0.0)))
        for label in labels
    ]
    winner, value = max(scored, key=lambda item: item[1])
    return winner if value > 0.0 else "none"


def _codegen_chunk_recovered(row: Mapping[str, Any], label: str) -> bool:
    return (
        float(row.get(f"{label}_pulls", 0.0)) >= CODEGEN_RECOVERY_MIN_PULLS
        and float(row.get(f"{label}_improvement_per_dollar", 0.0))
        > float(row.get("chunk2_improvement_per_dollar", 0.0))
    )


def _first_recovery_tokens(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> int | None:
    for row in rows:
        if _codegen_chunk_recovered(row, label):
            return int(float(row.get("mean_response_tokens", 0.0)))
    return None


def _mean_numeric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty_codegen_sweep_runs")
    if len(rows) == 1:
        return dict(rows[0])
    merged = dict(rows[0])
    keys = set().union(*(row.keys() for row in rows))
    for key in keys:
        values = [row.get(key) for row in rows]
        if all(isinstance(value, (int, float)) for value in values):
            merged[key] = fmean(float(value) for value in values)
    merged["runs"] = [dict(row) for row in rows]
    return merged


def _positive_int_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    parsed = tuple(sorted(_positive_int(value, name) for value in values))
    if not parsed:
        raise ValueError(f"{name}_empty")
    return parsed


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name}_must_be_positive")
    return parsed


def _metric(metrics: Mapping[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0))


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


_CODEGEN_TASKS = (
    CodegenTask(
        id="code_clamp",
        prompt=(
            "Write Python function solve(value, low, high) that clamps value "
            "inside the inclusive [low, high] interval."
        ),
        rollout_dollar_seconds=0.7,
        tests=(
            ((5, 1, 9), 5),
            ((-3, 0, 4), 0),
            ((12, 0, 4), 4),
            ((4, 4, 9), 4),
        ),
        candidates=(
            CodegenCandidate(
                compact="""
def solve(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value
""",
                structured="""
def solve(value, low, high):
    \"\"\"Clamp value into an inclusive numeric interval.\"\"\"
    below_limit = value < low
    above_limit = value > high
    if below_limit:
        return low
    if above_limit:
        return high
    return value
""",
                expanded="""
def solve(value, low, high):
    \"\"\"Clamp value into the inclusive range described by low and high.\"\"\"
    # Keep the branch order explicit so boundary cases are easy to inspect.
    is_below_lower_bound = value < low
    if is_below_lower_bound:
        clamped_value = low
        return clamped_value
    is_above_upper_bound = value > high
    if is_above_upper_bound:
        clamped_value = high
        return clamped_value
    clamped_value = value
    return clamped_value
""",
            ),
            CodegenCandidate(
                compact="""
def solve(value, low, high):
    if value < low:
        return high
    if value > high:
        return low
    return value
""",
                structured="""
def solve(value, low, high):
    \"\"\"Attempt to clamp value, but swaps the limits.\"\"\"
    if value < low:
        return high
    if value > high:
        return low
    return value
""",
                expanded="""
def solve(value, low, high):
    \"\"\"Incorrect clamp implementation with reversed boundary returns.\"\"\"
    is_below_lower_bound = value < low
    if is_below_lower_bound:
        return high
    is_above_upper_bound = value > high
    if is_above_upper_bound:
        return low
    return value
""",
            ),
            CodegenCandidate(
                compact="""
def solve(value, low, high):
    if value < low:
        return low
    return value
""",
                structured="""
def solve(value, low, high):
    \"\"\"Only applies the lower bound.\"\"\"
    below_limit = value < low
    if below_limit:
        return low
    return value
""",
                expanded="""
def solve(value, low, high):
    \"\"\"Incorrect clamp implementation that forgets the upper bound.\"\"\"
    below_limit = value < low
    if below_limit:
        lower_result = low
        return lower_result
    unchecked_result = value
    return unchecked_result
""",
            ),
        ),
    ),
    CodegenTask(
        id="code_vowels",
        prompt=(
            "Write Python function solve(text) that counts English vowels in "
            "text case-insensitively."
        ),
        rollout_dollar_seconds=0.9,
        tests=(
            (("Area",), 3),
            (("rhythm",), 0),
            (("OpenAI",), 4),
            (("",), 0),
        ),
        candidates=(
            CodegenCandidate(
                compact="""
def solve(text):
    total = 0
    for char in text.lower():
        if char in "aeiou":
            total += 1
    return total
""",
                structured="""
def solve(text):
    \"\"\"Count vowels without caring about letter case.\"\"\"
    vowels = "aeiou"
    normalized = text.lower()
    total = 0
    for char in normalized:
        if char in vowels:
            total += 1
    return total
""",
                expanded="""
def solve(text):
    \"\"\"Return the number of a, e, i, o, and u characters in text.\"\"\"
    vowels = "aeiou"
    normalized_text = text.lower()
    vowel_count = 0
    for character in normalized_text:
        # The verifier expects case-insensitive matching and excludes y.
        character_is_vowel = character in vowels
        if character_is_vowel:
            vowel_count = vowel_count + 1
    return vowel_count
""",
            ),
            CodegenCandidate(
                compact="""
def solve(text):
    total = 0
    for char in text:
        if char in "aeiou":
            total += 1
    return total
""",
                structured="""
def solve(text):
    \"\"\"Count lowercase vowels only.\"\"\"
    vowels = "aeiou"
    total = 0
    for char in text:
        if char in vowels:
            total += 1
    return total
""",
                expanded="""
def solve(text):
    \"\"\"Incorrectly counts vowels without normalizing uppercase letters.\"\"\"
    lowercase_vowels = "aeiou"
    total = 0
    for character in text:
        if character in lowercase_vowels:
            total = total + 1
    return total
""",
            ),
            CodegenCandidate(
                compact="""
def solve(text):
    total = 0
    for char in text.lower():
        if char in "aeiouy":
            total += 1
    return total
""",
                structured="""
def solve(text):
    \"\"\"Count vowels but mistakenly includes y.\"\"\"
    vowels = "aeiouy"
    normalized = text.lower()
    total = 0
    for char in normalized:
        if char in vowels:
            total += 1
    return total
""",
                expanded="""
def solve(text):
    \"\"\"Incorrectly treats y as a vowel for this verifier.\"\"\"
    vowels_plus_y = "aeiouy"
    normalized_text = text.lower()
    total = 0
    for character in normalized_text:
        if character in vowels_plus_y:
            total = total + 1
    return total
""",
            ),
        ),
    ),
    CodegenTask(
        id="code_dedupe",
        prompt=(
            "Write Python function solve(items) that removes duplicate values "
            "while preserving first-seen order."
        ),
        rollout_dollar_seconds=1.1,
        tests=(
            (([1, 2, 1, 3, 2],), [1, 2, 3]),
            ((["b", "a", "b"],), ["b", "a"]),
            (([],), []),
            (([4, 4, 4],), [4]),
        ),
        candidates=(
            CodegenCandidate(
                compact="""
def solve(items):
    seen = []
    result = []
    for item in items:
        if item not in seen:
            seen.append(item)
            result.append(item)
    return result
""",
                structured="""
def solve(items):
    \"\"\"Remove duplicates while keeping the first occurrence order.\"\"\"
    seen = []
    result = []
    for item in items:
        already_seen = item in seen
        if not already_seen:
            seen.append(item)
            result.append(item)
    return result
""",
                expanded="""
def solve(items):
    \"\"\"Return a list containing each input value at most once in original order.\"\"\"
    seen_items = []
    unique_items = []
    for item in items:
        # A list is used instead of sorting so the original order is preserved.
        item_has_been_seen = item in seen_items
        if item_has_been_seen:
            continue
        seen_items.append(item)
        unique_items.append(item)
    return unique_items
""",
            ),
            CodegenCandidate(
                compact="""
def solve(items):
    return sorted(set(items))
""",
                structured="""
def solve(items):
    \"\"\"Remove duplicates by sorting them.\"\"\"
    unique = set(items)
    return sorted(unique)
""",
                expanded="""
def solve(items):
    \"\"\"Incorrectly removes duplicates by sorting, which changes order.\"\"\"
    unique_values = set(items)
    sorted_values = sorted(unique_values)
    return sorted_values
""",
            ),
            CodegenCandidate(
                compact="""
def solve(items):
    return list(items)
""",
                structured="""
def solve(items):
    \"\"\"Return a copy but forget to remove duplicates.\"\"\"
    copied = list(items)
    return copied
""",
                expanded="""
def solve(items):
    \"\"\"Incorrectly keeps every element, including repeated values.\"\"\"
    copied_items = []
    for item in items:
        copied_items.append(item)
    return copied_items
""",
            ),
        ),
    ),
)
