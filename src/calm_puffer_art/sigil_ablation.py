from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Sequence as SequenceABC
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
from .sigil_integration import SigilCorpus, load_sigil_corpus, verify_sigil_code
from .types import (
    Message,
    PolicySnapshot,
    RunSummary,
    Scenario,
    TrainResult,
    Trajectory,
    TrajectoryGroup,
)


SIGIL_NORTH_STAR = (
    "north_star/published_policy_reward_improving_experience_per_dollar_second"
)
SIGIL_ACCOUNTED_NORTH_STAR = (
    "north_star/accounted_published_policy_reward_improving_experience_per_dollar_second"
)
DEFAULT_SIGIL_TRAIN_STEPS = 50
DEFAULT_SIGIL_ACTION_UNIT_DOLLAR_SECONDS = 0.08
DEFAULT_SIGIL_ROLLOUT_DOLLAR_SECONDS = 0.25
DEFAULT_SIGIL_MAX_TASKS: int | None = None
SIGIL_BUCKET_EASY = "sigil_easy"
SIGIL_BUCKET_MEDIUM = "sigil_medium"
SIGIL_BUCKET_HARD = "sigil_hard"
SIGIL_BUCKET_ORDER = (SIGIL_BUCKET_EASY, SIGIL_BUCKET_MEDIUM, SIGIL_BUCKET_HARD)
SIGIL_WORKLOAD_PROOF_SCOPE = "sigil_compiler_codegen_full_trinity_v0"
_SAFE_MODULE_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class SigilAblationTask:
    id: str
    prompt: str
    candidates: tuple[str, ...]
    rollout_dollar_seconds: float
    target_candidate: int
    source_id: str

    @property
    def target_code(self) -> str:
        return self.candidates[self.target_candidate]


class SigilAblationPolicy:
    """Tiny torch policy that selects a Sigil token sequence candidate."""

    def __init__(
        self,
        *,
        feature_dim: int = 16,
        candidates: int = 3,
        hidden_dim: int = 32,
        seed: int = 1337,
        model: Any | None = None,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.feature_dim = feature_dim
        self.candidates = candidates
        self.hidden_dim = hidden_dim
        self.seed = seed
        if model is None:
            torch.manual_seed(seed)
            self.model = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, candidates),
            )
        else:
            self.model = model
        self.model.eval()

    def clone(self) -> "SigilAblationPolicy":
        return SigilAblationPolicy(
            feature_dim=self.feature_dim,
            candidates=self.candidates,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
            model=copy.deepcopy(self.model),
        )

    def render(self, task: SigilAblationTask) -> tuple[int, str, tuple[float, ...]]:
        torch, _, _ = _import_torch()
        self.model.eval()
        features = _sigil_features(task, feature_dim=self.feature_dim)
        with torch.no_grad():
            tensor = torch.tensor([features], dtype=torch.float32)
            logits = self.model(tensor).squeeze(0)
        available = min(self.candidates, len(task.candidates))
        selected = max(
            range(available),
            key=lambda index: (float(logits[index].item()), -index),
        )
        return selected, task.candidates[selected], tuple(float(value) for value in logits)


class SigilTrainer:
    def __init__(
        self,
        *,
        scenarios: Sequence[Scenario],
        learning_rate: float = 0.15,
        update_epochs: int = 6,
    ) -> None:
        self.scenarios = tuple(scenarios)
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs

    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        torch, _, functional = _import_torch()
        current_policy = current.policy
        if not isinstance(current_policy, SigilAblationPolicy):
            raise TypeError("sigil_policy_required")
        policy = current_policy.clone()
        examples = [
            _sigil_example_from_trajectory(trajectory)
            for group in groups
            for trajectory in group.trajectories
        ]
        loss_value = 0.0
        if examples:
            policy.model.train()
            optimizer = torch.optim.SGD(policy.model.parameters(), lr=self.learning_rate)
            features = torch.tensor(
                [example["features"] for example in examples],
                dtype=torch.float32,
            )
            labels = torch.tensor(
                [int(example["target_candidate"]) for example in examples],
                dtype=torch.long,
            )
            for _ in range(self.update_epochs):
                optimizer.zero_grad()
                logits = policy.model(features)
                loss = functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.item())
            policy.model.eval()

        rewards = [
            trajectory.reward
            for group in groups
            for trajectory in group.trajectories
        ]
        eval_metrics = _evaluate_sigil_policy(policy, self.scenarios)
        return TrainResult(
            policy=policy,
            checkpoint_id=f"sigil-step-{current.step + 1}",
            metrics={
                "train/reward": eval_metrics["sigil/eval_target_accuracy"],
                "train/batch_reward": fmean(rewards) if rewards else 0.0,
                "train/loss": loss_value,
                "train/examples": float(len(examples)),
                "train/dollar_seconds": 0.35 + 0.03 * len(examples),
                **eval_metrics,
            },
        )


async def sigil_ablation_rollout(
    policy: SigilAblationPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    task, task_index = _select_task_from_scenario(scenario, context)
    difficulty_bucket = sigil_difficulty_bucket(task)
    selected, prediction_text, logits = policy.render(task)
    passed = verify_sigil_code(prediction_text)
    actions = context.action_codec.encode(prediction_text)
    for action in actions:
        if isinstance(action.metadata, dict):
            action.metadata.setdefault("sigil/task_id", task.id)
            action.metadata.setdefault("sigil/difficulty_bucket", difficulty_bucket)
            action.metadata.setdefault("sigil/candidate_index", selected)
    reconstruction_safe = not any(
        action.metadata.get("reconstruction/safe") is False for action in actions
    )
    fallback = any(action.metadata.get("action/fallback") is True for action in actions)
    reconstruction_accuracy = min(
        (
            float(action.metadata.get("reconstruction/accuracy", 1.0))
            for action in actions
            if not isinstance(action.metadata.get("reconstruction/accuracy"), bool)
        ),
        default=1.0,
    )
    rollout_cost = _sigil_rollout_dollar_seconds(
        task=task,
        scenario=scenario,
        action_units=len(actions),
    )
    metadata = {
        **dict(context.decision_metadata),
        "scenario_id": scenario.id,
        "sigil/workload": "compiler_checked_idiom_codegen",
        "sigil/task_id": task.id,
        "sigil/difficulty_bucket": difficulty_bucket,
        "sigil/task_index_in_bucket": task_index,
        "sigil/source_id": task.source_id,
        "sigil/prompt": task.prompt,
        "sigil/prediction_text": prediction_text,
        "sigil/candidate_index": selected,
        "sigil/target_candidate": task.target_candidate,
        "sigil/candidate_count": len(task.candidates),
        "sigil/features": _sigil_features(task, feature_dim=policy.feature_dim),
        "sigil/logits": logits,
        "sigil/source_tokens": sum(action.token_count for action in actions),
        "action/safe": True,
        "action/fallback": fallback,
        "reconstruction/accuracy": reconstruction_accuracy,
        "reconstruction/safe": reconstruction_safe,
        "verifier/passed": passed,
    }
    if fallback:
        metadata["failure/mode"] = "learned_chunk_fallback"
    if not passed:
        metadata["verifier/failure_mode"] = "sigil_check_failed"
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=[Message(role="user", content=task.prompt)],
        actions=actions,
        reward=1.0 if passed else 0.0,
        metrics={"rollout/dollar_seconds": rollout_cost},
        metadata=metadata,
    )


async def run_sigil_workload_ablation(
    *,
    corpus: SigilCorpus | None = None,
    tasks: Sequence[SigilAblationTask] | None = None,
    learned_bundle: Any | None = None,
    max_train_steps: int = DEFAULT_SIGIL_TRAIN_STEPS,
    max_tasks: int | None = DEFAULT_SIGIL_MAX_TASKS,
    action_unit_dollar_seconds: float = DEFAULT_SIGIL_ACTION_UNIT_DOLLAR_SECONDS,
    encoder_config: Any | None = None,
) -> dict[str, Any]:
    if max_train_steps < 1:
        raise ValueError("sigil_train_steps_must_be_positive")
    corpus = corpus or load_sigil_corpus()
    task_list = tuple(tasks) if tasks is not None else build_sigil_ablation_tasks(corpus, max_tasks=max_tasks)
    if not task_list:
        raise ValueError("sigil_tasks_empty")
    if learned_bundle is None:
        from .sigil_encoder import train_sigil_chunk_encoder

        learned_bundle = train_sigil_chunk_encoder(
            corpus.training_outputs,
            config=encoder_config,
        )
    bucket_counts = sigil_bucket_counts(task_list)
    static = await _run_sigil_condition(
        tasks=task_list,
        condition="static_art",
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
        max_train_steps=max_train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    scheduler_only = await _run_sigil_condition(
        tasks=task_list,
        condition="scheduler_only",
        scheduler=_objective_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec()],
        max_train_steps=max_train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    full_trinity_codecs = _full_trinity_codecs(learned_bundle)
    full_trinity = await _run_sigil_condition(
        tasks=task_list,
        condition="full_trinity",
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_min_pulls=6,
            demotion_decision_min_observations=6,
            seed_codecs=full_trinity_codecs[2:],
        ),
        action_codecs=full_trinity_codecs,
        max_train_steps=max_train_steps,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    summaries = {
        "static_art": sigil_summary_metrics(
            static,
            tasks=task_list,
            action_codecs=[TokenActionCodec()],
            scheduler_enabled=False,
        ),
        "scheduler_only": sigil_summary_metrics(
            scheduler_only,
            tasks=task_list,
            action_codecs=[TokenActionCodec()],
            scheduler_enabled=True,
        ),
        "full_trinity": sigil_summary_metrics(
            full_trinity,
            tasks=task_list,
            action_codecs=full_trinity_codecs,
            scheduler_enabled=True,
        ),
    }
    return {
        "proof_scope": SIGIL_WORKLOAD_PROOF_SCOPE,
        "task_count": len(task_list),
        "task_buckets": bucket_counts,
        "corpus": {
            "prompt_count": corpus.prompt_count,
            "training_output_count": corpus.training_output_count,
        },
        "encoder": _encoder_summary(learned_bundle),
        "conditions": summaries,
        "comparison": _condition_comparison(summaries),
    }


def build_sigil_ablation_tasks(
    corpus: SigilCorpus,
    *,
    max_tasks: int | None = DEFAULT_SIGIL_MAX_TASKS,
) -> tuple[SigilAblationTask, ...]:
    tasks: list[SigilAblationTask] = []
    for row_index, row in enumerate(corpus.idiom_rows):
        intent = row.get("intent")
        output = row.get("output")
        if not isinstance(intent, str) or not intent.strip():
            continue
        if not isinstance(output, str) or not output.strip():
            continue
        source_id = str(row.get("id", f"idiom_{row_index}"))
        valid_code = _wrap_sigil_module(output, row_index)
        if not verify_sigil_code(valid_code):
            continue
        invalid_candidates = _invalid_sigil_candidates(valid_code)
        target_index = len(tasks) % 3
        candidates = list(invalid_candidates[:2])
        candidates.insert(target_index, valid_code)
        tasks.append(
            SigilAblationTask(
                id=f"sigil_{len(tasks):03d}",
                prompt=intent.strip(),
                candidates=tuple(candidates),
                rollout_dollar_seconds=DEFAULT_SIGIL_ROLLOUT_DOLLAR_SECONDS,
                target_candidate=target_index,
                source_id=source_id,
            )
        )
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    return tuple(tasks)


def sigil_difficulty_bucket(task: SigilAblationTask) -> str:
    token_count = len(task.target_code.split())
    if token_count < 30:
        return SIGIL_BUCKET_EASY
    if token_count <= 80:
        return SIGIL_BUCKET_MEDIUM
    return SIGIL_BUCKET_HARD


def sigil_bucket_counts(tasks: Sequence[SigilAblationTask]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in SIGIL_BUCKET_ORDER}
    for task in tasks:
        counts[sigil_difficulty_bucket(task)] += 1
    return {bucket: count for bucket, count in counts.items() if count > 0}


def sigil_summary_metrics(
    summary: RunSummary,
    *,
    tasks: Sequence[SigilAblationTask] = (),
    action_codecs: Sequence[ActionCodec] = (),
    scheduler_enabled: bool = True,
) -> dict[str, float]:
    keys = [
        SIGIL_NORTH_STAR,
        SIGIL_ACCOUNTED_NORTH_STAR,
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
        "sigil/eval_target_accuracy",
        "action_space/active_codecs",
        "action_space/promotions",
        "action_space/demotions",
        "action_space/decision_payoff_demotions",
        "action_space/source_token_throughput_payoff_demotions",
        "action_space/demotion_min_pulls",
        "action_space/demotion_decision_min_observations",
        "action_space/max_chunk_size",
    ]
    metrics = {key: float(summary.metrics[key]) for key in keys if key in summary.metrics}
    if SIGIL_NORTH_STAR in metrics:
        metrics["north_star"] = metrics[SIGIL_NORTH_STAR]
    if SIGIL_ACCOUNTED_NORTH_STAR in metrics:
        metrics["accounted_north_star"] = metrics[SIGIL_ACCOUNTED_NORTH_STAR]
    metrics.update(_condition_codec_metrics(summary.metrics))
    metrics.update(
        _condition_task_and_arm_metrics(
            summary.metrics,
            tasks=tasks,
            action_codecs=action_codecs,
            scheduler_enabled=scheduler_enabled,
        )
    )
    return metrics


async def _run_sigil_condition(
    *,
    tasks: Sequence[SigilAblationTask],
    condition: str,
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec],
    max_train_steps: int,
    action_unit_dollar_seconds: float,
) -> RunSummary:
    scenarios = _bucketed_sigil_scenarios(
        tasks,
        condition=condition,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
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
        scenarios=scenarios,
        initial_policy=SigilAblationPolicy(),
        trainer=SigilTrainer(scenarios=scenarios),
        workflow=sigil_ablation_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


def _full_trinity_codecs(learned_bundle: Any) -> list[ActionCodec]:
    from .chunk_encoder import LearnedChunkActionCodec

    return [
        TokenActionCodec(),
        ChunkActionCodec(chunk_size=2),
        LearnedChunkActionCodec(learned_bundle),
        ChunkActionCodec(chunk_size=3),
        ChunkActionCodec(chunk_size=4),
    ]


def _objective_scheduler() -> ObjectiveScheduler:
    return ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        min_actor_count=1,
        max_actor_count=2,
        min_rollout_coverage_fraction=0.1,
        max_rollout_coverage_cost_fraction=0.5,
        exploration_bonus=0.0,
    )


def _bucketed_sigil_scenarios(
    tasks: Sequence[SigilAblationTask],
    *,
    condition: str,
    action_unit_dollar_seconds: float,
) -> tuple[Scenario, ...]:
    buckets: dict[str, list[SigilAblationTask]] = {
        bucket: [] for bucket in SIGIL_BUCKET_ORDER
    }
    for task in tasks:
        buckets[sigil_difficulty_bucket(task)].append(task)
    scenarios: list[Scenario] = []
    for bucket in SIGIL_BUCKET_ORDER:
        bucket_tasks = tuple(buckets[bucket])
        if not bucket_tasks:
            continue
        scenarios.append(
            Scenario(
                id=bucket,
                payload={
                    "condition": condition,
                    "task": bucket_tasks[0],
                    "tasks": bucket_tasks,
                    "sigil/difficulty_bucket": bucket,
                    "sigil/task_count": len(bucket_tasks),
                    "action_unit_dollar_seconds": action_unit_dollar_seconds,
                },
            )
        )
    return tuple(scenarios)


def _select_task_from_scenario(
    scenario: Scenario,
    context: RolloutContext,
) -> tuple[SigilAblationTask, int]:
    tasks = _tasks_from_scenario(scenario)
    if len(tasks) == 1:
        return tasks[0], 0
    selector = (
        f"{scenario.id}|{context.policy_step}|{context.actor_id}|"
        f"{context.scheduler_arm_id or ''}"
    )
    digest = hashlib.sha256(selector.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(tasks)
    return tasks[index], index


def _task_from_scenario(scenario: Scenario) -> SigilAblationTask:
    return _tasks_from_scenario(scenario)[0]


def _tasks_from_scenario(scenario: Scenario) -> tuple[SigilAblationTask, ...]:
    tasks = scenario.payload.get("tasks")
    if isinstance(tasks, SequenceABC) and not isinstance(
        tasks, (str, bytes, bytearray)
    ):
        typed_tasks = tuple(task for task in tasks if isinstance(task, SigilAblationTask))
        if typed_tasks:
            return typed_tasks
    task = scenario.payload.get("task")
    if not isinstance(task, SigilAblationTask):
        raise ValueError("sigil_task_missing")
    return (task,)


def _sigil_rollout_dollar_seconds(
    *,
    task: SigilAblationTask,
    scenario: Scenario,
    action_units: int,
) -> float:
    action_unit_cost = float(scenario.payload.get("action_unit_dollar_seconds", 0.0))
    return task.rollout_dollar_seconds + action_unit_cost * max(1, action_units)


def _evaluate_sigil_policy(
    policy: SigilAblationPolicy,
    scenarios: Sequence[Scenario],
) -> dict[str, float]:
    if not scenarios:
        return {"sigil/eval_target_accuracy": 0.0}
    tasks = tuple(
        task
        for scenario in scenarios
        for task in _tasks_from_scenario(scenario)
    )
    correct = 0
    for task in tasks:
        selected, _, _ = policy.render(task)
        if selected == task.target_candidate:
            correct += 1
    if not tasks:
        return {
            "sigil/eval_target_accuracy": 0.0,
            "sigil/eval_tasks": 0.0,
        }
    return {
        "sigil/eval_target_accuracy": correct / len(tasks),
        "sigil/eval_tasks": float(len(tasks)),
    }


def _sigil_example_from_trajectory(trajectory: Trajectory) -> dict[str, Any]:
    features = trajectory.metadata.get("sigil/features")
    if not isinstance(features, tuple):
        features = tuple(features) if isinstance(features, list) else ()
    return {
        "features": tuple(float(value) for value in features),
        "target_candidate": int(trajectory.metadata.get("sigil/target_candidate", 0)),
    }


def _sigil_features(
    task: SigilAblationTask,
    *,
    feature_dim: int,
) -> tuple[float, ...]:
    text = f"{task.id} {task.prompt}"
    tokens = text.split()
    features = [
        min(1.0, len(tokens) / 80.0),
        min(1.0, len(text) / 1200.0),
        1.0 if "record" in text.lower() else 0.0,
        1.0 if "enum" in text.lower() else 0.0,
        1.0 if "function" in text.lower() or "fn" in text.lower() else 0.0,
        min(1.0, len(task.candidates) / 4.0),
    ]
    buckets = [0.0 for _ in range(max(0, feature_dim - len(features)))]
    for token in tokens:
        if not buckets:
            break
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        buckets[int.from_bytes(digest[:4], "big") % len(buckets)] += 1.0
    if buckets:
        scale = max(1.0, max(buckets))
        features.extend(value / scale for value in buckets)
    return tuple(features[:feature_dim] + [0.0] * max(0, feature_dim - len(features)))


def _wrap_sigil_module(snippet: str, index: int) -> str:
    return f"module demo;\n{snippet.strip()}\n"


def _invalid_sigil_candidates(valid_code: str) -> tuple[str, str]:
    bad_module = valid_code.replace("module ", "modul ", 1)
    if "return" in valid_code:
        bad_body = valid_code.replace("return", "retun", 1)
    elif "pub " in valid_code:
        bad_body = valid_code.replace("pub ", "pbu ", 1)
    else:
        bad_body = valid_code + "\nlet __sigil_broken = ;\n"
    return bad_module, bad_body


def _condition_codec_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    labels = {
        "token": TokenActionCodec(),
        "chunk2": ChunkActionCodec(chunk_size=2),
        "chunk3": ChunkActionCodec(chunk_size=3),
        "chunk4": ChunkActionCodec(chunk_size=4),
    }
    row: dict[str, float] = {}
    for label, codec in labels.items():
        key = safe_metric_key(action_codec_key(codec))
        row[f"{label}_pulls"] = _codec_metric_sum(metrics, key, "pulls")
        row[f"{label}_improvement_per_dollar"] = _codec_metric_weighted(
            metrics,
            key,
            "total_improvement_per_dollar_second",
        )
        row[f"{label}_mean_rollout_dollar_seconds"] = _codec_metric_weighted(
            metrics,
            key,
            "mean_rollout_dollar_seconds",
        )
    learned_keys = [
        key
        for key in metrics
        if key.startswith("scheduler/arm/")
        and _is_learned_scheduler_arm_metric(key)
        and key.endswith("/pulls")
    ]
    row["learned_pulls"] = sum(float(metrics[key]) for key in learned_keys)
    learned_unsafe = sum(
        float(value)
        for key, value in metrics.items()
        if key.startswith("scheduler/arm/")
        and _is_learned_scheduler_arm_metric(key)
        and key.endswith("/unsafe")
    )
    learned_fallbacks = sum(
        float(value)
        for key, value in metrics.items()
        if key.startswith("scheduler/arm/")
        and _is_learned_scheduler_arm_metric(key)
        and key.endswith("/failure/learned_chunk_fallback")
    )
    row["learned_unsafe"] = learned_unsafe
    row["learned_fallbacks"] = learned_fallbacks
    row["learned_fallback_rate"] = (
        learned_fallbacks / row["learned_pulls"]
        if row["learned_pulls"]
        else 0.0
    )
    row["learned_reconstruction_max_drift"] = max(
        (
            float(value)
            for key, value in metrics.items()
            if key.startswith("scheduler/arm/")
            and _is_learned_scheduler_arm_metric(key)
            and key.endswith("/reconstruction_max_drift")
        ),
        default=0.0,
    )
    return row


def _condition_task_and_arm_metrics(
    metrics: Mapping[str, float],
    *,
    tasks: Sequence[SigilAblationTask],
    action_codecs: Sequence[ActionCodec],
    scheduler_enabled: bool,
) -> dict[str, float]:
    bucket_counts = sigil_bucket_counts(tasks)
    row: dict[str, float] = {
        "sigil/distinct_task_count": float(len(tasks)),
        "sigil/distinct_bucket_count": float(len(bucket_counts)),
        "scheduler/distinct_rollout_arms": float(
            len(_observed_scheduler_arm_keys(metrics))
        ),
        "scheduler/expected_rollout_arms": float(
            len(bucket_counts) * len(action_codecs) if scheduler_enabled else 0
        ),
    }
    for bucket in SIGIL_BUCKET_ORDER:
        count = bucket_counts.get(bucket, 0)
        row[f"sigil/tasks_per_bucket/{bucket}"] = float(count)
        for codec in action_codecs:
            codec_label = _codec_summary_label(codec)
            row[f"sigil/bucket/{bucket}/codec/{codec_label}/pulls"] = (
                _scheduler_arm_metric(
                    metrics,
                    arm_id=f"{bucket}|{_scheduler_codec_key(codec)}",
                    metric_name="pulls",
                )
            )
    return row


def _observed_scheduler_arm_keys(metrics: Mapping[str, float]) -> set[str]:
    prefix = "scheduler/arm/"
    suffix = "/pulls"
    return {
        key[len(prefix) : -len(suffix)]
        for key in metrics
        if key.startswith(prefix) and key.endswith(suffix)
    }


def _scheduler_arm_metric(
    metrics: Mapping[str, float],
    *,
    arm_id: str,
    metric_name: str,
) -> float:
    key = f"scheduler/arm/{safe_metric_key(arm_id)}/{metric_name}"
    return float(metrics.get(key, 0.0))


def _scheduler_codec_key(codec: ActionCodec) -> str:
    name = getattr(codec, "name", codec.__class__.__name__)
    values = getattr(codec, "__dict__", {})
    public_values = {
        key: value
        for key, value in values.items()
        if not key.startswith("_") and key != "name"
    }
    if not public_values:
        return str(name)
    suffix = ",".join(f"{key}={public_values[key]}" for key in sorted(public_values))
    return f"{name}({suffix})"


def _codec_summary_label(codec: ActionCodec) -> str:
    if isinstance(codec, TokenActionCodec):
        return "token"
    if isinstance(codec, ChunkActionCodec):
        return f"chunk{codec.chunk_size}"
    key = action_codec_key(codec)
    if key.startswith("learned_chunk"):
        return "learned"
    return safe_metric_key(key)


def _is_learned_scheduler_arm_metric(key: str) -> bool:
    arm_part = key[len("scheduler/arm/") :].split("/", 1)[0]
    return arm_part.endswith("learned_chunk") or "_learned_chunk_" in arm_part


def _codec_metric_sum(
    metrics: Mapping[str, float],
    codec_key: str,
    metric_name: str,
) -> float:
    return sum(
        float(value)
        for key, value in metrics.items()
        if key.startswith("scheduler/arm/")
        and f"_{codec_key}/" in key
        and key.endswith(f"/{metric_name}")
    )


def _codec_metric_weighted(
    metrics: Mapping[str, float],
    codec_key: str,
    metric_name: str,
) -> float:
    weighted = 0.0
    pulls = 0.0
    for key, value in metrics.items():
        if not (
            key.startswith("scheduler/arm/")
            and f"_{codec_key}/" in key
            and key.endswith(f"/{metric_name}")
        ):
            continue
        prefix = key[: -len(metric_name)]
        arm_pulls = float(metrics.get(prefix + "pulls", 0.0))
        weighted += arm_pulls * float(value)
        pulls += arm_pulls
    return weighted / pulls if pulls else 0.0


def _encoder_summary(bundle: Any) -> dict[str, Any]:
    report = bundle.training_report
    return {
        "proof_scope": report.proof_scope,
        "train_examples": report.train_examples,
        "train_reconstruction_accuracy": report.train_reconstruction_accuracy,
        "holdout_reconstruction_accuracy": report.holdout_reconstruction_accuracy,
        "nll_improvement": report.nll_improvement,
        "chunk_size": bundle.config.chunk_size,
        "latent_dim": bundle.config.latent_dim,
        "vocab_hash": bundle.vocabulary.hash,
        "vocab_size": report.vocab_size,
    }


def _condition_comparison(
    summaries: Mapping[str, Mapping[str, float]],
) -> dict[str, float | None]:
    static = float(summaries["static_art"].get(SIGIL_ACCOUNTED_NORTH_STAR, 0.0))
    scheduler = float(summaries["scheduler_only"].get(SIGIL_ACCOUNTED_NORTH_STAR, 0.0))
    full = float(summaries["full_trinity"].get(SIGIL_ACCOUNTED_NORTH_STAR, 0.0))
    return {
        "scheduler_vs_static_accounted_north_star_ratio": (
            scheduler / static if static > 0.0 else None
        ),
        "full_trinity_vs_static_accounted_north_star_ratio": (
            full / static if static > 0.0 else None
        ),
        "full_trinity_vs_scheduler_accounted_north_star_ratio": (
            full / scheduler if scheduler > 0.0 else None
        ),
        "full_trinity_accounted_north_star_delta_vs_scheduler": full - scheduler,
    }


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError(
            "Sigil ablations require torch. Install with "
            '`pip install -e ".[calm]"`.'
        ) from exc
    return torch, nn, functional


__all__ = [
    "DEFAULT_SIGIL_ACTION_UNIT_DOLLAR_SECONDS",
    "DEFAULT_SIGIL_MAX_TASKS",
    "DEFAULT_SIGIL_TRAIN_STEPS",
    "SIGIL_BUCKET_EASY",
    "SIGIL_BUCKET_HARD",
    "SIGIL_BUCKET_MEDIUM",
    "SIGIL_BUCKET_ORDER",
    "SIGIL_ACCOUNTED_NORTH_STAR",
    "SIGIL_NORTH_STAR",
    "SIGIL_WORKLOAD_PROOF_SCOPE",
    "SigilAblationPolicy",
    "SigilAblationTask",
    "SigilTrainer",
    "build_sigil_ablation_tasks",
    "run_sigil_workload_ablation",
    "sigil_ablation_rollout",
    "sigil_bucket_counts",
    "sigil_difficulty_bucket",
    "sigil_summary_metrics",
]
