from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import fmean
import time
from typing import Any, Mapping, Sequence

from .actions import (
    ActionCodec,
    ActionUnit,
    AdaptiveActionSpace,
    ChunkActionCodec,
    TokenActionCodec,
)
from .art_adapter import AsyncArtBackend, AsyncArtBackendConfig
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


NORTH_STAR = (
    "north_star/published_policy_reward_improving_experience_per_dollar_second"
)
ACCOUNTED_NORTH_STAR = (
    "north_star/accounted_published_policy_reward_improving_experience_per_dollar_second"
)
ART_NORTH_STAR = (
    "art_backend/published_policy_reward_improving_experience_per_dollar_second"
)
ART_ACCOUNTED_NORTH_STAR = (
    "art_backend/accounted_published_policy_reward_improving_experience_per_dollar_second"
)
BENCHMARK_NORTH_STAR = (
    "benchmark/published_policy_reward_improving_experience_per_dollar_second"
)
BENCHMARK_ACCOUNTED_NORTH_STAR = (
    "benchmark/accounted_published_policy_reward_improving_experience_per_dollar_second"
)

RUNTIME_CONTROL_CONTEXT_SUMMARY_KEYS = [
    "scheduler/control_context/keys",
    "scheduler/control_context/decisions",
    "scheduler/control_context/rollout_updates",
    "scheduler/control_context/train_updates",
    "scheduler/control_context/stale_updates",
    "scheduler/control_context/feedback_updates",
    "scheduler/control_context/total_objective",
    "scheduler/control_context/mean_objective_per_decision",
    "scheduler/control_context/mean_objective_per_feedback_update",
]


@dataclass(frozen=True)
class AblationPolicy:
    async def act(
        self,
        messages: Sequence[Message],
        *,
        scenario: Scenario,
        codec: ActionCodec,
    ):
        return codec.encode(f"{scenario.id} {codec.name}")


class MeanRewardTrainer:
    async def train(
        self,
        current: PolicySnapshot,
        groups: Sequence[TrajectoryGroup],
    ) -> TrainResult:
        rewards = [
            trajectory.reward
            for group in groups
            for trajectory in group.trajectories
        ]
        return TrainResult(
            policy=current.policy,
            checkpoint_id=f"ablation-step-{current.step + 1}",
            metrics={
                "train/reward": fmean(rewards),
                "train/dollar_seconds": 1.0,
            },
        )


@dataclass(frozen=True)
class RealAblationTask:
    scenario_id: str
    prompt: str
    features: tuple[float, ...]
    label: int
    answer: str
    rollout_dollar_seconds: float


class RealAblationPolicy:
    """Tiny torch policy for verifiable math ablations.

    This is intentionally not a language-model integration. It is a fast,
    dependency-optional model path that lets the scheduler face real inference,
    verifier reward, and supervised weight updates before ART/vLLM plumbing.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 10,
        answers: int = 4,
        hidden_dim: int = 16,
        seed: int = 1337,
        model: Any | None = None,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.feature_dim = feature_dim
        self.answers = answers
        self.hidden_dim = hidden_dim
        self.seed = seed
        if model is None:
            torch.manual_seed(seed)
            self.model = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, answers),
            )
        else:
            self.model = model
        self.model.eval()

    def clone(self) -> "RealAblationPolicy":
        import copy

        return RealAblationPolicy(
            feature_dim=self.feature_dim,
            answers=self.answers,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
            model=copy.deepcopy(self.model),
        )

    def predict(self, task: RealAblationTask) -> tuple[int, str, tuple[float, ...]]:
        torch, _, _ = _import_torch()
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor([task.features], dtype=torch.float32)
            logits = self.model(features).squeeze(0)
        prediction = int(torch.argmax(logits).item())
        return prediction, str(prediction), tuple(float(value) for value in logits)


class RealTrainer:
    def __init__(
        self,
        *,
        scenarios: Sequence[Scenario],
        learning_rate: float = 0.2,
        update_epochs: int = 8,
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
        policy = current.policy.clone()
        examples = [
            _real_example_from_trajectory(trajectory)
            for group in groups
            for trajectory in group.trajectories
        ]
        if examples:
            policy.model.train()
            optimizer = torch.optim.SGD(policy.model.parameters(), lr=self.learning_rate)
            features = torch.tensor(
                [example.features for example in examples],
                dtype=torch.float32,
            )
            labels = torch.tensor([example.label for example in examples], dtype=torch.long)
            loss_value = 0.0
            for _ in range(self.update_epochs):
                optimizer.zero_grad()
                logits = policy.model(features)
                loss = functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.item())
            policy.model.eval()
        else:
            loss_value = 0.0

        eval_metrics = _evaluate_real_policy(policy, self.scenarios)
        rewards = [
            trajectory.reward
            for group in groups
            for trajectory in group.trajectories
        ]
        return TrainResult(
            policy=policy,
            checkpoint_id=f"real-ablation-step-{current.step + 1}",
            metrics={
                "train/reward": eval_metrics["real/eval_weighted_accuracy"],
                "train/batch_reward": fmean(rewards) if rewards else 0.0,
                "train/loss": loss_value,
                "train/examples": float(len(examples)),
                "train/dollar_seconds": 0.2 + 0.05 * len(examples),
                **eval_metrics,
            },
        )


@dataclass(frozen=True)
class _AblationArtMessage:
    role: str
    content: str


@dataclass(frozen=True)
class _AblationArtChoice:
    message: _AblationArtMessage


@dataclass
class _AblationArtTrajectory:
    messages_and_choices: list[Any]
    reward: float
    initial_policy_version: int
    final_policy_version: int
    metrics: dict[str, float]
    metadata: dict[str, Any]


@dataclass
class _AblationArtGroup:
    trajectories: list[_AblationArtTrajectory]
    metadata: dict[str, Any]
    metrics: dict[str, float] | None = None

    def __iter__(self):
        return iter(self.trajectories)


@dataclass(frozen=True)
class _AblationArtTrainResult:
    step: int
    metrics: dict[str, float]
    checkpoint_path: str


class _AblationArtBackend:
    def __init__(self) -> None:
        self.step = 0
        self.calls = 0

    async def register(self, model: Any) -> None:
        return None

    async def _get_step(self, model: Any) -> int:
        return self.step

    async def train(
        self,
        model: Any,
        trajectory_groups: Sequence[_AblationArtGroup],
        **kwargs: Any,
    ) -> _AblationArtTrainResult:
        self.step += 1
        self.calls += 1
        rewards = [
            trajectory.reward
            for group in trajectory_groups
            for trajectory in group.trajectories
        ]
        reward = fmean(rewards) if rewards else 0.0
        return _AblationArtTrainResult(
            step=self.step,
            metrics={
                "train/reward": reward,
                "train/dollar_seconds": 1.0,
            },
            checkpoint_path=f".art/ablation/model/step_{self.step}",
        )


async def ablation_rollout(
    policy: AblationPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content="pick useful action bandwidth")]
    actions = await policy.act(messages, scenario=scenario, codec=context.action_codec)
    reward = float(scenario.payload.get(context.action_codec.name, 0.0))
    rollout_cost = float(
        scenario.payload.get(f"{context.action_codec.name}_dollar_seconds", 1.0)
    )
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
        metrics={"rollout/dollar_seconds": rollout_cost},
    )


async def action_space_ablation_rollout(
    policy: AblationPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    messages = [Message(role="user", content="pick useful semantic bandwidth")]
    actions = context.action_codec.encode(
        "alpha beta gamma delta epsilon zeta eta theta"
    )
    if isinstance(context.action_codec, ChunkActionCodec):
        codec_key = f"chunk_{context.action_codec.chunk_size}"
    else:
        codec_key = context.action_codec.name
    reward = float(scenario.payload.get(codec_key, 0.0))
    rollout_cost = float(scenario.payload.get(f"{codec_key}_dollar_seconds", 1.0))
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=messages,
        actions=actions,
        reward=reward,
        metrics={"rollout/dollar_seconds": rollout_cost},
    )


async def real_ablation_rollout(
    policy: RealAblationPolicy,
    scenario: Scenario,
    context: RolloutContext,
) -> Trajectory:
    task = _real_task_for_context(scenario, context)
    prediction, answer, logits = policy.predict(task)
    reward = 1.0 if prediction == task.label else 0.0
    actions = [
        ActionUnit(
            kind="token",
            payload=answer,
            token_count=1,
            text=answer,
            metadata={
                "real/prediction": prediction,
                "real/target": task.label,
                "real/correct": prediction == task.label,
            },
        )
    ]
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=context.policy_step,
        messages=[Message(role="user", content=task.prompt)],
        actions=actions,
        reward=reward,
        metrics={"rollout/dollar_seconds": task.rollout_dollar_seconds},
        metadata={
            **dict(context.decision_metadata),
            "scenario_id": scenario.id,
            "real/workload": "verifiable_math",
            "real/prompt": task.prompt,
            "real/features": task.features,
            "real/label": task.label,
            "real/answer": task.answer,
            "real/prediction": prediction,
            "real/logits": logits,
            "verifier/passed": prediction == task.label,
        },
    )


async def run_static_ablation() -> RunSummary:
    return await _run(scheduler=None)


async def run_objective_ablation() -> RunSummary:
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=2,
        min_actor_count=1,
        max_actor_count=2,
        exploration_bonus=0.0,
    )
    return await _run(scheduler=scheduler)


async def run_fixed_action_space_ablation() -> RunSummary:
    return await _run_action_space(
        scheduler=_objective_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
    )


async def run_adaptive_action_space_ablation() -> RunSummary:
    return await _run_action_space(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )


async def run_static_closed_loop_ablation() -> RunSummary:
    return await _run_closed_loop(
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
    )


async def run_objective_closed_loop_ablation() -> RunSummary:
    return await _run_closed_loop(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )


async def run_static_art_bridge_ablation() -> dict[str, float]:
    return await _run_art_bridge(
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
    )


async def run_objective_art_bridge_ablation() -> dict[str, float]:
    return await _run_art_bridge(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )


async def run_ablation() -> dict[str, Any]:
    static = await run_static_ablation()
    objective = await run_objective_ablation()
    static_score = float(static.metrics[NORTH_STAR])
    objective_score = float(objective.metrics[NORTH_STAR])
    accounted_static_score = float(static.metrics[ACCOUNTED_NORTH_STAR])
    accounted_objective_score = float(objective.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "north_star_absolute": objective_score - static_score,
            "north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
            "accounted_north_star_absolute": (
                accounted_objective_score - accounted_static_score
            ),
            "accounted_north_star_ratio": (
                accounted_objective_score / accounted_static_score
                if accounted_static_score > 0.0
                else None
            ),
        },
    }


async def run_action_space_ablation() -> dict[str, Any]:
    fixed = await run_fixed_action_space_ablation()
    adaptive = await run_adaptive_action_space_ablation()
    fixed_score = float(fixed.metrics[ACCOUNTED_NORTH_STAR])
    adaptive_score = float(adaptive.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "fixed": summary_metrics(fixed),
        "adaptive": summary_metrics(adaptive),
        "lift": {
            "accounted_north_star_absolute": adaptive_score - fixed_score,
            "accounted_north_star_ratio": (
                adaptive_score / fixed_score if fixed_score > 0.0 else None
            ),
        },
    }


async def run_closed_loop_ablation() -> dict[str, Any]:
    static = await run_static_closed_loop_ablation()
    objective = await run_objective_closed_loop_ablation()
    static_score = float(static.metrics[ACCOUNTED_NORTH_STAR])
    objective_score = float(objective.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "accounted_north_star_absolute": objective_score - static_score,
            "accounted_north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def run_real_ablation() -> dict[str, Any]:
    static = await _run(scheduler=None, real_workload=True)
    objective = await _run(scheduler=_objective_scheduler(), real_workload=True)
    static_score = float(static.metrics[ACCOUNTED_NORTH_STAR])
    objective_score = float(objective.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "accounted_north_star_absolute": objective_score - static_score,
            "accounted_north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def run_real_closed_loop_ablation() -> dict[str, Any]:
    static = await _run_closed_loop(
        scheduler=None,
        action_space=None,
        action_codecs=[TokenActionCodec()],
        real_workload=True,
    )
    objective = await _run_closed_loop(
        scheduler=_objective_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec()],
        real_workload=True,
    )
    static_score = float(static.metrics[ACCOUNTED_NORTH_STAR])
    objective_score = float(objective.metrics[ACCOUNTED_NORTH_STAR])
    return {
        "static": summary_metrics(static),
        "objective": summary_metrics(objective),
        "lift": {
            "accounted_north_star_absolute": objective_score - static_score,
            "accounted_north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def run_control_dimension_ablation() -> dict[str, Any]:
    full = await _run_closed_loop(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
    )
    fixed_policy_lag = await _run_closed_loop(
        scheduler=_objective_scheduler(max_policy_lag=1),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
        max_policy_lag=1,
    )
    shallow_queue = await _run_closed_loop(
        scheduler=_objective_scheduler(),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
        train_queue_capacity=1,
    )
    single_actor = await _run_closed_loop(
        scheduler=_objective_scheduler(max_actor_count=1),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        action_codecs=None,
        num_actors=1,
    )
    token_action_codec = await _run_closed_loop(
        scheduler=_objective_scheduler(),
        action_space=None,
        action_codecs=[TokenActionCodec()],
    )
    full_score = float(full.metrics[ACCOUNTED_NORTH_STAR])
    variants = {
        "fixed_policy_lag": _control_dimension_summary(
            fixed_policy_lag,
            num_actors=2,
            train_queue_capacity=2,
            max_policy_lag=1,
            adaptive_action_space=True,
        ),
        "shallow_queue": _control_dimension_summary(
            shallow_queue,
            num_actors=2,
            train_queue_capacity=1,
            max_policy_lag=2,
            adaptive_action_space=True,
        ),
        "single_actor": _control_dimension_summary(
            single_actor,
            num_actors=1,
            train_queue_capacity=2,
            max_policy_lag=2,
            adaptive_action_space=True,
        ),
        "token_action_codec": _control_dimension_summary(
            token_action_codec,
            num_actors=2,
            train_queue_capacity=2,
            max_policy_lag=2,
            adaptive_action_space=False,
        ),
    }
    return {
        "full": _control_dimension_summary(
            full,
            num_actors=2,
            train_queue_capacity=2,
            max_policy_lag=2,
            adaptive_action_space=True,
        ),
        **variants,
        "lift": {
            f"full_vs_{name}_accounted_north_star_absolute": (
                full_score - float(metrics[ACCOUNTED_NORTH_STAR])
            )
            for name, metrics in variants.items()
        }
        | {
            f"full_vs_{name}_accounted_north_star_ratio": (
                full_score / float(metrics[ACCOUNTED_NORTH_STAR])
                if float(metrics[ACCOUNTED_NORTH_STAR]) > 0.0
                else None
            )
            for name, metrics in variants.items()
        },
    }


async def run_art_bridge_ablation() -> dict[str, Any]:
    static = await run_static_art_bridge_ablation()
    objective = await run_objective_art_bridge_ablation()
    static_score = float(static[ART_ACCOUNTED_NORTH_STAR])
    objective_score = float(objective[ART_ACCOUNTED_NORTH_STAR])
    return {
        "static": static,
        "objective": objective,
        "lift": {
            "accounted_north_star_absolute": objective_score - static_score,
            "accounted_north_star_ratio": (
                objective_score / static_score if static_score > 0.0 else None
            ),
        },
    }


async def run_stock_art_benchmark() -> dict[str, float]:
    backend = _AblationArtBackend()
    await backend.register("art-model")
    sample_dollar_seconds = 0.0
    trainer_dollar_seconds = 0.0
    action_units = 0
    source_tokens = 0
    published_updates = 0
    baseline_score = 0.0
    reward_improving_experience = 0.0
    started_at = time.perf_counter()
    for _ in range(8):
        actions = _bridge_actions_for_codec(
            "token",
            "alpha beta gamma delta epsilon zeta eta theta",
        )
        group = _art_bridge_group_from_assignment(
            {
                "scheduler/scenario_id": "stock_art",
                "scheduler/action_codec": "token",
                "scheduler/policy_step": backend.step,
            }
        )
        sample_dollar_seconds += sum(
            float(trajectory.metrics.get("rollout/dollar_seconds", 0.0))
            for trajectory in group.trajectories
        )
        action_units += len(actions)
        source_tokens += sum(action.token_count for action in actions)
        result = await backend.train("art-model", [group])
        trainer_dollar_seconds += float(
            result.metrics.get("train/dollar_seconds", 0.0)
        )
        score = float(result.metrics.get("train/reward", 0.0))
        improvement = max(0.0, score - baseline_score)
        if improvement > 0.0:
            reward_improving_experience += improvement * len(group.trajectories)
            baseline_score = score
        published_updates += 1

    wall_s = max(time.perf_counter() - started_at, 1e-9)
    accounted_dollar_seconds = sample_dollar_seconds + trainer_dollar_seconds
    return {
        BENCHMARK_NORTH_STAR: reward_improving_experience / wall_s,
        BENCHMARK_ACCOUNTED_NORTH_STAR: (
            reward_improving_experience / accounted_dollar_seconds
            if accounted_dollar_seconds > 0.0
            else 0.0
        ),
        "benchmark/wall_clock_s": wall_s,
        "benchmark/accounted_dollar_seconds": accounted_dollar_seconds,
        "benchmark/sample_dollar_seconds": sample_dollar_seconds,
        "benchmark/trainer_dollar_seconds": trainer_dollar_seconds,
        "benchmark/submitted_groups": 8.0,
        "benchmark/completed_batches": float(published_updates),
        "benchmark/published_policy_updates": float(published_updates),
        "benchmark/published_policy_reward_improving_experience": (
            reward_improving_experience
        ),
        "benchmark/groups_per_s": 8.0 / wall_s,
        "benchmark/action_units": float(action_units),
        "benchmark/source_tokens": float(source_tokens),
        "benchmark/action_units_per_s": action_units / wall_s,
        "benchmark/source_tokens_per_s": source_tokens / wall_s,
        "actions/semantic_bandwidth_tokens_per_decision": (
            source_tokens / action_units if action_units else 0.0
        ),
    }


async def run_async_art_runtime_benchmark() -> dict[str, float]:
    return _benchmark_metrics_from_bridge(
        await _run_art_bridge(
            scheduler=_objective_scheduler(),
            action_space=None,
            action_codecs=[TokenActionCodec()],
        )
    )


async def run_async_semantic_art_benchmark() -> dict[str, float]:
    return _benchmark_metrics_from_bridge(
        await _run_art_bridge(
            scheduler=_objective_scheduler(),
            action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
            action_codecs=None,
        )
    )


async def run_art_runtime_benchmark() -> dict[str, Any]:
    stock = await run_stock_art_benchmark()
    async_runtime = await run_async_art_runtime_benchmark()
    async_semantic = await run_async_semantic_art_benchmark()
    stock_score = stock[BENCHMARK_ACCOUNTED_NORTH_STAR]
    async_score = async_runtime[BENCHMARK_ACCOUNTED_NORTH_STAR]
    semantic_score = async_semantic[BENCHMARK_ACCOUNTED_NORTH_STAR]
    return {
        "stock_art": stock,
        "art_async": async_runtime,
        "art_async_semantic": async_semantic,
        "lift": {
            "async_vs_stock_accounted_north_star_absolute": (
                async_score - stock_score
            ),
            "async_vs_stock_accounted_north_star_ratio": (
                async_score / stock_score if stock_score > 0.0 else None
            ),
            "async_semantic_vs_async_accounted_north_star_absolute": (
                semantic_score - async_score
            ),
            "async_semantic_vs_async_accounted_north_star_ratio": (
                semantic_score / async_score if async_score > 0.0 else None
            ),
            "async_semantic_vs_stock_accounted_north_star_absolute": (
                semantic_score - stock_score
            ),
            "async_semantic_vs_stock_accounted_north_star_ratio": (
                semantic_score / stock_score if stock_score > 0.0 else None
            ),
        },
    }


async def _run(
    scheduler: ObjectiveScheduler | None,
    *,
    real_workload: bool = False,
) -> RunSummary:
    if real_workload:
        return await _run_real_workload(scheduler=scheduler)
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=2,
            group_size=1,
            train_batch_groups=2,
            max_train_steps=8,
            queue_max_trajectories=4,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=[
            Scenario(
                id="bandwidth",
                payload={
                    "token": 0.1,
                    "chunk": 1.0,
                    "token_dollar_seconds": 1.0,
                    "chunk_dollar_seconds": 1.5,
                },
            )
        ],
        initial_policy=AblationPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=ablation_rollout,
        action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=2)],
        scheduler=scheduler,
    )


async def _run_action_space(
    *,
    scheduler: ObjectiveScheduler,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec] | None,
) -> RunSummary:
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=1,
            group_size=1,
            train_batch_groups=1,
            max_train_steps=8,
            queue_max_trajectories=4,
            train_queue_capacity=2,
            max_policy_lag=2,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=[
            Scenario(
                id="semantic",
                payload={
                    "token": 0.1,
                    "chunk_2": 1.0,
                    "chunk_4": 4.0,
                    "token_dollar_seconds": 1.0,
                    "chunk_2_dollar_seconds": 1.0,
                    "chunk_4_dollar_seconds": 1.0,
                },
            )
        ],
        initial_policy=AblationPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=action_space_ablation_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


async def _run_closed_loop(
    *,
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec] | None,
    num_actors: int = 2,
    train_queue_capacity: int = 2,
    max_policy_lag: int = 2,
    real_workload: bool = False,
) -> RunSummary:
    if real_workload:
        return await _run_real_workload(
            scheduler=scheduler,
            num_actors=num_actors,
            train_queue_capacity=train_queue_capacity,
            max_policy_lag=max_policy_lag,
        )
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=num_actors,
            group_size=1,
            train_batch_groups=1,
            max_train_steps=8,
            queue_max_trajectories=4,
            train_queue_capacity=train_queue_capacity,
            max_policy_lag=max_policy_lag,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=[
            Scenario(
                id="closed_loop",
                payload={
                    "token": 0.1,
                    "chunk_2": 1.0,
                    "chunk_4": 4.0,
                    "token_dollar_seconds": 1.0,
                    "chunk_2_dollar_seconds": 1.0,
                    "chunk_4_dollar_seconds": 1.0,
                },
            )
        ],
        initial_policy=AblationPolicy(),
        trainer=MeanRewardTrainer(),
        workflow=action_space_ablation_rollout,
        action_codecs=action_codecs,
        action_space=action_space,
        scheduler=scheduler,
    )


async def _run_real_workload(
    *,
    scheduler: ObjectiveScheduler | None,
    num_actors: int = 2,
    train_queue_capacity: int = 2,
    max_policy_lag: int = 2,
) -> RunSummary:
    scenarios = _real_workload_scenarios()
    runtime = ControlPlane(
        ControlPlaneConfig(
            num_actors=num_actors,
            group_size=1,
            train_batch_groups=2,
            max_train_steps=10,
            queue_max_trajectories=8,
            train_queue_capacity=train_queue_capacity,
            max_policy_lag=max_policy_lag,
            cost_per_second_usd=1.0,
        )
    )
    return await runtime.run(
        scenarios=scenarios,
        initial_policy=RealAblationPolicy(),
        trainer=RealTrainer(scenarios=scenarios),
        workflow=real_ablation_rollout,
        action_codecs=[TokenActionCodec()],
        scheduler=scheduler,
    )


async def _run_art_bridge(
    *,
    scheduler: ObjectiveScheduler | None,
    action_space: AdaptiveActionSpace | None,
    action_codecs: Sequence[ActionCodec] | None,
) -> dict[str, float]:
    backend = AsyncArtBackend(
        backend=_AblationArtBackend(),
        config=AsyncArtBackendConfig(
            train_queue_capacity=2,
            train_batch_groups=1,
            max_policy_lag=2,
            max_train_steps=8,
            cost_per_second_usd=1.0,
        ),
        scheduler=scheduler,
        action_space=action_space,
    )
    scenarios = [
        Scenario(
            id="art_bridge",
            payload={
                "token": 0.1,
                "chunk_2": 1.0,
                "chunk_4": 4.0,
                "token_dollar_seconds": 1.0,
                "chunk_2_dollar_seconds": 1.0,
                "chunk_4_dollar_seconds": 1.0,
            },
        )
    ]
    futures = []
    await backend.register("art-model")
    rollout_limit = 8 if scheduler is None else 64
    try:
        for actor_id in range(rollout_limit):
            assignment = await backend.admit_and_select_rollout(
                scenarios=scenarios,
                action_codecs=action_codecs,
                actor_id=actor_id % 2,
                configured_actor_count=2,
                trajectory_queue_pressure=backend.ring.pending_batches
                / backend.ring.capacity,
            )
            if not assignment.admitted or assignment.decision is None:
                break
            group = _art_bridge_group_from_assignment(assignment.metadata)
            futures.append(await backend.submit_group("art-model", group))
            await asyncio.sleep(0)
        await backend.flush_pending_groups()
        if futures:
            await asyncio.gather(*futures)
        return bridge_summary_metrics(backend.stats())
    finally:
        await backend.close()


def _art_bridge_group_from_assignment(
    metadata: Mapping[str, Any],
) -> _AblationArtGroup:
    scenario_id = str(metadata.get("scheduler/scenario_id", "art_bridge"))
    text = "alpha beta gamma delta epsilon zeta eta theta"
    codec_key = str(metadata.get("scheduler/action_codec", "token"))
    actions = _bridge_actions_for_codec(codec_key, text)
    reward_key = _payload_key_for_codec(codec_key)
    payload = {
        "token": 0.1,
        "chunk_2": 1.0,
        "chunk_4": 4.0,
        "token_dollar_seconds": 1.0,
        "chunk_2_dollar_seconds": 1.0,
        "chunk_4_dollar_seconds": 1.0,
    }
    reward = float(payload.get(reward_key, 0.0))
    rollout_cost = float(payload.get(f"{reward_key}_dollar_seconds", 1.0))
    policy_step = int(float(metadata.get("scheduler/policy_step", 0)))
    trajectory_metadata = {
        **metadata,
        "scenario_id": scenario_id,
    }
    trajectory = _AblationArtTrajectory(
        messages_and_choices=[
            _AblationArtMessage(role="user", content="pick useful ART action"),
            *[
                _AblationArtChoice(
                    _AblationArtMessage(role="assistant", content=action.text)
                )
                for action in actions
            ],
        ],
        reward=reward,
        initial_policy_version=policy_step,
        final_policy_version=policy_step,
        metrics={"rollout/dollar_seconds": rollout_cost},
        metadata=trajectory_metadata,
    )
    return _AblationArtGroup(
        trajectories=[trajectory],
        metadata={"scenario_id": scenario_id},
        metrics={},
    )


def _bridge_actions_for_codec(codec_key: str, text: str) -> tuple[ActionUnit, ...]:
    if codec_key == "chunk(chunk_size=4)":
        return tuple(ChunkActionCodec(chunk_size=4).encode(text))
    if codec_key == "chunk(chunk_size=2)":
        return tuple(ChunkActionCodec(chunk_size=2).encode(text))
    return tuple(TokenActionCodec().encode(text))


def _payload_key_for_codec(codec_key: str) -> str:
    if codec_key == "chunk(chunk_size=4)":
        return "chunk_4"
    if codec_key == "chunk(chunk_size=2)":
        return "chunk_2"
    return "token"


def _real_workload_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            id="easy_math",
            payload={
                "rollout_dollar_seconds": 0.2,
                "eval_weight": 0.9,
                "task_offset": 0,
            },
        ),
        Scenario(
            id="hard_math",
            payload={
                "rollout_dollar_seconds": 4.0,
                "eval_weight": 0.1,
                "task_offset": 11,
            },
        ),
    )


def _real_task_for_context(
    scenario: Scenario,
    context: RolloutContext,
) -> RealAblationTask:
    offset = int(scenario.payload.get("task_offset", 0))
    index = context.policy_step * 7 + context.actor_id * 3 + offset
    return _real_task(scenario, index)


def _real_task(scenario: Scenario, index: int) -> RealAblationTask:
    x = index % 4
    y = (index // 2) % 4
    if scenario.id == "easy_math":
        label = (x + y) % 2
        prompt = f"What is ({x} + {y}) mod 2?"
    elif scenario.id == "hard_math":
        label = (x + 2 * y + 1) % 4
        prompt = f"What is ({x} + 2*{y} + 1) mod 4?"
    else:
        raise ValueError(f"unknown real ablation scenario: {scenario.id}")
    return RealAblationTask(
        scenario_id=scenario.id,
        prompt=prompt,
        features=_real_features(scenario.id, x, y),
        label=label,
        answer=str(label),
        rollout_dollar_seconds=float(scenario.payload["rollout_dollar_seconds"]),
    )


def _real_features(scenario_id: str, x: int, y: int) -> tuple[float, ...]:
    scenario_bits = (1.0, 0.0) if scenario_id == "easy_math" else (0.0, 1.0)
    x_bits = tuple(1.0 if index == x else 0.0 for index in range(4))
    y_bits = tuple(1.0 if index == y else 0.0 for index in range(4))
    return scenario_bits + x_bits + y_bits


def _real_example_from_trajectory(trajectory: Trajectory) -> RealAblationTask:
    metadata = trajectory.metadata
    features = metadata.get("real/features")
    label = metadata.get("real/label")
    if (
        not isinstance(features, (list, tuple))
        or not all(isinstance(value, (int, float)) for value in features)
        or not isinstance(label, int)
    ):
        raise ValueError("real_ablation_training_example_missing")
    return RealAblationTask(
        scenario_id=trajectory.scenario_id,
        prompt=str(metadata.get("real/prompt", "")),
        features=tuple(float(value) for value in features),
        label=label,
        answer=str(metadata.get("real/answer", label)),
        rollout_dollar_seconds=float(
            trajectory.metrics.get("rollout/dollar_seconds", 0.0)
        ),
    )


def _evaluate_real_policy(
    policy: RealAblationPolicy,
    scenarios: Sequence[Scenario],
) -> dict[str, float]:
    total_weight = 0.0
    weighted_correct = 0.0
    metrics: dict[str, float] = {}
    for scenario in scenarios:
        correct = 0
        total = 0
        for index in range(16):
            task = _real_task(scenario, index)
            prediction, _, _ = policy.predict(task)
            correct += 1 if prediction == task.label else 0
            total += 1
        accuracy = correct / total if total else 0.0
        weight = float(scenario.payload.get("eval_weight", 1.0))
        total_weight += weight
        weighted_correct += weight * accuracy
        metrics[f"real/eval_accuracy/{scenario.id}"] = accuracy
    metrics["real/eval_weighted_accuracy"] = (
        weighted_correct / total_weight if total_weight else 0.0
    )
    return metrics


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError(
            "real workload ablations require torch. "
            'Install with `pip install -e ".[calm]"`.'
        ) from exc
    return torch, nn, functional


def _objective_scheduler(
    *,
    max_policy_lag: int = 2,
    max_actor_count: int = 2,
) -> ObjectiveScheduler:
    return ObjectiveScheduler(
        min_train_batch_groups=1,
        max_train_batch_groups=2,
        min_policy_lag=1,
        max_policy_lag=max(1, max_policy_lag),
        min_actor_count=1,
        max_actor_count=max(1, max_actor_count),
        exploration_bonus=0.0,
    )


def _control_dimension_summary(
    summary: RunSummary,
    *,
    num_actors: int,
    train_queue_capacity: int,
    max_policy_lag: int,
    adaptive_action_space: bool,
) -> dict[str, float]:
    metrics = summary_metrics(summary)
    metrics["ablation/config/num_actors"] = float(num_actors)
    metrics["ablation/config/train_queue_capacity"] = float(train_queue_capacity)
    metrics["ablation/config/max_policy_lag"] = float(max_policy_lag)
    metrics["ablation/config/adaptive_action_space"] = (
        1.0 if adaptive_action_space else 0.0
    )
    return metrics


def _benchmark_metrics_from_bridge(metrics: Mapping[str, float]) -> dict[str, float]:
    benchmark = dict(metrics)
    benchmark.update(
        {
            BENCHMARK_NORTH_STAR: float(metrics.get(ART_NORTH_STAR, 0.0)),
            BENCHMARK_ACCOUNTED_NORTH_STAR: float(
                metrics.get(ART_ACCOUNTED_NORTH_STAR, 0.0)
            ),
            "benchmark/wall_clock_s": float(
                metrics.get("art_backend/wall_clock_s", 0.0)
            ),
            "benchmark/accounted_dollar_seconds": float(
                metrics.get("art_backend/accounted_dollar_seconds", 0.0)
            ),
            "benchmark/sample_dollar_seconds": float(
                metrics.get("art_backend/sample_dollar_seconds", 0.0)
            ),
            "benchmark/trainer_dollar_seconds": float(
                metrics.get("art_backend/trainer_dollar_seconds", 0.0)
            ),
            "benchmark/submitted_groups": float(
                metrics.get("art_backend/submitted_groups", 0.0)
            ),
            "benchmark/completed_batches": float(
                metrics.get("art_backend/completed_batches", 0.0)
            ),
            "benchmark/published_policy_updates": float(
                metrics.get("art_backend/published_policy_updates", 0.0)
            ),
            "benchmark/published_policy_reward_improving_experience": float(
                metrics.get(
                    "art_backend/published_policy_reward_improving_experience",
                    0.0,
                )
            ),
            "benchmark/groups_per_s": float(
                metrics.get("art_backend/submitted_train_groups_per_s", 0.0)
            ),
            "benchmark/action_units": float(
                metrics.get("art_backend/action_units", 0.0)
            ),
            "benchmark/source_tokens": float(
                metrics.get("art_backend/source_tokens", 0.0)
            ),
            "benchmark/action_units_per_s": float(
                metrics.get("art_backend/action_units_per_s", 0.0)
            ),
            "benchmark/source_tokens_per_s": float(
                metrics.get("art_backend/source_tokens_per_s", 0.0)
            ),
        }
    )
    return benchmark


def summary_metrics(summary: RunSummary) -> dict[str, float]:
    keys = [
        NORTH_STAR,
        ACCOUNTED_NORTH_STAR,
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
        "real/eval_weighted_accuracy",
        "real/eval_accuracy/easy_math",
        "real/eval_accuracy/hard_math",
        "scheduler/arm/easy_math_token/pulls",
        "scheduler/arm/easy_math_token/mean_rollout_dollar_seconds",
        "scheduler/arm/easy_math_token/total_improvement_per_dollar_second",
        "scheduler/arm/hard_math_token/pulls",
        "scheduler/arm/hard_math_token/mean_rollout_dollar_seconds",
        "scheduler/arm/hard_math_token/total_improvement_per_dollar_second",
        "scheduler/arm/bandwidth_chunk_chunk_size_2/pulls",
        "scheduler/arm/bandwidth_chunk_chunk_size_2/mean_rollout_dollar_seconds",
        "scheduler/arm/bandwidth_chunk_chunk_size_2/total_improvement_per_dollar_second",
        "scheduler/arm/bandwidth_token/pulls",
        "scheduler/arm/bandwidth_token/mean_rollout_dollar_seconds",
        "scheduler/arm/bandwidth_token/total_improvement_per_dollar_second",
        "scheduler/arm/semantic_chunk_chunk_size_2/pulls",
        "scheduler/arm/semantic_chunk_chunk_size_2/mean_rollout_dollar_seconds",
        "scheduler/arm/semantic_chunk_chunk_size_2/total_improvement_per_dollar_second",
        "scheduler/arm/semantic_chunk_chunk_size_4/pulls",
        "scheduler/arm/semantic_chunk_chunk_size_4/mean_rollout_dollar_seconds",
        "scheduler/arm/semantic_chunk_chunk_size_4/total_improvement_per_dollar_second",
        "scheduler/arm/semantic_token/pulls",
        "scheduler/arm/semantic_token/mean_rollout_dollar_seconds",
        "scheduler/arm/semantic_token/total_improvement_per_dollar_second",
        "scheduler/arm/closed_loop_chunk_chunk_size_2/pulls",
        "scheduler/arm/closed_loop_chunk_chunk_size_2/mean_rollout_dollar_seconds",
        "scheduler/arm/closed_loop_chunk_chunk_size_2/total_improvement_per_dollar_second",
        "scheduler/arm/closed_loop_chunk_chunk_size_4/pulls",
        "scheduler/arm/closed_loop_chunk_chunk_size_4/mean_rollout_dollar_seconds",
        "scheduler/arm/closed_loop_chunk_chunk_size_4/total_improvement_per_dollar_second",
        "scheduler/arm/closed_loop_token/pulls",
        "scheduler/arm/closed_loop_token/mean_rollout_dollar_seconds",
        "scheduler/arm/closed_loop_token/total_improvement_per_dollar_second",
        "action_space/active_codecs",
        "action_space/promotions",
        "action_space/demotions",
        "action_space/decision_payoff_demotions",
        "action_space/max_chunk_size",
        "action_space/codec/chunk_chunk_size_4/active",
        "action_space/decision/decisions",
        "action_space/decision/post_decision_observations",
        "action_space/decision/realized_objective_payoff",
        "action_space/decision/mean_realized_objective_payoff_per_decision",
        "action_space/decision/"
        "mean_realized_objective_payoff_per_post_decision_observation",
        "action_space/decision/realized_source_token_throughput_payoff",
        "action_space/decision/"
        "mean_realized_source_token_throughput_payoff_per_decision",
        "action_space/decision/"
        "mean_realized_source_token_throughput_payoff_per_post_decision_observation",
        "promotion/decision/keys",
        "promotion/decision/decisions",
        "promotion/decision/promoted",
        "promotion/decision/rejected",
        "promotion/decision/positive_reward_improving_keys",
        "promotion/decision/total_candidate_improvement",
        "promotion/decision/total_published_policy_improvement",
        "promotion/decision/realized_reward_improving_experience",
        "promotion/decision/total_dollar_seconds",
        "promotion/decision/"
        "mean_realized_reward_improving_experience_per_decision",
        "promotion/decision/"
        "realized_reward_improving_experience_per_dollar_second",
        "scheduler/control/cadence_1/train_updates",
        "scheduler/control/cadence_1/mean_objective_per_decision",
        "scheduler/control/cadence_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_1/train_updates",
        "scheduler/control/policy_lag_1/mean_objective_per_decision",
        "scheduler/control/policy_lag_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_2/train_updates",
        "scheduler/control/policy_lag_2/mean_objective_per_decision",
        "scheduler/control/policy_lag_2/mean_objective_per_feedback_update",
        "scheduler/control/admission_delay_ms_0/rollout_updates",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_decision",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_1/rollout_updates",
        "scheduler/control/actor_count_1/mean_objective_per_decision",
        "scheduler/control/actor_count_1/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_2/rollout_updates",
        "scheduler/control/actor_count_2/mean_objective_per_decision",
        "scheduler/control/actor_count_2/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_1/score",
        "scheduler/control/actor_count_2/score",
        *RUNTIME_CONTROL_CONTEXT_SUMMARY_KEYS,
        "scheduler/coverage_control/keys",
        "scheduler/coverage_control/decisions",
        "scheduler/coverage_control/feedback_updates",
        "scheduler/coverage_control/positive_objective_keys",
        "scheduler/coverage_control/total_objective",
        "scheduler/coverage_control/mean_objective_per_decision",
        "scheduler/coverage_control/mean_objective_per_feedback_update",
        "scheduler/timing_response/keys",
        "scheduler/timing_response/decisions",
        "scheduler/timing_response/feedback_updates",
        "scheduler/timing_response/positive_objective_keys",
        "scheduler/timing_response/total_objective",
        "scheduler/timing_response/mean_objective_per_decision",
        "scheduler/timing_response/mean_objective_per_feedback_update",
        "scheduler/continuation/keys",
        "scheduler/continuation/decisions",
        "scheduler/continuation/feedback_updates",
        "scheduler/continuation/positive_objective_keys",
        "scheduler/continuation/total_objective",
        "scheduler/continuation/mean_objective_per_decision",
        "scheduler/continuation/mean_objective_per_feedback_update",
        "scheduler/train_selection/keys",
        "scheduler/train_selection/decisions",
        "scheduler/train_selection/feedback_updates",
        "scheduler/train_selection/positive_objective_keys",
        "scheduler/train_selection/total_objective",
        "scheduler/train_selection/mean_objective_per_decision",
        "scheduler/train_selection/mean_objective_per_feedback_update",
        "scheduler/last_train_batch_train_selection_score",
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
        "scheduler/joint_action/feedback_tuples",
        "scheduler/joint_action/positive_objective_tuples",
        "scheduler/joint_action/total_objective",
        "scheduler/joint_action/mean_objective_per_decision",
        "scheduler/joint_action/mean_objective_per_feedback_update",
        "scheduler/last_train_batch_joint_action_score",
    ]
    return {
        key: float(summary.metrics[key])
        for key in keys
        if key in summary.metrics
    }


def bridge_summary_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    keys = [
        ART_NORTH_STAR,
        ART_ACCOUNTED_NORTH_STAR,
        "art_backend/wall_clock_s",
        "art_backend/wall_clock_dollar_seconds",
        "art_backend/accounted_dollar_seconds",
        "art_backend/sample_dollar_seconds",
        "art_backend/trainer_dollar_seconds",
        "art_backend/submitted_groups",
        "art_backend/submitted_train_groups",
        "art_backend/completed_batches",
        "art_backend/submitted_batches_per_s",
        "art_backend/submitted_train_groups_per_s",
        "art_backend/completed_batches_per_s",
        "art_backend/sample_dollar_seconds_per_s",
        "art_backend/published_policy_updates",
        "art_backend/published_policy_reward_improving_experience",
        "art_backend/action_units",
        "art_backend/source_tokens",
        "art_backend/action_units_per_s",
        "art_backend/source_tokens_per_s",
        "art_backend/publication/decision/keys",
        "art_backend/publication/decision/decisions",
        "art_backend/publication/decision/published",
        "art_backend/publication/decision/positive_reward_improving_keys",
        "art_backend/publication/decision/total_candidate_improvement",
        "art_backend/publication/decision/total_published_policy_improvement",
        "art_backend/publication/decision/realized_reward_improving_experience",
        "art_backend/publication/decision/total_dollar_seconds",
        "art_backend/publication/decision/"
        "mean_realized_reward_improving_experience_per_decision",
        "art_backend/publication/decision/"
        "realized_reward_improving_experience_per_dollar_second",
        "actions/semantic_bandwidth_tokens_per_decision",
        "action_space/active_codecs",
        "action_space/promotions",
        "action_space/demotions",
        "action_space/decision_payoff_demotions",
        "action_space/max_chunk_size",
        "action_space/codec/chunk_chunk_size_4/active",
        "action_space/decision/decisions",
        "action_space/decision/post_decision_observations",
        "action_space/decision/realized_objective_payoff",
        "action_space/decision/mean_realized_objective_payoff_per_decision",
        "action_space/decision/"
        "mean_realized_objective_payoff_per_post_decision_observation",
        "scheduler/arm/art_bridge_chunk_chunk_size_2/pulls",
        "scheduler/arm/art_bridge_chunk_chunk_size_4/pulls",
        "scheduler/arm/art_bridge_token/pulls",
        "scheduler/control/cadence_1/train_updates",
        "scheduler/control/cadence_1/mean_objective_per_decision",
        "scheduler/control/cadence_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_1/train_updates",
        "scheduler/control/policy_lag_1/mean_objective_per_decision",
        "scheduler/control/policy_lag_1/mean_objective_per_feedback_update",
        "scheduler/control/policy_lag_2/train_updates",
        "scheduler/control/policy_lag_2/mean_objective_per_decision",
        "scheduler/control/policy_lag_2/mean_objective_per_feedback_update",
        "scheduler/control/admission_delay_ms_0/rollout_updates",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_decision",
        "scheduler/control/admission_delay_ms_0/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_1/rollout_updates",
        "scheduler/control/actor_count_1/mean_objective_per_decision",
        "scheduler/control/actor_count_1/mean_objective_per_feedback_update",
        "scheduler/control/actor_count_2/rollout_updates",
        "scheduler/control/actor_count_2/mean_objective_per_decision",
        "scheduler/control/actor_count_2/mean_objective_per_feedback_update",
        *RUNTIME_CONTROL_CONTEXT_SUMMARY_KEYS,
        "scheduler/coverage_control/keys",
        "scheduler/coverage_control/decisions",
        "scheduler/coverage_control/feedback_updates",
        "scheduler/coverage_control/positive_objective_keys",
        "scheduler/coverage_control/total_objective",
        "scheduler/coverage_control/mean_objective_per_decision",
        "scheduler/coverage_control/mean_objective_per_feedback_update",
        "scheduler/timing_response/keys",
        "scheduler/timing_response/decisions",
        "scheduler/timing_response/feedback_updates",
        "scheduler/timing_response/positive_objective_keys",
        "scheduler/timing_response/total_objective",
        "scheduler/timing_response/mean_objective_per_decision",
        "scheduler/timing_response/mean_objective_per_feedback_update",
        "scheduler/continuation/keys",
        "scheduler/continuation/decisions",
        "scheduler/continuation/feedback_updates",
        "scheduler/continuation/positive_objective_keys",
        "scheduler/continuation/total_objective",
        "scheduler/continuation/mean_objective_per_decision",
        "scheduler/continuation/mean_objective_per_feedback_update",
        "scheduler/train_selection/keys",
        "scheduler/train_selection/decisions",
        "scheduler/train_selection/feedback_updates",
        "scheduler/train_selection/positive_objective_keys",
        "scheduler/train_selection/total_objective",
        "scheduler/train_selection/mean_objective_per_decision",
        "scheduler/train_selection/mean_objective_per_feedback_update",
        "scheduler/last_train_batch_train_selection_score",
        "scheduler/joint_action/tuples",
        "scheduler/joint_action/decisions",
        "scheduler/joint_action/feedback_updates",
        "scheduler/joint_action/positive_objective_tuples",
        "scheduler/joint_action/total_objective",
        "scheduler/joint_action/mean_objective_per_decision",
        "scheduler/joint_action/mean_objective_per_feedback_update",
        "scheduler/last_train_batch_joint_action_score",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}
