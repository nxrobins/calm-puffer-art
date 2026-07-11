from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from calm_puffer_art.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    PricingConfig,
    TelemetryLedger,
)
from real_art_weight_update import (
    SYSTEM_MESSAGE,
    ChecksumTask,
    CompletionRecord,
    _complete,
    _evaluate,
    _float_mapping,
    _pricing_report,
)


CONDITIONS = ("base", "direct_art", "async_scheduler")
STRATA = ("easy", "medium", "hard", "challenge")
CALM_EXCLUSION = (
    "The learned chunk codec is reconstruction-smoke-only and does not yet "
    "change ART inference actions, policy logprobs, or the optimizer loss."
)
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


@dataclass(frozen=True)
class AblationTask:
    task: ChecksumTask
    stratum: str


@dataclass
class SampledGroup:
    group: Any
    records: list[CompletionRecord]
    task_id: str
    stratum: str
    reward_varied: bool


class RetryingServerlessBackend:
    """Retry transient managed-training failures without replaying rollouts."""

    def __init__(
        self,
        *,
        art: Any,
        delegate: Any,
        max_attempts: int,
        timeout_seconds: float,
        attempt_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.art = art
        self.delegate = delegate
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.attempt_observer = attempt_observer
        self.attempts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    async def train(
        self,
        model: Any,
        trajectory_groups: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        groups = list(trajectory_groups)
        initial_step = int(await self.delegate._get_step(model))
        for attempt in range(1, self.max_attempts + 1):
            if self.attempt_observer is not None:
                self.attempt_observer(
                    {
                        "event": "started",
                        "attempt": attempt,
                        "status": "started",
                        "initial_step": initial_step,
                    }
                )
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    self.delegate.train(model, groups, **kwargs),
                    timeout=self.timeout_seconds,
                )
                result_step = int(getattr(result, "step", initial_step))
                if result_step != initial_step + 1:
                    raise RuntimeError(
                        "managed training advanced an unexpected number of "
                        f"checkpoints: {initial_step} -> {result_step}"
                    )
                entry = {
                    "attempt": attempt,
                    "status": "completed",
                    "wall_s": time.perf_counter() - started,
                    "initial_step": initial_step,
                    "observed_step": result_step,
                    "step": result_step,
                    "metrics": _float_mapping(getattr(result, "metrics", {})),
                    "artifact_name": getattr(result, "artifact_name", None),
                }
                self._record_attempt(entry)
                return result
            except Exception as exc:
                observed_step = int(await self.delegate._get_step(model))
                entry = {
                    "attempt": attempt,
                    "status": "failed",
                    "wall_s": time.perf_counter() - started,
                    "initial_step": initial_step,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "observed_step": observed_step,
                }
                if observed_step > initial_step:
                    if observed_step != initial_step + 1:
                        self._record_attempt(entry)
                        raise RuntimeError(
                            "managed training advanced an unexpected number of "
                            f"checkpoints: {initial_step} -> {observed_step}"
                        ) from exc
                    entry["status"] = "recovered"
                    entry["initial_step"] = initial_step
                    entry["step"] = observed_step
                    artifact_name = (
                        f"{model.entity}/{model.project}/{model.name}:"
                        f"step{observed_step}"
                    )
                    entry["artifact_name"] = artifact_name
                    self._record_attempt(entry)
                    return self.art.ServerlessTrainResult(
                        step=observed_step,
                        metrics={"ablation/recovered_after_client_error": 1.0},
                        artifact_name=artifact_name,
                    )
                self._record_attempt(entry)
                if attempt >= self.max_attempts or not _retryable_train_error(exc):
                    raise
                await asyncio.sleep(5.0)
        raise RuntimeError("managed training exhausted retry attempts")

    def _record_attempt(self, entry: Mapping[str, Any]) -> None:
        payload = dict(entry)
        self.attempts.append(payload)
        if self.attempt_observer is not None:
            self.attempt_observer(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-seed fixed-budget comparison of base inference, "
            "direct ART, and ART through the adaptive async scheduler."
        )
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--env-path", type=Path)
    parser.add_argument("--project", default="calm-puffer-art-ablation")
    parser.add_argument("--base-model", default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--manifest-seed", type=int, default=20260711)
    parser.add_argument("--train-tasks", type=int, default=12)
    parser.add_argument("--heldout-tasks", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-group", type=int, default=4)
    parser.add_argument("--max-rollouts-per-group", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--inference-retries", type=int, default=4)
    parser.add_argument("--training-attempts", type=int, default=2)
    parser.add_argument("--training-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--input-usd-per-million-tokens", type=float, default=0.0
    )
    parser.add_argument(
        "--output-usd-per-million-tokens", type=float, default=0.0
    )
    parser.add_argument("--trainer-usd-per-hour", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--telemetry-path", type=Path)
    parser.add_argument("--telemetry-echo", action="store_true")
    args = parser.parse_args(argv)
    args.seed_values = _parse_seeds(parser, args.seeds)
    _validate_args(parser, args)
    return args


def _parse_seeds(
    parser: argparse.ArgumentParser,
    raw: str,
) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError:
        parser.error("--seeds must be a comma-separated list of integers")
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must not contain duplicates")
    return seeds


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "train-tasks": args.train_tasks,
        "heldout-tasks": args.heldout_tasks,
        "train-steps": args.train_steps,
        "groups-per-step": args.groups_per_step,
        "rollouts-per-group": args.rollouts_per_group,
        "max-rollouts-per-group": args.max_rollouts_per_group,
        "concurrency": args.concurrency,
        "max-tokens": args.max_tokens,
        "inference-retries": args.inference_retries,
        "training-attempts": args.training_attempts,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    expected_groups = args.train_steps * args.groups_per_step
    if args.train_tasks != expected_groups:
        parser.error(
            "--train-tasks must equal --train-steps * --groups-per-step"
        )
    if args.train_tasks % len(STRATA) != 0:
        parser.error(f"--train-tasks must be divisible by {len(STRATA)}")
    if args.max_rollouts_per_group < args.rollouts_per_group:
        parser.error(
            "--max-rollouts-per-group must be >= --rollouts-per-group"
        )
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.training_timeout_seconds <= 0.0:
        parser.error("--training-timeout-seconds must be positive")
    for name in (
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "trainer_usd_per_hour",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")


def build_manifest(
    *,
    seed: int,
    train_tasks: int,
    heldout_tasks: int,
) -> tuple[tuple[AblationTask, ...], tuple[AblationTask, ...]]:
    rng = random.Random(seed)
    train_counts = _balanced_counts(train_tasks)
    heldout_counts = _balanced_counts(heldout_tasks)
    train = tuple(
        _make_task(rng, split="train", stratum=stratum, index=index)
        for stratum, count in zip(STRATA, train_counts, strict=True)
        for index in range(1, count + 1)
    )
    heldout = tuple(
        _make_task(rng, split="heldout", stratum=stratum, index=index)
        for stratum, count in zip(STRATA, heldout_counts, strict=True)
        for index in range(1, count + 1)
    )
    return train, heldout


def _balanced_counts(total: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, len(STRATA))
    return tuple(quotient + (1 if index < remainder else 0) for index in range(4))


def _make_task(
    rng: random.Random,
    *,
    split: str,
    stratum: str,
    index: int,
) -> AblationTask:
    ranges = {
        "easy": (10, 99, (97, 101, 103, 107, 109)),
        "medium": (100, 999, (1009, 1013, 1019, 1021, 1031)),
        "hard": (1000, 9999, (10007, 10009, 10037, 10039, 10061)),
        "challenge": (10000, 99999, (65521, 65537, 65539, 65543, 65551)),
    }
    low, high, moduli = ranges[stratum]
    modulus = rng.choice(moduli)
    task = ChecksumTask(
        id=f"{split}-{stratum}-{index:02d}",
        seed=rng.randint(low, high),
        multiplier=rng.randint(max(2, low // 5), max(3, high // 3)),
        addend=rng.randint(low, high),
        square_offset=rng.randint(low, high),
        final_multiplier=rng.randint(max(2, low // 7), max(3, high // 4)),
        final_addend=rng.randint(low, high),
        modulus=modulus,
    )
    return AblationTask(task=task, stratum=stratum)


def manifest_fingerprint(
    train: Sequence[AblationTask],
    heldout: Sequence[AblationTask],
) -> str:
    payload = [
        {"split": split, "stratum": item.stratum, **asdict(item.task)}
        for split, values in (("train", train), ("heldout", heldout))
        for item in values
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    train, heldout = build_manifest(
        seed=args.manifest_seed,
        train_tasks=args.train_tasks,
        heldout_tasks=args.heldout_tasks,
    )
    art_installed = importlib.util.find_spec("art") is not None
    wandb_installed = importlib.util.find_spec("wandb") is not None
    art_version = None
    if art_installed:
        try:
            art_version = importlib.metadata.version("openpipe-art")
        except importlib.metadata.PackageNotFoundError:
            pass
    credential_ready = bool(os.environ.get("WANDB_API_KEY"))
    minimum_requests, maximum_requests = _inference_request_bounds(args)
    return {
        "ok": True,
        "mode": "preflight",
        "conditions": list(CONDITIONS),
        "excluded_conditions": {"calm": CALM_EXCLUSION},
        "seeds": list(args.seed_values),
        "train_tasks": len(train),
        "heldout_tasks": len(heldout),
        "train_steps_per_trained_condition": args.train_steps,
        "training_updates": (
            len(args.seed_values) * 2 * args.train_steps
        ),
        "minimum_inference_requests": minimum_requests,
        "maximum_inference_requests": maximum_requests,
        "manifest_fingerprint": manifest_fingerprint(train, heldout),
        "art_installed": art_installed,
        "art_version": art_version,
        "wandb_installed": wandb_installed,
        "credential_ready": credential_ready,
        "live_ready": art_installed and wandb_installed and credential_ready,
        "pricing": _pricing_report(args),
        "telemetry": {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_ledger": "append-only JSONL",
            "monetary_cost_requires_explicit_rates": True,
            "missing_price_is_null_not_zero": True,
            "token_cost_is_labeled_as_proxy": True,
            "captures": [
                "inference",
                "training_attempt",
                "scheduler_decision",
                "condition_lifecycle",
            ],
        },
    }


def _inference_request_bounds(args: argparse.Namespace) -> tuple[int, int]:
    minimum_per_seed = (
        2 * args.heldout_tasks
        + 2
        * (
            2 * args.heldout_tasks
            + args.train_tasks * args.rollouts_per_group
        )
    )
    maximum_per_seed = (
        2 * args.heldout_tasks
        + 2
        * (
            2 * args.heldout_tasks
            + args.train_tasks * args.max_rollouts_per_group
        )
    )
    seeds = len(args.seed_values)
    return minimum_per_seed * seeds, maximum_per_seed * seeds


def _telemetry_pricing(args: argparse.Namespace) -> PricingConfig:
    return PricingConfig(
        input_usd_per_million_tokens=(
            args.input_usd_per_million_tokens
            if args.input_usd_per_million_tokens > 0.0
            else None
        ),
        output_usd_per_million_tokens=(
            args.output_usd_per_million_tokens
            if args.output_usd_per_million_tokens > 0.0
            else None
        ),
        trainer_usd_per_hour=(
            args.trainer_usd_per_hour
            if args.trainer_usd_per_hour > 0.0
            else None
        ),
    )


async def _sample_group(
    *,
    art: Any,
    client: Any,
    inference_name: str,
    item: AblationTask,
    scenario_id: str,
    rollout_namespace: str,
    policy_step: int,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    initial_metadata: Sequence[Mapping[str, Any]],
    extra_metadata: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
) -> SampledGroup:
    async def sample(
        metadata: Mapping[str, Any],
        rollout_index: int,
    ) -> tuple[CompletionRecord, Mapping[str, Any], int]:
        request_seed = _rollout_seed(
            experiment_seed=args.request_seed,
            task_id=item.task.id,
            policy_step=policy_step,
            rollout_namespace=rollout_namespace,
            rollout_index=rollout_index,
        )
        record = await _complete(
            client=client,
            inference_name=inference_name,
            task=item.task,
            split="train",
            temperature=args.temperature,
            args=args,
            semaphore=semaphore,
            request_seed=request_seed,
        )
        return record, metadata, request_seed

    sampled = list(
        await asyncio.gather(
            *(
                sample(metadata, rollout_index)
                for rollout_index, metadata in enumerate(initial_metadata)
            )
        )
    )
    while (
        len({record.reward for record, _, _ in sampled}) < 2
        and len(sampled) < args.max_rollouts_per_group
    ):
        metadata = await extra_metadata() if extra_metadata is not None else {}
        sampled.append(await sample(metadata, len(sampled)))
    trajectories = []
    for record, metadata, request_seed in sampled:
        trajectories.append(
            art.Trajectory(
                messages_and_choices=[
                    SYSTEM_MESSAGE,
                    {"role": "user", "content": item.task.prompt},
                    record.choice,
                ],
                reward=record.reward,
                initial_policy_version=policy_step,
                final_policy_version=policy_step,
                metrics={
                    "rollout/dollar_seconds": _scheduler_cost_proxy(record),
                    "cost/api_usd": record.estimated_api_usd,
                    "duration": record.elapsed_s,
                    "usage/prompt_tokens": record.prompt_tokens,
                    "usage/completion_tokens": record.completion_tokens,
                },
                metadata={
                    **dict(metadata),
                    "scenario_id": scenario_id,
                    "ablation/task_id": item.task.id,
                    "ablation/stratum": item.stratum,
                    "ablation/request_seed": request_seed,
                },
            )
        )
    records = [record for record, _, _ in sampled]
    return SampledGroup(
        group=art.TrajectoryGroup(
            trajectories,
            metadata={
                "scenario_id": scenario_id,
                "ablation/task_id": item.task.id,
                "ablation/stratum": item.stratum,
            },
        ),
        records=records,
        task_id=item.task.id,
        stratum=item.stratum,
        reward_varied=len({record.reward for record in records}) >= 2,
    )


def _rollout_seed(
    *,
    experiment_seed: int,
    task_id: str,
    policy_step: int,
    rollout_namespace: str,
    rollout_index: int,
) -> int:
    payload = (
        f"{experiment_seed}|{task_id}|{policy_step}|"
        f"{rollout_namespace}|{rollout_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


async def _evaluate_manifest(
    *,
    client: Any,
    inference_name: str,
    heldout: Sequence[AblationTask],
    split: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    report = await _evaluate(
        client=client,
        inference_name=inference_name,
        tasks=[item.task for item in heldout],
        split=split,
        args=args,
        semaphore=semaphore,
    )
    report.update(_completion_diagnostics(report["records"]))
    task_strata = {item.task.id: item.stratum for item in heldout}
    by_stratum: dict[str, dict[str, float]] = {}
    for stratum in STRATA:
        rows = [
            row
            for row in report["records"]
            if task_strata.get(str(row["task_id"])) == stratum
        ]
        by_stratum[stratum] = {
            "requests": float(len(rows)),
            "exact_accuracy": (
                sum(bool(row["exact"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_reward": (
                sum(float(row["reward"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        }
    report["by_stratum"] = by_stratum
    return report


def _completion_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    parsed_rewards = [
        float(record["reward"])
        for record in records
        if record.get("parsed_answer") is not None
    ]
    return {
        "parsed_count": len(parsed_rewards),
        "parse_rate": len(parsed_rewards) / len(records) if records else 0.0,
        "mean_reward_given_parsed": (
            statistics.fmean(parsed_rewards) if parsed_rewards else 0.0
        ),
    }


async def _run_base_condition(
    *,
    art: Any,
    serverless_backend_cls: Any,
    heldout: Sequence[AblationTask],
    seed: int,
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    delegate = RetryingServerlessBackend(
        art=art,
        delegate=serverless_backend_cls(),
        max_attempts=args.training_attempts,
        timeout_seconds=args.training_timeout_seconds,
        attempt_observer=getattr(args, "training_attempt_observer", None),
    )
    model = art.TrainableModel(
        name=f"ablation-{run_id}-base-s{seed}",
        project=args.project,
        base_model=args.base_model,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    try:
        await model.register(delegate)
        initial_step = int(await delegate._get_step(model))
        inference_name = model.get_inference_name()
        before = await _evaluate_manifest(
            client=model.openai_client(),
            inference_name=inference_name,
            heldout=heldout,
            split="heldout_before",
            args=args,
            semaphore=semaphore,
        )
        after = await _evaluate_manifest(
            client=model.openai_client(),
            inference_name=inference_name,
            heldout=heldout,
            split="heldout_after_no_update",
            args=args,
            semaphore=semaphore,
        )
        return _condition_result(
            condition="base",
            model=model,
            initial_step=initial_step,
            final_step=initial_step,
            before=before,
            after=after,
            training=None,
            train_results=[],
            backend_attempts=delegate.attempts,
            wall_s=time.perf_counter() - started,
        )
    finally:
        await delegate.close()


async def _run_direct_condition(
    *,
    art: Any,
    serverless_backend_cls: Any,
    train: Sequence[AblationTask],
    heldout: Sequence[AblationTask],
    seed: int,
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    delegate = RetryingServerlessBackend(
        art=art,
        delegate=serverless_backend_cls(),
        max_attempts=args.training_attempts,
        timeout_seconds=args.training_timeout_seconds,
        attempt_observer=getattr(args, "training_attempt_observer", None),
    )
    model = art.TrainableModel(
        name=f"ablation-{run_id}-direct-s{seed}",
        project=args.project,
        base_model=args.base_model,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    sampled_groups: list[SampledGroup] = []
    train_results: list[dict[str, Any]] = []
    ordered_train = _round_robin_strata(train)
    try:
        await model.register(delegate)
        initial_step = int(await delegate._get_step(model))
        before = await _evaluate_manifest(
            client=model.openai_client(),
            inference_name=model.get_inference_name(),
            heldout=heldout,
            split="heldout_before",
            args=args,
            semaphore=semaphore,
        )
        current_step = initial_step
        for train_step in range(args.train_steps):
            batch_items = ordered_train[
                train_step * args.groups_per_step :
                (train_step + 1) * args.groups_per_step
            ]
            groups = list(
                await asyncio.gather(
                    *(
                        _sample_group(
                            art=art,
                            client=model.openai_client(),
                            inference_name=_checkpoint_inference_name(
                                model,
                                current_step=current_step,
                                initial_step=initial_step,
                            ),
                            item=item,
                            scenario_id=f"difficulty_{item.stratum}",
                            rollout_namespace=(
                                f"direct-{train_step}-{group_index}"
                            ),
                            policy_step=current_step,
                            args=args,
                            semaphore=semaphore,
                            initial_metadata=[{}] * args.rollouts_per_group,
                        )
                        for group_index, item in enumerate(batch_items)
                    )
                )
            )
            _require_trainable_batch(groups, train_step=train_step)
            sampled_groups.extend(groups)
            result = await delegate.train(
                model,
                [group.group for group in groups],
                learning_rate=args.learning_rate,
            )
            current_step = int(result.step)
            train_results.append(_train_result_report(result, train_step))
        after = await _evaluate_manifest(
            client=model.openai_client(),
            inference_name=model.get_inference_name(step=current_step),
            heldout=heldout,
            split="heldout_after",
            args=args,
            semaphore=semaphore,
        )
        return _condition_result(
            condition="direct_art",
            model=model,
            initial_step=initial_step,
            final_step=current_step,
            before=before,
            after=after,
            training=_training_report(sampled_groups),
            train_results=train_results,
            backend_attempts=delegate.attempts,
            wall_s=time.perf_counter() - started,
        )
    finally:
        await delegate.close()


async def _run_scheduler_condition(
    *,
    art: Any,
    serverless_backend_cls: Any,
    train: Sequence[AblationTask],
    heldout: Sequence[AblationTask],
    seed: int,
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from calm_puffer_art import (
        AsyncArtBackend,
        AsyncArtBackendConfig,
        ObjectiveScheduler,
        Scenario,
        TokenActionCodec,
    )

    delegate = RetryingServerlessBackend(
        art=art,
        delegate=serverless_backend_cls(),
        max_attempts=args.training_attempts,
        timeout_seconds=args.training_timeout_seconds,
        attempt_observer=getattr(args, "training_attempt_observer", None),
    )
    scheduler = ObjectiveScheduler(
        min_train_batch_groups=args.groups_per_step,
        max_train_batch_groups=args.groups_per_step,
        min_policy_lag=0,
        max_policy_lag=0,
        min_actor_count=args.max_rollouts_per_group,
        max_actor_count=args.max_rollouts_per_group,
        exploration_bonus=0.2,
        control_exploration_bonus=0.0,
    )
    bridge = AsyncArtBackend(
        backend=delegate,
        config=AsyncArtBackendConfig(
            train_queue_capacity=1,
            train_batch_groups=args.groups_per_step,
            max_policy_lag=0,
            max_train_steps=args.train_steps,
            cost_per_second_usd=args.trainer_usd_per_hour / 3600.0,
        ),
        scheduler=scheduler,
    )
    model = art.TrainableModel(
        name=f"ablation-{run_id}-scheduler-s{seed}",
        project=args.project,
        base_model=args.base_model,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    scenarios = tuple(
        Scenario(id=f"difficulty_{stratum}", payload={"stratum": stratum})
        for stratum in STRATA
    )
    codec = TokenActionCodec()
    tasks_by_stratum: dict[str, list[AblationTask]] = defaultdict(list)
    for item in train:
        tasks_by_stratum[item.stratum].append(item)
    task_offsets: Counter[str] = Counter()
    sampled_groups: list[SampledGroup] = []
    train_results: list[dict[str, Any]] = []
    allocations: list[str] = []
    started = time.perf_counter()
    try:
        await model.register(bridge)
        initial_step = int(await bridge._get_step(model))
        before = await _evaluate_manifest(
            client=model.openai_client(),
            inference_name=model.get_inference_name(),
            heldout=heldout,
            split="heldout_before",
            args=args,
            semaphore=semaphore,
        )
        current_step = initial_step
        for train_step in range(args.train_steps):
            step_groups: list[SampledGroup] = []
            for group_index in range(args.groups_per_step):
                first_assignment = await bridge.admit_and_select_rollout(
                    scenarios=scenarios,
                    action_codecs=[codec],
                    actor_id=0,
                    configured_actor_count=args.max_rollouts_per_group,
                    apply_delay=False,
                )
                if not first_assignment.admitted or first_assignment.decision is None:
                    raise RuntimeError("scheduler rejected a budgeted rollout group")
                selected = first_assignment.decision.scenario
                stratum = str(selected.payload["stratum"])
                allocations.append(stratum)
                items = tasks_by_stratum[stratum]
                item = items[task_offsets[stratum] % len(items)]
                task_offsets[stratum] += 1
                decision_observer = getattr(
                    args,
                    "scheduler_decision_observer",
                    None,
                )
                if decision_observer is not None:
                    decision_observer(
                        {
                            "train_step": train_step,
                            "group_index": group_index,
                            "selected_stratum": stratum,
                            "task_id": item.task.id,
                            "metadata": dict(first_assignment.metadata),
                        }
                    )
                metadata = [first_assignment.metadata]
                for actor_id in range(1, args.rollouts_per_group):
                    assignment = await bridge.admit_and_select_rollout(
                        scenarios=[selected],
                        action_codecs=[codec],
                        actor_id=actor_id,
                        configured_actor_count=args.max_rollouts_per_group,
                        apply_delay=False,
                    )
                    if not assignment.admitted or assignment.decision is None:
                        raise RuntimeError("scheduler rejected a grouped rollout")
                    metadata.append(assignment.metadata)

                extra_actor_id = args.rollouts_per_group

                async def extra_metadata() -> Mapping[str, Any]:
                    nonlocal extra_actor_id
                    assignment = await bridge.admit_and_select_rollout(
                        scenarios=[selected],
                        action_codecs=[codec],
                        actor_id=extra_actor_id,
                        configured_actor_count=args.max_rollouts_per_group,
                        apply_delay=False,
                    )
                    extra_actor_id += 1
                    if not assignment.admitted or assignment.decision is None:
                        raise RuntimeError("scheduler rejected an extra rollout")
                    return assignment.metadata

                step_groups.append(
                    await _sample_group(
                        art=art,
                        client=model.openai_client(),
                        inference_name=_checkpoint_inference_name(
                            model,
                            current_step=current_step,
                            initial_step=initial_step,
                        ),
                        item=item,
                        scenario_id=selected.id,
                        rollout_namespace=(
                            f"scheduler-{train_step}-{group_index}"
                        ),
                        policy_step=current_step,
                        args=args,
                        semaphore=semaphore,
                        initial_metadata=metadata,
                        extra_metadata=extra_metadata,
                    )
                )
            _require_trainable_batch(step_groups, train_step=train_step)
            sampled_groups.extend(step_groups)
            result = await bridge.train(
                model,
                [group.group for group in step_groups],
                learning_rate=args.learning_rate,
            )
            current_step = int(result.step)
            train_results.append(_train_result_report(result, train_step))
        after = await _evaluate_manifest(
            client=model.openai_client(),
            inference_name=model.get_inference_name(step=current_step),
            heldout=heldout,
            split="heldout_after",
            args=args,
            semaphore=semaphore,
        )
        training = _training_report(sampled_groups)
        training["scheduler_group_allocations"] = dict(Counter(allocations))
        training["scheduler_metrics"] = bridge.stats()
        return _condition_result(
            condition="async_scheduler",
            model=model,
            initial_step=initial_step,
            final_step=current_step,
            before=before,
            after=after,
            training=training,
            train_results=train_results,
            backend_attempts=delegate.attempts,
            wall_s=time.perf_counter() - started,
        )
    finally:
        await bridge.close()


def _round_robin_strata(train: Sequence[AblationTask]) -> list[AblationTask]:
    by_stratum = {
        stratum: [item for item in train if item.stratum == stratum]
        for stratum in STRATA
    }
    return [
        by_stratum[stratum][index]
        for index in range(len(next(iter(by_stratum.values()))))
        for stratum in STRATA
    ]


def _checkpoint_inference_name(
    model: Any,
    *,
    current_step: int,
    initial_step: int,
) -> str:
    if current_step == initial_step:
        return model.get_inference_name()
    return model.get_inference_name(step=current_step)


def _scheduler_cost_proxy(record: CompletionRecord) -> float:
    if record.estimated_api_usd > 0.0:
        return record.estimated_api_usd
    return max(record.total_tokens, 1) / 1_000_000.0


def _require_trainable_batch(
    groups: Sequence[SampledGroup],
    *,
    train_step: int,
) -> None:
    if not any(group.reward_varied for group in groups):
        raise RuntimeError(
            f"training step {train_step} had no non-uniform reward group"
        )


def _train_result_report(result: Any, train_step: int) -> dict[str, Any]:
    return {
        "train_step_index": train_step,
        "checkpoint_step": int(result.step),
        "artifact_name": getattr(result, "artifact_name", None),
        "metrics": _float_mapping(getattr(result, "metrics", {})),
    }


def _training_report(groups: Sequence[SampledGroup]) -> dict[str, Any]:
    records = [record for group in groups for record in group.records]
    requests = len(records)
    return {
        "groups": len(groups),
        "nonzero_advantage_groups": sum(group.reward_varied for group in groups),
        "uniform_reward_groups": sum(not group.reward_varied for group in groups),
        "requests": requests,
        "exact_accuracy": (
            sum(record.exact for record in records) / requests if requests else 0.0
        ),
        "mean_reward": (
            sum(record.reward for record in records) / requests if requests else 0.0
        ),
        "prompt_tokens": sum(record.prompt_tokens for record in records),
        "completion_tokens": sum(record.completion_tokens for record in records),
        "total_tokens": sum(record.total_tokens for record in records),
        "estimated_api_usd": sum(record.estimated_api_usd for record in records),
        "group_strata": dict(Counter(group.stratum for group in groups)),
        "task_ids": [group.task_id for group in groups],
    }


def _condition_result(
    *,
    condition: str,
    model: Any,
    initial_step: int,
    final_step: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    training: Mapping[str, Any] | None,
    train_results: Sequence[Mapping[str, Any]],
    backend_attempts: Sequence[Mapping[str, Any]],
    wall_s: float,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "condition": condition,
        "model": {
            "name": model.name,
            "id": model.id,
            "entity": model.entity,
            "project": model.project,
            "base_model": model.base_model,
        },
        "initial_step": initial_step,
        "final_step": final_step,
        "checkpoint_advanced": final_step > initial_step,
        "heldout_before": before,
        "heldout_after": after,
        "heldout_exact_accuracy_delta": (
            float(after["exact_accuracy"]) - float(before["exact_accuracy"])
        ),
        "heldout_mean_reward_delta": (
            float(after["mean_reward"]) - float(before["mean_reward"])
        ),
        "heldout_parse_rate_delta": (
            float(after["parse_rate"]) - float(before["parse_rate"])
        ),
        "heldout_reward_given_parsed_delta": (
            float(after["mean_reward_given_parsed"])
            - float(before["mean_reward_given_parsed"])
        ),
        "training": training,
        "train_results": list(train_results),
        "backend_attempts": list(backend_attempts),
        "usage": _condition_usage(before, after, training),
        "wall_s": wall_s,
    }


def _condition_usage(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    training: Mapping[str, Any] | None,
) -> dict[str, float]:
    phases = [before, after]
    if training is not None:
        phases.append(training)
    return {
        "requests": sum(float(phase.get("requests", 0.0) or 0.0) for phase in phases),
        "total_tokens": sum(
            float(phase.get("total_tokens", 0.0) or 0.0) for phase in phases
        ),
        "estimated_api_usd": sum(
            float(phase.get("estimated_api_usd", 0.0) or 0.0)
            for phase in phases
        ),
    }


def _condition_args_with_telemetry(
    args: argparse.Namespace,
    *,
    telemetry: TelemetryLedger,
    condition: str,
    seed: int,
    task_strata: Mapping[str, str],
) -> argparse.Namespace:
    condition_args = argparse.Namespace(**vars(args))

    def completion_observer(record: CompletionRecord) -> None:
        telemetry.record_inference(
            condition=condition,
            seed=seed,
            phase=record.split,
            task_id=record.task.id,
            stratum=task_strata.get(record.task.id),
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            latency_s=record.elapsed_s,
            attempts=record.attempts,
            reward=record.reward,
            exact=record.exact,
            parsed=record.parsed_answer is not None,
        )

    def training_attempt_observer(entry: Mapping[str, Any]) -> None:
        if entry.get("event") == "started":
            telemetry.emit(
                "training_attempt_started",
                dimensions={
                    "condition": condition,
                    "seed": seed,
                    "attempt": int(entry.get("attempt", 0) or 0),
                    "initial_step": int(entry.get("initial_step", 0) or 0),
                },
            )
            return
        metrics = entry.get("metrics")
        trainer_metrics = metrics if isinstance(metrics, Mapping) else {}
        reported_trainer_usd = trainer_metrics.get("cost/trainer_usd")
        telemetry.record_training_attempt(
            condition=condition,
            seed=seed,
            attempt=int(entry.get("attempt", 0) or 0),
            status=str(entry.get("status", "unknown")),
            duration_s=float(entry.get("wall_s", 0.0) or 0.0),
            initial_step=int(entry.get("initial_step", 0) or 0),
            observed_step=int(
                entry.get("observed_step", entry.get("step", 0)) or 0
            ),
            trainer_metrics=trainer_metrics,
            reported_trainer_usd=(
                float(reported_trainer_usd)
                if isinstance(reported_trainer_usd, (int, float))
                else None
            ),
            reported_cost_provenance="reported_by_trainer",
            error_type=(
                str(entry["error_type"])
                if entry.get("error_type") is not None
                else None
            ),
            error=(str(entry["error"]) if entry.get("error") is not None else None),
            artifact_name=(
                str(entry["artifact_name"])
                if entry.get("artifact_name") is not None
                else None
            ),
        )

    def scheduler_decision_observer(entry: Mapping[str, Any]) -> None:
        metadata = entry.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        estimated_cost = _first_finite_non_negative(
            metadata,
            (
                "scheduler/decision/estimated_rollout_dollar_seconds",
                "scheduler/decision/expected_rollout_dollar_seconds",
                "scheduler/decision/reserved_rollout_dollar_seconds",
            ),
        )
        telemetry.record_scheduler_decision(
            condition=condition,
            seed=seed,
            train_step=int(entry["train_step"]),
            group_index=int(entry["group_index"]),
            selected_stratum=str(entry["selected_stratum"]),
            task_id=str(entry["task_id"]),
            estimated_cost=estimated_cost,
            cost_provenance="scheduler_dollar_seconds_proxy",
            attributes=_scheduler_telemetry_attributes(metadata),
        )

    condition_args.completion_observer = completion_observer
    condition_args.training_attempt_observer = training_attempt_observer
    condition_args.scheduler_decision_observer = scheduler_decision_observer
    return condition_args


def _first_finite_non_negative(
    values: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)):
            converted = float(value)
            if math.isfinite(converted) and converted >= 0.0:
                return converted
    return None


def _scheduler_telemetry_attributes(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metadata.items()
        if str(key).startswith(("scheduler/decision/", "scheduler/control/"))
        and (
            value is None
            or isinstance(value, (str, int, float, bool))
        )
        and not (isinstance(value, float) and not math.isfinite(value))
    }


async def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for the controlled ablation")
    import art
    from art.serverless.backend import ServerlessBackend

    train, heldout = build_manifest(
        seed=args.manifest_seed,
        train_tasks=args.train_tasks,
        heldout_tasks=args.heldout_tasks,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output = args.output or Path("artifacts") / f"controlled_art_ablation_{run_id}.json"
    telemetry_path = args.telemetry_path or output.with_suffix(".telemetry.jsonl")
    telemetry = TelemetryLedger(
        run_id=run_id,
        pricing=_telemetry_pricing(args),
        path=telemetry_path,
        echo_to_stderr=args.telemetry_echo,
    )
    minimum_requests, maximum_requests = _inference_request_bounds(args)
    expected_requests = (
        minimum_requests if minimum_requests == maximum_requests else None
    )
    task_strata = {
        item.task.id: item.stratum for item in (*train, *heldout)
    }
    telemetry.emit(
        "run_started",
        attributes={
            "base_model": args.base_model,
            "conditions": list(CONDITIONS),
            "manifest_fingerprint": manifest_fingerprint(train, heldout),
            "minimum_inference_requests": minimum_requests,
            "maximum_inference_requests": maximum_requests,
            "expected_condition_runs": len(args.seed_values) * len(CONDITIONS),
            "expected_training_updates": (
                len(args.seed_values) * 2 * args.train_steps
            ),
            "expected_scheduler_decisions": (
                len(args.seed_values) * args.train_tasks
            ),
        },
    )
    report: dict[str, Any] = {
        "ok": False,
        "status": "running",
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "conditions": list(CONDITIONS),
        "excluded_conditions": {"calm": CALM_EXCLUSION},
        "config": {
            "base_model": args.base_model,
            "project": args.project,
            "seeds": list(args.seed_values),
            "manifest_seed": args.manifest_seed,
            "train_tasks": args.train_tasks,
            "heldout_tasks": args.heldout_tasks,
            "train_steps": args.train_steps,
            "groups_per_step": args.groups_per_step,
            "rollouts_per_group": args.rollouts_per_group,
            "max_rollouts_per_group": args.max_rollouts_per_group,
            "training_timeout_seconds": args.training_timeout_seconds,
            "learning_rate": args.learning_rate,
            "pricing": _pricing_report(args),
            "scheduler_cost_proxy": (
                "estimated API USD when supplied; otherwise total tokens / 1e6"
            ),
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        },
        "manifest_fingerprint": manifest_fingerprint(train, heldout),
        "manifest": {
            "train": [_manifest_row(item) for item in train],
            "heldout": [_manifest_row(item) for item in heldout],
        },
        "runs": {},
        "report_path": str(output.resolve()),
        "telemetry_path": str(telemetry_path.resolve()),
        "telemetry": telemetry.summary(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    condition_runners = {
        "base": _run_base_condition,
        "direct_art": _run_direct_condition,
        "async_scheduler": _run_scheduler_condition,
    }
    for seed in args.seed_values:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.request_seed = seed
        seed_key = str(seed)
        report["runs"][seed_key] = {}
        for condition in CONDITIONS:
            condition_args = _condition_args_with_telemetry(
                seed_args,
                telemetry=telemetry,
                condition=condition,
                seed=seed,
                task_strata=task_strata,
            )
            telemetry.emit(
                "condition_started",
                dimensions={"condition": condition, "seed": seed},
            )
            print(
                f"starting condition={condition} seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            started = time.perf_counter()
            try:
                kwargs = {
                    "art": art,
                    "serverless_backend_cls": ServerlessBackend,
                    "heldout": heldout,
                    "seed": seed,
                    "run_id": run_id,
                    "args": condition_args,
                }
                if condition != "base":
                    kwargs["train"] = train
                condition_report = await condition_runners[condition](**kwargs)
            except Exception as exc:
                condition_report = {
                    "status": "failed",
                    "condition": condition,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_s": time.perf_counter() - started,
                }
            telemetry.record_condition_finished(
                condition=condition,
                seed=seed,
                status=str(condition_report["status"]),
                wall_s=float(condition_report["wall_s"]),
                error_type=(
                    str(condition_report["error_type"])
                    if condition_report.get("error_type") is not None
                    else None
                ),
                error=(
                    str(condition_report["error"])
                    if condition_report.get("error") is not None
                    else None
                ),
            )
            report["runs"][seed_key][condition] = condition_report
            report["aggregate"] = aggregate_results(report["runs"])
            report["telemetry"] = telemetry.summary()
            _write_json(output, report)
            print(
                f"finished condition={condition} seed={seed} "
                f"status={condition_report['status']}",
                file=sys.stderr,
                flush=True,
            )
    report["aggregate"] = aggregate_results(report["runs"])
    conditions_ok = all(
        condition.get("status") == "completed"
        for seed_runs in report["runs"].values()
        for condition in seed_runs.values()
    )
    telemetry.emit(
        "run_finished",
        metrics={"conditions_ok": conditions_ok},
        attributes={"condition_runs": len(args.seed_values) * len(CONDITIONS)},
    )
    report["telemetry"] = telemetry.summary(
        expected_inference_requests=expected_requests
    )
    report["ok"] = conditions_ok and bool(report["telemetry"]["healthy"])
    if report["ok"]:
        report["status"] = "completed"
    elif conditions_ok:
        report["status"] = "completed_with_telemetry_errors"
    else:
        report["status"] = "completed_with_failures"
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(output, report)
    return report


def aggregate_results(runs: Mapping[str, Any]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    total_requests = 0.0
    total_tokens = 0.0
    for condition in CONDITIONS:
        completed = [
            seed_runs[condition]
            for seed_runs in runs.values()
            if condition in seed_runs
            and seed_runs[condition].get("status") == "completed"
        ]
        exact_deltas = [
            float(result["heldout_exact_accuracy_delta"]) for result in completed
        ]
        reward_deltas = [
            float(result["heldout_mean_reward_delta"]) for result in completed
        ]
        parse_rate_deltas = [
            float(result["heldout_parse_rate_delta"]) for result in completed
        ]
        parsed_reward_deltas = [
            float(result["heldout_reward_given_parsed_delta"])
            for result in completed
        ]
        condition_requests = sum(
            float(result["usage"]["requests"]) for result in completed
        )
        condition_tokens = sum(
            float(result["usage"]["total_tokens"]) for result in completed
        )
        total_requests += condition_requests
        total_tokens += condition_tokens
        aggregate[condition] = {
            "completed_seeds": len(completed),
            "exact_accuracy_delta": summarize_samples(exact_deltas),
            "mean_reward_delta": summarize_samples(reward_deltas),
            "parse_rate_delta": summarize_samples(parse_rate_deltas),
            "reward_given_parsed_delta": summarize_samples(parsed_reward_deltas),
            "requests": condition_requests,
            "total_tokens": condition_tokens,
            "checkpoint_advancements": sum(
                bool(result["checkpoint_advanced"]) for result in completed
            ),
        }
        if condition == "async_scheduler":
            allocations: Counter[str] = Counter()
            for result in completed:
                training = result.get("training")
                if isinstance(training, Mapping):
                    allocations.update(training.get("scheduler_group_allocations", {}))
            aggregate[condition]["scheduler_group_allocations"] = dict(allocations)
    aggregate["comparisons"] = {
        "direct_art_minus_base": _paired_comparison(
            runs, "direct_art", "base"
        ),
        "async_scheduler_minus_base": _paired_comparison(
            runs, "async_scheduler", "base"
        ),
        "async_scheduler_minus_direct_art": _paired_comparison(
            runs, "async_scheduler", "direct_art"
        ),
    }
    aggregate["total_requests"] = total_requests
    aggregate["total_tokens"] = total_tokens
    aggregate["interpretation"] = (
        "Condition deltas are paired within each model run. With three seeds, "
        "confidence intervals are descriptive and not publication-grade."
    )
    return aggregate


def _paired_comparison(
    runs: Mapping[str, Any],
    treatment: str,
    reference: str,
) -> dict[str, Any]:
    paired = [
        (seed_runs[treatment], seed_runs[reference])
        for seed_runs in runs.values()
        if treatment in seed_runs
        and reference in seed_runs
        and seed_runs[treatment].get("status") == "completed"
        and seed_runs[reference].get("status") == "completed"
    ]
    return {
        "paired_seeds": len(paired),
        "mean_reward_delta_difference": summarize_samples(
            [
                float(left["heldout_mean_reward_delta"])
                - float(right["heldout_mean_reward_delta"])
                for left, right in paired
            ]
        ),
        "parse_rate_delta_difference": summarize_samples(
            [
                float(left["heldout_parse_rate_delta"])
                - float(right["heldout_parse_rate_delta"])
                for left, right in paired
            ]
        ),
    }


def summarize_samples(values: Sequence[float]) -> dict[str, float | int | None]:
    count = len(values)
    if count == 0:
        return {
            "n": 0,
            "mean": None,
            "stddev": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = statistics.fmean(values)
    if count == 1:
        return {
            "n": 1,
            "mean": mean,
            "stddev": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    stddev = statistics.stdev(values)
    critical = T_CRITICAL_95.get(count - 1, 1.96)
    half_width = critical * stddev / math.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "stddev": stddev,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _manifest_row(item: AblationTask) -> dict[str, Any]:
    return {"stratum": item.stratum, **asdict(item.task), "answer": item.task.answer}


def _retryable_train_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "heartbeat timeout" in message or "temporarily unavailable" in message


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_result(report: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = report.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    return {
        "ok": report.get("ok"),
        "status": report.get("status"),
        "run_id": report.get("run_id"),
        "aggregate": report.get("aggregate"),
        "excluded_conditions": report.get("excluded_conditions"),
        "report_path": report.get("report_path"),
        "telemetry_path": report.get("telemetry_path"),
        "telemetry": {
            "healthy": telemetry.get("healthy"),
            "coverage": telemetry.get("coverage"),
            "cost_performance": telemetry.get("cost_performance"),
            "alerts": telemetry.get("alerts"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env_path is not None:
        from calm_puffer_art import load_env_file

        load_env_file(args.env_path)
    if args.preflight:
        result = preflight(args)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
        return 0
    try:
        result = asyncio.run(run_ablation(args))
    except Exception as exc:
        error = {
            "ok": False,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error, indent=2, sort_keys=True) if args.json else error)
        return 1
    payload = _compact_result(result)
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else payload)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
