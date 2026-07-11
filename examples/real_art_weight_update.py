from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an exact integer arithmetic agent. Work carefully, then return "
        "one line in the form FINAL=<integer>."
    ),
}
FINAL_PATTERN = re.compile(r"FINAL\s*=\s*(-?\d+)", re.IGNORECASE)


class ProofRunError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ChecksumTask:
    id: str
    seed: int
    multiplier: int
    addend: int
    square_offset: int
    final_multiplier: int
    final_addend: int
    modulus: int

    @property
    def answer(self) -> int:
        first = (self.seed * self.multiplier + self.addend) % self.modulus
        second = (first * first + self.square_offset) % self.modulus
        return (
            second * self.final_multiplier + self.final_addend
        ) % self.modulus

    @property
    def prompt(self) -> str:
        return (
            "Compute this checksum using integer arithmetic:\n"
            f"x1 = ({self.seed} * {self.multiplier} + {self.addend}) "
            f"mod {self.modulus}\n"
            f"x2 = (x1 * x1 + {self.square_offset}) mod {self.modulus}\n"
            f"x3 = (x2 * {self.final_multiplier} + {self.final_addend}) "
            f"mod {self.modulus}\n"
            "Return exactly FINAL=x3."
        )


@dataclass
class CompletionRecord:
    task: ChecksumTask
    split: str
    content: str
    parsed_answer: int | None
    reward: float
    exact: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_s: float
    estimated_api_usd: float
    attempts: int
    choice: Any

    def as_report(self) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "expected_answer": self.task.answer,
            "parsed_answer": self.parsed_answer,
            "exact": self.exact,
            "reward": self.reward,
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_s": self.elapsed_s,
            "estimated_api_usd": self.estimated_api_usd,
            "attempts": self.attempts,
        }


TRAIN_TASKS = (
    ChecksumTask("train-01", 437, 83, 191, 277, 41, 313, 1009),
    ChecksumTask("train-02", 829, 67, 359, 433, 53, 197, 1013),
    ChecksumTask("train-03", 613, 97, 271, 389, 47, 421, 1019),
    ChecksumTask("train-04", 947, 59, 463, 311, 61, 229, 1021),
    ChecksumTask("train-05", 751, 89, 337, 499, 43, 367, 1031),
    ChecksumTask("train-06", 883, 73, 409, 283, 71, 251, 1033),
)

HELDOUT_TASKS = (
    ChecksumTask("heldout-01", 541, 79, 223, 347, 37, 439, 1039),
    ChecksumTask("heldout-02", 907, 71, 383, 461, 59, 173, 1049),
    ChecksumTask("heldout-03", 677, 101, 293, 317, 43, 401, 1051),
    ChecksumTask("heldout-04", 967, 61, 449, 373, 67, 211, 1061),
    ChecksumTask("heldout-05", 787, 91, 331, 487, 47, 349, 1063),
    ChecksumTask("heldout-06", 919, 69, 397, 269, 73, 239, 1069),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real ART serverless RL update and compare a fixed held-out "
            "set before and after the checkpoint."
        )
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recover-report", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--env-path", type=Path)
    parser.add_argument("--project", default="calm-puffer-art-proof")
    parser.add_argument("--model-name")
    parser.add_argument("--entity")
    parser.add_argument(
        "--base-model",
        default="OpenPipe/Qwen3-14B-Instruct",
    )
    parser.add_argument("--train-task-limit", type=int, default=4)
    parser.add_argument("--heldout-task-limit", type=int, default=4)
    parser.add_argument("--rollouts-per-group", type=int, default=4)
    parser.add_argument("--max-rollouts-per-group", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--inference-retries", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--input-usd-per-million-tokens",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output-usd-per-million-tokens",
        type=float,
        default=0.0,
    )
    parser.add_argument("--trainer-usd-per-hour", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive_values = {
        "train-task-limit": args.train_task_limit,
        "heldout-task-limit": args.heldout_task_limit,
        "rollouts-per-group": args.rollouts_per_group,
        "max-rollouts-per-group": args.max_rollouts_per_group,
        "concurrency": args.concurrency,
        "max-tokens": args.max_tokens,
        "inference-retries": args.inference_retries,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.train_task_limit > len(TRAIN_TASKS):
        parser.error(f"--train-task-limit cannot exceed {len(TRAIN_TASKS)}")
    if args.heldout_task_limit > len(HELDOUT_TASKS):
        parser.error(f"--heldout-task-limit cannot exceed {len(HELDOUT_TASKS)}")
    if args.max_rollouts_per_group < args.rollouts_per_group:
        parser.error(
            "--max-rollouts-per-group must be >= --rollouts-per-group"
        )
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    for name in (
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "trainer_usd_per_hour",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")


def _parse_final(content: str) -> int | None:
    matches = FINAL_PATTERN.findall(content)
    return int(matches[-1]) if matches else None


def _reward(task: ChecksumTask, parsed_answer: int | None) -> float:
    if parsed_answer is None:
        return 0.0
    if parsed_answer == task.answer:
        return 1.0
    residue = parsed_answer % task.modulus
    direct = abs(residue - task.answer)
    distance = min(direct, task.modulus - direct)
    scale = max(1.0, task.modulus / 2.0)
    return max(0.0, 0.5 * (1.0 - distance / scale))


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    train_tasks = TRAIN_TASKS[: args.train_task_limit]
    heldout_tasks = HELDOUT_TASKS[: args.heldout_task_limit]
    train_ids = {task.id for task in train_tasks}
    heldout_ids = {task.id for task in heldout_tasks}
    dataset_valid = bool(
        train_ids and heldout_ids and train_ids.isdisjoint(heldout_ids)
    )
    art_installed = importlib.util.find_spec("art") is not None
    art_version = None
    if art_installed:
        try:
            art_version = importlib.metadata.version("openpipe-art")
        except importlib.metadata.PackageNotFoundError:
            pass
    credential_ready = bool(os.environ.get("WANDB_API_KEY"))
    return {
        "ok": dataset_valid,
        "mode": "preflight",
        "dataset_valid": dataset_valid,
        "train_tasks": len(train_tasks),
        "heldout_tasks": len(heldout_tasks),
        "minimum_planned_inference_requests": (
            len(train_tasks) * args.rollouts_per_group + 2 * len(heldout_tasks)
        ),
        "maximum_planned_inference_requests": (
            len(train_tasks) * args.max_rollouts_per_group
            + 2 * len(heldout_tasks)
        ),
        "art_installed": art_installed,
        "art_version": art_version,
        "credential_ready": credential_ready,
        "credential_name": "WANDB_API_KEY",
        "live_ready": dataset_valid and art_installed and credential_ready,
        "base_model": args.base_model,
        "pricing": _pricing_report(args),
    }


async def _complete(
    *,
    client: Any,
    inference_name: str,
    task: ChecksumTask,
    split: str,
    temperature: float,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> CompletionRecord:
    started = time.perf_counter()
    response = None
    attempts = 0
    for attempts in range(1, args.inference_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=inference_name,
                    messages=[
                        SYSTEM_MESSAGE,
                        {"role": "user", "content": task.prompt},
                    ],
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                )
            break
        except Exception:
            if attempts >= args.inference_retries:
                raise
            await asyncio.sleep(min(2 ** (attempts - 1), 8))
    if response is None or not response.choices:
        raise RuntimeError(f"inference returned no choices for {task.id}")
    elapsed_s = time.perf_counter() - started
    choice = response.choices[0]
    content = str(getattr(choice.message, "content", "") or "")
    parsed_answer = _parse_final(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or prompt_tokens + completion_tokens
    )
    estimated_api_usd = (
        prompt_tokens * args.input_usd_per_million_tokens
        + completion_tokens * args.output_usd_per_million_tokens
    ) / 1_000_000.0
    return CompletionRecord(
        task=task,
        split=split,
        content=content,
        parsed_answer=parsed_answer,
        reward=_reward(task, parsed_answer),
        exact=parsed_answer == task.answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        elapsed_s=elapsed_s,
        estimated_api_usd=estimated_api_usd,
        attempts=attempts,
        choice=choice,
    )


async def _evaluate(
    *,
    client: Any,
    inference_name: str,
    tasks: Sequence[ChecksumTask],
    split: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.perf_counter()
    records = await asyncio.gather(
        *(
            _complete(
                client=client,
                inference_name=inference_name,
                task=task,
                split=split,
                temperature=args.eval_temperature,
                args=args,
                semaphore=semaphore,
            )
            for task in tasks
        )
    )
    return _phase_report(records, wall_s=time.perf_counter() - started)


async def _sample_training_groups(
    *,
    art: Any,
    client: Any,
    inference_name: str,
    policy_step: int,
    tasks: Sequence[ChecksumTask],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> tuple[list[Any], dict[str, Any]]:
    async def sample_task(
        task: ChecksumTask,
    ) -> tuple[Any | None, list[CompletionRecord]]:
        records = list(
            await asyncio.gather(
                *(
                    _complete(
                        client=client,
                        inference_name=inference_name,
                        task=task,
                        split="train",
                        temperature=args.temperature,
                        args=args,
                        semaphore=semaphore,
                    )
                    for _ in range(args.rollouts_per_group)
                )
            )
        )
        while (
            len({record.reward for record in records}) < 2
            and len(records) < args.max_rollouts_per_group
        ):
            records.append(
                await _complete(
                    client=client,
                    inference_name=inference_name,
                    task=task,
                    split="train",
                    temperature=args.temperature,
                    args=args,
                    semaphore=semaphore,
                )
            )
        if len({record.reward for record in records}) < 2:
            return None, records
        trajectories = [
            art.Trajectory(
                messages_and_choices=[
                    SYSTEM_MESSAGE,
                    {"role": "user", "content": task.prompt},
                    record.choice,
                ],
                reward=record.reward,
                initial_policy_version=policy_step,
                final_policy_version=policy_step,
                metrics={
                    "rollout/dollar_seconds": record.estimated_api_usd,
                    "cost/api_usd": record.estimated_api_usd,
                    "duration": record.elapsed_s,
                    "usage/prompt_tokens": record.prompt_tokens,
                    "usage/completion_tokens": record.completion_tokens,
                },
                metadata={
                    "scenario_id": task.id,
                    "proof/split": "train",
                    "proof/expected_answer": task.answer,
                    "proof/parsed_answer": record.parsed_answer,
                    "proof/exact": record.exact,
                },
            )
            for record in records
        ]
        return (
            art.TrajectoryGroup(
                trajectories,
                metadata={"scenario_id": task.id, "proof/split": "train"},
            ),
            records,
        )

    started = time.perf_counter()
    sampled = await asyncio.gather(*(sample_task(task) for task in tasks))
    groups = [group for group, _ in sampled if group is not None]
    records = [record for _, task_records in sampled for record in task_records]
    report = _phase_report(records, wall_s=time.perf_counter() - started)
    report.update(
        {
            "requested_group_count": len(tasks),
            "nonzero_advantage_group_count": len(groups),
            "excluded_uniform_reward_groups": [
                task.id
                for task, (group, _) in zip(tasks, sampled, strict=True)
                if group is None
            ],
        }
    )
    return groups, report


def _phase_report(
    records: Sequence[CompletionRecord],
    *,
    wall_s: float,
) -> dict[str, Any]:
    count = len(records)
    return {
        "requests": count,
        "attempts": sum(record.attempts for record in records),
        "exact_count": sum(record.exact for record in records),
        "exact_accuracy": (
            sum(record.exact for record in records) / count if count else 0.0
        ),
        "mean_reward": (
            sum(record.reward for record in records) / count if count else 0.0
        ),
        "prompt_tokens": sum(record.prompt_tokens for record in records),
        "completion_tokens": sum(record.completion_tokens for record in records),
        "total_tokens": sum(record.total_tokens for record in records),
        "estimated_api_usd": sum(
            record.estimated_api_usd for record in records
        ),
        "wall_s": wall_s,
        "records": [record.as_report() for record in records],
    }


async def _run_live(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for a serverless ART run")

    import art
    from art.serverless.backend import ServerlessBackend

    from calm_puffer_art import (
        AsyncArtBackend,
        AsyncArtBackendConfig,
        ObjectiveScheduler,
        WeightBroadcastChannel,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    model_name = args.model_name or f"weight-proof-{timestamp}"
    delegate = ServerlessBackend()
    channel = WeightBroadcastChannel()
    updates = channel.subscribe()
    backend = AsyncArtBackend(
        backend=delegate,
        config=AsyncArtBackendConfig(
            train_queue_capacity=1,
            train_batch_groups=1,
            max_policy_lag=0,
            max_train_steps=1,
            cost_per_second_usd=args.trainer_usd_per_hour / 3600.0,
        ),
        scheduler=ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=args.train_task_limit,
            min_policy_lag=0,
            max_policy_lag=0,
            min_actor_count=1,
            max_actor_count=args.concurrency,
            exploration_bonus=0.0,
            control_exploration_bonus=0.0,
        ),
        weight_channel=channel,
    )
    model = art.TrainableModel(
        name=model_name,
        project=args.project,
        entity=args.entity,
        base_model=args.base_model,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    train_tasks = TRAIN_TASKS[: args.train_task_limit]
    heldout_tasks = HELDOUT_TASKS[: args.heldout_task_limit]
    registered_at = time.perf_counter()
    trainer_started: float | None = None
    report: dict[str, Any] = {
        "ok": False,
        "status": "starting",
        "mode": "serverless",
        "started_at_utc": timestamp,
        "model": {
            "name": model_name,
            "entity": args.entity,
            "project": args.project,
            "base_model": args.base_model,
        },
        "pricing": _pricing_report(args),
    }

    try:
        await model.register(backend)
        registration_wall_s = time.perf_counter() - registered_at
        initial_step = await backend._get_step(model)
        report.update(
            {
                "status": "registered",
                "registration_wall_s": registration_wall_s,
                "initial_step": initial_step,
                "model": {
                    "name": model.name,
                    "id": model.id,
                    "entity": model.entity,
                    "project": model.project,
                    "base_model": model.base_model,
                },
            }
        )
        client = model.openai_client()
        initial_inference_name = model.get_inference_name()
        before = await _evaluate(
            client=client,
            inference_name=initial_inference_name,
            tasks=heldout_tasks,
            split="heldout_before",
            args=args,
            semaphore=semaphore,
        )
        report["heldout_before"] = before
        groups, training_sampling = await _sample_training_groups(
            art=art,
            client=client,
            inference_name=initial_inference_name,
            policy_step=initial_step,
            tasks=train_tasks,
            args=args,
            semaphore=semaphore,
        )
        report["training_sampling"] = training_sampling
        report["claims"] = {
            "checkpoint_advanced": False,
            "artifact_identified": False,
            "verified_nonzero_advantage_groups": len(groups),
            "weight_update_evidence": False,
            "heldout_improved": False,
        }
        if not groups:
            raise RuntimeError(
                "all training groups had uniform reward; no ART update was "
                "submitted because the batch had no verified advantage signal"
            )

        trainer_started = time.perf_counter()
        train_result = await backend.train(
            model,
            groups,
            learning_rate=args.learning_rate,
        )
        trainer_wall_s = time.perf_counter() - trainer_started
        final_step = int(getattr(train_result, "step", initial_step))
        artifact_name = getattr(train_result, "artifact_name", None)
        checkpoint_advanced = final_step > initial_step
        trainer_estimated_usd = (
            trainer_wall_s * args.trainer_usd_per_hour / 3600.0
        )
        report.update(
            {
                "status": "trained",
                "final_step": final_step,
                "artifact_name": artifact_name,
                "trainer": {
                    "wall_s": trainer_wall_s,
                    "estimated_usd": trainer_estimated_usd,
                    "metrics": _float_mapping(
                        getattr(train_result, "metrics", {})
                    ),
                },
                "claims": {
                    "checkpoint_advanced": checkpoint_advanced,
                    "artifact_identified": bool(artifact_name),
                    "verified_nonzero_advantage_groups": len(groups),
                    "weight_update_evidence": (
                        checkpoint_advanced and bool(artifact_name) and bool(groups)
                    ),
                    "heldout_improved": False,
                },
            }
        )
        trainer_started = None
        final_inference_name = model.get_inference_name(step=final_step)
        after = await _evaluate(
            client=client,
            inference_name=final_inference_name,
            tasks=heldout_tasks,
            split="heldout_after",
            args=args,
            semaphore=semaphore,
        )
        report["heldout_after"] = after
        stats = backend.stats()
        published_update = None
        if not updates.empty():
            update = updates.get_nowait()
            published_update = {
                "step": update.step,
                "checkpoint_id": update.checkpoint_id,
                "art_step": update.metadata.get("art/step"),
                "artifact_name": update.metadata.get("art/artifact_name"),
            }
        heldout_delta = after["exact_accuracy"] - before["exact_accuracy"]
        estimated_api_usd = sum(
            phase["estimated_api_usd"]
            for phase in (before, training_sampling, after)
        )
        report.update(
            {
                "ok": checkpoint_advanced and bool(artifact_name),
                "status": "completed",
                "mode": "serverless",
                "started_at_utc": timestamp,
                "model": {
                    "name": model.name,
                    "id": model.id,
                    "entity": model.entity,
                    "project": model.project,
                    "base_model": model.base_model,
                },
                "registration_wall_s": registration_wall_s,
                "initial_step": initial_step,
                "final_step": final_step,
                "artifact_name": artifact_name,
                "initial_inference_name": initial_inference_name,
                "final_inference_name": final_inference_name,
                "training_sampling": training_sampling,
                "trainer": {
                    "wall_s": trainer_wall_s,
                    "estimated_usd": trainer_estimated_usd,
                    "metrics": _float_mapping(
                        getattr(train_result, "metrics", {})
                    ),
                },
                "heldout_before": before,
                "heldout_after": after,
                "heldout_exact_accuracy_delta": heldout_delta,
                "claims": {
                    "checkpoint_advanced": checkpoint_advanced,
                    "artifact_identified": bool(artifact_name),
                    "verified_nonzero_advantage_groups": len(groups),
                    "weight_update_evidence": (
                        checkpoint_advanced and bool(artifact_name) and bool(groups)
                    ),
                    "heldout_improved": heldout_delta > 0.0,
                },
                "pricing": _pricing_report(args),
                "costs": {
                    "estimated_api_usd": estimated_api_usd,
                    "estimated_trainer_usd": trainer_estimated_usd,
                    "estimated_total_usd": (
                        estimated_api_usd + trainer_estimated_usd
                    ),
                    "art_reported_cost_metrics": {
                        key: value
                        for key, value in _float_mapping(
                            getattr(train_result, "metrics", {})
                        ).items()
                        if "cost" in key or "dollar" in key
                    },
                },
                "published_update": published_update,
                "control_plane_metrics": stats,
            }
        )
        _write_report(report, args=args, model_name=model_name)
        return report
    except Exception as exc:
        trainer_wall_s = (
            time.perf_counter() - trainer_started
            if trainer_started is not None
            else 0.0
        )
        report.update(
            {
                "ok": False,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        try:
            report["control_plane_metrics"] = backend.stats()
        except Exception:
            pass
        _add_partial_costs(report, args=args, trainer_wall_s=trainer_wall_s)
        _write_report(report, args=args, model_name=model_name)
        raise ProofRunError(str(exc), report) from exc
    finally:
        await backend.close()


async def _recover_live(args: argparse.Namespace) -> dict[str, Any]:
    if args.recover_report is None:
        raise ValueError("--recover-report is required")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required to recover an ART run")

    import art
    from art.serverless.backend import ServerlessBackend

    report = json.loads(args.recover_report.read_text(encoding="utf-8"))
    model_info = report.get("model")
    if not isinstance(model_info, Mapping):
        raise ValueError("recovery report has no model metadata")
    before = report.get("heldout_before")
    if not isinstance(before, Mapping):
        raise ValueError("recovery report has no held-out baseline")
    records = before.get("records")
    if not isinstance(records, list):
        raise ValueError("recovery report has no held-out task records")
    tasks_by_id = {task.id: task for task in HELDOUT_TASKS}
    task_ids = [
        str(record.get("task_id"))
        for record in records
        if isinstance(record, Mapping)
    ]
    try:
        heldout_tasks = [tasks_by_id[task_id] for task_id in task_ids]
    except KeyError as exc:
        raise ValueError(f"unknown held-out task in recovery report: {exc}") from exc
    if not heldout_tasks:
        raise ValueError("recovery report selected no held-out tasks")

    backend = ServerlessBackend()
    model = art.TrainableModel(
        name=str(model_info["name"]),
        project=str(model_info["project"]),
        entity=(
            str(model_info["entity"])
            if model_info.get("entity") is not None
            else None
        ),
        base_model=str(model_info["base_model"]),
    )
    started = time.perf_counter()
    try:
        await model.register(backend)
        initial_step = int(report.get("initial_step", 0) or 0)
        final_step = int(await backend._get_step(model))
        if final_step <= initial_step:
            raise RuntimeError(
                "recovery found no checkpoint newer than the initial step"
            )
        final_inference_name = model.get_inference_name(step=final_step)
        after = await _evaluate(
            client=model.openai_client(),
            inference_name=final_inference_name,
            tasks=heldout_tasks,
            split="heldout_after_recovery",
            args=args,
            semaphore=asyncio.Semaphore(args.concurrency),
        )
        entity = str(model.entity)
        artifact_name = (
            f"{entity}/{model.project}/{model.name}:step{final_step}"
        )
        heldout_delta = (
            float(after["exact_accuracy"])
            - float(before.get("exact_accuracy", 0.0) or 0.0)
        )
        training_sampling = report.get("training_sampling", {})
        verified_groups = (
            int(training_sampling.get("nonzero_advantage_group_count", 0) or 0)
            if isinstance(training_sampling, Mapping)
            else 0
        )
        original_error = {
            "type": report.pop("error_type", None),
            "message": report.pop("error", None),
        }
        trainer = report.get("trainer")
        if not isinstance(trainer, dict):
            trainer = {}
            report["trainer"] = trainer
        trainer["checkpoint_recovered_after_client_error"] = True
        estimated_api_usd = sum(
            float(phase.get("estimated_api_usd", 0.0) or 0.0)
            for phase in (before, training_sampling, after)
            if isinstance(phase, Mapping)
        )
        trainer_estimated_usd = float(trainer.get("estimated_usd", 0.0) or 0.0)
        report.update(
            {
                "ok": True,
                "status": "recovered_completed",
                "final_step": final_step,
                "artifact_name": artifact_name,
                "final_inference_name": final_inference_name,
                "heldout_after": after,
                "heldout_exact_accuracy_delta": heldout_delta,
                "claims": {
                    "checkpoint_advanced": True,
                    "artifact_identified": True,
                    "verified_nonzero_advantage_groups": verified_groups,
                    "weight_update_evidence": verified_groups > 0,
                    "heldout_improved": heldout_delta > 0.0,
                },
                "pricing": _pricing_report(args),
                "costs": {
                    "estimated_api_usd": estimated_api_usd,
                    "estimated_trainer_usd": trainer_estimated_usd,
                    "estimated_total_usd": (
                        estimated_api_usd + trainer_estimated_usd
                    ),
                    "recovered_report": True,
                },
                "recovery": {
                    "original_client_error": original_error,
                    "checkpoint_verified_from_backend": True,
                    "control_plane_metrics_are_pre_recovery_failure_snapshot": True,
                    "wall_s": time.perf_counter() - started,
                },
            }
        )
        _write_report(report, args=args, model_name=model.name)
        return report
    finally:
        await backend.close()


def _float_mapping(values: Mapping[str, Any] | Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    result = {}
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[str(key)] = float(value)
    return result


def _add_partial_costs(
    report: dict[str, Any],
    *,
    args: argparse.Namespace,
    trainer_wall_s: float,
) -> None:
    estimated_api_usd = sum(
        float(phase.get("estimated_api_usd", 0.0) or 0.0)
        for key in ("heldout_before", "training_sampling", "heldout_after")
        if isinstance((phase := report.get(key)), Mapping)
    )
    trainer = report.get("trainer")
    if isinstance(trainer, Mapping):
        trainer_wall_s = max(
            trainer_wall_s,
            float(trainer.get("wall_s", 0.0) or 0.0),
        )
    trainer_estimated_usd = trainer_wall_s * args.trainer_usd_per_hour / 3600.0
    report["costs"] = {
        "estimated_api_usd": estimated_api_usd,
        "estimated_trainer_usd": trainer_estimated_usd,
        "estimated_total_usd": estimated_api_usd + trainer_estimated_usd,
        "partial_report": True,
    }
    if isinstance(trainer, dict):
        trainer.setdefault("wall_s", trainer_wall_s)
        trainer.setdefault("estimated_usd", trainer_estimated_usd)
    elif trainer_wall_s > 0.0:
        report["trainer"] = {
            "wall_s": trainer_wall_s,
            "estimated_usd": trainer_estimated_usd,
        }


def _write_report(
    report: dict[str, Any],
    *,
    args: argparse.Namespace,
    model_name: str,
) -> None:
    output = args.output or args.recover_report or (
        Path("artifacts") / f"real_art_weight_update_{model_name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(output.resolve())
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pricing_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_usd_per_million_tokens": args.input_usd_per_million_tokens,
        "output_usd_per_million_tokens": args.output_usd_per_million_tokens,
        "trainer_usd_per_hour": args.trainer_usd_per_hour,
        "monetary_costs_are_estimates": True,
        "zero_rate_means_no_monetary_rate_was_supplied": True,
    }


def _print_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if result.get("status") == "failed":
        print(f"ART proof failed: {result.get('error', 'unknown error')}")
        if result.get("report_path"):
            print(f"Partial report: {result['report_path']}")
        return
    if result.get("mode") == "preflight":
        print(
            "ART proof preflight: "
            f"dataset={result['dataset_valid']} "
            f"art={result['art_installed']} "
            f"credential={result['credential_ready']} "
            f"live_ready={result['live_ready']}"
        )
        return
    claims = result["claims"]
    print(
        "ART update proof: "
        f"step {result['initial_step']} -> {result['final_step']}; "
        f"artifact={claims['artifact_identified']}; "
        f"heldout_delta={result['heldout_exact_accuracy_delta']:+.3f}; "
        f"estimated_usd={result['costs']['estimated_total_usd']:.6f}"
    )
    print(f"Report: {result['report_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env_path is not None:
        try:
            from calm_puffer_art import load_env_file

            load_env_file(args.env_path)
        except Exception as exc:
            error = {
                "ok": False,
                "status": "failed",
                "mode": "preflight" if args.preflight else "serverless",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _print_result(error, as_json=args.json)
            return 1
    if args.preflight:
        result = _preflight(args)
        _print_result(result, as_json=args.json)
        return 0 if result["ok"] else 1
    if args.recover_report is not None:
        try:
            result = asyncio.run(_recover_live(args))
        except Exception as exc:
            error = {
                "ok": False,
                "status": "failed",
                "mode": "recovery",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _print_result(error, as_json=args.json)
            return 1
        _print_result(result, as_json=args.json)
        return 0 if result["ok"] else 1
    try:
        result = asyncio.run(_run_live(args))
    except ProofRunError as exc:
        _print_result(exc.report, as_json=args.json)
        return 1
    except Exception as exc:
        error = {
            "ok": False,
            "mode": "serverless",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _print_result(error, as_json=args.json)
        if not args.json:
            print(f"Error: {error['error']}", file=sys.stderr)
        return 1
    _print_result(result, as_json=args.json)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
