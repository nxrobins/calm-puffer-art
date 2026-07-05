from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from calm_puffer_art.foundry_codegen import (
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
    DEFAULT_FOUNDRY_TASK_SPLIT,
    DEFAULT_FOUNDRY_TRAIN_STEPS,
    DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES,
    DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S,
    FOUNDRY_CONDITIONS,
    FOUNDRY_PROMPT_CONTEXT_POLICIES,
    AzureFoundryCodegenConfig,
    run_azure_foundry_budget_race,
    run_azure_foundry_codegen_ablation,
)


class RunTimeoutError(RuntimeError):
    pass


@dataclass
class _TelemetrySink:
    path: Path | None = None
    echo_to_stderr: bool = False
    started_at: float = time.perf_counter()

    def elapsed_s(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "elapsed_s": round(self.elapsed_s(), 6),
            "timestamp_unix_s": round(time.time(), 6),
            **fields,
        }
        line = json.dumps(payload, sort_keys=True)
        if self.echo_to_stderr:
            print(line, file=sys.stderr, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


async def _run(
    *,
    env_path: Path,
    deployment: str,
    max_train_steps: int,
    task_limit: int,
    task_split: str,
    prompt_context_policy: str,
    conditions: tuple[str, ...],
    model_call_budget: int,
    max_completion_tokens: int,
    request_timeout_s: float,
    verify_timeout_s: float,
    request_dollar_seconds: float,
    action_unit_dollar_seconds: float,
    verify_memory_limit_bytes: int,
    budget_race: bool,
    budget_dollar_seconds: float,
) -> dict[str, Any]:
    config = AzureFoundryCodegenConfig(
        env_path=env_path,
        deployment=deployment,
        max_train_steps=max_train_steps,
        task_limit=task_limit,
        task_split=task_split,
        prompt_context_policy=prompt_context_policy,
        model_call_budget=model_call_budget,
        max_completion_tokens=max_completion_tokens,
        request_timeout_s=request_timeout_s,
        verify_timeout_s=verify_timeout_s,
        request_dollar_seconds=request_dollar_seconds,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
        verify_memory_limit_bytes=verify_memory_limit_bytes,
    )
    if budget_race:
        return await run_azure_foundry_budget_race(
            config=config,
            budget_dollar_seconds=budget_dollar_seconds,
            conditions=conditions,
        )
    return await run_azure_foundry_codegen_ablation(
        config=config,
        conditions=conditions,
    )


async def _run_with_watchdog(
    coro_factory: Callable[[], Awaitable[dict[str, Any]]],
    *,
    telemetry: _TelemetrySink,
    run_timeout_s: float,
    heartbeat_interval_s: float,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    telemetry.emit("run_started", **run_metadata)
    task = asyncio.create_task(coro_factory())
    heartbeat_task: asyncio.Task[None] | None = None
    if heartbeat_interval_s > 0.0:
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(
                task,
                telemetry=telemetry,
                interval_s=heartbeat_interval_s,
                run_timeout_s=run_timeout_s,
                run_metadata=run_metadata,
            )
        )
    try:
        if run_timeout_s > 0.0:
            payload = await asyncio.wait_for(task, timeout=run_timeout_s)
        else:
            payload = await task
    except asyncio.TimeoutError as exc:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        telemetry.emit(
            "run_timeout",
            timeout_s=run_timeout_s,
            **run_metadata,
        )
        raise RunTimeoutError(f"foundry_run_timeout_exceeded_s={run_timeout_s}") from exc
    except Exception as exc:
        telemetry.emit(
            "run_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            **run_metadata,
        )
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    telemetry.emit(
        "run_completed",
        measurement=str(payload.get("measurement", "")),
        ok=bool(payload.get("ok")),
        **run_metadata,
    )
    return payload


async def _heartbeat_loop(
    task: asyncio.Task[dict[str, Any]],
    *,
    telemetry: _TelemetrySink,
    interval_s: float,
    run_timeout_s: float,
    run_metadata: dict[str, Any],
) -> None:
    while not task.done():
        await asyncio.sleep(interval_s)
        if not task.done():
            telemetry.emit(
                "run_heartbeat",
                timeout_s=run_timeout_s,
                **run_metadata,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a live Azure Foundry Python repair ablation comparing static "
            "ART, scheduler-only token control, and scheduler plus adaptive "
            "semantic action bandwidth."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=DEFAULT_FOUNDRY_ENV_PATH,
        help="Path to a dotenv file containing AZURE_OPENAI_* keys.",
    )
    parser.add_argument(
        "--deployment",
        default=DEFAULT_FOUNDRY_DEPLOYMENT,
        help="Azure Foundry deployment name.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_FOUNDRY_TRAIN_STEPS,
        help="Maximum train steps per condition.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=DEFAULT_FOUNDRY_TASK_LIMIT,
        help="Number of embedded repair tasks to include.",
    )
    parser.add_argument(
        "--task-split",
        default=DEFAULT_FOUNDRY_TASK_SPLIT,
        help=(
            "Embedded repair task split: standard, standard_heldout, hard, "
            "hard_heldout, mixed_heldout, frontier_smoke, frontier_balanced, "
            "frontier_hard, or frontier_full."
        ),
    )
    parser.add_argument(
        "--prompt-context-policy",
        default=DEFAULT_FOUNDRY_PROMPT_CONTEXT_POLICY,
        choices=FOUNDRY_PROMPT_CONTEXT_POLICIES,
        help="Prompt context policy for live Foundry repair requests.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(FOUNDRY_CONDITIONS),
        choices=FOUNDRY_CONDITIONS,
        help="Condition preset(s) to execute.",
    )
    parser.add_argument(
        "--model-call-budget",
        type=int,
        default=DEFAULT_FOUNDRY_MODEL_CALL_BUDGET,
        help="Hard model-call budget per condition.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens per Foundry call.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_FOUNDRY_REQUEST_TIMEOUT_S,
        help="Per-request Azure Foundry timeout in seconds.",
    )
    parser.add_argument(
        "--verify-timeout-s",
        type=float,
        default=DEFAULT_FOUNDRY_VERIFY_TIMEOUT_S,
        help="Per-candidate verifier timeout in seconds.",
    )
    parser.add_argument(
        "--request-dollar-seconds",
        type=float,
        default=DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS,
        help="Cost proxy charged for each live Foundry request.",
    )
    parser.add_argument(
        "--action-unit-dollar-seconds",
        type=float,
        default=DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS,
        help="Cost proxy charged for each emitted action unit.",
    )
    parser.add_argument(
        "--verify-memory-limit-mib",
        type=int,
        default=DEFAULT_FOUNDRY_VERIFY_MEMORY_LIMIT_BYTES // (1024 * 1024),
        help="Per-candidate verifier memory limit in MiB.",
    )
    parser.add_argument(
        "--budget-race",
        action="store_true",
        help="Run fixed accounted-dollar budget race instead of train-step ablation.",
    )
    parser.add_argument(
        "--budget-dollar-seconds",
        type=float,
        default=DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS,
        help="Accounted dollar-second ceiling for --budget-race.",
    )
    parser.add_argument(
        "--run-timeout-s",
        type=float,
        default=0.0,
        help="Overall experiment wall-clock timeout in seconds; 0 disables it.",
    )
    parser.add_argument(
        "--heartbeat-interval-s",
        type=float,
        default=0.0,
        help="Emit JSONL heartbeat events to stderr every N seconds; 0 disables it.",
    )
    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=None,
        help="Optional JSONL file for run lifecycle telemetry.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_steps < 1:
        raise SystemExit("--train-steps values must be positive")
    if args.task_limit < 1:
        raise SystemExit("--task-limit values must be positive")
    if args.model_call_budget < 0:
        raise SystemExit("--model-call-budget must be non-negative")
    if args.max_completion_tokens < 1:
        raise SystemExit("--max-completion-tokens values must be positive")
    if args.request_timeout_s <= 0.0:
        raise SystemExit("--request-timeout-s must be positive")
    if args.verify_timeout_s <= 0.0:
        raise SystemExit("--verify-timeout-s must be positive")
    if args.request_dollar_seconds < 0.0:
        raise SystemExit("--request-dollar-seconds must be non-negative")
    if args.action_unit_dollar_seconds < 0.0:
        raise SystemExit("--action-unit-dollar-seconds must be non-negative")
    if args.verify_memory_limit_mib < 1:
        raise SystemExit("--verify-memory-limit-mib must be positive")
    if args.budget_dollar_seconds <= 0.0:
        raise SystemExit("--budget-dollar-seconds must be positive")
    if args.run_timeout_s < 0.0:
        raise SystemExit("--run-timeout-s must be non-negative")
    if args.heartbeat_interval_s < 0.0:
        raise SystemExit("--heartbeat-interval-s must be non-negative")


def _error_payload(exc: Exception, *, telemetry: _TelemetrySink) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "elapsed_s": telemetry.elapsed_s(),
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    telemetry = _TelemetrySink(
        path=args.telemetry_path,
        echo_to_stderr=args.heartbeat_interval_s > 0.0,
    )
    run_metadata = {
        "budget_race": bool(args.budget_race),
        "conditions": list(args.conditions),
        "deployment": args.deployment,
        "model_call_budget": args.model_call_budget,
        "prompt_context_policy": args.prompt_context_policy,
        "task_limit": args.task_limit,
        "task_split": args.task_split,
        "train_steps": args.train_steps,
    }
    try:
        payload = asyncio.run(
            _run_with_watchdog(
                lambda: _run(
                    env_path=args.env_path,
                    deployment=args.deployment,
                    max_train_steps=args.train_steps,
                    task_limit=args.task_limit,
                    task_split=args.task_split,
                    prompt_context_policy=args.prompt_context_policy,
                    conditions=tuple(args.conditions),
                    model_call_budget=args.model_call_budget,
                    max_completion_tokens=args.max_completion_tokens,
                    request_timeout_s=args.request_timeout_s,
                    verify_timeout_s=args.verify_timeout_s,
                    request_dollar_seconds=args.request_dollar_seconds,
                    action_unit_dollar_seconds=args.action_unit_dollar_seconds,
                    verify_memory_limit_bytes=args.verify_memory_limit_mib
                    * 1024
                    * 1024,
                    budget_race=args.budget_race,
                    budget_dollar_seconds=args.budget_dollar_seconds,
                ),
                telemetry=telemetry,
                run_timeout_s=args.run_timeout_s,
                heartbeat_interval_s=args.heartbeat_interval_s,
                run_metadata=run_metadata,
            )
        )
    except Exception as exc:
        if args.json:
            print(json.dumps(_error_payload(exc, telemetry=telemetry), sort_keys=True))
            raise SystemExit(1) from None
        raise SystemExit(str(exc)) from None
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
