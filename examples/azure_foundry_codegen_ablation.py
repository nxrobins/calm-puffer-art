from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from calm_puffer_art.foundry_codegen import (
    DEFAULT_FOUNDRY_ACTION_UNIT_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_BUDGET_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_DEPLOYMENT,
    DEFAULT_FOUNDRY_ENV_PATH,
    DEFAULT_FOUNDRY_MAX_COMPLETION_TOKENS,
    DEFAULT_FOUNDRY_MODEL_CALL_BUDGET,
    DEFAULT_FOUNDRY_REQUEST_DOLLAR_SECONDS,
    DEFAULT_FOUNDRY_TASK_LIMIT,
    DEFAULT_FOUNDRY_TRAIN_STEPS,
    AzureFoundryCodegenConfig,
    run_azure_foundry_budget_race,
    run_azure_foundry_codegen_ablation,
)


async def _run(
    *,
    env_path: Path,
    deployment: str,
    max_train_steps: int,
    task_limit: int,
    model_call_budget: int,
    max_completion_tokens: int,
    request_dollar_seconds: float,
    action_unit_dollar_seconds: float,
    budget_race: bool,
    budget_dollar_seconds: float,
) -> dict[str, Any]:
    config = AzureFoundryCodegenConfig(
        env_path=env_path,
        deployment=deployment,
        max_train_steps=max_train_steps,
        task_limit=task_limit,
        model_call_budget=model_call_budget,
        max_completion_tokens=max_completion_tokens,
        request_dollar_seconds=request_dollar_seconds,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )
    if budget_race:
        return await run_azure_foundry_budget_race(
            config=config,
            budget_dollar_seconds=budget_dollar_seconds,
        )
    return await run_azure_foundry_codegen_ablation(config=config)


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
        help="Path to a dotenv file containing COVENANT_AZURE_* keys.",
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
    if args.request_dollar_seconds < 0.0:
        raise SystemExit("--request-dollar-seconds must be non-negative")
    if args.action_unit_dollar_seconds < 0.0:
        raise SystemExit("--action-unit-dollar-seconds must be non-negative")
    if args.budget_dollar_seconds <= 0.0:
        raise SystemExit("--budget-dollar-seconds must be positive")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    payload = asyncio.run(
        _run(
            env_path=args.env_path,
            deployment=args.deployment,
            max_train_steps=args.train_steps,
            task_limit=args.task_limit,
            model_call_budget=args.model_call_budget,
            max_completion_tokens=args.max_completion_tokens,
            request_dollar_seconds=args.request_dollar_seconds,
            action_unit_dollar_seconds=args.action_unit_dollar_seconds,
            budget_race=args.budget_race,
            budget_dollar_seconds=args.budget_dollar_seconds,
        )
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
