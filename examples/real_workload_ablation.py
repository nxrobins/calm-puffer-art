from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from calm_puffer_art.objective_ablation import (
    DEFAULT_REAL_SWEEP_RESPONSE_MULTIPLIERS,
    DEFAULT_REAL_SWEEP_TRAIN_STEPS,
    run_real_ablation,
    run_real_closed_loop_ablation,
    run_real_semantic_sweeps,
)


async def _run(
    *,
    include_sweeps: bool = False,
    sweep_only: bool = False,
    sweep_repeats: int = 1,
    budget_train_steps: tuple[int, ...] = DEFAULT_REAL_SWEEP_TRAIN_STEPS,
    response_multipliers: tuple[int, ...] = DEFAULT_REAL_SWEEP_RESPONSE_MULTIPLIERS,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "proof_scope": "tiny_torch_verifiable_math",
    }
    if not sweep_only:
        payload["scheduler_control"] = await run_real_ablation()
        payload["closed_loop_control"] = await run_real_closed_loop_ablation()
    if include_sweeps or sweep_only:
        sweeps = await run_real_semantic_sweeps(
            train_steps=budget_train_steps,
            response_multipliers=response_multipliers,
            repeats=sweep_repeats,
        )
        payload["semantic_sweeps"] = sweeps
        payload["semantic_break_even_train_steps"] = sweeps[
            "semantic_break_even_train_steps"
        ]
        payload["chunk4_recovers_at_response_tokens"] = sweeps[
            "chunk4_recovers_at_response_tokens"
        ]
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny trainable workload ablation comparing static rollout "
            "configuration against the objective scheduler."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--include-sweeps",
        action="store_true",
        help="Also run semantic budget and chunk-length sweeps.",
    )
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Run only the semantic budget and chunk-length sweeps.",
    )
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=1,
        help="Deterministic repeats per sweep grid point.",
    )
    parser.add_argument(
        "--budget-train-steps",
        default=",".join(str(step) for step in DEFAULT_REAL_SWEEP_TRAIN_STEPS),
        help="Comma-separated train-step budgets for the crossover sweep.",
    )
    parser.add_argument(
        "--response-multipliers",
        default=",".join(
            str(multiplier)
            for multiplier in DEFAULT_REAL_SWEEP_RESPONSE_MULTIPLIERS
        ),
        help="Comma-separated response length multipliers for the chunk-4 sweep.",
    )
    return parser.parse_args()


def _parse_int_csv(value: str, flag_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must contain integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{flag_name} must not be empty")
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError(f"{flag_name} values must be positive")
    return parsed


def main() -> None:
    args = _parse_args()
    try:
        budget_train_steps = _parse_int_csv(
            args.budget_train_steps,
            "--budget-train-steps",
        )
        response_multipliers = _parse_int_csv(
            args.response_multipliers,
            "--response-multipliers",
        )
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.sweep_repeats < 1:
        raise SystemExit("--sweep-repeats values must be positive")
    payload = asyncio.run(
        _run(
            include_sweeps=args.include_sweeps,
            sweep_only=args.sweep_only,
            sweep_repeats=args.sweep_repeats,
            budget_train_steps=budget_train_steps,
            response_multipliers=response_multipliers,
        )
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
