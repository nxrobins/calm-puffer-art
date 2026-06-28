from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from calm_puffer_art.codegen_ablation import (
    DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
    DEFAULT_CODEGEN_RESPONSE_STYLES,
    DEFAULT_CODEGEN_TRAIN_STEPS,
    run_codegen_semantic_sweep,
)


async def _run(
    *,
    response_styles: tuple[int, ...] = DEFAULT_CODEGEN_RESPONSE_STYLES,
    max_train_steps: int = DEFAULT_CODEGEN_TRAIN_STEPS,
    sweep_repeats: int = 1,
    action_unit_dollar_seconds: float = DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
) -> dict[str, Any]:
    return await run_codegen_semantic_sweep(
        response_styles=response_styles,
        max_train_steps=max_train_steps,
        repeats=sweep_repeats,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the tiny unit-tested codegen semantic-bandwidth sweep for "
            "fixed token/chunk-2/chunk-3/chunk-4 action codecs."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--response-styles",
        default=",".join(str(style) for style in DEFAULT_CODEGEN_RESPONSE_STYLES),
        help="Comma-separated natural code verbosity styles to sweep.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_CODEGEN_TRAIN_STEPS,
        help="Maximum train steps per codegen sweep point.",
    )
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=1,
        help="Deterministic repeats per sweep point.",
    )
    parser.add_argument(
        "--action-unit-dollar-seconds",
        type=float,
        default=DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
        help="Explicit cost charged for each emitted action unit.",
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
        response_styles = _parse_int_csv(args.response_styles, "--response-styles")
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.train_steps < 1:
        raise SystemExit("--train-steps values must be positive")
    if args.sweep_repeats < 1:
        raise SystemExit("--sweep-repeats values must be positive")
    if args.action_unit_dollar_seconds < 0.0:
        raise SystemExit("--action-unit-dollar-seconds must be non-negative")
    payload = asyncio.run(
        _run(
            response_styles=response_styles,
            max_train_steps=args.train_steps,
            sweep_repeats=args.sweep_repeats,
            action_unit_dollar_seconds=args.action_unit_dollar_seconds,
        )
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
