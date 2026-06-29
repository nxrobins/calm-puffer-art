from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from calm_puffer_art.codegen_ablation import (
    DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
    DEFAULT_CODEGEN_SHOWCASE_RESPONSE_STYLE,
    DEFAULT_CODEGEN_SHOWCASE_TRAIN_STEPS,
    run_python_codegen_showcase,
)


async def _run(
    *,
    max_train_steps: int = DEFAULT_CODEGEN_SHOWCASE_TRAIN_STEPS,
    response_style: int = DEFAULT_CODEGEN_SHOWCASE_RESPONSE_STYLE,
    action_unit_dollar_seconds: float = DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
) -> dict[str, Any]:
    return await run_python_codegen_showcase(
        max_train_steps=max_train_steps,
        response_style=response_style,
        action_unit_dollar_seconds=action_unit_dollar_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic Python function-synthesis showcase comparing "
            "static token ART, token scheduler control, and scheduler plus "
            "adaptive semantic action bandwidth."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_CODEGEN_SHOWCASE_TRAIN_STEPS,
        help="Maximum train steps per condition.",
    )
    parser.add_argument(
        "--response-style",
        type=int,
        default=DEFAULT_CODEGEN_SHOWCASE_RESPONSE_STYLE,
        help="Code verbosity style: 1 compact, 2 structured, 3+ expanded.",
    )
    parser.add_argument(
        "--action-unit-dollar-seconds",
        type=float,
        default=DEFAULT_CODEGEN_ACTION_UNIT_DOLLAR_SECONDS,
        help="Explicit cost charged for each emitted action unit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.train_steps < 1:
        raise SystemExit("--train-steps values must be positive")
    if args.response_style < 1:
        raise SystemExit("--response-style values must be positive")
    if args.action_unit_dollar_seconds < 0.0:
        raise SystemExit("--action-unit-dollar-seconds must be non-negative")
    payload = asyncio.run(
        _run(
            max_train_steps=args.train_steps,
            response_style=args.response_style,
            action_unit_dollar_seconds=args.action_unit_dollar_seconds,
        )
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
