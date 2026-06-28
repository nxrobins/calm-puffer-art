from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from calm_puffer_art.objective_ablation import (
    run_real_ablation,
    run_real_closed_loop_ablation,
)


async def _run() -> dict[str, Any]:
    return {
        "proof_scope": "tiny_torch_verifiable_math",
        "scheduler_control": await run_real_ablation(),
        "closed_loop_control": await run_real_closed_loop_ablation(),
    }


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = asyncio.run(_run())
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
