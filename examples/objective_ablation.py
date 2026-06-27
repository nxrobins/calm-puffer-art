from __future__ import annotations

import asyncio
import json

from calm_puffer_art.objective_ablation import (
    run_ablation,
    run_action_space_ablation,
    run_art_runtime_benchmark,
    run_art_bridge_ablation,
    run_closed_loop_ablation,
)


async def main() -> None:
    print(
        json.dumps(
            {
                "scheduler_control": await run_ablation(),
                "action_space_control": await run_action_space_ablation(),
                "closed_loop_control": await run_closed_loop_ablation(),
                "art_bridge_control": await run_art_bridge_ablation(),
                "art_runtime_benchmark": await run_art_runtime_benchmark(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
