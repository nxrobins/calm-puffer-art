from __future__ import annotations

import asyncio
import json

from calm_puffer_art.objective_ablation import (
    run_ablation,
    run_action_space_ablation,
)


async def main() -> None:
    print(
        json.dumps(
            {
                "scheduler_control": await run_ablation(),
                "action_space_control": await run_action_space_ablation(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
