from __future__ import annotations

import asyncio
import json

from calm_puffer_art.objective_ablation import run_ablation


async def main() -> None:
    print(json.dumps(await run_ablation(), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
