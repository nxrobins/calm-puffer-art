from __future__ import annotations

import json

from calm_puffer_art import run_scheduler_scalability_profile


if __name__ == "__main__":
    print(json.dumps(run_scheduler_scalability_profile(), indent=2, sort_keys=True))
