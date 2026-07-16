from __future__ import annotations

import argparse
import json
from pathlib import Path

from calm_puffer_art.foundry_harness import (
    DEFAULT_FOUNDRY_RUNS_DIR,
    analyze_foundry_harness_runs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Foundry harness run artifacts for frontier diagnostics."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_FOUNDRY_RUNS_DIR,
        help="Directory containing Foundry harness run artifacts.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Only analyze run directories whose names start with this prefix.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = analyze_foundry_harness_runs(args.runs, run_prefix=args.run_prefix)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload.get("ok") else 1)

    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload.get("ok") else 1)


if __name__ == "__main__":
    main()
