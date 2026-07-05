from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from calm_puffer_art.foundry_harness import (
    DEFAULT_FOUNDRY_RUNS_DIR,
    compare_foundry_harness_runs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Foundry harness run artifacts by the configured objective."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_FOUNDRY_RUNS_DIR,
        help="Directory containing Foundry harness run artifact subdirectories.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Only compare run directories whose names start with this prefix.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = compare_foundry_harness_runs(args.runs, run_prefix=args.run_prefix)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        _print_table(payload)
    raise SystemExit(0 if payload["ok"] else 1)


def _print_table(payload: Mapping[str, Any]) -> None:
    print(f"runs_dir: {payload['runs_dir']}")
    print(f"objective_metric: {payload['objective_metric']}")
    ranking = payload.get("ranking", [])
    if not ranking:
        print("no runs found")
        return
    print()
    print("rank  candidate       source   score       condition      output")
    print("----  --------------  -------  ----------  -------------  ------")
    for item in ranking:
        score = item.get("ranking_score")
        if score is None:
            score = item.get("heldout_score")
        if score is None:
            score = item.get("primary_score")
        print(
            f"{item.get('rank', ''):<4}  "
            f"{str(item.get('candidate', ''))[:14]:<14}  "
            f"{str(item.get('ranking_score_source', ''))[:7]:<7}  "
            f"{_score_text(score):<10}  "
            f"{str(item.get('primary_condition', ''))[:13]:<13}  "
            f"{item.get('output_dir', '')}"
        )
    aggregates = payload.get("candidate_aggregates", [])
    if not aggregates:
        return
    print()
    print("candidate aggregates")
    print("rank  candidate       ok/runs  fail%    median      best        spend_mean")
    print("----  --------------  -------  -------  ----------  ----------  ----------")
    for item in aggregates:
        fail_rate = item.get("failure_rate")
        fail_pct = "n/a" if fail_rate is None else f"{float(fail_rate) * 100:.1f}"
        print(
            f"{item.get('rank', ''):<4}  "
            f"{str(item.get('candidate', ''))[:14]:<14}  "
            f"{item.get('ok_runs', 0)}/{item.get('runs', 0):<5}  "
            f"{fail_pct:<7}  "
            f"{_score_text(item.get('ranking_score_median')):<10}  "
            f"{_score_text(item.get('ranking_score_best')):<10}  "
            f"{_score_text(item.get('primary_accounted_dollar_seconds_mean')):<10}"
        )


def _score_text(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    main()
