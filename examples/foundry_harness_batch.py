from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calm_puffer_art.foundry_harness import (
    DEFAULT_FOUNDRY_HARNESS_DIR,
    DEFAULT_FOUNDRY_RUNS_DIR,
    compare_foundry_harness_runs,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_CLI = ROOT / "examples" / "foundry_harness_run.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Foundry harness candidates across sequential replicates."
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["baseline", "full_trinity"],
        help="Candidate names or manifest JSON paths to run.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Sequential replicate count per candidate.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_FOUNDRY_HARNESS_DIR,
        help="Directory containing public Foundry harness manifests.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_FOUNDRY_RUNS_DIR,
        help="Directory where replicate run artifacts are written.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=None,
        help="Override every candidate manifest dotenv path for this batch.",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Override every candidate manifest Azure Foundry deployment name.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Run directory prefix; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the batch after the first failed replicate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.replicates < 1:
        raise SystemExit("--replicates must be positive")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = args.run_prefix or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    records: list[dict[str, Any]] = []
    stopped_early = False
    for candidate in args.candidates:
        for replicate in range(1, args.replicates + 1):
            output_dir = args.runs_dir / (
                f"{run_prefix}-{_safe_path_label(candidate)}-r{replicate:02d}"
            )
            completed = _run_replicate(
                candidate=candidate,
                candidate_dir=args.candidate_dir,
                deployment=args.deployment,
                env_path=args.env_path,
                output_dir=output_dir,
            )
            payload = _parse_stdout(completed.stdout)
            first_failure = _first_failure(payload)
            record = {
                "candidate": candidate,
                "replicate": replicate,
                "ok": bool(payload.get("ok")),
                "returncode": completed.returncode,
                "output_dir": str(output_dir),
                "error": payload.get("error"),
                "error_type": payload.get("error_type"),
                "failure_counts": _failure_counts(payload),
                "first_failure_category": first_failure.get("category"),
                "first_failure_message": first_failure.get("message"),
            }
            records.append(record)
            if completed.returncode != 0 and args.stop_on_failure:
                stopped_early = True
                break
        if stopped_early:
            break

    comparison = compare_foundry_harness_runs(args.runs_dir, run_prefix=run_prefix)
    payload = {
        "ok": bool(records) and all(bool(record["ok"]) for record in records),
        "runs_dir": str(args.runs_dir),
        "run_prefix": run_prefix,
        "replicates": args.replicates,
        "stopped_early": stopped_early,
        "runs": records,
        "comparison": comparison,
    }
    batch_path = args.runs_dir / f"{run_prefix}-batch-summary.json"
    batch_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["batch_summary_path"] = str(batch_path)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["ok"] else 1)


def _run_replicate(
    *,
    candidate: str,
    candidate_dir: Path,
    deployment: str | None,
    env_path: Path | None,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    command = [
        sys.executable,
        str(RUN_CLI),
        "--candidate",
        candidate,
        "--candidate-dir",
        str(candidate_dir),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if env_path is not None:
        command.extend(["--env-path", str(env_path)])
    if deployment is not None:
        command.extend(["--deployment", deployment])
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def _parse_stdout(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "error_type": "JSONDecodeError",
        }
    return payload if isinstance(payload, dict) else {"ok": False}


def _failure_counts(payload: dict[str, Any]) -> dict[str, int]:
    failures = payload.get("failures")
    if not isinstance(failures, dict):
        return {}
    counts = failures.get("counts")
    if not isinstance(counts, dict):
        return {}
    parsed: dict[str, int] = {}
    for key, value in counts.items():
        try:
            parsed[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _first_failure(payload: dict[str, Any]) -> dict[str, Any]:
    failures = payload.get("failures")
    if not isinstance(failures, dict):
        return {}
    events = failures.get("events")
    if not isinstance(events, list) or not events:
        return {}
    first = events[0]
    return first if isinstance(first, dict) else {}


def _safe_path_label(value: str) -> str:
    label = Path(value).stem if Path(value).suffix else value
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "candidate"


if __name__ == "__main__":
    main()
