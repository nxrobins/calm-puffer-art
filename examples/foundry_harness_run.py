from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calm_puffer_art.foundry_harness import (
    DEFAULT_FOUNDRY_HARNESS_DIR,
    DEFAULT_FOUNDRY_RUNS_DIR,
    FOUNDRY_HARNESS_ARTIFACT_FILES,
    extract_foundry_harness_failures,
    foundry_harness_child_args,
    load_foundry_harness_manifest,
    summarize_foundry_harness_result,
)


ROOT = Path(__file__).resolve().parents[1]
FOUNDRY_CLI = ROOT / "examples" / "azure_foundry_codegen_ablation.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checked-in Foundry harness candidate through the fixed-budget "
            "race and write durable run artifacts."
        )
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate name from harnesses/foundry or a manifest JSON path.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_FOUNDRY_HARNESS_DIR,
        help="Directory containing public Foundry harness manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory; defaults to .codex/foundry-runs/<run-id>.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=None,
        help="Override the candidate manifest dotenv path for this run.",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Override the candidate manifest Azure Foundry deployment name.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        manifest = load_foundry_harness_manifest(
            args.candidate,
            candidate_dir=args.candidate_dir,
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        _print_payload(payload, as_json=args.json)
        raise SystemExit(2) from None
    if args.env_path is not None:
        manifest = replace(manifest, env_path=args.env_path)
    if args.deployment is not None:
        manifest = replace(manifest, deployment=args.deployment)
    if args.env_path is not None or args.deployment is not None:
        try:
            manifest.validate()
        except Exception as exc:
            payload = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            _print_payload(payload, as_json=args.json)
            raise SystemExit(2) from None

    output_dir = args.output_dir or _default_output_dir(manifest.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        name: output_dir / filename
        for name, filename in FOUNDRY_HARNESS_ARTIFACT_FILES.items()
    }
    _write_json(artifact_paths["manifest"], manifest.to_dict())

    command = [
        sys.executable,
        str(FOUNDRY_CLI),
        *foundry_harness_child_args(
            manifest,
            telemetry_path=artifact_paths["telemetry"],
        ),
    ]
    completed = _run_child(command, run_timeout_s=manifest.run_timeout_s)
    artifact_paths["stdout"].write_text(completed.stdout, encoding="utf-8")
    artifact_paths["stderr"].write_text(completed.stderr, encoding="utf-8")

    result = _parse_child_stdout(completed.stdout)
    _write_json(artifact_paths["result"], result)
    summary = summarize_foundry_harness_result(
        manifest,
        result,
        output_dir=output_dir,
        returncode=completed.returncode,
    )
    failures = extract_foundry_harness_failures(
        manifest,
        result,
        returncode=completed.returncode,
        stderr=completed.stderr,
    )
    _write_json(artifact_paths["summary"], summary)
    _write_json(artifact_paths["failures"], failures)

    payload = {
        "ok": bool(summary["ok"]),
        "candidate": manifest.name,
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "summary": summary,
        "failures": failures,
    }
    _print_payload(payload, as_json=args.json)
    raise SystemExit(0 if payload["ok"] else completed.returncode or 1)


def _run_child(
    command: list[str],
    *,
    run_timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    timeout = run_timeout_s + 10.0 if run_timeout_s > 0.0 else None
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = (
            stderr
            + "\nfoundry_harness_child_timeout_exceeded_s="
            + str(timeout)
            + "\n"
        )
        return subprocess.CompletedProcess(command, 124, stdout, stderr)


def _parse_child_stdout(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return {
            "ok": False,
            "error": "foundry_child_stdout_empty",
            "error_type": "OutputParseError",
        }
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "error_type": "JSONDecodeError",
            "raw_stdout_bytes": len(stdout.encode("utf-8")),
        }
    if isinstance(payload, dict):
        return payload
    return {
        "ok": False,
        "error": "foundry_child_stdout_json_not_object",
        "error_type": "OutputParseError",
    }


def _default_output_dir(candidate_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_FOUNDRY_RUNS_DIR / f"{stamp}-{candidate_name}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
