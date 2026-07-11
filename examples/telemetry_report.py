from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from calm_puffer_art.telemetry import (
    PricingConfig,
    load_telemetry_events,
    summarize_telemetry,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a calm-puffer-art JSONL evidence ledger, optionally "
            "applying monetary rates learned after the run."
        )
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-inference-requests", type=int)
    parser.add_argument("--input-usd-per-million-tokens", type=float)
    parser.add_argument("--output-usd-per-million-tokens", type=float)
    parser.add_argument("--trainer-usd-per-hour", type=float)
    parser.add_argument("--stale-after-seconds", type=float, default=600.0)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "expected_inference_requests",
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "trainer_usd_per_hour",
        "stale_after_seconds",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    events = load_telemetry_events(args.path)
    run_ids = sorted({str(event.get("run_id")) for event in events})
    run_id = args.run_id
    if run_id is None:
        if len(run_ids) != 1:
            raise ValueError(
                "telemetry path contains multiple run IDs; pass --run-id"
            )
        run_id = run_ids[0]
    selected = [event for event in events if str(event.get("run_id")) == run_id]
    if not selected:
        raise ValueError(f"run ID {run_id!r} was not found in telemetry path")
    summary = summarize_telemetry(
        selected,
        pricing=PricingConfig(
            input_usd_per_million_tokens=args.input_usd_per_million_tokens,
            output_usd_per_million_tokens=args.output_usd_per_million_tokens,
            trainer_usd_per_hour=args.trainer_usd_per_hour,
        ),
        path=args.path,
        expected_inference_requests=args.expected_inference_requests,
        stale_after_s=args.stale_after_seconds,
    )
    return {"run_id": run_id, **summary}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except Exception as exc:
        payload = {
            "healthy": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(report, indent=2 if args.json else None, sort_keys=True)
    )
    if args.fail_on_error and not report["healthy"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
