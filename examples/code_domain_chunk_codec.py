from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, checkpoint, reload, and evaluate the offline Python "
            "code-repair chunk codec proof."
        ),
    )
    parser.add_argument("--chunk-sizes", default="2,4")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "calm_domain_codec",
    )
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.chunk_size_values = tuple(
            int(value.strip())
            for value in args.chunk_sizes.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--chunk-sizes must be a comma-separated list of integers")
    if not args.chunk_size_values:
        parser.error("--chunk-sizes must contain at least one integer")
    if len(set(args.chunk_size_values)) != len(args.chunk_size_values):
        parser.error("--chunk-sizes must not contain duplicates")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from calm_puffer_art.calm_domain import run_code_domain_codec_proof

        report = run_code_domain_codec_proof(
            output_dir=args.output_dir,
            chunk_sizes=args.chunk_size_values,
            latent_dim=args.latent_dim,
        )
        report_path = args.report_path or args.output_dir / "report.json"
        report["report_path"] = str(report_path.resolve())
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        report = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    print(json.dumps(report, indent=2 if args.json else None, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
