from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from calm_puffer_art.sigil_ablation import (
    DEFAULT_SIGIL_ACTION_UNIT_DOLLAR_SECONDS,
    DEFAULT_SIGIL_TRAIN_STEPS,
    run_sigil_workload_ablation,
)
from calm_puffer_art.sigil_encoder import (
    DEFAULT_SIGIL_ENCODER_HIDDEN_DIM,
    DEFAULT_SIGIL_ENCODER_LATENT_DIM,
    DEFAULT_SIGIL_ENCODER_MAX_CHUNKS,
    DEFAULT_SIGIL_ENCODER_SCORER_HIDDEN_DIM,
    DEFAULT_SIGIL_ENCODER_TIMEOUT_S,
    DEFAULT_SIGIL_ENCODER_TRAIN_STEPS,
    SigilEncoderTrainingConfig,
)
from calm_puffer_art.sigil_integration import (
    DEFAULT_SIGIL_EXE,
    DEFAULT_SIGIL_IDIOM_JSONL,
    DEFAULT_SIGIL_IMPLEMENTATION_JSONL,
    load_sigil_corpus,
)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sigil_exe is not None:
        os.environ["SIGIL_EXE"] = str(args.sigil_exe)
    corpus = load_sigil_corpus(
        idiom_path=args.idiom_jsonl,
        implementation_path=args.implementation_jsonl,
    )
    encoder_config = SigilEncoderTrainingConfig(
        latent_dim=args.encoder_latent_dim,
        hidden_dim=args.encoder_hidden_dim,
        scorer_hidden_dim=args.encoder_scorer_hidden_dim,
        train_steps=args.encoder_train_steps,
        scorer_train_steps=args.encoder_train_steps,
        max_chunks=args.encoder_max_chunks,
        timeout_s=args.encoder_timeout_s,
    )
    return await run_sigil_workload_ablation(
        corpus=corpus,
        max_train_steps=args.train_steps,
        max_tasks=args.max_tasks,
        action_unit_dollar_seconds=args.action_unit_dollar_seconds,
        encoder_config=encoder_config,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Sigil compiler-verified Full Trinity ablation: static ART "
            "token baseline, scheduler-only token baseline, and scheduler plus "
            "token/fixed/learned chunk action space."
        )
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    parser.add_argument(
        "--idiom-jsonl",
        type=Path,
        default=DEFAULT_SIGIL_IDIOM_JSONL,
        help="Path to Sigil idiom.jsonl prompts.",
    )
    parser.add_argument(
        "--implementation-jsonl",
        type=Path,
        default=DEFAULT_SIGIL_IMPLEMENTATION_JSONL,
        help="Path to Sigil implementation.jsonl encoder-training outputs.",
    )
    parser.add_argument(
        "--sigil-exe",
        type=Path,
        default=DEFAULT_SIGIL_EXE,
        help="Path to sigil.exe. Also exported as SIGIL_EXE for verifier calls.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_SIGIL_TRAIN_STEPS,
        help="Maximum train steps per condition.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on compiler-verified idiom tasks.",
    )
    parser.add_argument(
        "--action-unit-dollar-seconds",
        type=float,
        default=DEFAULT_SIGIL_ACTION_UNIT_DOLLAR_SECONDS,
        help="Explicit cost charged for each emitted action unit.",
    )
    parser.add_argument(
        "--encoder-latent-dim",
        type=int,
        default=DEFAULT_SIGIL_ENCODER_LATENT_DIM,
        help="Latent dimension for the learned Sigil chunk encoder.",
    )
    parser.add_argument(
        "--encoder-hidden-dim",
        type=int,
        default=DEFAULT_SIGIL_ENCODER_HIDDEN_DIM,
        help="Autoencoder hidden dimension for the learned Sigil codec.",
    )
    parser.add_argument(
        "--encoder-scorer-hidden-dim",
        type=int,
        default=DEFAULT_SIGIL_ENCODER_SCORER_HIDDEN_DIM,
        help="Latent scorer hidden dimension for old/new/reference logprobs.",
    )
    parser.add_argument(
        "--encoder-train-steps",
        type=int,
        default=DEFAULT_SIGIL_ENCODER_TRAIN_STEPS,
        help="TinyChunkAutoencoder train steps for the learned Sigil codec.",
    )
    parser.add_argument(
        "--encoder-max-chunks",
        type=int,
        default=DEFAULT_SIGIL_ENCODER_MAX_CHUNKS,
        help="Bounded chunks sampled from the 987 validated implementation outputs.",
    )
    parser.add_argument(
        "--encoder-timeout-s",
        type=float,
        default=DEFAULT_SIGIL_ENCODER_TIMEOUT_S,
        help="Fail-fast timeout for learned encoder training.",
    )
    args = parser.parse_args()
    if args.train_steps < 1:
        raise SystemExit("--train-steps must be positive")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise SystemExit("--max-tasks must be positive when provided")
    if args.action_unit_dollar_seconds < 0.0:
        raise SystemExit("--action-unit-dollar-seconds must be non-negative")
    if not 1 <= args.encoder_latent_dim <= 64:
        raise SystemExit("--encoder-latent-dim must be in [1, 64]")
    if not 1 <= args.encoder_hidden_dim <= 512:
        raise SystemExit("--encoder-hidden-dim must be in [1, 512]")
    if not 1 <= args.encoder_scorer_hidden_dim <= 512:
        raise SystemExit("--encoder-scorer-hidden-dim must be in [1, 512]")
    if args.encoder_train_steps < 1:
        raise SystemExit("--encoder-train-steps must be positive")
    if args.encoder_max_chunks < 2:
        raise SystemExit("--encoder-max-chunks must be at least 2")
    if args.encoder_timeout_s <= 0.0:
        raise SystemExit("--encoder-timeout-s must be positive")
    return args


def main() -> None:
    args = _parse_args()
    payload = asyncio.run(_run(args))
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
