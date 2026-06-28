from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from calm_puffer_art.actions import action_logprob_stats, semantic_bandwidth


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and verify the optional learned chunk encoder smoke.",
    )
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--reconstruction-threshold", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    from calm_puffer_art.chunk_encoder import (
        LearnedChunkActionCodec,
        LearnedChunkEncoderConfig,
        SMOKE_PROOF_SCOPE,
        train_smoke_chunk_encoder,
        validate_learned_chunk_actions,
    )

    config = LearnedChunkEncoderConfig(
        chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
        reconstruction_threshold=args.reconstruction_threshold,
    )
    bundle = train_smoke_chunk_encoder(config=config)
    codec = LearnedChunkActionCodec(bundle)
    report = codec.encode_with_report("alpha beta gamma delta")
    stats = validate_learned_chunk_actions(report.actions)
    raw_stats = action_logprob_stats(report.actions)
    improved_chunks = sum(
        1
        for action in report.actions
        if action.old_logprob is not None
        and action.new_logprob is not None
        and action.new_logprob > action.old_logprob
    )
    old_reference_delta = [
        abs((action.old_logprob or 0.0) - (action.reference_logprob or 0.0))
        for action in report.actions
    ]
    output = {
        "ok": True,
        "proof_scope": SMOKE_PROOF_SCOPE,
        "used_torch": True,
        "chunk_size": config.chunk_size,
        "latent_dim": config.latent_dim,
        "train_reconstruction_accuracy": (
            bundle.training_report.train_reconstruction_accuracy
        ),
        "holdout_reconstruction_accuracy": (
            bundle.training_report.holdout_reconstruction_accuracy
        ),
        "passed_reconstruction_threshold": report.passed_reconstruction_threshold,
        "actions": len(report.actions),
        "semantic_bandwidth": semantic_bandwidth(report.actions),
        "old_logprob_coverage": stats.old_logprob_coverage,
        "new_logprob_coverage": stats.new_logprob_coverage,
        "reference_logprob_coverage": stats.reference_logprob_coverage,
        "new_logprob_improved_chunks": improved_chunks,
        "mean_old_reference_logprob_abs_delta": (
            sum(old_reference_delta) / len(old_reference_delta)
            if old_reference_delta
            else 0.0
        ),
        "old_new_logprob_delta_mean": raw_stats.old_new_logprob_delta_mean,
        "codec_identity": codec.identity,
        "nll_improvement": bundle.training_report.nll_improvement,
    }
    _assert_output_contract(output)
    return output


def _assert_output_contract(output: dict[str, object]) -> None:
    required = (
        "ok",
        "proof_scope",
        "used_torch",
        "chunk_size",
        "latent_dim",
        "train_reconstruction_accuracy",
        "holdout_reconstruction_accuracy",
        "passed_reconstruction_threshold",
        "actions",
        "semantic_bandwidth",
        "old_logprob_coverage",
        "new_logprob_coverage",
        "reference_logprob_coverage",
        "new_logprob_improved_chunks",
        "mean_old_reference_logprob_abs_delta",
    )
    missing = [key for key in required if key not in output]
    if missing:
        raise AssertionError(f"missing_chunk_encoder_smoke_output_keys:{missing}")
    if output["proof_scope"] != "smoke_only":
        raise AssertionError("missing_smoke_only_scope")
    if output["train_reconstruction_accuracy"] != 1.0:
        raise AssertionError("train_reconstruction_threshold_not_met")
    if output["holdout_reconstruction_accuracy"] != 1.0:
        raise AssertionError("holdout_reconstruction_threshold_not_met")
    if output["old_logprob_coverage"] != 1.0:
        raise AssertionError("missing_or_detached_chunk_logprobs")
    if output["new_logprob_coverage"] != 1.0:
        raise AssertionError("missing_or_detached_chunk_logprobs")
    if output["reference_logprob_coverage"] != 1.0:
        raise AssertionError("missing_or_detached_chunk_logprobs")
    if int(output["new_logprob_improved_chunks"]) <= 0:
        raise AssertionError("new_logprob_improvement_without_nll_improvement")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = run_smoke(args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"chunk encoder smoke failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
