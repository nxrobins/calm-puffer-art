from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .actions import action_logprob_stats
from .chunk_encoder import (
    DOMAIN_PROOF_SCOPE,
    LearnedChunkActionCodec,
    LearnedChunkEncoderConfig,
    load_chunk_encoder_checkpoint,
    save_chunk_encoder_checkpoint,
    train_chunk_encoder,
)


CODE_REPAIR_TRAIN_CORPUS: tuple[str, ...] = (
    "def add ( a , b ) : return a + b",
    "def subtract ( a , b ) : return a - b",
    "def multiply ( a , b ) : return a * b",
    "def divide ( a , b ) : return a / b",
    "def modulo ( a , b ) : return a % b",
    "def equal ( a , b ) : return a == b",
    "def greater ( a , b ) : return a > b",
    "def less ( a , b ) : return a < b",
)


CODE_REPAIR_HOLDOUT_CORPUS: tuple[str, ...] = (
    "def add ( a , b ) : return a + b def subtract ( a , b ) : return a - b",
    "def multiply ( a , b ) : return a * b def divide ( a , b ) : return a / b",
    "def modulo ( a , b ) : return a % b def equal ( a , b ) : return a == b",
    "def greater ( a , b ) : return a > b def less ( a , b ) : return a < b",
)


def evaluate_domain_codec(
    codec: LearnedChunkActionCodec,
    corpus: Sequence[str],
) -> dict[str, Any]:
    reports = [codec.encode_with_report(text) for text in corpus]
    learned_actions = [
        action
        for report in reports
        if not report.fallback
        for action in report.actions
    ]
    exact = [
        not report.fallback and report.decoded_text == " ".join(text.split())
        for text, report in zip(corpus, reports, strict=True)
    ]
    failures = Counter(
        str(report.metadata.get("failure/mode"))
        for report in reports
        if report.fallback
    )
    logprobs = action_logprob_stats(learned_actions)
    source_tokens = sum(len(text.split()) for text in corpus)
    action_units = sum(len(report.actions) for report in reports)
    reconstruction = [report.reconstruction_accuracy for report in reports]
    return {
        "examples": len(reports),
        "source_tokens": source_tokens,
        "action_units": action_units,
        "semantic_bandwidth_tokens_per_decision": (
            source_tokens / action_units if action_units else 0.0
        ),
        "exact_reconstructions": sum(exact),
        "exact_reconstruction_rate": sum(exact) / len(exact) if exact else 0.0,
        "mean_reconstruction_accuracy": (
            fmean(reconstruction) if reconstruction else 0.0
        ),
        "minimum_reconstruction_accuracy": min(reconstruction, default=0.0),
        "fallbacks": sum(report.fallback for report in reports),
        "fallback_rate": (
            sum(report.fallback for report in reports) / len(reports)
            if reports
            else 0.0
        ),
        "failure_modes": dict(failures),
        "old_logprob_coverage": logprobs.old_logprob_coverage,
        "new_logprob_coverage": logprobs.new_logprob_coverage,
        "reference_logprob_coverage": logprobs.reference_logprob_coverage,
    }


def run_code_domain_codec_proof(
    *,
    output_dir: Path,
    chunk_sizes: Sequence[int] = (2, 4),
    latent_dim: int = 32,
) -> dict[str, Any]:
    if not chunk_sizes or len(set(chunk_sizes)) != len(chunk_sizes):
        raise ValueError("chunk_sizes_must_be_unique_and_nonempty")
    rows: list[dict[str, Any]] = []
    for chunk_size in chunk_sizes:
        config = LearnedChunkEncoderConfig(
            chunk_size=int(chunk_size),
            latent_dim=latent_dim,
            proof_scope=DOMAIN_PROOF_SCOPE,
        )
        bundle = train_chunk_encoder(
            config=config,
            train_corpus=CODE_REPAIR_TRAIN_CORPUS,
            holdout_corpus=CODE_REPAIR_HOLDOUT_CORPUS,
        )
        checkpoint_path = output_dir / f"code-repair-chunk-{chunk_size}.pt"
        manifest = save_chunk_encoder_checkpoint(bundle, checkpoint_path)
        restored = load_chunk_encoder_checkpoint(checkpoint_path)
        codec = LearnedChunkActionCodec(restored)
        train_evaluation = evaluate_domain_codec(codec, CODE_REPAIR_TRAIN_CORPUS)
        holdout_evaluation = evaluate_domain_codec(
            codec,
            CODE_REPAIR_HOLDOUT_CORPUS,
        )
        fallback_probe = codec.encode_with_report(
            "def unseen_symbol ( value ) : return value"
        )
        eligible_for_live_bridge = (
            train_evaluation["exact_reconstruction_rate"] == 1.0
            and holdout_evaluation["exact_reconstruction_rate"] == 1.0
            and train_evaluation["fallback_rate"] == 0.0
            and holdout_evaluation["fallback_rate"] == 0.0
        )
        rows.append(
            {
                "chunk_size": chunk_size,
                "latent_dim": latent_dim,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_manifest": manifest,
                "roundtrip_identity_preserved": (
                    bundle.checkpoint_manifest() == restored.checkpoint_manifest()
                ),
                "eligible_for_live_bridge": eligible_for_live_bridge,
                "training_report": {
                    "train_examples": bundle.training_report.train_examples,
                    "holdout_examples": bundle.training_report.holdout_examples,
                    "train_reconstruction_accuracy": (
                        bundle.training_report.train_reconstruction_accuracy
                    ),
                    "holdout_reconstruction_accuracy": (
                        bundle.training_report.holdout_reconstruction_accuracy
                    ),
                    "train_steps": bundle.training_report.train_steps,
                    "scorer_train_steps": (
                        bundle.training_report.scorer_train_steps
                    ),
                    "nll_improvement": bundle.training_report.nll_improvement,
                },
                "train": train_evaluation,
                "holdout": holdout_evaluation,
                "unknown_token_fallback": {
                    "fallback": fallback_probe.fallback,
                    "failure_mode": fallback_probe.metadata.get("failure/mode"),
                },
            }
        )
    ok = all(
        row["roundtrip_identity_preserved"]
        and row["unknown_token_fallback"]["fallback"] is True
        and row["unknown_token_fallback"]["failure_mode"] == "unknown_token"
        for row in rows
    )
    return {
        "ok": ok,
        "proof_scope": DOMAIN_PROOF_SCOPE,
        "domain": "python_code_repair",
        "holdout_design": "unseen_sequence_recombination_of_seen_chunks",
        "tokenizer": "bounded_whitespace_vocabulary",
        "native_policy_logprobs": False,
        "art_loss_connected": False,
        "claim": "offline reconstruction and checkpoint roundtrip only",
        "eligible_chunk_sizes": [
            row["chunk_size"] for row in rows if row["eligible_for_live_bridge"]
        ],
        "all_candidates_eligible": all(
            row["eligible_for_live_bridge"] for row in rows
        ),
        "rows": rows,
    }
