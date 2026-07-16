from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a state-conditioned chunk policy and execute ART's real "
            "policy loss against chunk-action tensors."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts")
        / "calm_domain_codec"
        / "code-repair-chunk-2.pt",
    )
    parser.add_argument("--fit-steps", type=int, default=750)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts") / "calm_policy_adapter" / "report.json",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def run_proof(args: argparse.Namespace) -> dict[str, object]:
    import torch

    from calm_puffer_art.calm_domain import CODE_REPAIR_TRAIN_CORPUS
    from calm_puffer_art.calm_policy import (
        CalmPolicyAdapter,
        CalmPolicyConfig,
        StateConditionedChunkPolicy,
        build_art_chunk_loss_batch,
        deterministic_context_state,
        execute_art_chunk_loss,
        fit_policy_to_codec,
    )
    from calm_puffer_art.chunk_encoder import load_chunk_encoder_checkpoint

    torch.manual_seed(20260711)
    bundle = load_chunk_encoder_checkpoint(args.checkpoint)
    if bundle.config.chunk_size != 2:
        raise ValueError("state_conditioned_proof_requires_chunk_size_2")
    config = CalmPolicyConfig(
        state_dim=32,
        latent_dim=bundle.config.latent_dim,
        hidden_dim=64,
        log_std_init=-8.0,
    )
    states = []
    padded_chunks = []
    source_chunks: list[tuple[int, ...]] = []
    sequence_indexes: list[list[int]] = []
    for group_index, text in enumerate(CODE_REPAIR_TRAIN_CORPUS):
        token_ids, unknown = bundle.vocabulary.encode(text)
        if unknown:
            raise ValueError("state_conditioned_proof_unknown_token")
        indexes: list[int] = []
        for chunk_index in range(0, len(token_ids), bundle.config.chunk_size):
            source = tuple(token_ids[chunk_index : chunk_index + bundle.config.chunk_size])
            padded = source + (bundle.vocabulary.pad_id,) * (
                bundle.config.chunk_size - len(source)
            )
            context = f"group={group_index}|chunk={chunk_index}|text={text}"
            indexes.append(len(states))
            states.append(
                deterministic_context_state(context, state_dim=config.state_dim)
            )
            padded_chunks.append(padded)
            source_chunks.append(source)
        sequence_indexes.append(indexes)
    state_tensor = torch.stack(states)
    source_tensor = torch.tensor(padded_chunks, dtype=torch.long)
    policy = StateConditionedChunkPolicy(config)
    fit = fit_policy_to_codec(
        policy=policy,
        bundle=bundle,
        states=state_tensor,
        source_chunks=source_tensor,
        steps=args.fit_steps,
    )
    adapter = CalmPolicyAdapter.from_fitted_policy(bundle=bundle, policy=policy)
    sequences = []
    actions = []
    for group_index, indexes in enumerate(sequence_indexes):
        sequence = []
        for local_index, example_index in enumerate(indexes):
            step = adapter.rollout_step(
                state=state_tensor[example_index],
                source_token_ids=source_chunks[example_index],
                advantage=1.0 if (group_index + local_index) % 2 == 0 else -0.5,
                group_id=group_index + 1,
            )
            sequence.append(step)
            actions.append(step.action)
        sequences.append(sequence)
    exact_actions = sum(step.reconstruction_exact for sequence in sequences for step in sequence)
    batch = build_art_chunk_loss_batch(
        policy=adapter.current_policy,
        sequences=sequences,
    )
    before = _module_state_id(adapter.current_policy)
    optimizer = torch.optim.Adam(adapter.current_policy.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = execute_art_chunk_loss(
        batch,
        experimental_config={"ppo": True, "epsilon": 0.2},
    )
    loss.policy_loss.backward()
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().pow(2).sum().item())
            for parameter in adapter.current_policy.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()
    after = _module_state_id(adapter.current_policy)
    action_count = batch.action_count
    return {
        "ok": (
            exact_actions == action_count
            and math.isfinite(float(loss.policy_loss.item()))
            and gradient_norm > 0.0
            and before != after
        ),
        "proof_scope": "local_state_conditioned_art_loss_only",
        "checkpoint": str(args.checkpoint.resolve()),
        "state_conditioned": True,
        "state_source": "deterministic_context_features",
        "serving_model_hidden_states": False,
        "art_loss_executed": True,
        "art_loss_module": "art.loss.loss_fn",
        "art_serverless_custom_action_supported": False,
        "actions": action_count,
        "exact_reconstruction_actions": exact_actions,
        "reconstruction_exact_rate": exact_actions / action_count,
        "old_logprob_coverage": sum(action.old_logprob is not None for action in actions)
        / action_count,
        "new_logprob_coverage": sum(action.new_logprob is not None for action in actions)
        / action_count,
        "reference_logprob_coverage": sum(
            action.reference_logprob is not None for action in actions
        )
        / action_count,
        "fit": fit,
        "policy_loss": float(loss.policy_loss.item()),
        "gradient_norm": gradient_norm,
        "policy_state_changed": before != after,
    }


def _module_state_id(module: object) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_proof(args)
        report["report_path"] = str(args.report_path.resolve())
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
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
