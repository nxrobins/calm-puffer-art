from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised without calm extra.
    raise ImportError(
        "calm_puffer_art.calm_policy requires the optional calm extra. "
        'Install with `pip install -e ".[calm]"`.'
    ) from exc

from .chunk_encoder import LearnedChunkEncoderBundle
from .types import ActionUnit


@dataclass(frozen=True)
class CalmPolicyConfig:
    state_dim: int = 32
    latent_dim: int = 32
    hidden_dim: int = 64
    log_std_init: float = -5.0

    def validate(self) -> None:
        if min(self.state_dim, self.latent_dim, self.hidden_dim) <= 0:
            raise ValueError("calm_policy_dimension_invalid")
        if not math.isfinite(self.log_std_init) or not -8.0 <= self.log_std_init <= 2.0:
            raise ValueError("calm_policy_log_std_invalid")


class StateConditionedChunkPolicy(nn.Module):
    def __init__(self, config: CalmPolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.log_std = nn.Parameter(
            torch.full((config.latent_dim,), config.log_std_init)
        )

    def mean(self, states: torch.Tensor) -> torch.Tensor:
        _validate_states(states, self.config.state_dim)
        return self.network(states)

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        mean = self.mean(states)
        std = torch.exp(torch.clamp(self.log_std, min=-8.0, max=2.0))
        return mean + torch.randn_like(mean) * std

    def logprob(self, states: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean(states)
        if latents.shape != mean.shape:
            raise ValueError("calm_policy_latent_shape_invalid")
        log_std = torch.clamp(self.log_std, min=-8.0, max=2.0)
        variance = torch.exp(2.0 * log_std)
        return -0.5 * (
            ((latents - mean) ** 2) / variance
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)

    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        self.mean(states)
        log_std = torch.clamp(self.log_std, min=-8.0, max=2.0)
        value = log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
        return value.sum().expand(states.shape[0])


@dataclass(frozen=True)
class ChunkPolicyStep:
    state: torch.Tensor
    latent: torch.Tensor
    action: ActionUnit
    old_logprob: float
    reference_logprob: float
    advantage: float
    group_id: int
    reconstruction_exact: bool


@dataclass(frozen=True)
class ArtChunkLossBatch:
    inputs: Mapping[str, torch.Tensor]
    new_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    entropies: torch.Tensor
    action_count: int


class CalmPolicyAdapter:
    def __init__(
        self,
        *,
        bundle: LearnedChunkEncoderBundle,
        current_policy: StateConditionedChunkPolicy,
        behavior_policy: StateConditionedChunkPolicy,
        reference_policy: StateConditionedChunkPolicy,
    ) -> None:
        if current_policy.config.latent_dim != bundle.config.latent_dim:
            raise ValueError("calm_policy_codec_latent_mismatch")
        if not (
            current_policy.config == behavior_policy.config == reference_policy.config
        ):
            raise ValueError("calm_policy_snapshot_config_mismatch")
        self.bundle = bundle
        self.current_policy = current_policy
        self.behavior_policy = behavior_policy
        self.reference_policy = reference_policy
        for policy in (self.behavior_policy, self.reference_policy):
            for parameter in policy.parameters():
                parameter.requires_grad = False
            policy.eval()

    @classmethod
    def from_fitted_policy(
        cls,
        *,
        bundle: LearnedChunkEncoderBundle,
        policy: StateConditionedChunkPolicy,
    ) -> "CalmPolicyAdapter":
        return cls(
            bundle=bundle,
            current_policy=copy.deepcopy(policy),
            behavior_policy=copy.deepcopy(policy),
            reference_policy=copy.deepcopy(policy),
        )

    def rollout_step(
        self,
        *,
        state: torch.Tensor,
        source_token_ids: Sequence[int],
        advantage: float,
        group_id: int,
    ) -> ChunkPolicyStep:
        if len(source_token_ids) == 0 or len(source_token_ids) > self.bundle.config.chunk_size:
            raise ValueError("calm_policy_source_chunk_invalid")
        state_batch = state.detach().reshape(1, -1)
        with torch.no_grad():
            latent = self.behavior_policy.sample(state_batch)
            old_logprob = self.behavior_policy.logprob(state_batch, latent)
            reference_logprob = self.reference_policy.logprob(state_batch, latent)
            current_logprob = self.current_policy.logprob(state_batch, latent)
            decoded = self.bundle.autoencoder.decode_ids(latent).squeeze(0).tolist()
        decoded_ids = tuple(int(value) for value in decoded[: len(source_token_ids)])
        source_ids = tuple(int(value) for value in source_token_ids)
        exact = decoded_ids == source_ids
        text = self.bundle.vocabulary.decode(source_ids)
        action = ActionUnit(
            kind="state_conditioned_chunk" if exact else "token_fallback",
            payload=(
                tuple(round(float(value), 6) for value in latent.squeeze(0))
                if exact
                else source_ids
            ),
            token_count=len(source_ids),
            text=text,
            metadata={
                "action/fallback": not exact,
                "reconstruction/safe": exact,
                "reconstruction/accuracy": 1.0 if exact else 0.0,
                "failure/mode": None if exact else "reconstruction_drift",
                "codec/identity": self.bundle.checkpoint_manifest(),
                "state_conditioned": True,
            },
            old_logprob=float(old_logprob.item()) if exact else None,
            new_logprob=float(current_logprob.item()) if exact else None,
            reference_logprob=float(reference_logprob.item()) if exact else None,
        )
        return ChunkPolicyStep(
            state=state_batch.squeeze(0),
            latent=latent.squeeze(0),
            action=action,
            old_logprob=float(old_logprob.item()),
            reference_logprob=float(reference_logprob.item()),
            advantage=float(advantage),
            group_id=int(group_id),
            reconstruction_exact=exact,
        )


def fit_policy_to_codec(
    *,
    policy: StateConditionedChunkPolicy,
    bundle: LearnedChunkEncoderBundle,
    states: torch.Tensor,
    source_chunks: torch.Tensor,
    steps: int = 500,
    learning_rate: float = 0.01,
) -> dict[str, float]:
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("calm_policy_fit_config_invalid")
    _validate_states(states, policy.config.state_dim)
    if source_chunks.ndim != 2 or source_chunks.shape[0] != states.shape[0]:
        raise ValueError("calm_policy_source_chunk_invalid")
    with torch.no_grad():
        target_latents = bundle.autoencoder.encode(source_chunks)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    initial_loss = float(F.mse_loss(policy.mean(states), target_latents).item())
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(policy.mean(states), target_latents)
        loss.backward()
        optimizer.step()
    final_loss = float(F.mse_loss(policy.mean(states), target_latents).item())
    return {
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "improvement": initial_loss - final_loss,
        "steps": float(steps),
    }


def build_art_chunk_loss_batch(
    *,
    policy: StateConditionedChunkPolicy,
    sequences: Sequence[Sequence[ChunkPolicyStep]],
) -> ArtChunkLossBatch:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("calm_policy_loss_batch_empty")
    if any(not step.reconstruction_exact for sequence in sequences for step in sequence):
        raise ValueError("calm_policy_fallback_not_trainable")
    max_actions = max(len(sequence) for sequence in sequences)
    width = max_actions + 1
    old = torch.full((len(sequences), width), float("nan"))
    advantages = torch.zeros((len(sequences), width))
    assistant_mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    weights = torch.zeros((len(sequences), width))
    group_ids = torch.zeros((len(sequences), width), dtype=torch.long)
    new_rows: list[torch.Tensor] = []
    reference_rows: list[torch.Tensor] = []
    entropy_rows: list[torch.Tensor] = []
    action_count = 0
    for row, sequence in enumerate(sequences):
        states = torch.stack([step.state for step in sequence])
        latents = torch.stack([step.latent for step in sequence])
        new_values = policy.logprob(states, latents)
        entropy_values = policy.entropy(states)
        pad = width - len(sequence)
        new_rows.append(F.pad(new_values, (0, pad)))
        entropy_rows.append(F.pad(entropy_values, (0, pad)))
        reference_rows.append(
            F.pad(
                torch.tensor([step.reference_logprob for step in sequence]),
                (0, pad),
            )
        )
        for column, step in enumerate(sequence, start=1):
            old[row, column] = step.old_logprob
            advantages[row, column] = step.advantage
            assistant_mask[row, column] = True
            weights[row, column] = 1.0
            group_ids[row, column] = step.group_id
            action_count += 1
    return ArtChunkLossBatch(
        inputs={
            "logprobs": old,
            "advantages": advantages,
            "assistant_mask": assistant_mask,
            "weights": weights,
            "group_ids": group_ids,
        },
        new_logprobs=torch.stack(new_rows),
        reference_logprobs=torch.stack(reference_rows),
        entropies=torch.stack(entropy_rows),
        action_count=action_count,
    )


def execute_art_chunk_loss(
    batch: ArtChunkLossBatch,
    *,
    experimental_config: Mapping[str, Any] | None = None,
) -> Any:
    try:
        from art.loss import loss_fn
    except ImportError as exc:
        raise ImportError(
            "execute_art_chunk_loss requires the optional art extra"
        ) from exc
    config = {
        "importance_sampling_level": "token",
        "kl_penalty_coef": 0.0,
        "ppo": True,
        **dict(experimental_config or {}),
    }
    return loss_fn(
        batch.inputs,
        batch.new_logprobs,
        batch.reference_logprobs,
        batch.entropies,
        config,
    )


def deterministic_context_state(text: str, *, state_dim: int) -> torch.Tensor:
    if state_dim <= 0:
        raise ValueError("calm_policy_dimension_invalid")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [digest[index % len(digest)] / 127.5 - 1.0 for index in range(state_dim)]
    return torch.tensor(values, dtype=torch.float32)


def _validate_states(states: torch.Tensor, state_dim: int) -> None:
    if states.ndim != 2 or states.shape[1] != state_dim:
        raise ValueError("calm_policy_state_shape_invalid")
