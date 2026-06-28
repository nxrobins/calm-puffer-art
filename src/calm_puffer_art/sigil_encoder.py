from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .chunk_encoder import (
    LearnedChunkEncoderBundle,
    LearnedChunkEncoderConfig,
    LatentChunkScorer,
    SmokeTrainingReport,
    TinyChunkAutoencoder,
    WhitespaceVocabulary,
    validate_checkpoint_manifest,
)


SIGIL_ENCODER_PROOF_SCOPE = "sigil_corpus_v0"
DEFAULT_SIGIL_ENCODER_CHUNK_SIZE = 3
DEFAULT_SIGIL_ENCODER_LATENT_DIM = 64
DEFAULT_SIGIL_ENCODER_HIDDEN_DIM = 256
DEFAULT_SIGIL_ENCODER_SCORER_HIDDEN_DIM = 256
DEFAULT_SIGIL_ENCODER_TRAIN_STEPS = 100
DEFAULT_SIGIL_ENCODER_MAX_CHUNKS = 1024
DEFAULT_SIGIL_ENCODER_TIMEOUT_S = 180.0
DEFAULT_SIGIL_ENCODER_MAX_INPUT_BYTES = 128 * 1024
DEFAULT_SIGIL_ENCODER_MAX_SOURCE_TOKENS = 12_000
SIGIL_ENCODER_NLL_IMPROVEMENT_EPS = 1e-6


@dataclass(frozen=True)
class SigilEncoderTrainingConfig:
    chunk_size: int = DEFAULT_SIGIL_ENCODER_CHUNK_SIZE
    latent_dim: int = DEFAULT_SIGIL_ENCODER_LATENT_DIM
    hidden_dim: int = DEFAULT_SIGIL_ENCODER_HIDDEN_DIM
    scorer_hidden_dim: int = DEFAULT_SIGIL_ENCODER_SCORER_HIDDEN_DIM
    train_steps: int = DEFAULT_SIGIL_ENCODER_TRAIN_STEPS
    scorer_train_steps: int = DEFAULT_SIGIL_ENCODER_TRAIN_STEPS
    max_chunks: int = DEFAULT_SIGIL_ENCODER_MAX_CHUNKS
    timeout_s: float = DEFAULT_SIGIL_ENCODER_TIMEOUT_S
    seed: int = 1337

    def validate(self) -> None:
        if not 1 <= self.chunk_size <= 8:
            raise ValueError("sigil_encoder_chunk_size_out_of_range")
        if not 1 <= self.latent_dim <= 64:
            raise ValueError("sigil_encoder_latent_dim_out_of_range")
        if not 1 <= self.hidden_dim <= 512:
            raise ValueError("sigil_encoder_hidden_dim_out_of_range")
        if not 1 <= self.scorer_hidden_dim <= 512:
            raise ValueError("sigil_encoder_scorer_hidden_dim_out_of_range")
        if self.train_steps < 1 or self.scorer_train_steps < 1:
            raise ValueError("sigil_encoder_train_steps_must_be_positive")
        if self.max_chunks < 2:
            raise ValueError("sigil_encoder_max_chunks_must_be_at_least_two")
        if self.timeout_s <= 0.0:
            raise ValueError("sigil_encoder_timeout_must_be_positive")


def train_sigil_chunk_encoder(
    training_outputs: Sequence[str],
    config: SigilEncoderTrainingConfig | None = None,
) -> LearnedChunkEncoderBundle:
    """Train a TinyChunkAutoencoder bundle from validated Sigil outputs."""

    sigil_config = config or SigilEncoderTrainingConfig()
    sigil_config.validate()
    corpus = tuple(text.strip() for text in training_outputs if text.strip())
    if not corpus:
        raise ValueError("sigil_training_outputs_empty")

    started = time.monotonic()
    _seed_torch(sigil_config.seed)
    torch.set_num_threads(1)
    vocabulary = WhitespaceVocabulary.build(corpus)
    chunk_rows = _bounded_training_chunks(
        corpus,
        vocabulary=vocabulary,
        config=sigil_config,
    )
    train_chunks, holdout_chunks = _split_chunks(chunk_rows)
    train_tensor = torch.tensor(train_chunks, dtype=torch.long)
    holdout_tensor = torch.tensor(holdout_chunks, dtype=torch.long)
    autoencoder = TinyChunkAutoencoder(
        vocab_size=len(vocabulary.token_to_id),
        chunk_size=sigil_config.chunk_size,
        latent_dim=sigil_config.latent_dim,
        hidden_dim=sigil_config.hidden_dim,
    )
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.003)
    train_steps = 0
    for step in range(1, sigil_config.train_steps + 1):
        _check_timeout(started, sigil_config.timeout_s)
        optimizer.zero_grad()
        logits, _ = autoencoder(train_tensor)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            train_tensor.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        train_steps = step
    train_accuracy = _accuracy(autoencoder, train_tensor)
    holdout_accuracy = _accuracy(autoencoder, holdout_tensor)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    with torch.no_grad():
        train_latents = autoencoder.encode(train_tensor)
    reference_scorer = LatentChunkScorer(
        vocab_size=len(vocabulary.token_to_id),
        chunk_size=sigil_config.chunk_size,
        latent_dim=sigil_config.latent_dim,
        hidden_dim=sigil_config.scorer_hidden_dim,
    )
    old_scorer = copy.deepcopy(reference_scorer)
    new_scorer = copy.deepcopy(reference_scorer)
    old_nll = float(old_scorer.nll(train_tensor, train_latents).item())
    scorer_optimizer = torch.optim.Adam(new_scorer.parameters(), lr=0.01)
    scorer_train_steps = 0
    for step in range(1, sigil_config.scorer_train_steps + 1):
        _check_timeout(started, sigil_config.timeout_s)
        scorer_optimizer.zero_grad()
        nll = new_scorer.nll(train_tensor, train_latents.detach())
        nll.backward()
        scorer_optimizer.step()
        scorer_train_steps = step
    new_nll = float(new_scorer.nll(train_tensor, train_latents).item())
    nll_improvement = old_nll - new_nll
    if nll_improvement < SIGIL_ENCODER_NLL_IMPROVEMENT_EPS:
        raise AssertionError("sigil_encoder_missing_nll_improvement")
    for scorer in (reference_scorer, old_scorer, new_scorer):
        scorer.eval()
        for parameter in scorer.parameters():
            parameter.requires_grad = False

    learned_config = LearnedChunkEncoderConfig(
        chunk_size=sigil_config.chunk_size,
        latent_dim=sigil_config.latent_dim,
        seed=sigil_config.seed,
        max_train_steps=sigil_config.train_steps,
        timeout_s=sigil_config.timeout_s,
        max_input_bytes=DEFAULT_SIGIL_ENCODER_MAX_INPUT_BYTES,
        max_source_tokens=DEFAULT_SIGIL_ENCODER_MAX_SOURCE_TOKENS,
        proof_scope=SIGIL_ENCODER_PROOF_SCOPE,
        reconstruction_threshold=-1.0,
    )
    report = SmokeTrainingReport(
        proof_scope=SIGIL_ENCODER_PROOF_SCOPE,
        train_examples=len(corpus),
        holdout_examples=max(1, len(holdout_chunks)),
        train_reconstruction_accuracy=train_accuracy,
        holdout_reconstruction_accuracy=holdout_accuracy,
        old_nll=old_nll,
        new_nll=new_nll,
        nll_improvement=nll_improvement,
        train_steps=train_steps,
        scorer_train_steps=scorer_train_steps,
        vocab_size=len(vocabulary.token_to_id),
        vocab_hash=vocabulary.hash,
    )
    bundle = LearnedChunkEncoderBundle(
        config=learned_config,
        vocabulary=vocabulary,
        autoencoder=autoencoder,
        reference_scorer=reference_scorer,
        old_scorer=old_scorer,
        new_scorer=new_scorer,
        training_report=report,
        reference_scorer_state_id=_state_id(reference_scorer),
        old_scorer_state_id=_state_id(old_scorer),
        new_scorer_state_id=_state_id(new_scorer),
        autoencoder_state_id=_state_id(autoencoder),
    )
    validate_checkpoint_manifest(bundle.checkpoint_manifest())
    return bundle


def _bounded_training_chunks(
    corpus: Sequence[str],
    *,
    vocabulary: WhitespaceVocabulary,
    config: SigilEncoderTrainingConfig,
) -> list[list[int]]:
    per_example: list[list[list[int]]] = []
    for text in corpus:
        token_ids, unknown = vocabulary.encode(text)
        if unknown:
            raise ValueError("sigil_encoder_unknown_token")
        chunks = _chunk_token_ids(
            token_ids,
            chunk_size=config.chunk_size,
            pad_id=vocabulary.pad_id,
        )
        if chunks:
            per_example.append(chunks)
    if not per_example:
        raise ValueError("sigil_encoder_no_chunks")

    selected: list[list[int]] = []
    cursor = 0
    while len(selected) < config.max_chunks:
        added = False
        for chunks in per_example:
            if cursor < len(chunks):
                selected.append(chunks[cursor])
                added = True
                if len(selected) >= config.max_chunks:
                    break
        if not added:
            break
        cursor += 1
    if len(selected) < 2:
        raise ValueError("sigil_encoder_no_holdout_chunks")
    return selected


def _chunk_token_ids(
    token_ids: Sequence[int],
    *,
    chunk_size: int,
    pad_id: int,
) -> list[list[int]]:
    return [
        list(token_ids[index : index + chunk_size])
        + [pad_id] * max(0, chunk_size - len(token_ids[index : index + chunk_size]))
        for index in range(0, len(token_ids), chunk_size)
    ]


def _split_chunks(chunks: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[int]]]:
    holdout_count = max(1, len(chunks) // 5)
    train_count = len(chunks) - holdout_count
    if train_count <= 0:
        raise ValueError("sigil_encoder_no_train_chunks")
    return (
        [list(chunk) for chunk in chunks[:train_count]],
        [list(chunk) for chunk in chunks[train_count:]],
    )


@torch.no_grad()
def _accuracy(model: TinyChunkAutoencoder, chunks: torch.Tensor) -> float:
    logits, _ = model(chunks)
    predictions = logits.argmax(dim=-1)
    return float((predictions == chunks).float().mean().item())


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def _check_timeout(started: float, timeout_s: float) -> None:
    if time.monotonic() - started > timeout_s:
        raise TimeoutError("sigil_encoder_training_timeout")


def _state_id(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


__all__ = [
    "DEFAULT_SIGIL_ENCODER_CHUNK_SIZE",
    "DEFAULT_SIGIL_ENCODER_HIDDEN_DIM",
    "DEFAULT_SIGIL_ENCODER_LATENT_DIM",
    "DEFAULT_SIGIL_ENCODER_MAX_CHUNKS",
    "DEFAULT_SIGIL_ENCODER_SCORER_HIDDEN_DIM",
    "DEFAULT_SIGIL_ENCODER_TIMEOUT_S",
    "DEFAULT_SIGIL_ENCODER_TRAIN_STEPS",
    "SIGIL_ENCODER_PROOF_SCOPE",
    "SigilEncoderTrainingConfig",
    "train_sigil_chunk_encoder",
]
