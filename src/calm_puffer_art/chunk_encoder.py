from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised in no-calm installs.
    raise ImportError(
        "calm_puffer_art.chunk_encoder requires the optional calm extra. "
        'Install with `pip install -e ".[calm]"`.'
    ) from exc

from .actions import (
    ActionLogprobStats,
    TokenActionCodec,
    action_logprob_stats,
    semantic_bandwidth,
)
from .types import ActionUnit


SMOKE_PROOF_SCOPE = "smoke_only"
DOMAIN_PROOF_SCOPE = "offline_domain_reconstruction_only"
CHUNK_CHECKPOINT_SCHEMA_VERSION = 1
SMOKE_SEED = 1337
MAX_SMOKE_EXAMPLES = 128
MAX_TRAIN_STEPS = 1000
SMOKE_TIMEOUT_S = 30.0
MAX_INPUT_BYTES = 4096
MAX_SOURCE_TOKENS = 256
MAX_CHUNK_SIZE = 8
MAX_LATENT_DIM = 64
NLL_IMPROVEMENT_EPS = 1e-6
PAD_TOKEN = "<pad>"


SMOKE_CORPUS: tuple[str, ...] = (
    "alpha beta alpha beta",
    "gamma delta gamma delta",
    "beta alpha beta alpha",
    "delta gamma delta gamma",
    "alpha gamma alpha gamma",
    "beta delta beta delta",
    "gamma alpha gamma alpha",
    "delta beta delta beta",
    "alpha beta gamma delta",
    "beta alpha delta gamma",
)


@dataclass(frozen=True)
class LearnedChunkEncoderConfig:
    chunk_size: int = 2
    latent_dim: int = 16
    reconstruction_threshold: float = 1.0
    seed: int = SMOKE_SEED
    max_train_steps: int = MAX_TRAIN_STEPS
    timeout_s: float = SMOKE_TIMEOUT_S
    max_input_bytes: int = MAX_INPUT_BYTES
    max_source_tokens: int = MAX_SOURCE_TOKENS
    max_unknown_tokens: int = 0
    proof_scope: str = SMOKE_PROOF_SCOPE
    streaming: bool = False
    byte_level: bool = False
    syntax_aware: bool = False
    multilingual_normalization: bool = False
    overlapping_chunks: bool = False

    def validate(self) -> None:
        if not 1 <= self.chunk_size <= MAX_CHUNK_SIZE:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if not 1 <= self.latent_dim <= MAX_LATENT_DIM:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.max_train_steps > MAX_TRAIN_STEPS:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.timeout_s > SMOKE_TIMEOUT_S:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.max_input_bytes > MAX_INPUT_BYTES:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.max_source_tokens > MAX_SOURCE_TOKENS:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.max_unknown_tokens != 0:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.reconstruction_threshold != 1.0:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if self.proof_scope not in {SMOKE_PROOF_SCOPE, DOMAIN_PROOF_SCOPE}:
            raise ValueError("chunk_encoder_input_limit_exceeded")
        if (
            self.streaming
            or self.byte_level
            or self.syntax_aware
            or self.multilingual_normalization
            or self.overlapping_chunks
        ):
            raise NotImplementedError("unsupported_chunk_encoder_mode")


@dataclass(frozen=True)
class SmokeTrainingReport:
    proof_scope: str
    train_examples: int
    holdout_examples: int
    train_reconstruction_accuracy: float
    holdout_reconstruction_accuracy: float
    old_nll: float
    new_nll: float
    nll_improvement: float
    train_steps: int
    scorer_train_steps: int
    vocab_size: int
    vocab_hash: str


@dataclass(frozen=True)
class ChunkEncodeReport:
    actions: list[ActionUnit]
    decoded_text: str
    reconstruction_accuracy: float
    passed_reconstruction_threshold: bool
    fallback: bool
    metrics: dict[str, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WhitespaceVocabulary:
    token_to_id: Mapping[str, int]

    @classmethod
    def build(cls, corpus: Sequence[str]) -> "WhitespaceVocabulary":
        tokens = sorted({token for text in corpus for token in text.split()})
        token_to_id = {PAD_TOKEN: 0}
        token_to_id.update({token: index + 1 for index, token in enumerate(tokens)})
        return cls(token_to_id=token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def hash(self) -> str:
        payload = json.dumps(dict(sorted(self.token_to_id.items())), separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def encode(self, text: str) -> tuple[list[int], list[str]]:
        token_ids: list[int] = []
        unknown: list[str] = []
        for token in text.split():
            token_id = self.token_to_id.get(token)
            if token_id is None:
                unknown.append(token)
            else:
                token_ids.append(token_id)
        return token_ids, unknown

    def decode(self, token_ids: Sequence[int]) -> str:
        lookup = self.id_to_token
        tokens = [
            lookup[token_id]
            for token_id in token_ids
            if token_id != self.pad_id and token_id in lookup
        ]
        return " ".join(tokens)


class TinyChunkAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        chunk_size: int,
        latent_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(chunk_size * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
        )
        self.output_heads = nn.ModuleList(
            nn.Linear(hidden_dim, vocab_size) for _ in range(chunk_size)
        )

    def encode(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(chunk_ids)
        return self.encoder(embeddings.reshape(embeddings.shape[0], -1))

    def forward(self, chunk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(chunk_ids)
        hidden = self.decoder(latents)
        logits = torch.stack([head(hidden) for head in self.output_heads], dim=1)
        return logits, latents

    def decode_ids(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(latents)
        logits = torch.stack([head(hidden) for head in self.output_heads], dim=1)
        return logits.argmax(dim=-1)


class LatentChunkScorer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        chunk_size: int,
        latent_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(chunk_size * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(latent_dim))

    def mean(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(chunk_ids)
        return self.network(embeddings.reshape(embeddings.shape[0], -1))

    def logprob(self, chunk_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean(chunk_ids)
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        variance = torch.exp(2.0 * log_std)
        log_two_pi = math.log(2.0 * math.pi)
        return -0.5 * (
            ((latents - mean) ** 2) / variance + 2.0 * log_std + log_two_pi
        ).sum(dim=-1)

    def nll(self, chunk_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return -self.logprob(chunk_ids, latents).mean()


@dataclass
class LearnedChunkEncoderBundle:
    config: LearnedChunkEncoderConfig
    vocabulary: WhitespaceVocabulary
    autoencoder: TinyChunkAutoencoder
    reference_scorer: LatentChunkScorer
    old_scorer: LatentChunkScorer
    new_scorer: LatentChunkScorer
    training_report: SmokeTrainingReport
    reference_scorer_state_id: str
    old_scorer_state_id: str
    new_scorer_state_id: str
    autoencoder_state_id: str

    def checkpoint_manifest(self) -> dict[str, Any]:
        return {
            "encoder": self.autoencoder_state_id,
            "decoder": self.autoencoder_state_id,
            "reference_scorer": self.reference_scorer_state_id,
            "old_scorer": self.old_scorer_state_id,
            "new_scorer": self.new_scorer_state_id,
            "vocab": {
                "hash": self.vocabulary.hash,
                "token_to_id": dict(self.vocabulary.token_to_id),
            },
            "config": {
                "chunk_size": self.config.chunk_size,
                "latent_dim": self.config.latent_dim,
                "reconstruction_threshold": self.config.reconstruction_threshold,
                "proof_scope": self.config.proof_scope,
            },
        }


@dataclass
class LearnedChunkActionCodec:
    _bundle: LearnedChunkEncoderBundle = field(repr=False)
    name: str = "learned_chunk"

    def __post_init__(self) -> None:
        identity = self.identity
        required = {
            "vocab_hash",
            "chunk_size",
            "latent_dim",
            "reconstruction_threshold",
            "reference_scorer_state_id",
            "old_scorer_state_id",
            "new_scorer_state_id",
        }
        if required - set(identity):
            raise ValueError("learned_codec_identity_missing_required_fields")

    @property
    def chunk_size(self) -> int:
        return self._bundle.config.chunk_size

    @property
    def latent_dim(self) -> int:
        return self._bundle.config.latent_dim

    @property
    def reconstruction_threshold(self) -> float:
        return self._bundle.config.reconstruction_threshold

    @property
    def vocab_hash(self) -> str:
        return self._bundle.vocabulary.hash

    @property
    def reference_scorer_state_id(self) -> str:
        return self._bundle.reference_scorer_state_id

    @property
    def old_scorer_state_id(self) -> str:
        return self._bundle.old_scorer_state_id

    @property
    def new_scorer_state_id(self) -> str:
        return self._bundle.new_scorer_state_id

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "vocab_hash": self.vocab_hash,
            "chunk_size": self.chunk_size,
            "latent_dim": self.latent_dim,
            "reconstruction_threshold": self.reconstruction_threshold,
            "reference_scorer_state_id": self.reference_scorer_state_id,
            "old_scorer_state_id": self.old_scorer_state_id,
            "new_scorer_state_id": self.new_scorer_state_id,
        }

    def encode(self, value: str) -> list[ActionUnit]:
        return self.encode_with_report(value).actions

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        return " ".join(action.text for action in actions).strip()

    def encode_with_report(self, value: str) -> ChunkEncodeReport:
        _validate_input_limits(value, self._bundle.config)
        token_ids, unknown_tokens = self._bundle.vocabulary.encode(value)
        if unknown_tokens:
            return _fallback_report(
                value,
                failure_mode="unknown_token",
                decoded_text=value,
                token_count=len(value.split()),
            )
        chunks, lengths = _chunk_token_ids(
            token_ids,
            chunk_size=self.chunk_size,
            pad_id=self._bundle.vocabulary.pad_id,
        )
        if self._bundle.vocabulary.pad_id in token_ids:
            raise ValueError("invalid_reconstruction_target")
        if not chunks:
            return _fallback_report(
                value,
                failure_mode="reconstruction_drift",
                decoded_text="",
                token_count=0,
            )
        with torch.no_grad():
            chunk_tensor = torch.tensor(chunks, dtype=torch.long)
            latents = self._bundle.autoencoder.encode(chunk_tensor)
            decoded_padded = self._bundle.autoencoder.decode_ids(latents).tolist()
        decoded_token_ids = _unpad_decoded_chunks(decoded_padded, lengths)
        accuracy = compute_reconstruction_accuracy(token_ids, decoded_token_ids)
        decoded_text = self._bundle.vocabulary.decode(decoded_token_ids)
        if accuracy < self.reconstruction_threshold:
            return _fallback_report(
                value,
                failure_mode="reconstruction_drift",
                decoded_text=decoded_text,
                token_count=len(token_ids),
                reconstruction_accuracy=accuracy,
            )
        actions = self._actions_for_chunks(
            chunks=chunks,
            lengths=lengths,
            latents=latents,
            decoded_token_ids=decoded_token_ids,
        )
        stats = validate_learned_chunk_actions(actions)
        improved = sum(
            1
            for action in actions
            if action.new_logprob is not None
            and action.old_logprob is not None
            and action.new_logprob > action.old_logprob
        )
        old_reference_delta = [
            abs((action.old_logprob or 0.0) - (action.reference_logprob or 0.0))
            for action in actions
        ]
        metadata = {
            "action/fallback": False,
            "reconstruction/accuracy": accuracy,
            "reconstruction/safe": True,
            "proof_scope": self._bundle.config.proof_scope,
            "codec/identity": self.identity,
        }
        return ChunkEncodeReport(
            actions=actions,
            decoded_text=decoded_text,
            reconstruction_accuracy=accuracy,
            passed_reconstruction_threshold=True,
            fallback=False,
            metrics={
                "actions": float(len(actions)),
                "semantic_bandwidth": semantic_bandwidth(actions),
                "old_logprob_coverage": stats.old_logprob_coverage,
                "new_logprob_coverage": stats.new_logprob_coverage,
                "reference_logprob_coverage": stats.reference_logprob_coverage,
                "new_logprob_improved_chunks": float(improved),
                "mean_old_reference_logprob_abs_delta": (
                    sum(old_reference_delta) / len(old_reference_delta)
                    if old_reference_delta
                    else 0.0
                ),
            },
            metadata=metadata,
        )

    def _actions_for_chunks(
        self,
        *,
        chunks: Sequence[Sequence[int]],
        lengths: Sequence[int],
        latents: torch.Tensor,
        decoded_token_ids: Sequence[int],
    ) -> list[ActionUnit]:
        chunk_tensor = torch.tensor(chunks, dtype=torch.long)
        with torch.no_grad():
            reference_logprobs = self._bundle.reference_scorer.logprob(
                chunk_tensor,
                latents,
            )
            old_logprobs = self._bundle.old_scorer.logprob(chunk_tensor, latents)
            new_logprobs = self._bundle.new_scorer.logprob(chunk_tensor, latents)
        actions: list[ActionUnit] = []
        cursor = 0
        for index, (chunk, length) in enumerate(zip(chunks, lengths)):
            chunk_text_ids = list(decoded_token_ids[cursor : cursor + length])
            cursor += length
            text = self._bundle.vocabulary.decode(chunk_text_ids)
            reference = float(reference_logprobs[index].item())
            old = float(old_logprobs[index].item())
            new = float(new_logprobs[index].item())
            if not all(math.isfinite(value) for value in (reference, old, new)):
                raise ValueError("missing_or_detached_chunk_logprobs")
            actions.append(
                ActionUnit(
                    kind="learned_chunk",
                    payload=tuple(round(float(value), 6) for value in latents[index]),
                    token_count=length,
                    text=text,
                    metadata={
                        "chunk_index": index,
                        "chunk_size": self.chunk_size,
                        "latent_dim": self.latent_dim,
                        "source_token_ids": tuple(
                            int(token_id) for token_id in chunk[:length]
                        ),
                        "action/fallback": False,
                        "reconstruction/accuracy": 1.0,
                        "reconstruction/safe": True,
                        "vocab_hash": self.vocab_hash,
                        "reference_scorer_state_id": self.reference_scorer_state_id,
                        "old_scorer_state_id": self.old_scorer_state_id,
                        "new_scorer_state_id": self.new_scorer_state_id,
                    },
                    old_logprob=old,
                    new_logprob=new,
                    reference_logprob=reference,
                )
            )
        return actions


def train_chunk_encoder(
    *,
    config: LearnedChunkEncoderConfig,
    train_corpus: Sequence[str],
    holdout_corpus: Sequence[str],
    vocabulary_corpus: Sequence[str] | None = None,
) -> LearnedChunkEncoderBundle:
    config.validate()
    train_texts = tuple(train_corpus)
    holdout_texts = tuple(holdout_corpus)
    if (
        not train_texts
        or not holdout_texts
        or len(train_texts) + len(holdout_texts) > MAX_SMOKE_EXAMPLES
    ):
        raise ValueError("chunk_encoder_input_limit_exceeded")
    vocabulary_texts = tuple(vocabulary_corpus or (*train_texts, *holdout_texts))
    if not vocabulary_texts or len(vocabulary_texts) > MAX_SMOKE_EXAMPLES:
        raise ValueError("chunk_encoder_input_limit_exceeded")
    start = time.monotonic()
    _seed_torch(config.seed)
    torch.set_num_threads(1)
    vocabulary = WhitespaceVocabulary.build(vocabulary_texts)
    train_chunks, _ = _texts_to_chunks(train_texts, vocabulary, config)
    holdout_chunks, _ = _texts_to_chunks(holdout_texts, vocabulary, config)
    autoencoder = TinyChunkAutoencoder(
        vocab_size=len(vocabulary.token_to_id),
        chunk_size=config.chunk_size,
        latent_dim=config.latent_dim,
    )
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.05)
    train_tensor = torch.tensor(train_chunks, dtype=torch.long)
    holdout_tensor = torch.tensor(holdout_chunks, dtype=torch.long)
    train_steps = 0
    for step in range(1, config.max_train_steps + 1):
        _check_timeout(start, config.timeout_s)
        optimizer.zero_grad()
        logits, _ = autoencoder(train_tensor)
        loss = _chunk_cross_entropy(logits, train_tensor)
        loss.backward()
        optimizer.step()
        train_steps = step
        if step % 25 == 0 or step == config.max_train_steps:
            train_accuracy = _autoencoder_accuracy(autoencoder, train_tensor)
            holdout_accuracy = _autoencoder_accuracy(autoencoder, holdout_tensor)
            if train_accuracy == 1.0 and holdout_accuracy == 1.0:
                break
    train_accuracy = _autoencoder_accuracy(autoencoder, train_tensor)
    holdout_accuracy = _autoencoder_accuracy(autoencoder, holdout_tensor)
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False
    autoencoder.eval()
    with torch.no_grad():
        train_latents = autoencoder.encode(train_tensor)
    reference_scorer = LatentChunkScorer(
        vocab_size=len(vocabulary.token_to_id),
        chunk_size=config.chunk_size,
        latent_dim=config.latent_dim,
    )
    old_scorer = copy.deepcopy(reference_scorer)
    new_scorer = copy.deepcopy(reference_scorer)
    old_nll = float(old_scorer.nll(train_tensor, train_latents).item())
    scorer_optimizer = torch.optim.Adam(new_scorer.parameters(), lr=0.05)
    scorer_train_steps = 0
    for step in range(1, config.max_train_steps + 1):
        _check_timeout(start, config.timeout_s)
        scorer_optimizer.zero_grad()
        nll = new_scorer.nll(train_tensor, train_latents.detach())
        nll.backward()
        scorer_optimizer.step()
        scorer_train_steps = step
        if step % 25 == 0 or step == config.max_train_steps:
            new_nll_probe = float(new_scorer.nll(train_tensor, train_latents).item())
            if old_nll - new_nll_probe >= NLL_IMPROVEMENT_EPS:
                break
    new_nll = float(new_scorer.nll(train_tensor, train_latents).item())
    nll_improvement = old_nll - new_nll
    if nll_improvement < NLL_IMPROVEMENT_EPS:
        raise AssertionError("new_logprob_improvement_without_nll_improvement")
    for scorer in (reference_scorer, old_scorer, new_scorer):
        for parameter in scorer.parameters():
            parameter.requires_grad = False
        scorer.eval()
    autoencoder_state_id = _state_id(autoencoder)
    reference_state_id = _state_id(reference_scorer)
    old_state_id = _state_id(old_scorer)
    new_state_id = _state_id(new_scorer)
    report = SmokeTrainingReport(
        proof_scope=config.proof_scope,
        train_examples=len(train_texts),
        holdout_examples=len(holdout_texts),
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
        config=config,
        vocabulary=vocabulary,
        autoencoder=autoencoder,
        reference_scorer=reference_scorer,
        old_scorer=old_scorer,
        new_scorer=new_scorer,
        training_report=report,
        reference_scorer_state_id=reference_state_id,
        old_scorer_state_id=old_state_id,
        new_scorer_state_id=new_state_id,
        autoencoder_state_id=autoencoder_state_id,
    )
    validate_checkpoint_manifest(bundle.checkpoint_manifest())
    return bundle


def train_smoke_chunk_encoder(
    config: LearnedChunkEncoderConfig | None = None,
    corpus: Sequence[str] = SMOKE_CORPUS,
) -> LearnedChunkEncoderBundle:
    config = config or LearnedChunkEncoderConfig()
    train_texts, holdout_texts = _split_corpus(corpus)
    bundle = train_chunk_encoder(
        config=config,
        train_corpus=train_texts,
        holdout_corpus=holdout_texts,
        vocabulary_corpus=corpus,
    )
    if (
        bundle.training_report.train_reconstruction_accuracy != 1.0
        or bundle.training_report.holdout_reconstruction_accuracy != 1.0
    ):
        raise AssertionError("missing_holdout_reconstruction_report")
    return bundle


def save_chunk_encoder_checkpoint(
    bundle: LearnedChunkEncoderBundle,
    path: Path,
) -> dict[str, Any]:
    manifest = bundle.checkpoint_manifest()
    validate_checkpoint_manifest(manifest)
    payload = {
        "schema_version": CHUNK_CHECKPOINT_SCHEMA_VERSION,
        "manifest": manifest,
        "config": asdict(bundle.config),
        "training_report": asdict(bundle.training_report),
        "state_dicts": {
            "autoencoder": bundle.autoencoder.state_dict(),
            "reference_scorer": bundle.reference_scorer.state_dict(),
            "old_scorer": bundle.old_scorer.state_dict(),
            "new_scorer": bundle.new_scorer.state_dict(),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return manifest


def load_chunk_encoder_checkpoint(path: Path) -> LearnedChunkEncoderBundle:
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "manifest",
        "config",
        "training_report",
        "state_dicts",
    }:
        raise ValueError("learned_chunk_checkpoint_invalid")
    if payload.get("schema_version") != CHUNK_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("learned_chunk_checkpoint_schema_unsupported")
    manifest = payload.get("manifest")
    config_payload = payload.get("config")
    report_payload = payload.get("training_report")
    state_dicts = payload.get("state_dicts")
    if not all(
        isinstance(value, Mapping)
        for value in (manifest, config_payload, report_payload, state_dicts)
    ):
        raise ValueError("learned_chunk_checkpoint_invalid")
    assert isinstance(manifest, Mapping)
    assert isinstance(config_payload, Mapping)
    assert isinstance(report_payload, Mapping)
    assert isinstance(state_dicts, Mapping)
    validate_checkpoint_manifest(manifest)
    if set(state_dicts) != {
        "autoencoder",
        "reference_scorer",
        "old_scorer",
        "new_scorer",
    }:
        raise ValueError("learned_chunk_checkpoint_invalid")
    try:
        config = LearnedChunkEncoderConfig(**dict(config_payload))
        training_report = SmokeTrainingReport(**dict(report_payload))
    except (TypeError, ValueError) as exc:
        raise ValueError("learned_chunk_checkpoint_invalid") from exc
    config.validate()
    vocab_payload = manifest["vocab"]
    assert isinstance(vocab_payload, Mapping)
    token_to_id = vocab_payload["token_to_id"]
    assert isinstance(token_to_id, Mapping)
    vocabulary = WhitespaceVocabulary(
        token_to_id={str(token): int(index) for token, index in token_to_id.items()}
    )
    if vocabulary.hash != vocab_payload.get("hash"):
        raise ValueError("learned_chunk_checkpoint_identity_mismatch")
    autoencoder = TinyChunkAutoencoder(
        vocab_size=len(vocabulary.token_to_id),
        chunk_size=config.chunk_size,
        latent_dim=config.latent_dim,
    )
    scorers = [
        LatentChunkScorer(
            vocab_size=len(vocabulary.token_to_id),
            chunk_size=config.chunk_size,
            latent_dim=config.latent_dim,
        )
        for _ in range(3)
    ]
    modules = {
        "autoencoder": autoencoder,
        "reference_scorer": scorers[0],
        "old_scorer": scorers[1],
        "new_scorer": scorers[2],
    }
    try:
        for name, module in modules.items():
            module.load_state_dict(state_dicts[name], strict=True)
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError("learned_chunk_checkpoint_invalid") from exc
    bundle = LearnedChunkEncoderBundle(
        config=config,
        vocabulary=vocabulary,
        autoencoder=autoencoder,
        reference_scorer=scorers[0],
        old_scorer=scorers[1],
        new_scorer=scorers[2],
        training_report=training_report,
        reference_scorer_state_id=_state_id(scorers[0]),
        old_scorer_state_id=_state_id(scorers[1]),
        new_scorer_state_id=_state_id(scorers[2]),
        autoencoder_state_id=_state_id(autoencoder),
    )
    if bundle.checkpoint_manifest() != dict(manifest):
        raise ValueError("learned_chunk_checkpoint_identity_mismatch")
    return bundle


def validate_learned_chunk_actions(
    actions: Sequence[ActionUnit],
) -> ActionLogprobStats:
    if any(
        action.kind != "learned_chunk" or action.metadata.get("action/fallback") is True
        for action in actions
    ):
        raise ValueError("fallback_actions_in_learned_chunk_metrics")
    stats = action_logprob_stats(actions)
    if (
        not actions
        or stats.old_logprob_coverage != 1.0
        or stats.new_logprob_coverage != 1.0
        or stats.reference_logprob_coverage != 1.0
    ):
        raise ValueError("missing_or_detached_chunk_logprobs")
    return stats


def validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "encoder",
        "decoder",
        "reference_scorer",
        "old_scorer",
        "new_scorer",
        "vocab",
        "config",
    }
    if set(manifest) != required or any(manifest.get(key) is None for key in required):
        raise NotImplementedError("learned_chunk_checkpoint_incomplete")
    if any(
        not isinstance(manifest[key], str) or not manifest[key]
        for key in (
            "encoder",
            "decoder",
            "reference_scorer",
            "old_scorer",
            "new_scorer",
        )
    ):
        raise ValueError("learned_chunk_checkpoint_invalid")
    vocabulary = manifest["vocab"]
    config = manifest["config"]
    if not isinstance(vocabulary, Mapping) or set(vocabulary) != {
        "hash",
        "token_to_id",
    }:
        raise ValueError("learned_chunk_checkpoint_invalid")
    if not isinstance(vocabulary["hash"], str) or not isinstance(
        vocabulary["token_to_id"], Mapping
    ):
        raise ValueError("learned_chunk_checkpoint_invalid")
    if not isinstance(config, Mapping) or set(config) != {
        "chunk_size",
        "latent_dim",
        "reconstruction_threshold",
        "proof_scope",
    }:
        raise ValueError("learned_chunk_checkpoint_invalid")


def compute_reconstruction_accuracy(
    original_unpadded_token_ids: Sequence[int],
    reconstructed_unpadded_token_ids: Sequence[int],
) -> float:
    if any(token_id == 0 for token_id in original_unpadded_token_ids):
        raise ValueError("invalid_reconstruction_target")
    if not original_unpadded_token_ids:
        return 0.0
    matches = sum(
        1
        for expected, actual in zip(
            original_unpadded_token_ids,
            reconstructed_unpadded_token_ids,
        )
        if expected == actual
    )
    if len(original_unpadded_token_ids) != len(reconstructed_unpadded_token_ids):
        return min(matches, len(original_unpadded_token_ids) - 1) / len(
            original_unpadded_token_ids
        )
    return matches / len(original_unpadded_token_ids)


def _fallback_report(
    text: str,
    *,
    failure_mode: str,
    decoded_text: str,
    token_count: int,
    reconstruction_accuracy: float = 0.0,
) -> ChunkEncodeReport:
    token_actions = [
        ActionUnit(
            kind=action.kind,
            payload=action.payload,
            token_count=action.token_count,
            text=action.text,
            metadata={
                **dict(action.metadata),
                "action/fallback": True,
                "reconstruction/safe": False,
                "failure/mode": failure_mode,
            },
        )
        for action in TokenActionCodec().encode(text)
    ]
    metadata = {
        "action/fallback": True,
        "reconstruction/accuracy": reconstruction_accuracy,
        "reconstruction/safe": False,
        "failure/mode": failure_mode,
    }
    return ChunkEncodeReport(
        actions=token_actions,
        decoded_text=decoded_text,
        reconstruction_accuracy=reconstruction_accuracy,
        passed_reconstruction_threshold=False,
        fallback=True,
        metrics={
            "actions": float(len(token_actions)),
            "semantic_bandwidth": semantic_bandwidth(token_actions),
            "old_logprob_coverage": 0.0,
            "new_logprob_coverage": 0.0,
            "reference_logprob_coverage": 0.0,
            "new_logprob_improved_chunks": 0.0,
            "mean_old_reference_logprob_abs_delta": 0.0,
            "fallback_source_tokens": float(token_count),
        },
        metadata=metadata,
    )


def _split_corpus(corpus: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not corpus:
        raise ValueError("chunk_encoder_input_limit_exceeded")
    holdout_count = max(1, math.ceil(len(corpus) * 0.2))
    train_count = len(corpus) - holdout_count
    if train_count <= 0:
        raise ValueError("chunk_encoder_input_limit_exceeded")
    return tuple(corpus[:train_count]), tuple(corpus[train_count:])


def _texts_to_chunks(
    texts: Sequence[str],
    vocabulary: WhitespaceVocabulary,
    config: LearnedChunkEncoderConfig,
) -> tuple[list[list[int]], list[int]]:
    chunks: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        _validate_input_limits(text, config)
        token_ids, unknown = vocabulary.encode(text)
        if unknown:
            raise ValueError("unknown_token")
        text_chunks, text_lengths = _chunk_token_ids(
            token_ids,
            chunk_size=config.chunk_size,
            pad_id=vocabulary.pad_id,
        )
        chunks.extend(text_chunks)
        lengths.extend(text_lengths)
    return chunks, lengths


def _chunk_token_ids(
    token_ids: Sequence[int],
    *,
    chunk_size: int,
    pad_id: int,
) -> tuple[list[list[int]], list[int]]:
    chunks: list[list[int]] = []
    lengths: list[int] = []
    for index in range(0, len(token_ids), chunk_size):
        chunk = list(token_ids[index : index + chunk_size])
        lengths.append(len(chunk))
        chunks.append(chunk + [pad_id] * (chunk_size - len(chunk)))
    return chunks, lengths


def _unpad_decoded_chunks(
    decoded_chunks: Sequence[Sequence[int]],
    lengths: Sequence[int],
) -> list[int]:
    token_ids: list[int] = []
    for chunk, length in zip(decoded_chunks, lengths):
        token_ids.extend(int(token_id) for token_id in chunk[:length])
    return token_ids


def _chunk_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def _autoencoder_accuracy(model: TinyChunkAutoencoder, chunks: torch.Tensor) -> float:
    logits, _ = model(chunks)
    predictions = logits.argmax(dim=-1)
    return float((predictions == chunks).float().mean().item())


def _validate_input_limits(text: str, config: LearnedChunkEncoderConfig) -> None:
    if len(text.encode("utf-8")) > config.max_input_bytes:
        raise ValueError("chunk_encoder_input_limit_exceeded")
    if len(text.split()) > config.max_source_tokens:
        raise ValueError("chunk_encoder_input_limit_exceeded")


def _check_timeout(start: float, timeout_s: float) -> None:
    if time.monotonic() - start > timeout_s:
        raise TimeoutError("chunk_encoder_smoke_timeout")


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def _state_id(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]
