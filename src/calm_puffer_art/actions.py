from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence

from .types import ActionUnit


ACTION_SPACE_STATE_KEY = "action_space/state"


class ActionCodec(Protocol):
    """Encodes text or structured outputs into higher-level policy decisions."""

    name: str

    def encode(self, value: str) -> list[ActionUnit]:
        ...

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        ...


def semantic_bandwidth(actions: Sequence[ActionUnit]) -> float:
    """Average source-token payload per policy decision."""

    if not actions:
        return 0.0
    return sum(max(0, action.token_count) for action in actions) / len(actions)


def action_codec_key(codec: ActionCodec) -> str:
    """Stable key used for scheduler arms and action-space telemetry."""

    name = getattr(codec, "name", codec.__class__.__name__)
    values = getattr(codec, "__dict__", {})
    public_values = {
        key: value
        for key, value in values.items()
        if not key.startswith("_") and key != "name"
    }
    if not public_values:
        return str(name)
    suffix = ",".join(f"{key}={public_values[key]}" for key in sorted(public_values))
    return f"{name}({suffix})"


def safe_metric_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _tokens(text: str) -> list[str]:
    return text.split()


def _chunked(items: Sequence[str], chunk_size: int) -> list[list[str]]:
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


@dataclass(frozen=True)
class TokenActionCodec:
    """Baseline token-ish codec for comparing semantic bandwidth."""

    name: str = "token"

    def encode(self, value: str) -> list[ActionUnit]:
        return [
            ActionUnit(kind="token", payload=token, token_count=1, text=token)
            for token in _tokens(value)
        ]

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        return " ".join(str(action.payload) for action in actions)


@dataclass(frozen=True)
class ChunkActionCodec:
    """Groups K tokens into one decision unit."""

    chunk_size: int = 4
    name: str = "chunk"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def encode(self, value: str) -> list[ActionUnit]:
        actions: list[ActionUnit] = []
        for index, chunk in enumerate(_chunked(_tokens(value), self.chunk_size)):
            text = " ".join(chunk)
            actions.append(
                ActionUnit(
                    kind="chunk",
                    payload=tuple(chunk),
                    token_count=len(chunk),
                    text=text,
                    metadata={"chunk_index": index, "chunk_size": self.chunk_size},
                )
            )
        return actions

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        tokens: list[str] = []
        for action in actions:
            if isinstance(action.payload, (list, tuple)):
                tokens.extend(str(token) for token in action.payload)
            else:
                tokens.extend(_tokens(action.text or str(action.payload)))
        return " ".join(tokens)


@dataclass(frozen=True)
class LatentPatchActionCodec:
    """Inspectable stand-in for CALM-style chunk embeddings.

    The vector is deterministic and non-learned. The original text is retained
    so examples and tests can round-trip without implementing a neural decoder.
    """

    patch_size: int = 4
    latent_size: int = 8
    name: str = "latent_patch"

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.latent_size <= 0:
            raise ValueError("latent_size must be positive")

    def encode(self, value: str) -> list[ActionUnit]:
        actions: list[ActionUnit] = []
        for index, chunk in enumerate(_chunked(_tokens(value), self.patch_size)):
            text = " ".join(chunk)
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = tuple(round(byte / 255.0, 6) for byte in digest[: self.latent_size])
            actions.append(
                ActionUnit(
                    kind="latent_patch",
                    payload=vector,
                    token_count=len(chunk),
                    text=text,
                    metadata={
                        "patch_index": index,
                        "patch_size": self.patch_size,
                        "latent_size": self.latent_size,
                        "reconstruction_text": text,
                    },
                )
            )
        return actions

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        return " ".join(
            str(action.metadata.get("reconstruction_text", action.text))
            for action in actions
        ).strip()


@dataclass(frozen=True)
class CommandActionCodec:
    """Represents tool calls or workflow commands as single decisions."""

    name: str = "command"

    def encode(self, value: str) -> list[ActionUnit]:
        parsed = self._parse(value)
        commands = parsed if isinstance(parsed, list) else [parsed]
        actions: list[ActionUnit] = []
        for index, command in enumerate(commands):
            text = json.dumps(command, sort_keys=True, separators=(",", ":"))
            actions.append(
                ActionUnit(
                    kind="command",
                    payload=command,
                    token_count=max(1, len(_tokens(text))),
                    text=text,
                    metadata={"command_index": index},
                )
            )
        return actions

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        return "\n".join(
            json.dumps(action.payload, sort_keys=True, separators=(",", ":"))
            for action in actions
        )

    @staticmethod
    def _parse(value: str) -> Any:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"name": "say", "args": {"text": value}}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
        return {"name": "emit", "args": {"value": parsed}}


@dataclass(frozen=True)
class ReasoningStepCodec:
    """Compresses one line of reasoning or planning into one action unit."""

    name: str = "reasoning_step"

    def encode(self, value: str) -> list[ActionUnit]:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines and value.strip():
            lines = [value.strip()]
        return [
            ActionUnit(
                kind="reasoning_step",
                payload=line,
                token_count=max(1, len(_tokens(line))),
                text=line,
                metadata={"step_index": index},
            )
            for index, line in enumerate(lines)
        ]

    def decode(self, actions: Sequence[ActionUnit]) -> str:
        return "\n".join(str(action.payload) for action in actions)


@dataclass
class AdaptiveActionSpace:
    """Promotes higher-bandwidth action codecs from objective feedback.

    This is intentionally conservative: the baseline token codec remains active,
    larger chunk codecs are added only after the current chunk size has positive
    objective signal, strong action quality, and an acceptable unsafe rate in
    scheduler metrics. Opt-in latent-patch candidates use the same evidence
    gate and can be retired from their own objective feedback.
    """

    min_chunk_size: int = 2
    max_chunk_size: int = 8
    promotion_objective_threshold: float = 0.0
    promotion_parent_margin: float = 0.0
    promotion_quality_threshold: float = 0.95
    promotion_semantic_bandwidth_threshold: float = 1.0
    promotion_max_reconstruction_drift: float = 0.05
    promotion_min_pulls: int = 1
    unsafe_rate_threshold: float = 0.0
    demotion_objective_threshold: float = 0.0
    demotion_parent_margin: float = 0.0
    demotion_quality_threshold: float = 0.5
    demotion_semantic_bandwidth_threshold: float = 1.0
    demotion_max_reconstruction_drift: float = 0.05
    demotion_min_pulls: int = 2
    promote_latent_patches: bool = False
    latent_patch_latent_size: int = 8
    include_token: bool = True
    seed_codecs: Sequence[ActionCodec] = field(default_factory=tuple)
    _codecs: list[ActionCodec] = field(default_factory=list, init=False)
    _disabled_codec_keys: set[str] = field(default_factory=set, init=False)
    _promotions: int = field(default=0, init=False)
    _demotions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if self.promotion_quality_threshold < 0:
            raise ValueError("promotion_quality_threshold must be non-negative")
        if self.promotion_parent_margin < 0:
            raise ValueError("promotion_parent_margin must be non-negative")
        if self.promotion_semantic_bandwidth_threshold < 0:
            raise ValueError(
                "promotion_semantic_bandwidth_threshold must be non-negative"
            )
        if self.promotion_max_reconstruction_drift < 0:
            raise ValueError("promotion_max_reconstruction_drift must be non-negative")
        if self.promotion_min_pulls < 0:
            raise ValueError("promotion_min_pulls must be non-negative")
        if self.unsafe_rate_threshold < 0:
            raise ValueError("unsafe_rate_threshold must be non-negative")
        if self.demotion_quality_threshold < 0:
            raise ValueError("demotion_quality_threshold must be non-negative")
        if self.demotion_parent_margin < 0:
            raise ValueError("demotion_parent_margin must be non-negative")
        if self.demotion_semantic_bandwidth_threshold < 0:
            raise ValueError(
                "demotion_semantic_bandwidth_threshold must be non-negative"
            )
        if self.demotion_max_reconstruction_drift < 0:
            raise ValueError("demotion_max_reconstruction_drift must be non-negative")
        if self.demotion_min_pulls < 0:
            raise ValueError("demotion_min_pulls must be non-negative")
        if self.latent_patch_latent_size <= 0:
            raise ValueError("latent_patch_latent_size must be positive")
        if self.include_token:
            self.add_codec(TokenActionCodec())
        self.add_codec(ChunkActionCodec(chunk_size=self.min_chunk_size))
        for codec in self.seed_codecs:
            self.add_codec(codec)

    @property
    def codecs(self) -> tuple[ActionCodec, ...]:
        return tuple(self._codecs)

    def add_codec(self, codec: ActionCodec) -> bool:
        key = action_codec_key(codec)
        if any(action_codec_key(existing) == key for existing in self._codecs):
            return False
        self._disabled_codec_keys.discard(key)
        self._codecs.append(codec)
        return True

    def update_from_metrics(self, metrics: Mapping[str, float]) -> None:
        disabled_this_update = self._demote_from_metrics(metrics)
        for codec in tuple(self._codecs):
            if not isinstance(codec, ChunkActionCodec):
                continue
            signal = self._codec_signal(codec, metrics)
            if signal.pulls < self.promotion_min_pulls:
                continue
            if signal.objective <= self.promotion_objective_threshold:
                continue
            if signal.quality < self.promotion_quality_threshold:
                continue
            if (
                signal.semantic_bandwidth
                < self.promotion_semantic_bandwidth_threshold
            ):
                continue
            if signal.reconstruction_drift > self.promotion_max_reconstruction_drift:
                continue
            if signal.unsafe_rate > self.unsafe_rate_threshold:
                continue
            if not self._parent_allows_promotion(codec, signal, metrics):
                continue
            if codec.chunk_size < self.max_chunk_size:
                next_size = min(self.max_chunk_size, codec.chunk_size * 2)
                candidate = ChunkActionCodec(chunk_size=next_size)
                candidate_key = action_codec_key(candidate)
                if (
                    candidate_key in self._disabled_codec_keys
                    or candidate_key in disabled_this_update
                ):
                    pass
                elif self.add_codec(candidate):
                    self._promotions += 1
            if self.promote_latent_patches:
                latent_candidate = LatentPatchActionCodec(
                    patch_size=codec.chunk_size,
                    latent_size=self.latent_patch_latent_size,
                )
                latent_key = action_codec_key(latent_candidate)
                if (
                    latent_key in self._disabled_codec_keys
                    or latent_key in disabled_this_update
                ):
                    continue
                if self.add_codec(latent_candidate):
                    self._promotions += 1

    def _parent_allows_promotion(
        self,
        codec: ActionCodec,
        signal: "_CodecSignal",
        metrics: Mapping[str, float],
    ) -> bool:
        parent = self._parent_codec(codec)
        if parent is None:
            return True
        parent_signal = self._codec_signal(parent, metrics)
        if parent_signal.pulls <= 0.0:
            return True
        if parent_signal.unsafe_rate > self.unsafe_rate_threshold:
            return True
        return signal.objective > (
            parent_signal.objective + self.promotion_parent_margin
        )

    def _demote_from_metrics(self, metrics: Mapping[str, float]) -> set[str]:
        disabled: set[str] = set()
        for codec in tuple(self._codecs):
            if self._is_protected_codec(codec):
                continue
            signal = self._codec_signal(codec, metrics)
            if signal.pulls < self.demotion_min_pulls:
                continue
            if not self._should_demote(codec, signal, metrics):
                continue
            for candidate in tuple(self._codecs):
                if self._should_disable_with(codec, candidate):
                    key = action_codec_key(candidate)
                    self._codecs.remove(candidate)
                    self._disabled_codec_keys.add(key)
                    disabled.add(key)
                    self._demotions += 1
        return disabled

    def _should_demote(
        self,
        codec: ActionCodec,
        signal: "_CodecSignal",
        metrics: Mapping[str, float],
    ) -> bool:
        if signal.unsafe_rate > self.unsafe_rate_threshold:
            return True
        if signal.quality < self.demotion_quality_threshold:
            return True
        if signal.semantic_bandwidth < self.demotion_semantic_bandwidth_threshold:
            return True
        if signal.reconstruction_drift > self.demotion_max_reconstruction_drift:
            return True
        if self._parent_outperforms(codec, signal, metrics):
            return True
        return signal.objective <= self.demotion_objective_threshold

    def _parent_outperforms(
        self,
        codec: ActionCodec,
        signal: "_CodecSignal",
        metrics: Mapping[str, float],
    ) -> bool:
        parent = self._parent_codec(codec)
        if parent is None:
            return False
        parent_signal = self._codec_signal(parent, metrics)
        if parent_signal.pulls < self.demotion_min_pulls:
            return False
        if parent_signal.unsafe_rate > self.unsafe_rate_threshold:
            return False
        if parent_signal.quality < self.demotion_quality_threshold:
            return False
        return parent_signal.objective > (
            signal.objective + self.demotion_parent_margin
        )

    def _is_protected_codec(self, codec: ActionCodec) -> bool:
        if isinstance(codec, TokenActionCodec):
            return True
        return (
            isinstance(codec, ChunkActionCodec)
            and codec.chunk_size <= self.min_chunk_size
        )

    def _should_disable_with(
        self,
        demoted_codec: ActionCodec,
        candidate: ActionCodec,
    ) -> bool:
        if self._is_protected_codec(candidate):
            return False
        if action_codec_key(candidate) == action_codec_key(demoted_codec):
            return True
        if not isinstance(demoted_codec, ChunkActionCodec):
            return False
        if isinstance(candidate, ChunkActionCodec):
            return candidate.chunk_size >= demoted_codec.chunk_size
        return (
            isinstance(candidate, LatentPatchActionCodec)
            and candidate.patch_size >= demoted_codec.chunk_size
        )

    def _parent_codec(self, codec: ActionCodec) -> ActionCodec | None:
        if isinstance(codec, ChunkActionCodec):
            return self._nearest_smaller_chunk(codec) or self._token_codec()
        if isinstance(codec, LatentPatchActionCodec):
            chunks = [
                candidate
                for candidate in self._codecs
                if (
                    isinstance(candidate, ChunkActionCodec)
                    and candidate.chunk_size <= codec.patch_size
                )
            ]
            if not chunks:
                return None
            return max(chunks, key=lambda candidate: candidate.chunk_size)
        return None

    def _token_codec(self) -> TokenActionCodec | None:
        for codec in self._codecs:
            if isinstance(codec, TokenActionCodec):
                return codec
        return None

    def _nearest_smaller_chunk(
        self,
        codec: ChunkActionCodec,
    ) -> ChunkActionCodec | None:
        smaller = [
            candidate
            for candidate in self._codecs
            if (
                isinstance(candidate, ChunkActionCodec)
                and candidate.chunk_size < codec.chunk_size
            )
        ]
        if not smaller:
            return None
        return max(smaller, key=lambda candidate: candidate.chunk_size)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-friendly action-space state for checkpoint/resume."""

        return {
            "version": 1,
            "config": {
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "promotion_objective_threshold": (
                    self.promotion_objective_threshold
                ),
                "promotion_parent_margin": self.promotion_parent_margin,
                "promotion_quality_threshold": self.promotion_quality_threshold,
                "promotion_semantic_bandwidth_threshold": (
                    self.promotion_semantic_bandwidth_threshold
                ),
                "promotion_max_reconstruction_drift": (
                    self.promotion_max_reconstruction_drift
                ),
                "promotion_min_pulls": self.promotion_min_pulls,
                "unsafe_rate_threshold": self.unsafe_rate_threshold,
                "demotion_objective_threshold": self.demotion_objective_threshold,
                "demotion_parent_margin": self.demotion_parent_margin,
                "demotion_quality_threshold": self.demotion_quality_threshold,
                "demotion_semantic_bandwidth_threshold": (
                    self.demotion_semantic_bandwidth_threshold
                ),
                "demotion_max_reconstruction_drift": (
                    self.demotion_max_reconstruction_drift
                ),
                "demotion_min_pulls": self.demotion_min_pulls,
                "promote_latent_patches": self.promote_latent_patches,
                "latent_patch_latent_size": self.latent_patch_latent_size,
                "include_token": self.include_token,
            },
            "active_codecs": [_codec_to_state(codec) for codec in self._codecs],
            "disabled_codec_keys": sorted(self._disabled_codec_keys),
            "learning_state": {
                "promotions": self._promotions,
                "demotions": self._demotions,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load state produced by :meth:`state_dict`.

        Built-in codecs are reconstructed directly. Unknown custom codecs are
        restored only if an equivalent codec is already present on this action
        space, preserving the user-code ownership of arbitrary action units.
        """

        config = _mapping_state(state.get("config"))
        self.min_chunk_size = _state_int(
            config.get("min_chunk_size"),
            self.min_chunk_size,
        )
        self.max_chunk_size = _state_int(
            config.get("max_chunk_size"),
            self.max_chunk_size,
        )
        self.promotion_objective_threshold = _state_float(
            config.get("promotion_objective_threshold"),
            self.promotion_objective_threshold,
        )
        self.promotion_parent_margin = _state_float(
            config.get("promotion_parent_margin"),
            self.promotion_parent_margin,
        )
        self.promotion_quality_threshold = _state_float(
            config.get("promotion_quality_threshold"),
            self.promotion_quality_threshold,
        )
        self.promotion_semantic_bandwidth_threshold = _state_float(
            config.get("promotion_semantic_bandwidth_threshold"),
            self.promotion_semantic_bandwidth_threshold,
        )
        self.promotion_max_reconstruction_drift = _state_float(
            config.get("promotion_max_reconstruction_drift"),
            self.promotion_max_reconstruction_drift,
        )
        self.promotion_min_pulls = _state_int(
            config.get("promotion_min_pulls"),
            self.promotion_min_pulls,
        )
        self.unsafe_rate_threshold = _state_float(
            config.get("unsafe_rate_threshold"),
            self.unsafe_rate_threshold,
        )
        self.demotion_objective_threshold = _state_float(
            config.get("demotion_objective_threshold"),
            self.demotion_objective_threshold,
        )
        self.demotion_parent_margin = _state_float(
            config.get("demotion_parent_margin"),
            self.demotion_parent_margin,
        )
        self.demotion_quality_threshold = _state_float(
            config.get("demotion_quality_threshold"),
            self.demotion_quality_threshold,
        )
        self.demotion_semantic_bandwidth_threshold = _state_float(
            config.get("demotion_semantic_bandwidth_threshold"),
            self.demotion_semantic_bandwidth_threshold,
        )
        self.demotion_max_reconstruction_drift = _state_float(
            config.get("demotion_max_reconstruction_drift"),
            self.demotion_max_reconstruction_drift,
        )
        self.demotion_min_pulls = _state_int(
            config.get("demotion_min_pulls"),
            self.demotion_min_pulls,
        )
        self.promote_latent_patches = _state_bool(
            config.get("promote_latent_patches"),
            self.promote_latent_patches,
        )
        self.latent_patch_latent_size = _state_int(
            config.get("latent_patch_latent_size"),
            self.latent_patch_latent_size,
        )
        self.include_token = _state_bool(
            config.get("include_token"),
            self.include_token,
        )
        self.min_chunk_size = max(1, self.min_chunk_size)
        self.max_chunk_size = max(self.min_chunk_size, self.max_chunk_size)
        self.promotion_semantic_bandwidth_threshold = max(
            0.0,
            self.promotion_semantic_bandwidth_threshold,
        )
        self.promotion_parent_margin = max(0.0, self.promotion_parent_margin)
        self.promotion_max_reconstruction_drift = max(
            0.0,
            self.promotion_max_reconstruction_drift,
        )
        self.promotion_min_pulls = max(0, self.promotion_min_pulls)
        self.demotion_semantic_bandwidth_threshold = max(
            0.0,
            self.demotion_semantic_bandwidth_threshold,
        )
        self.demotion_parent_margin = max(0.0, self.demotion_parent_margin)
        self.demotion_max_reconstruction_drift = max(
            0.0,
            self.demotion_max_reconstruction_drift,
        )
        self.latent_patch_latent_size = max(1, self.latent_patch_latent_size)

        active_codec_states = state.get("active_codecs")
        if isinstance(active_codec_states, (list, tuple)) and not isinstance(
            active_codec_states,
            (str, bytes),
        ):
            existing_by_key = {
                action_codec_key(codec): codec for codec in self._codecs
            }
            restored_codecs: list[ActionCodec] = []
            restored_keys: set[str] = set()
            for codec_state in active_codec_states:
                codec = _codec_from_state(codec_state, existing_by_key)
                if codec is None:
                    continue
                key = action_codec_key(codec)
                if key in restored_keys:
                    continue
                restored_codecs.append(codec)
                restored_keys.add(key)
            if restored_codecs:
                self._codecs = restored_codecs
                disabled = _string_set_state(state.get("disabled_codec_keys"))
                self._disabled_codec_keys = disabled - restored_keys

        learning_state = _mapping_state(state.get("learning_state"))
        self._promotions = _state_int(
            learning_state.get("promotions"),
            self._promotions,
        )
        self._demotions = _state_int(
            learning_state.get("demotions"),
            self._demotions,
        )

    def metrics(self) -> dict[str, float]:
        values = {
            "action_space/active_codecs": float(len(self._codecs)),
            "action_space/promotions": float(self._promotions),
            "action_space/demotions": float(self._demotions),
            "action_space/disabled_codecs": float(len(self._disabled_codec_keys)),
            "action_space/max_chunk_size": float(self._active_max_chunk_size()),
        }
        for codec in self._codecs:
            values[
                f"action_space/codec/{safe_metric_key(action_codec_key(codec))}/active"
            ] = 1.0
        for key in self._disabled_codec_keys:
            values[f"action_space/codec/{safe_metric_key(key)}/disabled"] = 1.0
        return values

    def _active_max_chunk_size(self) -> int:
        chunk_sizes = [
            codec.chunk_size
            for codec in self._codecs
            if isinstance(codec, ChunkActionCodec)
        ]
        return max(chunk_sizes) if chunk_sizes else 0

    def _codec_signal(
        self,
        codec: ActionCodec,
        metrics: Mapping[str, float],
    ) -> "_CodecSignal":
        fragment = safe_metric_key(action_codec_key(codec))
        objective_values: list[float] = []
        quality_values: list[float] = []
        unsafe_values: list[float] = []
        failure_values: list[float] = []
        pull_values: list[float] = []
        semantic_bandwidth_values: list[float] = []
        reconstruction_drift_values: list[float] = []
        for key, value in metrics.items():
            if not key.startswith("scheduler/arm/") or f"_{fragment}/" not in key:
                continue
            if key.endswith("/policy_improvement_objective_ema"):
                objective_values.append(float(value))
            elif key.endswith("/objective_score"):
                objective_values.append(float(value))
            elif key.endswith("/marginal_objective_ema"):
                objective_values.append(float(value))
            elif key.endswith("/action_quality_ema"):
                quality_values.append(float(value))
            elif key.endswith("/unsafe_rate"):
                unsafe_values.append(float(value))
            elif key.endswith("/failure_rate"):
                failure_values.append(float(value))
            elif key.endswith("/pulls"):
                pull_values.append(float(value))
            elif key.endswith("/semantic_bandwidth_tokens_per_decision"):
                semantic_bandwidth_values.append(float(value))
            elif key.endswith("/reconstruction_max_drift"):
                reconstruction_drift_values.append(float(value))
            elif key.endswith("/reconstruction_drift_ema"):
                reconstruction_drift_values.append(float(value))
        return _CodecSignal(
            objective=max(objective_values) if objective_values else 0.0,
            quality=min(quality_values) if quality_values else 0.0,
            unsafe_rate=max(unsafe_values + failure_values)
            if unsafe_values or failure_values
            else 0.0,
            pulls=sum(pull_values),
            semantic_bandwidth=(
                max(semantic_bandwidth_values)
                if semantic_bandwidth_values
                else 0.0
            ),
            reconstruction_drift=(
                max(reconstruction_drift_values)
                if reconstruction_drift_values
                else 0.0
            ),
        )


@dataclass(frozen=True)
class _CodecSignal:
    objective: float
    quality: float
    unsafe_rate: float
    pulls: float
    semantic_bandwidth: float
    reconstruction_drift: float


def action_space_checkpoint_metadata(action_space: Any | None) -> dict[str, Any]:
    """Return checkpoint metadata for action spaces with snapshot support."""

    if action_space is None:
        return {}
    state_dict = getattr(action_space, "state_dict", None)
    if state_dict is None:
        return {}
    state = state_dict()
    if not isinstance(state, Mapping):
        return {}
    return {ACTION_SPACE_STATE_KEY: state}


def _codec_to_state(codec: ActionCodec) -> dict[str, Any]:
    key = action_codec_key(codec)
    if isinstance(codec, TokenActionCodec):
        return {"type": "token", "key": key}
    if isinstance(codec, ChunkActionCodec):
        return {
            "type": "chunk",
            "key": key,
            "chunk_size": codec.chunk_size,
        }
    if isinstance(codec, LatentPatchActionCodec):
        return {
            "type": "latent_patch",
            "key": key,
            "patch_size": codec.patch_size,
            "latent_size": codec.latent_size,
        }
    if isinstance(codec, CommandActionCodec):
        return {"type": "command", "key": key}
    if isinstance(codec, ReasoningStepCodec):
        return {"type": "reasoning_step", "key": key}
    return {"type": "unknown", "key": key}


def _codec_from_state(
    value: Any,
    existing_by_key: Mapping[str, ActionCodec],
) -> ActionCodec | None:
    state = _mapping_state(value)
    codec_type = state.get("type")
    key = str(state.get("key", ""))
    if codec_type == "token":
        return TokenActionCodec()
    if codec_type == "chunk":
        return ChunkActionCodec(
            chunk_size=max(1, _state_int(state.get("chunk_size"), 1))
        )
    if codec_type == "latent_patch":
        return LatentPatchActionCodec(
            patch_size=max(1, _state_int(state.get("patch_size"), 1)),
            latent_size=max(1, _state_int(state.get("latent_size"), 1)),
        )
    if codec_type == "command":
        return CommandActionCodec()
    if codec_type == "reasoning_step":
        return ReasoningStepCodec()
    return existing_by_key.get(key)


def _mapping_state(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, MappingABC) else {}


def _string_set_state(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        return set()
    return {str(item) for item in value if item is not None}


def _state_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if isfinite(candidate) else default


def _state_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default
