from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import fmean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Scenario:
    """One task instance that a user-defined agent workflow can attempt."""

    id: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """OpenAI/ART-style message content retained inside trajectories."""

    role: str
    content: str


@dataclass(frozen=True)
class ActionUnit:
    """A policy decision at token, chunk, latent-patch, command, or step level."""

    kind: str
    payload: Any
    token_count: int
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    old_logprob: float | None = None
    new_logprob: float | None = None
    reference_logprob: float | None = None


@dataclass
class Trajectory:
    """Rewarded rollout record submitted to the trainer."""

    scenario_id: str
    policy_step: int
    messages: list[Message]
    actions: list[ActionUnit]
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    exception: str | None = None

    @property
    def action_units(self) -> int:
        return len(self.actions)

    @property
    def token_count(self) -> int:
        return sum(max(0, action.token_count) for action in self.actions)


@dataclass(frozen=True)
class TrajectoryGroup:
    """Same-scenario trajectories trained together, ART/GRPO style."""

    scenario_id: str
    trajectories: tuple[Trajectory, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(t.reward for t in self.trajectories if isfinite(t.reward))

    @property
    def mean_reward(self) -> float:
        rewards = self.rewards
        return fmean(rewards) if rewards else 0.0

    @property
    def policy_steps(self) -> tuple[int, ...]:
        return tuple(t.policy_step for t in self.trajectories)


@dataclass(frozen=True)
class PolicySnapshot:
    """A point-in-time served policy reference."""

    step: int
    policy: Any
    checkpoint_id: str
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainResult:
    """Trainer output for one consumed batch of trajectory groups."""

    policy: Any | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    checkpoint_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionDecision:
    """Decision to publish or reject a trained candidate policy."""

    promoted: bool
    score: float
    baseline_score: float = 0.0
    improvement: float = 0.0
    dollar_seconds: float = 0.0
    reason: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)
    trajectories: tuple[Trajectory, ...] = ()


@dataclass(frozen=True)
class Checkpoint:
    """Published policy checkpoint metadata."""

    step: int
    checkpoint_id: str
    created_at: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSummary:
    """Serializable runtime outcome and telemetry rollup."""

    latest_step: int
    checkpoints: tuple[Checkpoint, ...]
    metrics: Mapping[str, float]
    pending_trajectories: int
    pending_groups: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest_step": self.latest_step,
            "checkpoints": [
                {
                    "step": checkpoint.step,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "created_at": checkpoint.created_at,
                    "metrics": dict(checkpoint.metrics),
                    "metadata": dict(checkpoint.metadata),
                }
                for checkpoint in self.checkpoints
            ],
            "metrics": dict(self.metrics),
            "pending_trajectories": self.pending_trajectories,
            "pending_groups": self.pending_groups,
        }


def mean(values: Sequence[float]) -> float:
    finite_values = [value for value in values if isfinite(value)]
    return fmean(finite_values) if finite_values else 0.0
