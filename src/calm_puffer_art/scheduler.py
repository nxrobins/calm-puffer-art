from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Protocol, Sequence

from .actions import (
    ActionCodec,
    ActionLogprobStats,
    action_logprob_stats,
    safe_metric_key,
)
from .types import Scenario, TrainResult, Trajectory, TrajectoryGroup, mean


SCHEDULER_STATE_KEY = "scheduler/state"
_UNOBSERVED_ARM_SCORE = 1_000_000_000.0
_UNOBSERVED_ARM_INFLIGHT_PENALTY = 1_000_000.0
_UNOBSERVED_ARM_COST_PENALTY_CAP = 999_999.0


@dataclass(frozen=True)
class SchedulerDecision:
    """One closed-loop rollout/control decision."""

    scenario: Scenario
    action_codec: ActionCodec
    arm_id: str
    target_train_batch_groups: int
    max_policy_lag: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def scheduling_action_key(
    *,
    arm_id: str,
    target_train_batch_groups: int,
    max_policy_lag: int,
    active_actor_count: int,
    admission_delay_ms: int,
    action_space_key: str | None = None,
) -> str:
    """Stable key for the full rollout/runtime/action scheduling tuple."""

    key = (
        f"arm={arm_id}"
        f"|cadence={max(1, int(target_train_batch_groups))}"
        f"|lag={max(0, int(max_policy_lag))}"
        f"|actors={max(0, int(active_actor_count))}"
        f"|admission_ms={max(0, int(admission_delay_ms))}"
    )
    normalized_action_space_key = _normalize_key_component(action_space_key)
    if normalized_action_space_key is not None:
        key = f"{key}|action_space={normalized_action_space_key}"
    return key


class AdaptiveScheduler(Protocol):
    """Chooses rollout work and runtime controls from objective feedback."""

    def select_rollout(
        self,
        *,
        scenarios: Sequence[Scenario],
        action_codecs: Sequence[ActionCodec],
        actor_id: int,
        policy_step: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        configured_train_batch_groups: int,
        configured_max_policy_lag: int,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> SchedulerDecision:
        ...

    def target_train_batch_groups(
        self,
        *,
        configured: int,
        pending_groups: int,
        train_queue_pressure: float,
        policy_step: int,
        max_policy_lag: int | None = None,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> int:
        ...

    def max_policy_lag(
        self,
        *,
        configured: int,
        train_queue_pressure: float,
        policy_step: int,
        target_train_batch_groups: int | None = None,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> int:
        ...

    def rollout_admission_delay_s(
        self,
        *,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
        active_actor_count: int | None = None,
        action_space_key: str | None = None,
    ) -> float:
        ...

    def active_actor_count(
        self,
        *,
        configured: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
        action_space_key: str | None = None,
    ) -> int:
        ...

    def cancel_actor_count_decision(self, active_actor_count: int) -> None:
        ...

    def observe_rollout_admission_delay(
        self,
        *,
        seconds: float,
        dollar_seconds: float,
    ) -> None:
        ...

    def observe_rollout(
        self,
        trajectory: Trajectory,
        *,
        accepted: bool,
        dollar_seconds: float,
        queue_wait_dollar_seconds: float = 0.0,
    ) -> None:
        ...

    def cancel_rollout_decision(self, decision: SchedulerDecision) -> None:
        ...

    def observe_train(
        self,
        *,
        groups: Sequence[TrajectoryGroup],
        result: TrainResult,
        duration_s: float,
        dollar_seconds: float,
        policy_step: int,
    ) -> None:
        ...

    def observe_stale_batch(
        self,
        *,
        groups: Sequence[TrajectoryGroup],
        policy_step: int,
        reason: str,
        additional_dollar_seconds: float = 0.0,
    ) -> None:
        ...

    def score_train_groups(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        policy_step: int,
    ) -> float:
        ...

    def record_train_batch_selection(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        priority: float,
        policy_step: int,
    ) -> None:
        ...

    def should_continue_training(
        self,
        *,
        policy_step: int,
        max_train_steps: int,
        pending_train_batches: int,
        train_queue_pressure: float,
    ) -> bool:
        ...

    def metrics(self) -> dict[str, float]:
        ...


@dataclass
class ArmStats:
    decisions: int = 0
    inflight: int = 0
    reserved_rollout_dollar_seconds: float = 0.0
    pulls: int = 0
    accepted: int = 0
    unsafe: int = 0
    failed_rollouts: int = 0
    failure_modes: dict[str, int] = field(default_factory=dict)
    reconstruction_observations: int = 0
    reconstruction_accuracy_ema: float = 1.0
    reconstruction_drift_ema: float = 0.0
    total_reconstruction_accuracy: float = 0.0
    min_reconstruction_accuracy: float = 1.0
    max_reconstruction_drift: float = 0.0
    train_updates: int = 0
    stale_updates: int = 0
    stale_batches: int = 0
    stale_trajectories: int = 0
    reward_ema: float = 0.0
    effective_reward_ema: float = 0.0
    min_effective_reward: float = 0.0
    max_effective_reward: float = 0.0
    last_reward_scale: float = 1.0
    last_normalized_positive_improvement: float = 0.0
    action_quality_ema: float = 1.0
    reward_efficiency_ema: float = 0.0
    marginal_objective_ema: float = 0.0
    policy_improvement_objective_ema: float = 0.0
    objective_observations: int = 0
    objective_mean: float = 0.0
    objective_m2: float = 0.0
    dollar_seconds_ema: float = 0.0
    train_reward_observations: int = 0
    min_train_reward: float = 0.0
    max_train_reward: float = 0.0
    last_train_reward: float = 0.0
    last_train_reward_scale: float = 1.0
    last_train_reward_improvement: float = 0.0
    last_normalized_train_reward_improvement: float = 0.0
    total_reward: float = 0.0
    total_effective_reward: float = 0.0
    total_positive_improvement: float = 0.0
    total_normalized_positive_improvement: float = 0.0
    total_reward_improving_experience: float = 0.0
    total_normalized_reward_improving_experience: float = 0.0
    total_policy_improvement_objective: float = 0.0
    total_stale_penalty_objective: float = 0.0
    total_dollar_seconds: float = 0.0
    rollout_dollar_seconds: float = 0.0
    queue_wait_dollar_seconds: float = 0.0
    admission_dollar_seconds: float = 0.0
    action_units: int = 0
    source_tokens: int = 0
    old_logprob_units: int = 0
    new_logprob_units: int = 0
    reference_logprob_units: int = 0
    old_new_logprob_pairs: int = 0
    old_reference_logprob_pairs: int = 0
    old_logprob_sum: float = 0.0
    new_logprob_sum: float = 0.0
    reference_logprob_sum: float = 0.0
    old_new_logprob_delta_sum: float = 0.0
    old_new_logprob_abs_delta_sum: float = 0.0
    old_reference_logprob_delta_sum: float = 0.0
    importance_ratio_sum: float = 0.0
    stale_experience: float = 0.0


@dataclass
class ControlStats:
    decisions: int = 0
    rollout_updates: int = 0
    train_updates: int = 0
    stale_updates: int = 0
    objective_ema: float = 0.0
    total_objective: float = 0.0
    total_stale_penalty_objective: float = 0.0
    stale_experience: float = 0.0


@dataclass
class ActorStats:
    decisions: int = 0
    inflight: int = 0
    pulls: int = 0
    accepted: int = 0
    unsafe: int = 0
    failed_rollouts: int = 0
    train_updates: int = 0
    stale_updates: int = 0
    stale_batches: int = 0
    stale_trajectories: int = 0
    objective_ema: float = 0.0
    rollout_objective_ema: float = 0.0
    train_objective_ema: float = 0.0
    total_objective: float = 0.0
    total_rollout_objective: float = 0.0
    total_train_objective: float = 0.0
    total_stale_penalty_objective: float = 0.0
    total_dollar_seconds: float = 0.0
    rollout_dollar_seconds: float = 0.0
    queue_wait_dollar_seconds: float = 0.0
    admission_dollar_seconds: float = 0.0
    stale_experience: float = 0.0
    action_units: int = 0
    source_tokens: int = 0


class ObjectiveScheduler:
    """Bandit-style scheduler for reward improvement per dollar-second.

    The scheduler treats each `(scenario, action_codec)` pair as an arm. It
    explores untried arms first, then chooses the arm with the best weighted
    estimate of marginal reward improvement per dollar-second plus a small
    UCB-style exploration bonus. Raw reward efficiency is an explicit optional
    term, not part of the default objective.
    """

    def __init__(
        self,
        *,
        min_train_batch_groups: int = 1,
        max_train_batch_groups: int | None = None,
        min_policy_lag: int = 1,
        max_policy_lag: int | None = None,
        min_actor_count: int = 1,
        max_actor_count: int | None = None,
        ema_alpha: float = 0.25,
        exploration_bonus: float = 0.2,
        objective_threshold: float = 1e-6,
        unsafe_penalty: float = 1.0,
        rollout_objective_weight: float = 1.0,
        train_objective_weight: float = 1.0,
        reward_efficiency_weight: float = 0.0,
        stale_penalty_weight: float = 1.0,
        staleness_priority_weight: float = 1.0,
        off_policy_priority_weight: float = 1.0,
        off_policy_cadence_tightening_threshold: float = 0.0,
        off_policy_lag_tightening_threshold: float = 0.0,
        confidence_penalty_weight: float = 0.0,
        control_exploration_bonus: float = 0.1,
        rollout_cadence_lag_control_weight: float = 0.0,
        joint_action_objective_weight: float = 1.0,
        train_selection_objective_weight: float = 1.0,
        max_control_candidate_values: int = 8,
        min_rollout_coverage_fraction: float = 0.0,
        max_rollout_coverage_cost_fraction: float | None = None,
        min_train_steps: int = 1,
        roi_patience: int | None = None,
        min_train_objective: float = 0.0,
        continuation_objective: str = "accounted",
        control_train_objective: str = "accounted",
        max_accounted_dollar_seconds: float | None = None,
        max_rollout_admission_delay_s: float = 0.0,
        rollout_admission_pressure_threshold: float = 0.75,
        rollout_admission_positive_signal_scale: float = 0.25,
        reconstruction_drift_threshold: float = 0.95,
        reward_scale_normalization: str = "none",
    ) -> None:
        if min_train_batch_groups <= 0:
            raise ValueError("min_train_batch_groups must be positive")
        if min_policy_lag < 0:
            raise ValueError("min_policy_lag must be non-negative")
        if min_actor_count <= 0:
            raise ValueError("min_actor_count must be positive")
        if max_actor_count is not None and max_actor_count < min_actor_count:
            raise ValueError("max_actor_count must be >= min_actor_count")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if exploration_bonus < 0:
            raise ValueError("exploration_bonus must be non-negative")
        if unsafe_penalty < 0:
            raise ValueError("unsafe_penalty must be non-negative")
        if rollout_objective_weight < 0:
            raise ValueError("rollout_objective_weight must be non-negative")
        if train_objective_weight < 0:
            raise ValueError("train_objective_weight must be non-negative")
        if reward_efficiency_weight < 0:
            raise ValueError("reward_efficiency_weight must be non-negative")
        if stale_penalty_weight < 0:
            raise ValueError("stale_penalty_weight must be non-negative")
        if staleness_priority_weight < 0:
            raise ValueError("staleness_priority_weight must be non-negative")
        if off_policy_priority_weight < 0:
            raise ValueError("off_policy_priority_weight must be non-negative")
        if off_policy_cadence_tightening_threshold < 0:
            raise ValueError(
                "off_policy_cadence_tightening_threshold must be non-negative"
            )
        if off_policy_lag_tightening_threshold < 0:
            raise ValueError(
                "off_policy_lag_tightening_threshold must be non-negative"
            )
        if confidence_penalty_weight < 0:
            raise ValueError("confidence_penalty_weight must be non-negative")
        if control_exploration_bonus < 0:
            raise ValueError("control_exploration_bonus must be non-negative")
        if rollout_cadence_lag_control_weight < 0:
            raise ValueError(
                "rollout_cadence_lag_control_weight must be non-negative"
            )
        if joint_action_objective_weight < 0:
            raise ValueError("joint_action_objective_weight must be non-negative")
        if train_selection_objective_weight < 0:
            raise ValueError(
                "train_selection_objective_weight must be non-negative"
            )
        if max_control_candidate_values <= 0:
            raise ValueError("max_control_candidate_values must be positive")
        if not 0 <= min_rollout_coverage_fraction <= 1:
            raise ValueError(
                "min_rollout_coverage_fraction must be in [0, 1]"
            )
        if (
            max_rollout_coverage_cost_fraction is not None
            and not 0 < max_rollout_coverage_cost_fraction <= 1
        ):
            raise ValueError(
                "max_rollout_coverage_cost_fraction must be in (0, 1] when set"
            )
        if min_train_steps < 0:
            raise ValueError("min_train_steps must be non-negative")
        if roi_patience is not None and roi_patience <= 0:
            raise ValueError("roi_patience must be positive when set")
        if continuation_objective not in {"train", "accounted"}:
            raise ValueError("continuation_objective must be 'train' or 'accounted'")
        if control_train_objective not in {"train", "accounted"}:
            raise ValueError(
                "control_train_objective must be 'train' or 'accounted'"
            )
        if (
            max_accounted_dollar_seconds is not None
            and max_accounted_dollar_seconds <= 0
        ):
            raise ValueError(
                "max_accounted_dollar_seconds must be positive when set"
            )
        if max_rollout_admission_delay_s < 0:
            raise ValueError("max_rollout_admission_delay_s must be non-negative")
        if not 0 <= rollout_admission_pressure_threshold < 1:
            raise ValueError(
                "rollout_admission_pressure_threshold must be in [0, 1)"
            )
        if not 0 <= rollout_admission_positive_signal_scale <= 1:
            raise ValueError(
                "rollout_admission_positive_signal_scale must be in [0, 1]"
            )
        if not 0 <= reconstruction_drift_threshold <= 1:
            raise ValueError("reconstruction_drift_threshold must be in [0, 1]")
        if reward_scale_normalization not in {"none", "arm_range"}:
            raise ValueError(
                "reward_scale_normalization must be 'none' or 'arm_range'"
            )
        self.min_train_batch_groups = min_train_batch_groups
        self.max_train_batch_groups = max_train_batch_groups
        self.min_policy_lag = min_policy_lag
        self.max_policy_lag_limit = max_policy_lag
        self.min_actor_count = min_actor_count
        self.max_actor_count_limit = max_actor_count
        self.ema_alpha = ema_alpha
        self.exploration_bonus = exploration_bonus
        self.objective_threshold = objective_threshold
        self.unsafe_penalty = unsafe_penalty
        self.rollout_objective_weight = rollout_objective_weight
        self.train_objective_weight = train_objective_weight
        self.reward_efficiency_weight = reward_efficiency_weight
        self.stale_penalty_weight = stale_penalty_weight
        self.staleness_priority_weight = staleness_priority_weight
        self.off_policy_priority_weight = off_policy_priority_weight
        self.off_policy_cadence_tightening_threshold = (
            off_policy_cadence_tightening_threshold
        )
        self.off_policy_lag_tightening_threshold = (
            off_policy_lag_tightening_threshold
        )
        self.confidence_penalty_weight = confidence_penalty_weight
        self.control_exploration_bonus = control_exploration_bonus
        self.rollout_cadence_lag_control_weight = (
            rollout_cadence_lag_control_weight
        )
        self.joint_action_objective_weight = joint_action_objective_weight
        self.train_selection_objective_weight = train_selection_objective_weight
        self.max_control_candidate_values = max_control_candidate_values
        self.min_rollout_coverage_fraction = min_rollout_coverage_fraction
        self.max_rollout_coverage_cost_fraction = (
            max_rollout_coverage_cost_fraction
        )
        self.min_train_steps = min_train_steps
        self.roi_patience = roi_patience
        self.min_train_objective = min_train_objective
        self.continuation_objective = continuation_objective
        self.control_train_objective = control_train_objective
        self.max_accounted_dollar_seconds = max_accounted_dollar_seconds
        self.max_rollout_admission_delay_s = max_rollout_admission_delay_s
        self.rollout_admission_pressure_threshold = (
            rollout_admission_pressure_threshold
        )
        self.rollout_admission_positive_signal_scale = (
            rollout_admission_positive_signal_scale
        )
        self.reconstruction_drift_threshold = reconstruction_drift_threshold
        self.reward_scale_normalization = reward_scale_normalization
        self._arms: dict[str, ArmStats] = {}
        self._total_decisions = 0
        self._total_pulls = 0
        self._global_objective_ema = 0.0
        self._train_reward_ema = 0.0
        self._train_objective_ema = 0.0
        self._train_observations = 0
        self._last_train_reward = 0.0
        self._last_train_objective = 0.0
        self._last_train_reward_improvement = 0.0
        self._last_train_experience_count = 0.0
        self._last_train_reward_improving_experience = 0.0
        self._last_train_control_reward_improving_experience = 0.0
        self._accounted_objective_ema = 0.0
        self._last_accounted_objective = 0.0
        self._last_accounted_reward_improving_experience = 0.0
        self._last_accounted_control_reward_improving_experience = 0.0
        self._last_accounted_dollar_seconds = 0.0
        self._last_continuation_objective = 0.0
        self._previous_accounted_dollar_seconds = 0.0
        self._last_stale_penalty_objective = 0.0
        self._last_stale_experience_count = 0.0
        self._last_stale_lost_reward_improving_experience = 0.0
        self._last_stale_sample_dollar_seconds = 0.0
        self._last_stale_unobserved_sample_dollar_seconds = 0.0
        self._last_stale_policy_step = -1
        self._last_stale_reason = ""
        self._last_decision: SchedulerDecision | None = None
        self._last_decision_snapshot: dict[str, Any] | None = None
        self._coverage_forced_decisions = 0
        self._last_rollout_coverage_target = 0.0
        self._last_rollout_coverage_share = 0.0
        self._last_rollout_coverage_deficit = 0.0
        self._last_rollout_coverage_cost_share = 0.0
        self._last_rollout_coverage_cost_limited = False
        self._last_train_batch_priority = 0.0
        self._last_train_batch_policy_lag = 0
        self._last_train_batch_lag_limit = -1
        self._last_train_batch_staleness_urgency = 0.0
        self._last_train_batch_staleness_bonus = 0.0
        self._last_train_batch_old_new_logprob_coverage = 0.0
        self._last_train_batch_off_policy_drift = 0.0
        self._last_train_batch_off_policy_penalty = 0.0
        self._last_train_batch_priority_before_off_policy = 0.0
        self._last_cadence_off_policy_penalty = 0.0
        self._last_cadence_off_policy_tightened = False
        self._last_policy_lag_off_policy_penalty = 0.0
        self._last_policy_lag_off_policy_tightened = False
        self._last_train_batch_reward_improving_experience = 0.0
        self._last_train_batch_sample_dollar_seconds = 0.0
        self._last_train_batch_cost_normalized_priority = 0.0
        self._last_train_batch_joint_action_score = 0.0
        self._last_train_batch_train_selection_score = 0.0
        self._last_continuation_decision_continue = False
        self._last_continuation_decision_key = ""
        self._last_continuation_decision_reason = ""
        self._last_continuation_pending_train_batches = 0
        self._last_continuation_train_queue_pressure = 0.0
        self._last_cadence_response_key = ""
        self._last_cadence_response_reason = ""
        self._last_policy_lag_response_key = ""
        self._last_policy_lag_response_reason = ""
        self._global_action_quality_ema = 1.0
        self._low_roi_train_steps = 0
        self._stop_recommended = False
        self._rollout_dollar_seconds = 0.0
        self._queue_wait_dollar_seconds = 0.0
        self._rollout_admission_decisions = 0
        self._rollout_admission_delay_s = 0.0
        self._rollout_admission_dollar_seconds = 0.0
        self._last_rollout_admission_delay_s = 0.0
        self._last_rollout_admission_pressure = 0.0
        self._train_dollar_seconds = 0.0
        self._stale_batches = 0
        self._stale_trajectories = 0
        self._stale_experience = 0.0
        self._stale_lost_reward_improving_experience = 0.0
        self._stale_sample_dollar_seconds = 0.0
        self._stale_unobserved_sample_dollar_seconds = 0.0
        self._stale_additional_dollar_seconds = 0.0
        self._last_stale_additional_dollar_seconds = 0.0
        self._cadence_controls: dict[int, ControlStats] = {}
        self._lag_controls: dict[int, ControlStats] = {}
        self._admission_controls: dict[int, ControlStats] = {}
        self._actor_count_controls: dict[int, ControlStats] = {}
        self._joint_action_controls: dict[str, ControlStats] = {}
        self._train_selection_controls: dict[str, ControlStats] = {}
        self._continuation_controls: dict[str, ControlStats] = {}
        self._coverage_controls: dict[str, ControlStats] = {}
        self._timing_response_controls: dict[str, ControlStats] = {}
        self._pending_continuation_decisions: dict[int, str] = {}
        self._recorded_continuation_stop_decisions: set[tuple[int, str]] = set()
        self._actors: dict[int, ActorStats] = {}

    def select_rollout(
        self,
        *,
        scenarios: Sequence[Scenario],
        action_codecs: Sequence[ActionCodec],
        actor_id: int,
        policy_step: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        configured_train_batch_groups: int,
        configured_max_policy_lag: int,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> SchedulerDecision:
        if not scenarios:
            raise ValueError("at least one scenario is required")
        if not action_codecs:
            raise ValueError("at least one action codec is required")
        action_space_key = _normalize_key_component(action_space_key)

        target_train_batch_groups = self.target_train_batch_groups(
            configured=configured_train_batch_groups,
            pending_groups=0,
            train_queue_pressure=train_queue_pressure,
            policy_step=policy_step,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        max_policy_lag = self.max_policy_lag(
            configured=configured_max_policy_lag,
            train_queue_pressure=train_queue_pressure,
            policy_step=policy_step,
            target_train_batch_groups=target_train_batch_groups,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        arms = self._arm_candidates(scenarios, action_codecs)
        coverage_selection, coverage_cost_limited = self._coverage_candidate(
            arms,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        arm_ids = [candidate[0] for candidate in arms]
        if coverage_selection is None:
            arm_id, scenario, codec = max(
                arms,
                key=lambda arm: self._score_arm(
                    arm[0],
                    arm[1],
                    arm[2],
                    target_train_batch_groups=target_train_batch_groups,
                    max_policy_lag=max_policy_lag,
                    active_actor_count=active_actor_count,
                    rollout_admission_delay_ms=rollout_admission_delay_ms,
                    action_space_key=action_space_key,
                ),
            )
            coverage_forced = False
            coverage_target = self._effective_coverage_target(len(arms))
            coverage_share = self._arm_decision_share(arm_id)
            coverage_deficit = max(0.0, coverage_target - coverage_share)
            coverage_cost_share = self._arm_sample_dollar_share(arm_id, arm_ids)
        else:
            (
                arm_id,
                scenario,
                codec,
                coverage_target,
                coverage_share,
                coverage_deficit,
                coverage_cost_share,
            ) = coverage_selection
            coverage_forced = True
            self._coverage_forced_decisions += 1
        coverage_control_key = (
            _coverage_control_key(arm_id) if coverage_forced else None
        )
        coverage_cost_limit = self.max_rollout_coverage_cost_fraction or 0.0
        selected_stats = self._arms[arm_id]
        estimated_rollout_dollar_seconds = (
            self._estimated_rollout_dollar_seconds(arm_id, scenario, codec)
        )
        unobserved_rollout_cost_penalty = (
            self._unobserved_rollout_cost_penalty(arm_id, scenario, codec)
        )
        decision_score = self._score_arm(
            arm_id,
            scenario,
            codec,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        objective_score = self._arm_value(selected_stats)
        exploration_score = self._exploration_value(selected_stats)
        joint_action_score = self._joint_action_score(
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        joint_action_key = self._candidate_joint_action_key(
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        reserved_rollout_dollar_seconds = max(0.0, estimated_rollout_dollar_seconds)
        self._record_arm_decision(
            selected_stats,
            reserved_rollout_dollar_seconds=reserved_rollout_dollar_seconds,
        )
        actor_stats = self._actors.setdefault(actor_id, ActorStats())
        actor_stats.decisions += 1
        actor_stats.inflight += 1
        self._last_rollout_coverage_target = coverage_target
        self._last_rollout_coverage_share = coverage_share
        self._last_rollout_coverage_deficit = coverage_deficit
        self._last_rollout_coverage_cost_share = coverage_cost_share
        self._last_rollout_coverage_cost_limited = coverage_cost_limited
        decision = SchedulerDecision(
            scenario=scenario,
            action_codec=codec,
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            metadata={
                "actor_id": actor_id,
                "policy_step": policy_step,
                "trajectory_queue_pressure": trajectory_queue_pressure,
                "train_queue_pressure": train_queue_pressure,
                "score": decision_score,
                "objective_score": objective_score,
                "exploration_score": exploration_score,
                "joint_action_score": joint_action_score,
                "joint_action_score_weight": self.joint_action_objective_weight,
                "inflight_rollouts": selected_stats.inflight,
                "coverage_forced": coverage_forced,
                "coverage_target": coverage_target,
                "coverage_share": coverage_share,
                "coverage_deficit": coverage_deficit,
                "coverage_cost_share": coverage_cost_share,
                "coverage_cost_limit": coverage_cost_limit,
                "coverage_cost_limited": coverage_cost_limited,
                "expected_rollout_dollar_seconds": (
                    estimated_rollout_dollar_seconds
                ),
                "estimated_rollout_dollar_seconds": (
                    estimated_rollout_dollar_seconds
                ),
                "reserved_rollout_dollar_seconds": (
                    reserved_rollout_dollar_seconds
                ),
                "unobserved_rollout_cost_penalty": (
                    unobserved_rollout_cost_penalty
                ),
                "unobserved_rollout_cost_estimated": (
                    selected_stats.pulls == 0
                    and estimated_rollout_dollar_seconds > 0.0
                ),
            },
        )
        if action_space_key is not None:
            decision = replace(
                decision,
                metadata={
                    **decision.metadata,
                    "action_space_key": action_space_key,
                },
            )
        if coverage_control_key is not None:
            decision = replace(
                decision,
                metadata={
                    **decision.metadata,
                    "coverage_control_key": coverage_control_key,
                },
            )
            self._record_coverage_decision(coverage_control_key)
        timing_metadata = self.timing_response_metadata()
        if timing_metadata:
            decision = replace(
                decision,
                metadata={
                    **decision.metadata,
                    **{
                        key.removeprefix("scheduler/"): value
                        for key, value in timing_metadata.items()
                    },
                },
            )
        if joint_action_key is not None:
            decision = replace(
                decision,
                metadata={
                    **decision.metadata,
                    "joint_action_key": joint_action_key,
                },
            )
            self._record_joint_action_decision(joint_action_key)
        self._last_decision = decision
        self._last_decision_snapshot = _decision_to_state(decision)
        return decision

    def cancel_rollout_decision(self, decision: SchedulerDecision) -> None:
        """Cancel a selected rollout before any rollout work was produced."""

        metadata = _mapping_state(decision.metadata)
        stats = self._arms.get(decision.arm_id)
        if stats is not None:
            reserved_cost = _decision_reserved_rollout_dollar_seconds(decision)
            stats.inflight = max(0, stats.inflight - 1)
            stats.reserved_rollout_dollar_seconds = max(
                0.0,
                stats.reserved_rollout_dollar_seconds - reserved_cost,
            )
            stats.decisions = max(0, stats.decisions - 1)
        self._total_decisions = max(0, self._total_decisions - 1)

        actor_id = _state_optional_int(decision.metadata.get("actor_id"), None)
        actor_stats = self._actors.get(actor_id) if actor_id is not None else None
        if actor_stats is not None:
            actor_stats.inflight = max(0, actor_stats.inflight - 1)
            actor_stats.decisions = max(0, actor_stats.decisions - 1)

        joint_action_key = _joint_action_key_from_metadata(metadata)
        if joint_action_key is not None:
            self._cancel_control_decision(
                self._joint_action_controls,
                joint_action_key,
            )
        coverage_control_key = _coverage_control_key_from_metadata(metadata)
        if coverage_control_key is not None:
            self._cancel_control_decision(
                self._coverage_controls,
                coverage_control_key,
            )
        for timing_response_key in _timing_response_keys_from_metadata(metadata):
            self._cancel_control_decision(
                self._timing_response_controls,
                timing_response_key,
            )

        self._cancel_control_decision(
            self._cadence_controls,
            decision.target_train_batch_groups,
        )
        self._cancel_control_decision(self._lag_controls, decision.max_policy_lag)
        actor_count = _first_int_metadata(
            metadata,
            ("scheduler/active_actor_count", "active_actor_count"),
        )
        if actor_count is not None and actor_count > 0:
            self._cancel_control_decision(self._actor_count_controls, actor_count)
        admission_delay_ms = _first_int_metadata(
            metadata,
            (
                "scheduler/active_rollout_admission_delay_ms",
                "rollout_admission_delay_ms",
            ),
        )
        if (
            admission_delay_ms is not None
            and not _state_bool(metadata.get("scheduler/admission_observed"), False)
        ):
            self._cancel_control_decision(
                self._admission_controls,
                admission_delay_ms,
            )
            self._rollout_admission_decisions = max(
                0,
                self._rollout_admission_decisions - 1,
            )
            self._last_rollout_admission_delay_s = 0.0

        if _state_bool(
            metadata.get("coverage_forced"),
            False,
        ):
            self._coverage_forced_decisions = max(
                0,
                self._coverage_forced_decisions - 1,
            )

        if _decision_matches_snapshot_for_cancel(
            self._last_decision_snapshot,
            decision,
        ):
            self._last_decision = None
            self._last_decision_snapshot = None
            self._last_rollout_coverage_target = 0.0
            self._last_rollout_coverage_share = 0.0
            self._last_rollout_coverage_deficit = 0.0
            self._last_rollout_coverage_cost_share = 0.0
            self._last_rollout_coverage_cost_limited = False

    def target_train_batch_groups(
        self,
        *,
        configured: int,
        pending_groups: int,
        train_queue_pressure: float,
        policy_step: int,
        max_policy_lag: int | None = None,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> int:
        upper = self.max_train_batch_groups or configured
        upper = max(self.min_train_batch_groups, upper)
        configured = min(max(configured, self.min_train_batch_groups), upper)
        candidates = self._control_candidates(
            min_value=self.min_train_batch_groups,
            configured=configured,
            upper=upper,
        )
        off_policy_penalty = self._last_train_batch_off_policy_penalty
        self._last_cadence_off_policy_penalty = off_policy_penalty
        self._last_cadence_off_policy_tightened = False
        if self._has_positive_objective_signal():
            preferred = self.min_train_batch_groups
            preference_reason = "positive_signal"
        elif off_policy_penalty > self.off_policy_cadence_tightening_threshold:
            preferred = self.min_train_batch_groups
            preference_reason = "off_policy_tighten"
            self._last_cadence_off_policy_tightened = True
        elif train_queue_pressure >= 0.75:
            preferred = upper
            preference_reason = "train_queue_pressure"
        elif pending_groups >= upper:
            preferred = upper
            preference_reason = "pending_groups"
        else:
            preferred = configured
            preference_reason = "configured"
        joint_scores = self._joint_action_scores_for_control_values(
            "cadence",
            candidates,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        if (
            preferred == upper
            and not self._has_positive_objective_signal()
            and not self._control_family_has_feedback(
                self._cadence_controls,
                candidates,
            )
            and not joint_scores
        ):
            selected = upper
        else:
            selected = self._select_control_value(
                self._cadence_controls,
                candidates,
                preferred=preferred,
                joint_scores=joint_scores,
            )
        self._record_timing_response_decision(
            knob="cadence",
            value=selected,
            preference_reason=preference_reason,
            train_queue_pressure=train_queue_pressure,
            pending_groups=pending_groups,
        )
        return self._record_control_decision(self._cadence_controls, selected)

    def max_policy_lag(
        self,
        *,
        configured: int,
        train_queue_pressure: float,
        policy_step: int,
        target_train_batch_groups: int | None = None,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> int:
        upper = self.max_policy_lag_limit
        if upper is None:
            upper = configured
        upper = max(self.min_policy_lag, upper)
        configured = min(max(configured, self.min_policy_lag), upper)
        candidates = self._control_candidates(
            min_value=self.min_policy_lag,
            configured=configured,
            upper=upper,
        )
        off_policy_penalty = self._last_train_batch_off_policy_penalty
        self._last_policy_lag_off_policy_penalty = off_policy_penalty
        self._last_policy_lag_off_policy_tightened = False
        protecting_unaccepted_arm = self._has_unaccepted_known_arm()
        if protecting_unaccepted_arm:
            preferred = configured
            preference_reason = "protect_unaccepted"
        elif train_queue_pressure >= 0.75:
            preferred = self.min_policy_lag
            preference_reason = "train_queue_pressure"
        elif off_policy_penalty > self.off_policy_lag_tightening_threshold:
            preferred = self.min_policy_lag
            preference_reason = "off_policy_tighten"
            self._last_policy_lag_off_policy_tightened = True
        elif self._has_positive_objective_signal():
            preferred = self.min_policy_lag
            preference_reason = "positive_signal"
        else:
            preferred = configured
            preference_reason = "configured"
        joint_scores = self._joint_action_scores_for_control_values(
            "lag",
            candidates,
            target_train_batch_groups=target_train_batch_groups,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        if (
            protecting_unaccepted_arm
            and not self._control_family_has_feedback(
                self._lag_controls,
                candidates,
            )
            and not joint_scores
        ):
            selected = configured
        else:
            selected = self._select_control_value(
                self._lag_controls,
                candidates,
                preferred=preferred,
                joint_scores=joint_scores,
            )
        self._record_timing_response_decision(
            knob="policy_lag",
            value=selected,
            preference_reason=preference_reason,
            train_queue_pressure=train_queue_pressure,
            pending_groups=0,
        )
        return self._record_control_decision(self._lag_controls, selected)

    def active_actor_count(
        self,
        *,
        configured: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
        action_space_key: str | None = None,
    ) -> int:
        upper = self.max_actor_count_limit
        if upper is None:
            upper = configured
        upper = max(self.min_actor_count, upper)
        configured = min(max(configured, self.min_actor_count), upper)
        candidates = self._control_candidates(
            min_value=self.min_actor_count,
            configured=configured,
            upper=upper,
        )
        pressure = max(
            0.0,
            min(1.0, trajectory_queue_pressure),
            min(1.0, train_queue_pressure),
        )
        if (
            self._total_pulls > 0
            and not self._has_positive_objective_signal()
            and self._low_roi_train_steps > 0
        ):
            preferred = self.min_actor_count
        elif pressure >= 0.75 and not self._has_positive_objective_signal():
            preferred = self.min_actor_count
        else:
            preferred = configured
        joint_scores = self._joint_action_scores_for_control_values(
            "actors",
            candidates,
            action_space_key=action_space_key,
        )
        if (
            preferred == self.min_actor_count
            and not self._has_positive_objective_signal()
            and not self._control_family_has_feedback(
                self._actor_count_controls,
                candidates,
            )
            and not joint_scores
        ):
            return self._record_control_decision(
                self._actor_count_controls,
                preferred,
            )
        return self._record_control_decision(
            self._actor_count_controls,
            self._select_control_value(
                self._actor_count_controls,
                candidates,
                preferred=preferred,
                joint_scores=joint_scores,
            ),
        )

    def cancel_actor_count_decision(self, active_actor_count: int) -> None:
        """Cancel an actor-count control choice that admitted no rollout work."""

        active_count = max(0, int(active_actor_count))
        if active_count <= 0:
            return
        self._cancel_control_decision(self._actor_count_controls, active_count)

    def timing_response_metadata(self) -> dict[str, str]:
        """Return metadata keys for the most recent cadence/lag responses."""

        metadata: dict[str, str] = {}
        if self._last_cadence_response_key:
            metadata["scheduler/cadence_response_key"] = (
                self._last_cadence_response_key
            )
        if self._last_policy_lag_response_key:
            metadata["scheduler/policy_lag_response_key"] = (
                self._last_policy_lag_response_key
            )
        return metadata

    def record_train_batch_flush(
        self,
        *,
        flushed_groups: int,
        pending_groups: int,
        train_queue_pressure: float,
        reason: str = "manual_flush",
    ) -> dict[str, str]:
        """Record a forced partial train-batch flush as a timing response."""

        flushed = max(0, int(flushed_groups))
        if flushed <= 0:
            return {}
        key = self._record_timing_response_decision(
            knob="batch_flush",
            value=flushed,
            preference_reason=reason,
            train_queue_pressure=train_queue_pressure,
            pending_groups=pending_groups,
        )
        return {"scheduler/batch_flush_response_key": key}

    def observe_rollout(
        self,
        trajectory: Trajectory,
        *,
        accepted: bool,
        dollar_seconds: float,
        queue_wait_dollar_seconds: float = 0.0,
    ) -> None:
        arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
        stats = self._arms.setdefault(arm_id, ArmStats())
        if stats.inflight > 0:
            reserved_cost = _trajectory_reserved_rollout_dollar_seconds(trajectory)
            if (
                reserved_cost <= 0.0
                and stats.reserved_rollout_dollar_seconds > 0.0
            ):
                reserved_cost = (
                    stats.reserved_rollout_dollar_seconds / stats.inflight
                )
            stats.inflight -= 1
            stats.reserved_rollout_dollar_seconds = max(
                0.0,
                stats.reserved_rollout_dollar_seconds - reserved_cost,
            )
        rollout_cost = max(dollar_seconds, 1e-12)
        queue_wait_cost = max(0.0, queue_wait_dollar_seconds)
        admission_cost = _trajectory_admission_dollar_seconds(trajectory)
        cost = max(rollout_cost + queue_wait_cost + admission_cost, 1e-12)
        trajectory.metadata["scheduler/rollout_observed"] = True
        self._rollout_dollar_seconds += rollout_cost
        self._queue_wait_dollar_seconds += queue_wait_cost
        quality = action_quality(trajectory)
        failure_modes = trajectory_failure_modes(
            trajectory,
            reconstruction_drift_threshold=self.reconstruction_drift_threshold,
        )
        reconstruction_accuracy = trajectory_reconstruction_accuracy(trajectory)
        reward = trajectory.reward if accepted else 0.0
        effective_reward = reward * quality
        previous_reward = stats.effective_reward_ema if stats.pulls else 0.0
        positive_improvement = max(0.0, effective_reward - previous_reward)
        reward_scale = self._arm_effective_reward_scale(stats, effective_reward)
        normalized_positive_improvement = self._normalized_reward_improvement(
            positive_improvement,
            reward_scale,
        )
        marginal_objective = normalized_positive_improvement / cost
        reward_efficiency = max(0.0, effective_reward) / cost

        stats.pulls += 1
        self._observe_arm_effective_reward(stats, effective_reward)
        stats.last_reward_scale = reward_scale
        stats.last_normalized_positive_improvement = (
            normalized_positive_improvement
        )
        self._total_pulls += 1
        if accepted:
            stats.accepted += 1
        if quality <= 0.0:
            stats.unsafe += 1
        if failure_modes:
            stats.failed_rollouts += 1
            for mode in failure_modes:
                stats.failure_modes[mode] = stats.failure_modes.get(mode, 0) + 1
        if reconstruction_accuracy is not None:
            reconstruction_drift = max(0.0, 1.0 - reconstruction_accuracy)
            stats.reconstruction_observations += 1
            stats.total_reconstruction_accuracy += reconstruction_accuracy
            stats.min_reconstruction_accuracy = min(
                stats.min_reconstruction_accuracy,
                reconstruction_accuracy,
            )
            stats.max_reconstruction_drift = max(
                stats.max_reconstruction_drift,
                reconstruction_drift,
            )
            stats.reconstruction_accuracy_ema = self._ema(
                stats.reconstruction_accuracy_ema,
                reconstruction_accuracy,
                stats.reconstruction_observations,
            )
            stats.reconstruction_drift_ema = self._ema(
                stats.reconstruction_drift_ema,
                reconstruction_drift,
                stats.reconstruction_observations,
            )
        stats.total_reward += reward
        stats.total_effective_reward += effective_reward
        stats.total_positive_improvement += positive_improvement
        stats.total_normalized_positive_improvement += (
            normalized_positive_improvement
        )
        stats.total_dollar_seconds += cost
        stats.rollout_dollar_seconds += rollout_cost
        stats.queue_wait_dollar_seconds += queue_wait_cost
        stats.admission_dollar_seconds += admission_cost
        stats.action_units += trajectory.action_units
        stats.source_tokens += trajectory.token_count
        logprob_stats = action_logprob_stats(trajectory.actions)
        stats.old_logprob_units += logprob_stats.old_logprob_units
        stats.new_logprob_units += logprob_stats.new_logprob_units
        stats.reference_logprob_units += logprob_stats.reference_logprob_units
        stats.old_new_logprob_pairs += logprob_stats.old_new_pairs
        stats.old_reference_logprob_pairs += logprob_stats.old_reference_pairs
        stats.old_logprob_sum += logprob_stats.old_logprob_sum
        stats.new_logprob_sum += logprob_stats.new_logprob_sum
        stats.reference_logprob_sum += logprob_stats.reference_logprob_sum
        stats.old_new_logprob_delta_sum += (
            logprob_stats.old_new_logprob_delta_sum
        )
        stats.old_new_logprob_abs_delta_sum += (
            logprob_stats.old_new_logprob_abs_delta_sum
        )
        stats.old_reference_logprob_delta_sum += (
            logprob_stats.old_reference_logprob_delta_sum
        )
        stats.importance_ratio_sum += logprob_stats.importance_ratio_sum
        stats.dollar_seconds_ema = self._ema(
            stats.dollar_seconds_ema,
            cost,
            stats.pulls,
        )
        stats.reward_ema = self._ema(stats.reward_ema, reward, stats.pulls)
        stats.effective_reward_ema = self._ema(
            stats.effective_reward_ema,
            effective_reward,
            stats.pulls,
        )
        stats.action_quality_ema = self._ema(
            stats.action_quality_ema,
            quality,
            stats.pulls,
        )
        stats.marginal_objective_ema = self._ema(
            stats.marginal_objective_ema,
            marginal_objective,
            stats.pulls,
        )
        self._observe_objective_sample(stats, marginal_objective)
        stats.reward_efficiency_ema = self._ema(
            stats.reward_efficiency_ema,
            reward_efficiency,
            stats.pulls,
        )
        if self.rollout_cadence_lag_control_weight > 0.0:
            cadence_lag_objective = (
                marginal_objective * self.rollout_cadence_lag_control_weight
            )
            self._credit_rollout_objective_to_cadence_control(
                trajectory,
                cadence_lag_objective,
            )
            self._credit_rollout_objective_to_lag_control(
                trajectory,
                cadence_lag_objective,
            )
            self._credit_rollout_objective_to_timing_responses(
                trajectory,
                cadence_lag_objective,
            )
        self._credit_rollout_objective_to_admission_control(
            trajectory,
            marginal_objective,
        )
        self._credit_rollout_objective_to_actor_count_control(
            trajectory,
            marginal_objective,
        )
        self._credit_rollout_objective_to_joint_action(
            trajectory,
            marginal_objective,
        )
        self._credit_rollout_objective_to_coverage_control(
            trajectory,
            marginal_objective,
        )
        self._credit_rollout_objective_to_actor(
            trajectory,
            marginal_objective,
            accepted=accepted,
            rollout_cost=rollout_cost,
            queue_wait_cost=queue_wait_cost,
            admission_cost=admission_cost,
            quality=quality,
            failure_modes=failure_modes,
        )
        self._global_objective_ema = self._ema(
            self._global_objective_ema,
            marginal_objective,
            self._total_pulls,
        )
        self._global_action_quality_ema = self._ema(
            self._global_action_quality_ema,
            quality,
            self._total_pulls,
        )

    def rollout_admission_delay_s(
        self,
        *,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
        active_actor_count: int | None = None,
        action_space_key: str | None = None,
    ) -> float:
        """Return a pre-rollout delay when downstream buffers are saturated."""

        pressure = max(
            0.0,
            min(1.0, trajectory_queue_pressure),
            min(1.0, train_queue_pressure),
        )
        self._rollout_admission_decisions += 1
        self._last_rollout_admission_pressure = pressure
        if (
            self.max_rollout_admission_delay_s <= 0.0
            or pressure <= self.rollout_admission_pressure_threshold
        ):
            self._last_rollout_admission_delay_s = 0.0
            self._record_control_decision(self._admission_controls, 0)
            return 0.0

        span = max(1e-12, 1.0 - self.rollout_admission_pressure_threshold)
        delay_fraction = min(
            1.0,
            (pressure - self.rollout_admission_pressure_threshold) / span,
        )
        if self._has_positive_objective_signal():
            delay_fraction *= self.rollout_admission_positive_signal_scale
        preferred_ms = _seconds_to_milliseconds(
            self.max_rollout_admission_delay_s * delay_fraction
        )
        max_delay_ms = _seconds_to_milliseconds(
            self.max_rollout_admission_delay_s
        )
        candidates = self._control_candidates(
            min_value=0,
            configured=preferred_ms,
            upper=max_delay_ms,
        )
        joint_scores = self._joint_action_scores_for_control_values(
            "admission_ms",
            candidates,
            active_actor_count=active_actor_count,
            action_space_key=action_space_key,
        )
        selected_ms = self._select_control_value(
            self._admission_controls,
            candidates,
            preferred=preferred_ms,
            joint_scores=joint_scores,
        )
        self._record_control_decision(self._admission_controls, selected_ms)
        delay = selected_ms / 1000.0
        self._last_rollout_admission_delay_s = delay
        return delay

    def observe_rollout_admission_delay(
        self,
        *,
        seconds: float,
        dollar_seconds: float,
    ) -> None:
        delay_s = max(0.0, seconds)
        delay_cost = max(0.0, dollar_seconds)
        self._rollout_admission_delay_s += delay_s
        self._rollout_admission_dollar_seconds += delay_cost

    def observe_train(
        self,
        *,
        groups: Sequence[TrajectoryGroup],
        result: TrainResult,
        duration_s: float,
        dollar_seconds: float,
        policy_step: int,
    ) -> None:
        metric_reward = result.metrics.get("promotion/score")
        if metric_reward is None:
            metric_reward = result.metrics.get("train/reward")
        reward = (
            float(metric_reward)
            if metric_reward is not None and math.isfinite(float(metric_reward))
            else mean([group.mean_reward for group in groups])
        )
        cost = max(dollar_seconds, 1e-12)
        self._train_dollar_seconds += cost
        self._train_observations += 1
        (
            objective,
            experience_count,
            reward_improving_experience,
            arm_objectives,
            arm_weights,
            control_reward_improving_experience,
        ) = self._credit_train_objective_to_arms(
            groups,
            reward=reward,
            cost=cost,
        )
        improvement = (
            reward_improving_experience / experience_count
            if experience_count > 0.0
            else 0.0
        )
        self._last_train_reward = reward
        self._last_train_objective = objective
        self._last_train_reward_improvement = improvement
        self._last_train_experience_count = experience_count
        self._last_train_reward_improving_experience = reward_improving_experience
        self._last_train_control_reward_improving_experience = (
            control_reward_improving_experience
        )
        accounted_objective = self._observe_accounted_train_objective(
            control_reward_improving_experience,
            raw_reward_improving_experience=reward_improving_experience,
        )
        continuation_objective = (
            accounted_objective
            if self.continuation_objective == "accounted"
            else objective
        )
        self._last_continuation_objective = continuation_objective
        self._credit_train_objective_to_controls(
            groups,
            objective=(
                accounted_objective
                if self.control_train_objective == "accounted"
                else None
            ),
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
        )
        self._credit_train_objective_to_actors(
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
        )
        self._credit_train_objective_to_train_selection(
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
        )
        self._credit_train_objective_to_continuation(
            policy_step=policy_step,
            objective=continuation_objective,
        )
        if continuation_objective <= self.min_train_objective:
            self._low_roi_train_steps += 1
        else:
            self._low_roi_train_steps = 0
        self._train_reward_ema = self._ema(
            self._train_reward_ema,
            reward,
            policy_step + 1,
        )
        self._train_objective_ema = self._ema(
            self._train_objective_ema,
            objective,
            policy_step + 1,
        )

    def observe_stale_batch(
        self,
        *,
        groups: Sequence[TrajectoryGroup],
        policy_step: int,
        reason: str,
        additional_dollar_seconds: float = 0.0,
    ) -> None:
        """Record useful experience that was produced but never trained."""

        stale_trajectories = sum(
            len(group.trajectories)
            for group in groups
        )
        stale_experience = _useful_experience_count(groups)
        stale_sample_cost = _groups_sample_dollar_seconds(groups)
        stale_unobserved_sample_cost = _groups_unobserved_sample_dollar_seconds(
            groups
        )
        stale_overhead_cost = max(0.0, additional_dollar_seconds)
        stale_cost = stale_sample_cost + stale_overhead_cost
        if stale_cost <= 0.0:
            stale_cost = max(stale_experience, 1.0)
        lost_reward_improving_experience = (
            self._estimate_stale_lost_reward_improving_experience(
                groups,
                stale_cost=stale_cost,
                stale_experience=stale_experience,
            )
        )
        penalty_experience = (
            lost_reward_improving_experience
            if lost_reward_improving_experience > 0.0
            else stale_experience
        )
        penalty_objective = -(
            self.stale_penalty_weight
            * penalty_experience
            / max(stale_cost, 1e-12)
        )

        self._stale_batches += 1
        self._stale_trajectories += stale_trajectories
        self._stale_experience += stale_experience
        self._stale_lost_reward_improving_experience += penalty_experience
        self._stale_sample_dollar_seconds += stale_sample_cost
        self._stale_unobserved_sample_dollar_seconds += stale_unobserved_sample_cost
        self._stale_additional_dollar_seconds += stale_overhead_cost
        self._last_stale_penalty_objective = penalty_objective
        self._last_stale_experience_count = stale_experience
        self._last_stale_lost_reward_improving_experience = penalty_experience
        self._last_stale_sample_dollar_seconds = stale_sample_cost
        self._last_stale_unobserved_sample_dollar_seconds = (
            stale_unobserved_sample_cost
        )
        self._last_stale_additional_dollar_seconds = stale_overhead_cost
        self._last_stale_policy_step = policy_step
        self._last_stale_reason = str(reason)

        self._credit_objective_to_arms(
            groups,
            penalty_objective,
            stale_feedback=True,
            stale_experience=stale_experience,
        )
        self._credit_objective_to_controls(
            groups,
            penalty_objective,
            stale_feedback=True,
            stale_experience=stale_experience,
        )
        self._credit_objective_to_actors(
            groups,
            penalty_objective,
            stale_feedback=True,
            stale_experience=stale_experience,
        )

    def score_train_groups(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        policy_step: int,
    ) -> float:
        """Estimate train-batch value before the trainer consumes it."""

        arm_values: list[float] = []
        joint_action_scores: list[float] = []
        rewards: list[float] = []
        for group in groups:
            rewards.append(group.mean_reward)
            for trajectory in group.trajectories:
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                stats = self._arms.get(arm_id)
                quality = action_quality(trajectory)
                if stats is None or stats.accepted == 0:
                    arm_values.append(max(0.0, trajectory.reward) * quality)
                else:
                    arm_value = self._arm_value(stats)
                    if quality <= 0.0:
                        arm_values.append(min(arm_value, -self.unsafe_penalty))
                    else:
                        arm_values.append(arm_value * quality)
                joint_action_scores.append(
                    self._joint_action_priority_score(trajectory) * quality
                )
        raw_reward_component = max(
            0.0,
            mean(
                [
                    trajectory.reward * action_quality(trajectory)
                    for group in groups
                    for trajectory in group.trajectories
                ]
            ),
        )
        arm_component = mean(arm_values)
        joint_action_component = mean(joint_action_scores)
        train_selection_component = self._train_selection_priority_score(groups)
        uncosted_base_priority = (
            arm_component
            + joint_action_component
            + train_selection_component
            + self.reward_efficiency_weight * raw_reward_component
        )
        experience_count = _useful_experience_count(groups)
        batch_reward_improving_experience = (
            uncosted_base_priority * max(experience_count, 1.0)
            if groups
            else 0.0
        )
        sample_dollar_seconds = _groups_sample_dollar_seconds(groups)
        base_priority = (
            batch_reward_improving_experience / sample_dollar_seconds
            if sample_dollar_seconds > 0.0
            else uncosted_base_priority
        )
        policy_lag, lag_limit, staleness_urgency = _batch_staleness_state(
            groups,
            policy_step=policy_step,
            fallback_lag_limit=self.max_policy_lag_limit,
        )
        staleness_bonus = (
            max(0.0, base_priority)
            * self.staleness_priority_weight
            * staleness_urgency
        )
        priority_before_off_policy = base_priority + staleness_bonus
        logprob_stats = _groups_action_logprob_stats(groups)
        old_new_logprob_coverage = (
            logprob_stats.old_new_pairs / logprob_stats.action_units
            if logprob_stats.action_units
            else 0.0
        )
        off_policy_drift = logprob_stats.old_new_logprob_abs_delta_mean
        off_policy_penalty = (
            self.off_policy_priority_weight
            * old_new_logprob_coverage
            * off_policy_drift
        )
        if priority_before_off_policy >= 0.0:
            priority = priority_before_off_policy / (1.0 + off_policy_penalty)
        else:
            priority = priority_before_off_policy * (1.0 + off_policy_penalty)
        self._last_train_batch_priority = priority
        self._last_train_batch_policy_lag = policy_lag
        self._last_train_batch_lag_limit = (
            lag_limit if lag_limit is not None else -1
        )
        self._last_train_batch_staleness_urgency = staleness_urgency
        self._last_train_batch_staleness_bonus = staleness_bonus
        self._last_train_batch_old_new_logprob_coverage = (
            old_new_logprob_coverage
        )
        self._last_train_batch_off_policy_drift = off_policy_drift
        self._last_train_batch_off_policy_penalty = off_policy_penalty
        self._last_train_batch_priority_before_off_policy = (
            priority_before_off_policy
        )
        self._last_train_batch_reward_improving_experience = (
            batch_reward_improving_experience
        )
        self._last_train_batch_sample_dollar_seconds = sample_dollar_seconds
        self._last_train_batch_cost_normalized_priority = base_priority
        self._last_train_batch_joint_action_score = joint_action_component
        self._last_train_batch_train_selection_score = train_selection_component
        return priority

    def record_train_batch_selection(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        priority: float,
        policy_step: int,
    ) -> None:
        """Record the batch the trainer selected from the ready queue."""

        if not groups:
            return
        key = _train_selection_key(groups)
        stats = self._train_selection_controls.setdefault(key, ControlStats())
        stats.decisions += 1
        selected_priority = priority if math.isfinite(priority) else 0.0
        for group in groups:
            for trajectory in group.trajectories:
                trajectory.metadata.setdefault("scheduler/train_selection_key", key)
                trajectory.metadata.setdefault(
                    "scheduler/train_selection_priority",
                    selected_priority,
                )
                trajectory.metadata.setdefault(
                    "scheduler/train_selection_policy_step",
                    policy_step,
                )

    def should_continue_training(
        self,
        *,
        policy_step: int,
        max_train_steps: int,
        pending_train_batches: int,
        train_queue_pressure: float,
    ) -> bool:
        if policy_step >= max_train_steps:
            self._stop_recommended = True
            return self._record_continuation_decision(
                policy_step=policy_step,
                continue_training=False,
                reason="max_steps",
                pending_train_batches=pending_train_batches,
                train_queue_pressure=train_queue_pressure,
            )
        if self._accounted_budget_exhausted():
            self._stop_recommended = True
            return self._record_continuation_decision(
                policy_step=policy_step,
                continue_training=False,
                reason="budget",
                pending_train_batches=pending_train_batches,
                train_queue_pressure=train_queue_pressure,
            )
        if self.roi_patience is None:
            return self._record_continuation_decision(
                policy_step=policy_step,
                continue_training=True,
                reason="no_patience",
                pending_train_batches=pending_train_batches,
                train_queue_pressure=train_queue_pressure,
            )
        if policy_step < self.min_train_steps:
            return self._record_continuation_decision(
                policy_step=policy_step,
                continue_training=True,
                reason="warmup",
                pending_train_batches=pending_train_batches,
                train_queue_pressure=train_queue_pressure,
            )
        if self._has_unaccepted_known_arm():
            return self._record_continuation_decision(
                policy_step=policy_step,
                continue_training=True,
                reason="exploration",
                pending_train_batches=pending_train_batches,
                train_queue_pressure=train_queue_pressure,
            )
        if self._low_roi_train_steps >= self.roi_patience:
            self._stop_recommended = True
            return self._record_continuation_decision(
                policy_step=policy_step,
                continue_training=False,
                reason="low_roi",
                pending_train_batches=pending_train_batches,
                train_queue_pressure=train_queue_pressure,
            )
        return self._record_continuation_decision(
            policy_step=policy_step,
            continue_training=True,
            reason="roi_ok",
            pending_train_batches=pending_train_batches,
            train_queue_pressure=train_queue_pressure,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-friendly scheduler state for checkpoint/resume."""

        return {
            "version": 1,
            "config": {
                "min_train_batch_groups": self.min_train_batch_groups,
                "max_train_batch_groups": self.max_train_batch_groups,
                "min_policy_lag": self.min_policy_lag,
                "max_policy_lag": self.max_policy_lag_limit,
                "min_actor_count": self.min_actor_count,
                "max_actor_count": self.max_actor_count_limit,
                "ema_alpha": self.ema_alpha,
                "exploration_bonus": self.exploration_bonus,
                "objective_threshold": self.objective_threshold,
                "unsafe_penalty": self.unsafe_penalty,
                "rollout_objective_weight": self.rollout_objective_weight,
                "train_objective_weight": self.train_objective_weight,
                "reward_efficiency_weight": self.reward_efficiency_weight,
                "stale_penalty_weight": self.stale_penalty_weight,
                "staleness_priority_weight": self.staleness_priority_weight,
                "off_policy_priority_weight": self.off_policy_priority_weight,
                "off_policy_cadence_tightening_threshold": (
                    self.off_policy_cadence_tightening_threshold
                ),
                "off_policy_lag_tightening_threshold": (
                    self.off_policy_lag_tightening_threshold
                ),
                "confidence_penalty_weight": self.confidence_penalty_weight,
                "control_exploration_bonus": self.control_exploration_bonus,
                "rollout_cadence_lag_control_weight": (
                    self.rollout_cadence_lag_control_weight
                ),
                "joint_action_objective_weight": (
                    self.joint_action_objective_weight
                ),
                "train_selection_objective_weight": (
                    self.train_selection_objective_weight
                ),
                "max_control_candidate_values": (
                    self.max_control_candidate_values
                ),
                "min_rollout_coverage_fraction": (
                    self.min_rollout_coverage_fraction
                ),
                "max_rollout_coverage_cost_fraction": (
                    self.max_rollout_coverage_cost_fraction
                ),
                "min_train_steps": self.min_train_steps,
                "roi_patience": self.roi_patience,
                "min_train_objective": self.min_train_objective,
                "continuation_objective": self.continuation_objective,
                "control_train_objective": self.control_train_objective,
                "max_accounted_dollar_seconds": (
                    self.max_accounted_dollar_seconds
                ),
                "max_rollout_admission_delay_s": (
                    self.max_rollout_admission_delay_s
                ),
                "rollout_admission_pressure_threshold": (
                    self.rollout_admission_pressure_threshold
                ),
                "rollout_admission_positive_signal_scale": (
                    self.rollout_admission_positive_signal_scale
                ),
                "reconstruction_drift_threshold": (
                    self.reconstruction_drift_threshold
                ),
                "reward_scale_normalization": self.reward_scale_normalization,
            },
            "learning_state": {
                "total_decisions": self._total_decisions,
                "total_pulls": self._total_pulls,
                "global_objective_ema": self._global_objective_ema,
                "train_observations": self._train_observations,
                "train_reward_ema": self._train_reward_ema,
                "train_objective_ema": self._train_objective_ema,
                "last_train_reward": self._last_train_reward,
                "last_train_objective": self._last_train_objective,
                "last_train_reward_improvement": (
                    self._last_train_reward_improvement
                ),
                "last_train_experience_count": self._last_train_experience_count,
                "last_train_reward_improving_experience": (
                    self._last_train_reward_improving_experience
                ),
                "last_train_control_reward_improving_experience": (
                    self._last_train_control_reward_improving_experience
                ),
                "accounted_objective_ema": self._accounted_objective_ema,
                "last_accounted_objective": self._last_accounted_objective,
                "last_accounted_reward_improving_experience": (
                    self._last_accounted_reward_improving_experience
                ),
                "last_accounted_control_reward_improving_experience": (
                    self._last_accounted_control_reward_improving_experience
                ),
                "last_accounted_dollar_seconds": (
                    self._last_accounted_dollar_seconds
                ),
                "last_continuation_objective": self._last_continuation_objective,
                "previous_accounted_dollar_seconds": (
                    self._previous_accounted_dollar_seconds
                ),
                "last_stale_penalty_objective": (
                    self._last_stale_penalty_objective
                ),
                "last_stale_experience_count": self._last_stale_experience_count,
                "last_stale_lost_reward_improving_experience": (
                    self._last_stale_lost_reward_improving_experience
                ),
                "last_stale_sample_dollar_seconds": (
                    self._last_stale_sample_dollar_seconds
                ),
                "last_stale_unobserved_sample_dollar_seconds": (
                    self._last_stale_unobserved_sample_dollar_seconds
                ),
                "last_stale_additional_dollar_seconds": (
                    self._last_stale_additional_dollar_seconds
                ),
                "last_stale_policy_step": self._last_stale_policy_step,
                "last_stale_reason": self._last_stale_reason,
                "last_train_batch_priority": self._last_train_batch_priority,
                "last_train_batch_policy_lag": (
                    self._last_train_batch_policy_lag
                ),
                "last_train_batch_lag_limit": self._last_train_batch_lag_limit,
                "last_train_batch_staleness_urgency": (
                    self._last_train_batch_staleness_urgency
                ),
                "last_train_batch_staleness_bonus": (
                    self._last_train_batch_staleness_bonus
                ),
                "last_train_batch_old_new_logprob_coverage": (
                    self._last_train_batch_old_new_logprob_coverage
                ),
                "last_train_batch_off_policy_drift": (
                    self._last_train_batch_off_policy_drift
                ),
                "last_train_batch_off_policy_penalty": (
                    self._last_train_batch_off_policy_penalty
                ),
                "last_train_batch_priority_before_off_policy": (
                    self._last_train_batch_priority_before_off_policy
                ),
                "last_cadence_off_policy_penalty": (
                    self._last_cadence_off_policy_penalty
                ),
                "last_cadence_off_policy_tightened": (
                    self._last_cadence_off_policy_tightened
                ),
                "last_policy_lag_off_policy_penalty": (
                    self._last_policy_lag_off_policy_penalty
                ),
                "last_policy_lag_off_policy_tightened": (
                    self._last_policy_lag_off_policy_tightened
                ),
                "last_train_batch_reward_improving_experience": (
                    self._last_train_batch_reward_improving_experience
                ),
                "last_train_batch_sample_dollar_seconds": (
                    self._last_train_batch_sample_dollar_seconds
                ),
                "last_train_batch_cost_normalized_priority": (
                    self._last_train_batch_cost_normalized_priority
                ),
                "last_train_batch_joint_action_score": (
                    self._last_train_batch_joint_action_score
                ),
                "last_train_batch_train_selection_score": (
                    self._last_train_batch_train_selection_score
                ),
                "last_continuation_decision_continue": (
                    self._last_continuation_decision_continue
                ),
                "last_continuation_decision_key": (
                    self._last_continuation_decision_key
                ),
                "last_continuation_decision_reason": (
                    self._last_continuation_decision_reason
                ),
                "last_continuation_pending_train_batches": (
                    self._last_continuation_pending_train_batches
                ),
                "last_continuation_train_queue_pressure": (
                    self._last_continuation_train_queue_pressure
                ),
                "last_cadence_response_key": self._last_cadence_response_key,
                "last_cadence_response_reason": (
                    self._last_cadence_response_reason
                ),
                "last_policy_lag_response_key": (
                    self._last_policy_lag_response_key
                ),
                "last_policy_lag_response_reason": (
                    self._last_policy_lag_response_reason
                ),
                "coverage_forced_decisions": self._coverage_forced_decisions,
                "last_rollout_coverage_target": (
                    self._last_rollout_coverage_target
                ),
                "last_rollout_coverage_share": self._last_rollout_coverage_share,
                "last_rollout_coverage_deficit": (
                    self._last_rollout_coverage_deficit
                ),
                "last_rollout_coverage_cost_share": (
                    self._last_rollout_coverage_cost_share
                ),
                "last_rollout_coverage_cost_limited": (
                    self._last_rollout_coverage_cost_limited
                ),
                "global_action_quality_ema": self._global_action_quality_ema,
                "low_roi_train_steps": self._low_roi_train_steps,
                "stop_recommended": self._stop_recommended,
                "rollout_dollar_seconds": self._rollout_dollar_seconds,
                "queue_wait_dollar_seconds": self._queue_wait_dollar_seconds,
                "rollout_admission_decisions": (
                    self._rollout_admission_decisions
                ),
                "rollout_admission_delay_s": self._rollout_admission_delay_s,
                "rollout_admission_dollar_seconds": (
                    self._rollout_admission_dollar_seconds
                ),
                "last_rollout_admission_delay_s": (
                    self._last_rollout_admission_delay_s
                ),
                "last_rollout_admission_pressure": (
                    self._last_rollout_admission_pressure
                ),
                "train_dollar_seconds": self._train_dollar_seconds,
                "stale_batches": self._stale_batches,
                "stale_trajectories": self._stale_trajectories,
                "stale_experience": self._stale_experience,
                "stale_lost_reward_improving_experience": (
                    self._stale_lost_reward_improving_experience
                ),
                "stale_sample_dollar_seconds": (
                    self._stale_sample_dollar_seconds
                ),
                "stale_unobserved_sample_dollar_seconds": (
                    self._stale_unobserved_sample_dollar_seconds
                ),
                "stale_additional_dollar_seconds": (
                    self._stale_additional_dollar_seconds
                ),
            },
            "arms": {
                arm_id: _arm_stats_state(stats)
                for arm_id, stats in self._arms.items()
            },
            "cadence_controls": {
                str(value): _dataclass_state(stats)
                for value, stats in self._cadence_controls.items()
            },
            "lag_controls": {
                str(value): _dataclass_state(stats)
                for value, stats in self._lag_controls.items()
            },
            "admission_controls": {
                str(value): _dataclass_state(stats)
                for value, stats in self._admission_controls.items()
            },
            "actor_count_controls": {
                str(value): _dataclass_state(stats)
                for value, stats in self._actor_count_controls.items()
            },
            "joint_action_controls": {
                key: _dataclass_state(stats)
                for key, stats in self._joint_action_controls.items()
            },
            "train_selection_controls": {
                key: _dataclass_state(stats)
                for key, stats in self._train_selection_controls.items()
            },
            "continuation_controls": {
                key: _dataclass_state(stats)
                for key, stats in self._continuation_controls.items()
            },
            "coverage_controls": {
                key: _dataclass_state(stats)
                for key, stats in self._coverage_controls.items()
            },
            "timing_response_controls": {
                key: _dataclass_state(stats)
                for key, stats in self._timing_response_controls.items()
            },
            "actors": {
                str(actor_id): _actor_stats_state(stats)
                for actor_id, stats in self._actors.items()
            },
            "last_decision": (
                dict(self._last_decision_snapshot)
                if self._last_decision_snapshot is not None
                else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load state produced by :meth:`state_dict`.

        Missing sections are tolerated so older checkpoints can still restore
        the useful parts of scheduler memory.
        """

        config = _mapping_state(state.get("config"))
        self.min_train_batch_groups = _state_int(
            config.get("min_train_batch_groups"),
            self.min_train_batch_groups,
        )
        self.max_train_batch_groups = _state_optional_int(
            config.get("max_train_batch_groups"),
            self.max_train_batch_groups,
        )
        self.min_policy_lag = _state_int(
            config.get("min_policy_lag"),
            self.min_policy_lag,
        )
        self.max_policy_lag_limit = _state_optional_int(
            config.get("max_policy_lag"),
            self.max_policy_lag_limit,
        )
        self.min_actor_count = max(
            1,
            _state_int(config.get("min_actor_count"), self.min_actor_count),
        )
        restored_max_actor_count = _state_optional_int(
            config.get("max_actor_count"),
            self.max_actor_count_limit,
        )
        if (
            restored_max_actor_count is not None
            and restored_max_actor_count < self.min_actor_count
        ):
            restored_max_actor_count = self.min_actor_count
        self.max_actor_count_limit = restored_max_actor_count
        self.ema_alpha = _state_float(config.get("ema_alpha"), self.ema_alpha)
        self.exploration_bonus = _state_float(
            config.get("exploration_bonus"),
            self.exploration_bonus,
        )
        self.objective_threshold = _state_float(
            config.get("objective_threshold"),
            self.objective_threshold,
        )
        self.unsafe_penalty = _state_float(
            config.get("unsafe_penalty"),
            self.unsafe_penalty,
        )
        self.rollout_objective_weight = _state_float(
            config.get("rollout_objective_weight"),
            self.rollout_objective_weight,
        )
        self.train_objective_weight = _state_float(
            config.get("train_objective_weight"),
            self.train_objective_weight,
        )
        self.reward_efficiency_weight = _state_float(
            config.get("reward_efficiency_weight"),
            self.reward_efficiency_weight,
        )
        self.stale_penalty_weight = _state_float(
            config.get("stale_penalty_weight"),
            self.stale_penalty_weight,
        )
        self.staleness_priority_weight = max(
            0.0,
            _state_float(
                config.get("staleness_priority_weight"),
                self.staleness_priority_weight,
            ),
        )
        self.off_policy_priority_weight = max(
            0.0,
            _state_float(
                config.get("off_policy_priority_weight"),
                self.off_policy_priority_weight,
            ),
        )
        self.off_policy_cadence_tightening_threshold = max(
            0.0,
            _state_float(
                config.get("off_policy_cadence_tightening_threshold"),
                self.off_policy_cadence_tightening_threshold,
            ),
        )
        self.off_policy_lag_tightening_threshold = max(
            0.0,
            _state_float(
                config.get("off_policy_lag_tightening_threshold"),
                self.off_policy_lag_tightening_threshold,
            ),
        )
        self.confidence_penalty_weight = max(
            0.0,
            _state_float(
                config.get("confidence_penalty_weight"),
                self.confidence_penalty_weight,
            ),
        )
        self.control_exploration_bonus = max(
            0.0,
            _state_float(
                config.get("control_exploration_bonus"),
                self.control_exploration_bonus,
            ),
        )
        self.rollout_cadence_lag_control_weight = max(
            0.0,
            _state_float(
                config.get("rollout_cadence_lag_control_weight"),
                self.rollout_cadence_lag_control_weight,
            ),
        )
        self.joint_action_objective_weight = max(
            0.0,
            _state_float(
                config.get("joint_action_objective_weight"),
                self.joint_action_objective_weight,
            ),
        )
        self.train_selection_objective_weight = max(
            0.0,
            _state_float(
                config.get("train_selection_objective_weight"),
                self.train_selection_objective_weight,
            ),
        )
        self.max_control_candidate_values = max(
            1,
            _state_int(
                config.get("max_control_candidate_values"),
                self.max_control_candidate_values,
            ),
        )
        self.min_rollout_coverage_fraction = min(
            1.0,
            max(
                0.0,
                _state_float(
                    config.get("min_rollout_coverage_fraction"),
                    self.min_rollout_coverage_fraction,
                ),
            ),
        )
        restored_coverage_cost_fraction = _state_optional_float(
            config.get("max_rollout_coverage_cost_fraction"),
            self.max_rollout_coverage_cost_fraction,
        )
        if (
            restored_coverage_cost_fraction is not None
            and not 0 < restored_coverage_cost_fraction <= 1
        ):
            restored_coverage_cost_fraction = None
        self.max_rollout_coverage_cost_fraction = restored_coverage_cost_fraction
        self.min_train_steps = _state_int(
            config.get("min_train_steps"),
            self.min_train_steps,
        )
        self.roi_patience = _state_optional_int(
            config.get("roi_patience"),
            self.roi_patience,
        )
        self.min_train_objective = _state_float(
            config.get("min_train_objective"),
            self.min_train_objective,
        )
        continuation_objective = str(
            config.get("continuation_objective", self.continuation_objective)
        )
        if continuation_objective in {"train", "accounted"}:
            self.continuation_objective = continuation_objective
        control_train_objective = str(
            config.get("control_train_objective", self.control_train_objective)
        )
        if control_train_objective in {"train", "accounted"}:
            self.control_train_objective = control_train_objective
        restored_budget = _state_optional_float(
            config.get("max_accounted_dollar_seconds"),
            self.max_accounted_dollar_seconds,
        )
        if restored_budget is not None and restored_budget <= 0.0:
            restored_budget = None
        self.max_accounted_dollar_seconds = restored_budget
        self.max_rollout_admission_delay_s = _state_float(
            config.get("max_rollout_admission_delay_s"),
            self.max_rollout_admission_delay_s,
        )
        self.rollout_admission_pressure_threshold = min(
            0.999999,
            max(
                0.0,
                _state_float(
                    config.get("rollout_admission_pressure_threshold"),
                    self.rollout_admission_pressure_threshold,
                ),
            ),
        )
        self.rollout_admission_positive_signal_scale = min(
            1.0,
            max(
                0.0,
                _state_float(
                    config.get("rollout_admission_positive_signal_scale"),
                    self.rollout_admission_positive_signal_scale,
                ),
            ),
        )
        self.reconstruction_drift_threshold = min(
            1.0,
            max(
                0.0,
                _state_float(
                    config.get("reconstruction_drift_threshold"),
                    self.reconstruction_drift_threshold,
                ),
            ),
        )
        reward_scale_normalization = str(
            config.get(
                "reward_scale_normalization",
                self.reward_scale_normalization,
            )
        )
        if reward_scale_normalization in {"none", "arm_range"}:
            self.reward_scale_normalization = reward_scale_normalization

        arms = _mapping_state(state.get("arms"))
        self._arms = {
            str(arm_id): _arm_stats_from_state(stats)
            for arm_id, stats in arms.items()
            if str(arm_id)
        }
        self._cadence_controls = _control_family_from_state(
            state.get("cadence_controls")
        )
        self._lag_controls = _control_family_from_state(state.get("lag_controls"))
        self._admission_controls = _control_family_from_state(
            state.get("admission_controls")
        )
        self._actor_count_controls = _control_family_from_state(
            state.get("actor_count_controls")
        )
        self._joint_action_controls = _string_control_family_from_state(
            state.get("joint_action_controls")
        )
        self._train_selection_controls = _string_control_family_from_state(
            state.get("train_selection_controls")
        )
        self._continuation_controls = _string_control_family_from_state(
            state.get("continuation_controls")
        )
        self._coverage_controls = _string_control_family_from_state(
            state.get("coverage_controls")
        )
        self._timing_response_controls = _string_control_family_from_state(
            state.get("timing_response_controls")
        )
        self._pending_continuation_decisions = {}
        self._recorded_continuation_stop_decisions = set()
        self._actors = _actor_family_from_state(state.get("actors"))

        learning_state = _mapping_state(state.get("learning_state"))
        total_pulls_default = sum(stats.pulls for stats in self._arms.values())
        total_decisions_default = sum(
            stats.decisions for stats in self._arms.values()
        )
        if total_decisions_default == 0:
            total_decisions_default = total_pulls_default
        self._total_pulls = _state_int(
            learning_state.get("total_pulls"),
            total_pulls_default,
        )
        self._total_decisions = _state_int(
            learning_state.get("total_decisions"),
            total_decisions_default,
        )
        self._global_objective_ema = _state_float(
            learning_state.get("global_objective_ema"),
            self._global_objective_ema,
        )
        self._train_observations = _state_int(
            learning_state.get("train_observations"),
            self._train_observations,
        )
        self._train_reward_ema = _state_float(
            learning_state.get("train_reward_ema"),
            self._train_reward_ema,
        )
        self._train_objective_ema = _state_float(
            learning_state.get("train_objective_ema"),
            self._train_objective_ema,
        )
        self._last_train_reward = _state_float(
            learning_state.get("last_train_reward"),
            self._last_train_reward,
        )
        self._last_train_objective = _state_float(
            learning_state.get("last_train_objective"),
            self._last_train_objective,
        )
        self._last_train_reward_improvement = _state_float(
            learning_state.get("last_train_reward_improvement"),
            self._last_train_reward_improvement,
        )
        self._last_train_experience_count = _state_float(
            learning_state.get("last_train_experience_count"),
            self._last_train_experience_count,
        )
        self._last_train_reward_improving_experience = _state_float(
            learning_state.get("last_train_reward_improving_experience"),
            self._last_train_reward_improving_experience,
        )
        self._last_train_control_reward_improving_experience = _state_float(
            learning_state.get(
                "last_train_control_reward_improving_experience"
            ),
            self._last_train_control_reward_improving_experience,
        )
        self._accounted_objective_ema = _state_float(
            learning_state.get("accounted_objective_ema"),
            self._accounted_objective_ema,
        )
        self._last_accounted_objective = _state_float(
            learning_state.get("last_accounted_objective"),
            self._last_accounted_objective,
        )
        self._last_accounted_reward_improving_experience = _state_float(
            learning_state.get("last_accounted_reward_improving_experience"),
            self._last_accounted_reward_improving_experience,
        )
        self._last_accounted_control_reward_improving_experience = _state_float(
            learning_state.get(
                "last_accounted_control_reward_improving_experience"
            ),
            self._last_accounted_control_reward_improving_experience,
        )
        self._last_accounted_dollar_seconds = _state_float(
            learning_state.get("last_accounted_dollar_seconds"),
            self._last_accounted_dollar_seconds,
        )
        self._last_continuation_objective = _state_float(
            learning_state.get("last_continuation_objective"),
            self._last_continuation_objective,
        )
        self._previous_accounted_dollar_seconds = _state_float(
            learning_state.get("previous_accounted_dollar_seconds"),
            self._previous_accounted_dollar_seconds,
        )
        self._last_stale_penalty_objective = _state_float(
            learning_state.get("last_stale_penalty_objective"),
            self._last_stale_penalty_objective,
        )
        self._last_stale_experience_count = _state_float(
            learning_state.get("last_stale_experience_count"),
            self._last_stale_experience_count,
        )
        self._last_stale_lost_reward_improving_experience = _state_float(
            learning_state.get(
                "last_stale_lost_reward_improving_experience"
            ),
            self._last_stale_lost_reward_improving_experience,
        )
        self._last_stale_sample_dollar_seconds = _state_float(
            learning_state.get("last_stale_sample_dollar_seconds"),
            self._last_stale_sample_dollar_seconds,
        )
        self._last_stale_unobserved_sample_dollar_seconds = _state_float(
            learning_state.get("last_stale_unobserved_sample_dollar_seconds"),
            self._last_stale_unobserved_sample_dollar_seconds,
        )
        self._last_stale_additional_dollar_seconds = _state_float(
            learning_state.get("last_stale_additional_dollar_seconds"),
            self._last_stale_additional_dollar_seconds,
        )
        self._last_stale_policy_step = _state_int(
            learning_state.get("last_stale_policy_step"),
            self._last_stale_policy_step,
        )
        self._last_stale_reason = str(
            learning_state.get("last_stale_reason", self._last_stale_reason)
        )
        self._last_train_batch_priority = _state_float(
            learning_state.get("last_train_batch_priority"),
            self._last_train_batch_priority,
        )
        self._last_train_batch_policy_lag = _state_int(
            learning_state.get("last_train_batch_policy_lag"),
            self._last_train_batch_policy_lag,
        )
        self._last_train_batch_lag_limit = _state_int(
            learning_state.get("last_train_batch_lag_limit"),
            self._last_train_batch_lag_limit,
        )
        self._last_train_batch_staleness_urgency = _state_float(
            learning_state.get("last_train_batch_staleness_urgency"),
            self._last_train_batch_staleness_urgency,
        )
        self._last_train_batch_staleness_bonus = _state_float(
            learning_state.get("last_train_batch_staleness_bonus"),
            self._last_train_batch_staleness_bonus,
        )
        self._last_train_batch_old_new_logprob_coverage = _state_float(
            learning_state.get("last_train_batch_old_new_logprob_coverage"),
            self._last_train_batch_old_new_logprob_coverage,
        )
        self._last_train_batch_off_policy_drift = _state_float(
            learning_state.get("last_train_batch_off_policy_drift"),
            self._last_train_batch_off_policy_drift,
        )
        self._last_train_batch_off_policy_penalty = _state_float(
            learning_state.get("last_train_batch_off_policy_penalty"),
            self._last_train_batch_off_policy_penalty,
        )
        self._last_train_batch_priority_before_off_policy = _state_float(
            learning_state.get("last_train_batch_priority_before_off_policy"),
            self._last_train_batch_priority_before_off_policy,
        )
        self._last_cadence_off_policy_penalty = _state_float(
            learning_state.get("last_cadence_off_policy_penalty"),
            self._last_cadence_off_policy_penalty,
        )
        self._last_cadence_off_policy_tightened = _state_bool(
            learning_state.get("last_cadence_off_policy_tightened"),
            self._last_cadence_off_policy_tightened,
        )
        self._last_policy_lag_off_policy_penalty = _state_float(
            learning_state.get("last_policy_lag_off_policy_penalty"),
            self._last_policy_lag_off_policy_penalty,
        )
        self._last_policy_lag_off_policy_tightened = _state_bool(
            learning_state.get("last_policy_lag_off_policy_tightened"),
            self._last_policy_lag_off_policy_tightened,
        )
        self._last_train_batch_reward_improving_experience = _state_float(
            learning_state.get("last_train_batch_reward_improving_experience"),
            self._last_train_batch_reward_improving_experience,
        )
        self._last_train_batch_sample_dollar_seconds = _state_float(
            learning_state.get("last_train_batch_sample_dollar_seconds"),
            self._last_train_batch_sample_dollar_seconds,
        )
        self._last_train_batch_cost_normalized_priority = _state_float(
            learning_state.get("last_train_batch_cost_normalized_priority"),
            self._last_train_batch_cost_normalized_priority,
        )
        self._last_train_batch_joint_action_score = _state_float(
            learning_state.get("last_train_batch_joint_action_score"),
            self._last_train_batch_joint_action_score,
        )
        self._last_train_batch_train_selection_score = _state_float(
            learning_state.get("last_train_batch_train_selection_score"),
            self._last_train_batch_train_selection_score,
        )
        self._last_continuation_decision_continue = _state_bool(
            learning_state.get("last_continuation_decision_continue"),
            self._last_continuation_decision_continue,
        )
        self._last_continuation_decision_key = str(
            learning_state.get(
                "last_continuation_decision_key",
                self._last_continuation_decision_key,
            )
        )
        self._last_continuation_decision_reason = str(
            learning_state.get(
                "last_continuation_decision_reason",
                self._last_continuation_decision_reason,
            )
        )
        self._last_continuation_pending_train_batches = _state_int(
            learning_state.get("last_continuation_pending_train_batches"),
            self._last_continuation_pending_train_batches,
        )
        self._last_continuation_train_queue_pressure = _state_float(
            learning_state.get("last_continuation_train_queue_pressure"),
            self._last_continuation_train_queue_pressure,
        )
        self._last_cadence_response_key = str(
            learning_state.get(
                "last_cadence_response_key",
                self._last_cadence_response_key,
            )
            or ""
        )
        self._last_cadence_response_reason = str(
            learning_state.get(
                "last_cadence_response_reason",
                self._last_cadence_response_reason,
            )
            or ""
        )
        self._last_policy_lag_response_key = str(
            learning_state.get(
                "last_policy_lag_response_key",
                self._last_policy_lag_response_key,
            )
            or ""
        )
        self._last_policy_lag_response_reason = str(
            learning_state.get(
                "last_policy_lag_response_reason",
                self._last_policy_lag_response_reason,
            )
            or ""
        )
        self._coverage_forced_decisions = _state_int(
            learning_state.get("coverage_forced_decisions"),
            self._coverage_forced_decisions,
        )
        self._last_rollout_coverage_target = _state_float(
            learning_state.get("last_rollout_coverage_target"),
            self._last_rollout_coverage_target,
        )
        self._last_rollout_coverage_share = _state_float(
            learning_state.get("last_rollout_coverage_share"),
            self._last_rollout_coverage_share,
        )
        self._last_rollout_coverage_deficit = _state_float(
            learning_state.get("last_rollout_coverage_deficit"),
            self._last_rollout_coverage_deficit,
        )
        self._last_rollout_coverage_cost_share = _state_float(
            learning_state.get("last_rollout_coverage_cost_share"),
            self._last_rollout_coverage_cost_share,
        )
        self._last_rollout_coverage_cost_limited = _state_bool(
            learning_state.get("last_rollout_coverage_cost_limited"),
            self._last_rollout_coverage_cost_limited,
        )
        self._global_action_quality_ema = _state_float(
            learning_state.get("global_action_quality_ema"),
            self._global_action_quality_ema,
        )
        self._low_roi_train_steps = _state_int(
            learning_state.get("low_roi_train_steps"),
            self._low_roi_train_steps,
        )
        self._stop_recommended = _state_bool(
            learning_state.get("stop_recommended"),
            self._stop_recommended,
        )
        self._rollout_dollar_seconds = _state_float(
            learning_state.get("rollout_dollar_seconds"),
            self._rollout_dollar_seconds,
        )
        self._queue_wait_dollar_seconds = _state_float(
            learning_state.get("queue_wait_dollar_seconds"),
            self._queue_wait_dollar_seconds,
        )
        self._rollout_admission_decisions = _state_int(
            learning_state.get("rollout_admission_decisions"),
            self._rollout_admission_decisions,
        )
        self._rollout_admission_delay_s = _state_float(
            learning_state.get("rollout_admission_delay_s"),
            self._rollout_admission_delay_s,
        )
        self._rollout_admission_dollar_seconds = _state_float(
            learning_state.get("rollout_admission_dollar_seconds"),
            self._rollout_admission_dollar_seconds,
        )
        self._last_rollout_admission_delay_s = _state_float(
            learning_state.get("last_rollout_admission_delay_s"),
            self._last_rollout_admission_delay_s,
        )
        self._last_rollout_admission_pressure = _state_float(
            learning_state.get("last_rollout_admission_pressure"),
            self._last_rollout_admission_pressure,
        )
        self._train_dollar_seconds = _state_float(
            learning_state.get("train_dollar_seconds"),
            self._train_dollar_seconds,
        )
        self._stale_batches = _state_int(
            learning_state.get("stale_batches"),
            self._stale_batches,
        )
        self._stale_trajectories = _state_int(
            learning_state.get("stale_trajectories"),
            self._stale_trajectories,
        )
        self._stale_experience = _state_float(
            learning_state.get("stale_experience"),
            self._stale_experience,
        )
        self._stale_lost_reward_improving_experience = _state_float(
            learning_state.get("stale_lost_reward_improving_experience"),
            self._stale_lost_reward_improving_experience,
        )
        self._stale_sample_dollar_seconds = _state_float(
            learning_state.get("stale_sample_dollar_seconds"),
            self._stale_sample_dollar_seconds,
        )
        self._stale_unobserved_sample_dollar_seconds = _state_float(
            learning_state.get("stale_unobserved_sample_dollar_seconds"),
            self._stale_unobserved_sample_dollar_seconds,
        )
        self._stale_additional_dollar_seconds = _state_float(
            learning_state.get("stale_additional_dollar_seconds"),
            self._stale_additional_dollar_seconds,
        )
        self._last_decision = None
        self._last_decision_snapshot = _decision_state_from_mapping(
            state.get("last_decision")
        )

    def metrics(self) -> dict[str, float]:
        inflight_rollouts = sum(stats.inflight for stats in self._arms.values())
        failure_rollouts = sum(
            stats.failed_rollouts for stats in self._arms.values()
        )
        reconstruction_observations = sum(
            stats.reconstruction_observations for stats in self._arms.values()
        )
        total_reconstruction_accuracy = sum(
            stats.total_reconstruction_accuracy for stats in self._arms.values()
        )
        max_reconstruction_drift = max(
            (stats.max_reconstruction_drift for stats in self._arms.values()),
            default=0.0,
        )
        accounted_dollar_seconds = self._accounted_dollar_seconds()
        reserved_inflight_dollar_seconds = (
            self._reserved_inflight_rollout_dollar_seconds()
        )
        projected_accounted_dollar_seconds = (
            accounted_dollar_seconds + reserved_inflight_dollar_seconds
        )
        budget_limit = self.max_accounted_dollar_seconds or 0.0
        budget_remaining = (
            max(0.0, budget_limit - projected_accounted_dollar_seconds)
            if self.max_accounted_dollar_seconds is not None
            else 0.0
        )
        budget_fraction = (
            projected_accounted_dollar_seconds / budget_limit
            if budget_limit > 0.0
            else 0.0
        )
        failure_modes: dict[str, int] = {}
        for stats in self._arms.values():
            for mode, count in stats.failure_modes.items():
                failure_modes[mode] = failure_modes.get(mode, 0) + count
        metrics: dict[str, float] = {
            "scheduler/total_rollout_decisions": float(self._total_decisions),
            "scheduler/total_rollout_observations": float(self._total_pulls),
            "scheduler/total_inflight_rollouts": float(inflight_rollouts),
            "scheduler/failure_rollouts": float(failure_rollouts),
            "scheduler/failure_rate": (
                failure_rollouts / self._total_pulls
                if self._total_pulls
                else 0.0
            ),
            "scheduler/reconstruction_observations": float(
                reconstruction_observations
            ),
            "scheduler/reconstruction_accuracy_mean": (
                total_reconstruction_accuracy / reconstruction_observations
                if reconstruction_observations
                else 0.0
            ),
            "scheduler/reconstruction_max_drift": max_reconstruction_drift,
            "scheduler/global_marginal_objective_ema": self._global_objective_ema,
            "scheduler/global_action_quality_ema": self._global_action_quality_ema,
            "scheduler/train_observations": float(self._train_observations),
            "scheduler/train_reward_ema": self._train_reward_ema,
            "scheduler/train_marginal_objective_ema": self._train_objective_ema,
            "scheduler/train_last_objective": self._last_train_objective,
            "scheduler/train_last_reward_improvement": (
                self._last_train_reward_improvement
            ),
            "scheduler/train_last_experience_count": self._last_train_experience_count,
            "scheduler/train_last_reward_improving_experience": (
                self._last_train_reward_improving_experience
            ),
            "scheduler/train_last_control_reward_improving_experience": (
                self._last_train_control_reward_improving_experience
            ),
            "scheduler/accounted_objective_ema": self._accounted_objective_ema,
            "scheduler/accounted_last_objective": self._last_accounted_objective,
            "scheduler/accounted_last_reward_improving_experience": (
                self._last_accounted_reward_improving_experience
            ),
            "scheduler/accounted_last_control_reward_improving_experience": (
                self._last_accounted_control_reward_improving_experience
            ),
            "scheduler/accounted_last_dollar_seconds": (
                self._last_accounted_dollar_seconds
            ),
            "scheduler/continuation_last_objective": (
                self._last_continuation_objective
            ),
            "scheduler/continuation/objective_accounted": (
                1.0 if self.continuation_objective == "accounted" else 0.0
            ),
            "scheduler/continuation/last_decision_continue": (
                1.0 if self._last_continuation_decision_continue else 0.0
            ),
            "scheduler/continuation/last_pending_train_batches": float(
                self._last_continuation_pending_train_batches
            ),
            "scheduler/continuation/last_train_queue_pressure": (
                self._last_continuation_train_queue_pressure
            ),
            "scheduler/budget/max_accounted_dollar_seconds": budget_limit,
            "scheduler/budget/accounted_dollar_seconds": accounted_dollar_seconds,
            "scheduler/budget/reserved_inflight_rollout_dollar_seconds": (
                reserved_inflight_dollar_seconds
            ),
            "scheduler/budget/projected_accounted_dollar_seconds": (
                projected_accounted_dollar_seconds
            ),
            "scheduler/budget/remaining_accounted_dollar_seconds": (
                budget_remaining
            ),
            "scheduler/budget/accounted_fraction": budget_fraction,
            "scheduler/budget/accounted_exhausted": (
                1.0 if self._accounted_budget_exhausted() else 0.0
            ),
            "scheduler/control/train_objective_accounted": (
                1.0 if self.control_train_objective == "accounted" else 0.0
            ),
            "scheduler/stale_batches": float(self._stale_batches),
            "scheduler/stale_trajectories": float(self._stale_trajectories),
            "scheduler/stale_experience": self._stale_experience,
            "scheduler/stale_lost_reward_improving_experience": (
                self._stale_lost_reward_improving_experience
            ),
            "scheduler/stale_sample_dollar_seconds": (
                self._stale_sample_dollar_seconds
            ),
            "scheduler/stale_unobserved_sample_dollar_seconds": (
                self._stale_unobserved_sample_dollar_seconds
            ),
            "scheduler/stale_additional_dollar_seconds": (
                self._stale_additional_dollar_seconds
            ),
            "scheduler/stale_total_dollar_seconds": (
                self._stale_sample_dollar_seconds
                + self._stale_additional_dollar_seconds
            ),
            "scheduler/stale_last_penalty_objective": (
                self._last_stale_penalty_objective
            ),
            "scheduler/stale_last_experience_count": (
                self._last_stale_experience_count
            ),
            "scheduler/stale_last_lost_reward_improving_experience": (
                self._last_stale_lost_reward_improving_experience
            ),
            "scheduler/stale_last_sample_dollar_seconds": (
                self._last_stale_sample_dollar_seconds
            ),
            "scheduler/stale_last_unobserved_sample_dollar_seconds": (
                self._last_stale_unobserved_sample_dollar_seconds
            ),
            "scheduler/stale_last_additional_dollar_seconds": (
                self._last_stale_additional_dollar_seconds
            ),
            "scheduler/stale_last_total_dollar_seconds": (
                self._last_stale_sample_dollar_seconds
                + self._last_stale_additional_dollar_seconds
            ),
            "scheduler/stale_last_policy_step": float(self._last_stale_policy_step),
            "scheduler/last_train_batch_priority": self._last_train_batch_priority,
            "scheduler/last_train_batch_policy_lag": float(
                self._last_train_batch_policy_lag
            ),
            "scheduler/last_train_batch_lag_limit": float(
                self._last_train_batch_lag_limit
            ),
            "scheduler/last_train_batch_staleness_urgency": (
                self._last_train_batch_staleness_urgency
            ),
            "scheduler/last_train_batch_staleness_bonus": (
                self._last_train_batch_staleness_bonus
            ),
            "scheduler/last_train_batch_old_new_logprob_coverage": (
                self._last_train_batch_old_new_logprob_coverage
            ),
            "scheduler/last_train_batch_off_policy_drift": (
                self._last_train_batch_off_policy_drift
            ),
            "scheduler/last_train_batch_off_policy_penalty": (
                self._last_train_batch_off_policy_penalty
            ),
            "scheduler/last_train_batch_priority_before_off_policy": (
                self._last_train_batch_priority_before_off_policy
            ),
            "scheduler/cadence/last_off_policy_penalty": (
                self._last_cadence_off_policy_penalty
            ),
            "scheduler/cadence/off_policy_tightened": (
                1.0 if self._last_cadence_off_policy_tightened else 0.0
            ),
            "scheduler/cadence/off_policy_tightening_threshold": (
                self.off_policy_cadence_tightening_threshold
            ),
            "scheduler/policy_lag/last_off_policy_penalty": (
                self._last_policy_lag_off_policy_penalty
            ),
            "scheduler/policy_lag/off_policy_tightened": (
                1.0 if self._last_policy_lag_off_policy_tightened else 0.0
            ),
            "scheduler/policy_lag/off_policy_tightening_threshold": (
                self.off_policy_lag_tightening_threshold
            ),
            "scheduler/last_train_batch_reward_improving_experience": (
                self._last_train_batch_reward_improving_experience
            ),
            "scheduler/last_train_batch_sample_dollar_seconds": (
                self._last_train_batch_sample_dollar_seconds
            ),
            "scheduler/last_train_batch_cost_normalized_priority": (
                self._last_train_batch_cost_normalized_priority
            ),
            "scheduler/last_train_batch_joint_action_score": (
                self._last_train_batch_joint_action_score
            ),
            "scheduler/last_train_batch_train_selection_score": (
                self._last_train_batch_train_selection_score
            ),
            "scheduler/low_roi_train_steps": float(self._low_roi_train_steps),
            "scheduler/stop_recommended": 1.0 if self._stop_recommended else 0.0,
            "scheduler/weights/rollout_objective": self.rollout_objective_weight,
            "scheduler/weights/train_objective": self.train_objective_weight,
            "scheduler/weights/reward_efficiency": self.reward_efficiency_weight,
            "scheduler/weights/stale_penalty": self.stale_penalty_weight,
            "scheduler/weights/staleness_priority": (
                self.staleness_priority_weight
            ),
            "scheduler/weights/off_policy_priority": (
                self.off_policy_priority_weight
            ),
            "scheduler/weights/confidence_penalty": (
                self.confidence_penalty_weight
            ),
            "scheduler/weights/control_exploration": (
                self.control_exploration_bonus
            ),
            "scheduler/weights/rollout_cadence_lag_control": (
                self.rollout_cadence_lag_control_weight
            ),
            "scheduler/weights/joint_action_objective": (
                self.joint_action_objective_weight
            ),
            "scheduler/weights/train_selection_objective": (
                self.train_selection_objective_weight
            ),
            "scheduler/reward_scale_normalization/arm_range": (
                1.0
                if self.reward_scale_normalization == "arm_range"
                else 0.0
            ),
            "scheduler/coverage/min_fraction": (
                self.min_rollout_coverage_fraction
            ),
            "scheduler/coverage/max_cost_fraction": (
                self.max_rollout_coverage_cost_fraction or 0.0
            ),
            "scheduler/coverage/forced_decisions": float(
                self._coverage_forced_decisions
            ),
            "scheduler/coverage/last_target": (
                self._last_rollout_coverage_target
            ),
            "scheduler/coverage/last_share": self._last_rollout_coverage_share,
            "scheduler/coverage/last_deficit": (
                self._last_rollout_coverage_deficit
            ),
            "scheduler/coverage/last_cost_share": (
                self._last_rollout_coverage_cost_share
            ),
            "scheduler/coverage/last_cost_limited": (
                1.0 if self._last_rollout_coverage_cost_limited else 0.0
            ),
            "scheduler/max_control_candidate_values": float(
                self.max_control_candidate_values
            ),
            "scheduler/weights/unsafe_penalty": self.unsafe_penalty,
            "scheduler/reconstruction_drift_threshold": (
                self.reconstruction_drift_threshold
            ),
            "scheduler/costs/rollout_dollar_seconds": self._rollout_dollar_seconds,
            "scheduler/costs/queue_wait_dollar_seconds": (
                self._queue_wait_dollar_seconds
            ),
            "scheduler/admission/decisions": float(
                self._rollout_admission_decisions
            ),
            "scheduler/admission/total_delay_s": (
                self._rollout_admission_delay_s
            ),
            "scheduler/admission/last_delay_s": (
                self._last_rollout_admission_delay_s
            ),
            "scheduler/admission/last_pressure": (
                self._last_rollout_admission_pressure
            ),
            "scheduler/costs/rollout_admission_dollar_seconds": (
                self._rollout_admission_dollar_seconds
            ),
            "scheduler/costs/train_dollar_seconds": self._train_dollar_seconds,
            "scheduler/costs/stale_unobserved_sample_dollar_seconds": (
                self._stale_unobserved_sample_dollar_seconds
            ),
            "scheduler/costs/stale_additional_dollar_seconds": (
                self._stale_additional_dollar_seconds
            ),
            "scheduler/costs/stale_total_dollar_seconds": (
                self._stale_sample_dollar_seconds
                + self._stale_additional_dollar_seconds
            ),
            "scheduler/costs/total_dollar_seconds": accounted_dollar_seconds,
        }
        if self._last_decision_snapshot is not None:
            last_arm_id = str(self._last_decision_snapshot.get("arm_id", ""))
            last_metadata = _mapping_state(
                self._last_decision_snapshot.get("metadata")
            )
            metrics[
                f"scheduler/last_arm/{_safe_metric_key(last_arm_id)}"
            ] = 1.0
            metrics["scheduler/last_target_train_batch_groups"] = float(
                self._last_decision_snapshot.get("target_train_batch_groups", 0)
            )
            metrics["scheduler/last_max_policy_lag"] = float(
                self._last_decision_snapshot.get("max_policy_lag", 0)
            )
            metrics["scheduler/last_rollout_estimated_dollar_seconds"] = (
                _state_float(
                    last_metadata.get("estimated_rollout_dollar_seconds"),
                    0.0,
                )
            )
            metrics["scheduler/last_rollout_unobserved_cost_penalty"] = (
                _state_float(
                    last_metadata.get("unobserved_rollout_cost_penalty"),
                    0.0,
                )
            )
            metrics["scheduler/last_rollout_unobserved_cost_estimated"] = (
                1.0
                if _state_bool(
                    last_metadata.get("unobserved_rollout_cost_estimated"),
                    False,
                )
                else 0.0
            )
        for mode, count in failure_modes.items():
            safe_mode = _safe_metric_key(mode)
            metrics[f"scheduler/failure/{safe_mode}"] = float(count)
            metrics[f"scheduler/failure/{safe_mode}_rate"] = (
                count / self._total_pulls if self._total_pulls else 0.0
            )
        for arm_id, stats in self._arms.items():
            prefix = f"scheduler/arm/{_safe_metric_key(arm_id)}"
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/decision_share"] = self._arm_decision_share(
                arm_id
            )
            metrics[f"{prefix}/inflight"] = float(stats.inflight)
            metrics[f"{prefix}/reserved_rollout_dollar_seconds"] = (
                stats.reserved_rollout_dollar_seconds
            )
            metrics[f"{prefix}/pulls"] = float(stats.pulls)
            metrics[f"{prefix}/accepted"] = float(stats.accepted)
            metrics[f"{prefix}/unsafe"] = float(stats.unsafe)
            metrics[f"{prefix}/unsafe_rate"] = (
                stats.unsafe / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/failure_rollouts"] = float(stats.failed_rollouts)
            metrics[f"{prefix}/failure_rate"] = (
                stats.failed_rollouts / stats.pulls if stats.pulls else 0.0
            )
            for mode, count in stats.failure_modes.items():
                safe_mode = _safe_metric_key(mode)
                metrics[f"{prefix}/failure/{safe_mode}"] = float(count)
                metrics[f"{prefix}/failure/{safe_mode}_rate"] = (
                    count / stats.pulls if stats.pulls else 0.0
                )
            metrics[f"{prefix}/reconstruction_observations"] = float(
                stats.reconstruction_observations
            )
            metrics[f"{prefix}/reconstruction_accuracy_ema"] = (
                stats.reconstruction_accuracy_ema
                if stats.reconstruction_observations
                else 0.0
            )
            metrics[f"{prefix}/reconstruction_accuracy_mean"] = (
                stats.total_reconstruction_accuracy
                / stats.reconstruction_observations
                if stats.reconstruction_observations
                else 0.0
            )
            metrics[f"{prefix}/reconstruction_accuracy_min"] = (
                stats.min_reconstruction_accuracy
                if stats.reconstruction_observations
                else 0.0
            )
            metrics[f"{prefix}/reconstruction_drift_ema"] = (
                stats.reconstruction_drift_ema
                if stats.reconstruction_observations
                else 0.0
            )
            metrics[f"{prefix}/reconstruction_max_drift"] = (
                stats.max_reconstruction_drift
                if stats.reconstruction_observations
                else 0.0
            )
            metrics[f"{prefix}/reward_ema"] = stats.reward_ema
            metrics[f"{prefix}/effective_reward_ema"] = stats.effective_reward_ema
            metrics[f"{prefix}/effective_reward_min"] = (
                stats.min_effective_reward if stats.pulls else 0.0
            )
            metrics[f"{prefix}/effective_reward_max"] = (
                stats.max_effective_reward if stats.pulls else 0.0
            )
            metrics[f"{prefix}/last_reward_scale"] = stats.last_reward_scale
            metrics[f"{prefix}/last_normalized_positive_improvement"] = (
                stats.last_normalized_positive_improvement
            )
            metrics[f"{prefix}/action_quality_ema"] = stats.action_quality_ema
            metrics[f"{prefix}/marginal_objective_ema"] = (
                stats.marginal_objective_ema
            )
            metrics[f"{prefix}/policy_improvement_objective_ema"] = (
                stats.policy_improvement_objective_ema
            )
            metrics[f"{prefix}/last_train_reward"] = stats.last_train_reward
            metrics[f"{prefix}/train_reward_min"] = (
                stats.min_train_reward if stats.train_reward_observations else 0.0
            )
            metrics[f"{prefix}/train_reward_max"] = (
                stats.max_train_reward if stats.train_reward_observations else 0.0
            )
            metrics[f"{prefix}/last_train_reward_scale"] = (
                stats.last_train_reward_scale
            )
            metrics[f"{prefix}/last_train_reward_improvement"] = (
                stats.last_train_reward_improvement
            )
            metrics[f"{prefix}/last_normalized_train_reward_improvement"] = (
                stats.last_normalized_train_reward_improvement
            )
            metrics[f"{prefix}/reward_efficiency_ema"] = stats.reward_efficiency_ema
            metrics[f"{prefix}/objective_score"] = self._arm_value(stats)
            metrics[f"{prefix}/raw_objective_score"] = self._raw_arm_value(stats)
            metrics[f"{prefix}/confidence_penalty"] = self._confidence_penalty(stats)
            metrics[f"{prefix}/objective_observations"] = float(
                stats.objective_observations
            )
            metrics[f"{prefix}/objective_mean"] = stats.objective_mean
            metrics[f"{prefix}/objective_stddev"] = _objective_stddev(stats)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _arm_feedback_updates(stats)
            )
            metrics[f"{prefix}/stale_batches"] = float(stats.stale_batches)
            metrics[f"{prefix}/stale_trajectories"] = float(
                stats.stale_trajectories
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
            metrics[f"{prefix}/sample_dollar_seconds"] = stats.total_dollar_seconds
            metrics[f"{prefix}/sample_dollar_share"] = (
                self._arm_sample_dollar_share(arm_id)
            )
            metrics[f"{prefix}/rollout_dollar_seconds"] = stats.rollout_dollar_seconds
            metrics[f"{prefix}/queue_wait_dollar_seconds"] = (
                stats.queue_wait_dollar_seconds
            )
            metrics[f"{prefix}/admission_dollar_seconds"] = (
                stats.admission_dollar_seconds
            )
            metrics[f"{prefix}/sample_dollar_seconds_ema"] = (
                stats.dollar_seconds_ema
            )
            metrics[f"{prefix}/rollout_dollar_seconds_ema"] = (
                stats.dollar_seconds_ema
            )
            metrics[f"{prefix}/mean_rollout_dollar_seconds"] = (
                stats.rollout_dollar_seconds / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/mean_sample_dollar_seconds"] = (
                stats.total_dollar_seconds / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/mean_queue_wait_dollar_seconds"] = (
                stats.queue_wait_dollar_seconds / stats.pulls
                if stats.pulls
                else 0.0
            )
            metrics[f"{prefix}/mean_admission_dollar_seconds"] = (
                stats.admission_dollar_seconds / stats.pulls
                if stats.pulls
                else 0.0
            )
            metrics[f"{prefix}/estimated_rollout_dollar_seconds"] = (
                self._estimated_rollout_dollar_seconds(arm_id)
            )
            metrics[f"{prefix}/unobserved_rollout_cost_penalty"] = (
                self._unobserved_rollout_cost_penalty(arm_id)
            )
            metrics[f"{prefix}/action_units"] = float(stats.action_units)
            metrics[f"{prefix}/source_tokens"] = float(stats.source_tokens)
            metrics[f"{prefix}/semantic_bandwidth_tokens_per_decision"] = (
                stats.source_tokens / stats.action_units
                if stats.action_units
                else 0.0
            )
            metrics[f"{prefix}/old_logprob_coverage"] = (
                stats.old_logprob_units / stats.action_units
                if stats.action_units
                else 0.0
            )
            metrics[f"{prefix}/new_logprob_coverage"] = (
                stats.new_logprob_units / stats.action_units
                if stats.action_units
                else 0.0
            )
            metrics[f"{prefix}/reference_logprob_coverage"] = (
                stats.reference_logprob_units / stats.action_units
                if stats.action_units
                else 0.0
            )
            metrics[f"{prefix}/old_new_logprob_delta_mean"] = (
                stats.old_new_logprob_delta_sum / stats.old_new_logprob_pairs
                if stats.old_new_logprob_pairs
                else 0.0
            )
            metrics[f"{prefix}/old_new_logprob_abs_delta_mean"] = (
                stats.old_new_logprob_abs_delta_sum
                / stats.old_new_logprob_pairs
                if stats.old_new_logprob_pairs
                else 0.0
            )
            metrics[f"{prefix}/importance_ratio_mean"] = (
                stats.importance_ratio_sum / stats.old_new_logprob_pairs
                if stats.old_new_logprob_pairs
                else 0.0
            )
            metrics[f"{prefix}/old_reference_logprob_delta_mean"] = (
                stats.old_reference_logprob_delta_sum
                / stats.old_reference_logprob_pairs
                if stats.old_reference_logprob_pairs
                else 0.0
            )
            metrics[f"{prefix}/action_units_per_dollar_second"] = (
                stats.action_units / stats.total_dollar_seconds
                if stats.total_dollar_seconds > 0.0
                else 0.0
            )
            metrics[f"{prefix}/source_tokens_per_dollar_second"] = (
                stats.source_tokens / stats.total_dollar_seconds
                if stats.total_dollar_seconds > 0.0
                else 0.0
            )
            metrics[f"{prefix}/total_positive_improvement"] = (
                stats.total_positive_improvement
            )
            metrics[f"{prefix}/total_normalized_positive_improvement"] = (
                stats.total_normalized_positive_improvement
            )
            metrics[f"{prefix}/total_reward_improving_experience"] = (
                stats.total_reward_improving_experience
            )
            metrics[f"{prefix}/total_normalized_reward_improving_experience"] = (
                stats.total_normalized_reward_improving_experience
            )
            metrics[f"{prefix}/total_improvement_per_dollar_second"] = (
                stats.total_positive_improvement / stats.total_dollar_seconds
                if stats.total_dollar_seconds > 0.0
                else 0.0
            )
            metrics[f"{prefix}/total_policy_improvement_objective"] = (
                stats.total_policy_improvement_objective
            )
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
        for actor_id, stats in sorted(self._actors.items()):
            prefix = f"scheduler/actor/{_actor_metric_key(actor_id)}"
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/inflight"] = float(stats.inflight)
            metrics[f"{prefix}/pulls"] = float(stats.pulls)
            metrics[f"{prefix}/accepted"] = float(stats.accepted)
            metrics[f"{prefix}/unsafe"] = float(stats.unsafe)
            metrics[f"{prefix}/unsafe_rate"] = (
                stats.unsafe / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/failure_rollouts"] = float(stats.failed_rollouts)
            metrics[f"{prefix}/failure_rate"] = (
                stats.failed_rollouts / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _actor_feedback_updates(stats)
            )
            metrics[f"{prefix}/stale_batches"] = float(stats.stale_batches)
            metrics[f"{prefix}/stale_trajectories"] = float(
                stats.stale_trajectories
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/rollout_objective_ema"] = (
                stats.rollout_objective_ema
            )
            metrics[f"{prefix}/train_objective_ema"] = stats.train_objective_ema
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_rollout_objective"] = (
                stats.total_rollout_objective
            )
            metrics[f"{prefix}/total_train_objective"] = (
                stats.total_train_objective
            )
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/sample_dollar_seconds"] = stats.total_dollar_seconds
            metrics[f"{prefix}/rollout_dollar_seconds"] = stats.rollout_dollar_seconds
            metrics[f"{prefix}/queue_wait_dollar_seconds"] = (
                stats.queue_wait_dollar_seconds
            )
            metrics[f"{prefix}/admission_dollar_seconds"] = (
                stats.admission_dollar_seconds
            )
            metrics[f"{prefix}/mean_sample_dollar_seconds"] = (
                stats.total_dollar_seconds / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/mean_queue_wait_dollar_seconds"] = (
                stats.queue_wait_dollar_seconds / stats.pulls
                if stats.pulls
                else 0.0
            )
            metrics[f"{prefix}/mean_admission_dollar_seconds"] = (
                stats.admission_dollar_seconds / stats.pulls
                if stats.pulls
                else 0.0
            )
            metrics[f"{prefix}/action_units"] = float(stats.action_units)
            metrics[f"{prefix}/source_tokens"] = float(stats.source_tokens)
            metrics[f"{prefix}/semantic_bandwidth_tokens_per_decision"] = (
                stats.source_tokens / stats.action_units
                if stats.action_units
                else 0.0
            )
        for value, stats in sorted(self._cadence_controls.items()):
            prefix = f"scheduler/control/cadence_{value}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._cadence_controls,
                value,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(self._cadence_controls, stats)
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        for value, stats in sorted(self._lag_controls.items()):
            prefix = f"scheduler/control/policy_lag_{value}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._lag_controls,
                value,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(self._lag_controls, stats)
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        for value, stats in sorted(self._admission_controls.items()):
            prefix = f"scheduler/control/admission_delay_ms_{value}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._admission_controls,
                value,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._admission_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        for value, stats in sorted(self._actor_count_controls.items()):
            prefix = f"scheduler/control/actor_count_{value}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._actor_count_controls,
                value,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._actor_count_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        coverage_feedback_updates = {
            key: _control_feedback_updates(stats)
            for key, stats in self._coverage_controls.items()
        }
        coverage_decisions = sum(
            stats.decisions for stats in self._coverage_controls.values()
        )
        coverage_rollout_updates = sum(
            stats.rollout_updates for stats in self._coverage_controls.values()
        )
        coverage_train_updates = sum(
            stats.train_updates for stats in self._coverage_controls.values()
        )
        coverage_stale_updates = sum(
            stats.stale_updates for stats in self._coverage_controls.values()
        )
        coverage_total_feedback_updates = sum(coverage_feedback_updates.values())
        coverage_total_objective = sum(
            stats.total_objective for stats in self._coverage_controls.values()
        )
        coverage_total_stale_penalty_objective = sum(
            stats.total_stale_penalty_objective
            for stats in self._coverage_controls.values()
        )
        metrics["scheduler/coverage_control/keys"] = float(
            len(self._coverage_controls)
        )
        metrics["scheduler/coverage_control/decisions"] = float(
            coverage_decisions
        )
        metrics["scheduler/coverage_control/rollout_updates"] = float(
            coverage_rollout_updates
        )
        metrics["scheduler/coverage_control/train_updates"] = float(
            coverage_train_updates
        )
        metrics["scheduler/coverage_control/stale_updates"] = float(
            coverage_stale_updates
        )
        metrics["scheduler/coverage_control/feedback_updates"] = float(
            coverage_total_feedback_updates
        )
        metrics["scheduler/coverage_control/feedback_keys"] = float(
            sum(1 for updates in coverage_feedback_updates.values() if updates > 0)
        )
        metrics["scheduler/coverage_control/positive_objective_keys"] = float(
            sum(
                1
                for stats in self._coverage_controls.values()
                if stats.total_objective > 0.0
            )
        )
        metrics["scheduler/coverage_control/total_objective"] = (
            coverage_total_objective
        )
        metrics["scheduler/coverage_control/mean_objective_per_decision"] = (
            coverage_total_objective / coverage_decisions
            if coverage_decisions
            else 0.0
        )
        metrics[
            "scheduler/coverage_control/mean_objective_per_feedback_update"
        ] = (
            coverage_total_objective / coverage_total_feedback_updates
            if coverage_total_feedback_updates
            else 0.0
        )
        metrics["scheduler/coverage_control/total_stale_penalty_objective"] = (
            coverage_total_stale_penalty_objective
        )
        for key, stats in sorted(self._coverage_controls.items()):
            prefix = f"scheduler/coverage_control/{_safe_metric_key(key)}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._coverage_controls,
                key,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._coverage_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        timing_feedback_updates = {
            key: _control_feedback_updates(stats)
            for key, stats in self._timing_response_controls.items()
        }
        timing_decisions = sum(
            stats.decisions for stats in self._timing_response_controls.values()
        )
        timing_rollout_updates = sum(
            stats.rollout_updates for stats in self._timing_response_controls.values()
        )
        timing_train_updates = sum(
            stats.train_updates for stats in self._timing_response_controls.values()
        )
        timing_stale_updates = sum(
            stats.stale_updates for stats in self._timing_response_controls.values()
        )
        timing_total_feedback_updates = sum(timing_feedback_updates.values())
        timing_total_objective = sum(
            stats.total_objective for stats in self._timing_response_controls.values()
        )
        timing_total_stale_penalty_objective = sum(
            stats.total_stale_penalty_objective
            for stats in self._timing_response_controls.values()
        )
        metrics["scheduler/timing_response/keys"] = float(
            len(self._timing_response_controls)
        )
        metrics["scheduler/timing_response/decisions"] = float(timing_decisions)
        metrics["scheduler/timing_response/rollout_updates"] = float(
            timing_rollout_updates
        )
        metrics["scheduler/timing_response/train_updates"] = float(
            timing_train_updates
        )
        metrics["scheduler/timing_response/stale_updates"] = float(
            timing_stale_updates
        )
        metrics["scheduler/timing_response/feedback_updates"] = float(
            timing_total_feedback_updates
        )
        metrics["scheduler/timing_response/feedback_keys"] = float(
            sum(1 for updates in timing_feedback_updates.values() if updates > 0)
        )
        metrics["scheduler/timing_response/positive_objective_keys"] = float(
            sum(
                1
                for stats in self._timing_response_controls.values()
                if stats.total_objective > 0.0
            )
        )
        metrics["scheduler/timing_response/total_objective"] = (
            timing_total_objective
        )
        metrics["scheduler/timing_response/mean_objective_per_decision"] = (
            timing_total_objective / timing_decisions
            if timing_decisions
            else 0.0
        )
        metrics[
            "scheduler/timing_response/mean_objective_per_feedback_update"
        ] = (
            timing_total_objective / timing_total_feedback_updates
            if timing_total_feedback_updates
            else 0.0
        )
        metrics["scheduler/timing_response/total_stale_penalty_objective"] = (
            timing_total_stale_penalty_objective
        )
        metrics["scheduler/timing_response/last_cadence_has_key"] = (
            1.0 if self._last_cadence_response_key else 0.0
        )
        metrics["scheduler/timing_response/last_policy_lag_has_key"] = (
            1.0 if self._last_policy_lag_response_key else 0.0
        )
        for key, stats in sorted(self._timing_response_controls.items()):
            prefix = f"scheduler/timing_response/{_safe_metric_key(key)}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._timing_response_controls,
                key,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._timing_response_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        continuation_feedback_updates = {
            key: _control_feedback_updates(stats)
            for key, stats in self._continuation_controls.items()
        }
        continuation_decisions = sum(
            stats.decisions for stats in self._continuation_controls.values()
        )
        continuation_train_updates = sum(
            stats.train_updates for stats in self._continuation_controls.values()
        )
        continuation_total_feedback_updates = sum(
            continuation_feedback_updates.values()
        )
        continuation_total_objective = sum(
            stats.total_objective
            for stats in self._continuation_controls.values()
        )
        metrics["scheduler/continuation/keys"] = float(
            len(self._continuation_controls)
        )
        metrics["scheduler/continuation/decisions"] = float(
            continuation_decisions
        )
        metrics["scheduler/continuation/train_updates"] = float(
            continuation_train_updates
        )
        metrics["scheduler/continuation/feedback_updates"] = float(
            continuation_total_feedback_updates
        )
        metrics["scheduler/continuation/positive_objective_keys"] = float(
            sum(
                1
                for stats in self._continuation_controls.values()
                if stats.total_objective > 0.0
            )
        )
        metrics["scheduler/continuation/total_objective"] = (
            continuation_total_objective
        )
        metrics["scheduler/continuation/mean_objective_per_decision"] = (
            continuation_total_objective / continuation_decisions
            if continuation_decisions
            else 0.0
        )
        metrics[
            "scheduler/continuation/mean_objective_per_feedback_update"
        ] = (
            continuation_total_objective / continuation_total_feedback_updates
            if continuation_total_feedback_updates
            else 0.0
        )
        for key, stats in sorted(self._continuation_controls.items()):
            prefix = f"scheduler/continuation/{_safe_metric_key(key)}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._continuation_controls,
                key,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._continuation_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
        train_selection_feedback_updates = {
            key: _control_feedback_updates(stats)
            for key, stats in self._train_selection_controls.items()
        }
        train_selection_decisions = sum(
            stats.decisions for stats in self._train_selection_controls.values()
        )
        train_selection_train_updates = sum(
            stats.train_updates for stats in self._train_selection_controls.values()
        )
        train_selection_total_feedback_updates = sum(
            train_selection_feedback_updates.values()
        )
        train_selection_total_objective = sum(
            stats.total_objective
            for stats in self._train_selection_controls.values()
        )
        metrics["scheduler/train_selection/keys"] = float(
            len(self._train_selection_controls)
        )
        metrics["scheduler/train_selection/decisions"] = float(
            train_selection_decisions
        )
        metrics["scheduler/train_selection/train_updates"] = float(
            train_selection_train_updates
        )
        metrics["scheduler/train_selection/feedback_updates"] = float(
            train_selection_total_feedback_updates
        )
        metrics["scheduler/train_selection/positive_objective_keys"] = float(
            sum(
                1
                for stats in self._train_selection_controls.values()
                if stats.total_objective > 0.0
            )
        )
        metrics["scheduler/train_selection/total_objective"] = (
            train_selection_total_objective
        )
        metrics["scheduler/train_selection/mean_objective_per_decision"] = (
            train_selection_total_objective / train_selection_decisions
            if train_selection_decisions
            else 0.0
        )
        metrics[
            "scheduler/train_selection/mean_objective_per_feedback_update"
        ] = (
            train_selection_total_objective / train_selection_total_feedback_updates
            if train_selection_total_feedback_updates
            else 0.0
        )
        for key, stats in sorted(self._train_selection_controls.items()):
            prefix = f"scheduler/train_selection/{_safe_metric_key(key)}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._train_selection_controls,
                key,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._train_selection_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
        joint_feedback_updates = {
            key: _control_feedback_updates(stats)
            for key, stats in self._joint_action_controls.items()
        }
        joint_decisions = sum(
            stats.decisions for stats in self._joint_action_controls.values()
        )
        joint_rollout_updates = sum(
            stats.rollout_updates for stats in self._joint_action_controls.values()
        )
        joint_train_updates = sum(
            stats.train_updates for stats in self._joint_action_controls.values()
        )
        joint_stale_updates = sum(
            stats.stale_updates for stats in self._joint_action_controls.values()
        )
        joint_total_feedback_updates = sum(joint_feedback_updates.values())
        joint_total_objective = sum(
            stats.total_objective for stats in self._joint_action_controls.values()
        )
        joint_total_stale_penalty_objective = sum(
            stats.total_stale_penalty_objective
            for stats in self._joint_action_controls.values()
        )
        metrics["scheduler/joint_action/tuples"] = float(
            len(self._joint_action_controls)
        )
        metrics["scheduler/joint_action/decisions"] = float(joint_decisions)
        metrics["scheduler/joint_action/rollout_updates"] = float(
            joint_rollout_updates
        )
        metrics["scheduler/joint_action/train_updates"] = float(joint_train_updates)
        metrics["scheduler/joint_action/stale_updates"] = float(joint_stale_updates)
        metrics["scheduler/joint_action/feedback_updates"] = float(
            joint_total_feedback_updates
        )
        metrics["scheduler/joint_action/feedback_tuples"] = float(
            sum(1 for updates in joint_feedback_updates.values() if updates > 0)
        )
        metrics["scheduler/joint_action/positive_objective_tuples"] = float(
            sum(
                1
                for stats in self._joint_action_controls.values()
                if stats.total_objective > 0.0
            )
        )
        metrics["scheduler/joint_action/total_objective"] = joint_total_objective
        metrics["scheduler/joint_action/mean_objective_per_decision"] = (
            joint_total_objective / joint_decisions if joint_decisions else 0.0
        )
        metrics["scheduler/joint_action/mean_objective_per_feedback_update"] = (
            joint_total_objective / joint_total_feedback_updates
            if joint_total_feedback_updates
            else 0.0
        )
        metrics["scheduler/joint_action/total_stale_penalty_objective"] = (
            joint_total_stale_penalty_objective
        )
        for key, stats in sorted(self._joint_action_controls.items()):
            prefix = f"scheduler/joint_action/{_safe_metric_key(key)}"
            feedback_updates = _control_feedback_updates(stats)
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(feedback_updates)
            metrics[f"{prefix}/mean_objective_per_decision"] = (
                stats.total_objective / stats.decisions
                if stats.decisions
                else 0.0
            )
            metrics[f"{prefix}/mean_objective_per_feedback_update"] = (
                stats.total_objective / feedback_updates
                if feedback_updates
                else 0.0
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/score"] = self._score_control_value(
                self._joint_action_controls,
                key,
            )
            metrics[f"{prefix}/exploration_score"] = (
                self._control_exploration_value(
                    self._joint_action_controls,
                    stats,
                )
            )
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        return metrics

    def _arm_candidates(
        self,
        scenarios: Sequence[Scenario],
        action_codecs: Sequence[ActionCodec],
    ) -> list[tuple[str, Scenario, ActionCodec]]:
        return [
            (_arm_id(scenario, codec), scenario, codec)
            for scenario in scenarios
            for codec in action_codecs
        ]

    def _coverage_candidate(
        self,
        arms: Sequence[tuple[str, Scenario, ActionCodec]],
        *,
        target_train_batch_groups: int,
        max_policy_lag: int,
        active_actor_count: int | None,
        rollout_admission_delay_ms: int | None,
        action_space_key: str | None = None,
    ) -> tuple[
        tuple[str, Scenario, ActionCodec, float, float, float, float] | None,
        bool,
    ]:
        if self.min_rollout_coverage_fraction <= 0.0 or not arms:
            return None, False
        if self._total_decisions < len(arms):
            return None, False
        for arm_id, _, _ in arms:
            stats = self._arms.get(arm_id)
            if stats is None or stats.decisions == 0:
                return None, False
        target = self._effective_coverage_target(len(arms))
        if target <= 0.0:
            return None, False

        most_undercovered: (
            tuple[str, Scenario, ActionCodec, float, float, float, float] | None
        ) = None
        arm_ids = [arm_id for arm_id, _, _ in arms]
        cost_limited = False
        for arm_id, scenario, codec in arms:
            self._arms.setdefault(arm_id, ArmStats())
            share = self._arm_decision_share(arm_id)
            deficit = target - share
            if deficit <= 0.0:
                continue
            cost_share = self._arm_sample_dollar_share(arm_id, arm_ids)
            if self._coverage_cost_limited(cost_share):
                cost_limited = True
                continue
            candidate = (
                arm_id,
                scenario,
                codec,
                target,
                share,
                deficit,
                cost_share,
            )
            if most_undercovered is None:
                most_undercovered = candidate
                continue
            if deficit > most_undercovered[5]:
                most_undercovered = candidate
                continue
            if (
                math.isclose(deficit, most_undercovered[5])
                and self._score_arm(
                    arm_id,
                    scenario,
                    codec,
                    target_train_batch_groups=target_train_batch_groups,
                    max_policy_lag=max_policy_lag,
                    active_actor_count=active_actor_count,
                    rollout_admission_delay_ms=rollout_admission_delay_ms,
                    action_space_key=action_space_key,
                )
                > self._score_arm(
                    most_undercovered[0],
                    most_undercovered[1],
                    most_undercovered[2],
                    target_train_batch_groups=target_train_batch_groups,
                    max_policy_lag=max_policy_lag,
                    active_actor_count=active_actor_count,
                    rollout_admission_delay_ms=rollout_admission_delay_ms,
                    action_space_key=action_space_key,
                )
            ):
                most_undercovered = candidate
        return most_undercovered, cost_limited

    def _effective_coverage_target(self, arm_count: int) -> float:
        if arm_count <= 0:
            return 0.0
        return min(self.min_rollout_coverage_fraction, 1.0 / arm_count)

    def _arm_decision_share(self, arm_id: str) -> float:
        if self._total_decisions <= 0:
            return 0.0
        stats = self._arms.get(arm_id)
        if stats is None:
            return 0.0
        return stats.decisions / self._total_decisions

    def _arm_sample_dollar_share(
        self,
        arm_id: str,
        arm_ids: Sequence[str] | None = None,
    ) -> float:
        selected_ids = list(arm_ids) if arm_ids is not None else list(self._arms)
        total = sum(
            max(0.0, self._arms.get(candidate, ArmStats()).total_dollar_seconds)
            for candidate in selected_ids
        )
        if total <= 0.0:
            return 0.0
        stats = self._arms.get(arm_id)
        if stats is None:
            return 0.0
        return max(0.0, stats.total_dollar_seconds) / total

    def _coverage_cost_limited(self, cost_share: float) -> bool:
        if self.max_rollout_coverage_cost_fraction is None:
            return False
        return cost_share >= self.max_rollout_coverage_cost_fraction

    def _score_arm(
        self,
        arm_id: str,
        scenario: Scenario | None = None,
        codec: ActionCodec | None = None,
        *,
        target_train_batch_groups: int | None = None,
        max_policy_lag: int | None = None,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> float:
        stats = self._arms.setdefault(arm_id, ArmStats())
        if stats.pulls == 0:
            return (
                _UNOBSERVED_ARM_SCORE
                - stats.inflight * _UNOBSERVED_ARM_INFLIGHT_PENALTY
                - self._unobserved_rollout_cost_penalty(
                    arm_id,
                    scenario,
                    codec,
                )
            )
        exploitation = self._arm_value(stats)
        exploration = self._exploration_value(stats)
        return exploitation + exploration + self._joint_action_score(
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )

    def _joint_action_score(
        self,
        *,
        arm_id: str,
        target_train_batch_groups: int | None,
        max_policy_lag: int | None,
        active_actor_count: int | None,
        rollout_admission_delay_ms: int | None,
        action_space_key: str | None = None,
    ) -> float:
        if self.joint_action_objective_weight <= 0.0:
            return 0.0
        key = self._candidate_joint_action_key(
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            rollout_admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )
        if key is None:
            return 0.0
        stats = self._joint_action_controls.get(key)
        if stats is None or _control_feedback_updates(stats) <= 0:
            return 0.0
        return self.joint_action_objective_weight * self._score_control_value(
            self._joint_action_controls,
            key,
        )

    def _joint_action_priority_score(self, trajectory: Trajectory) -> float:
        if self.joint_action_objective_weight <= 0.0:
            return 0.0
        key = _joint_action_key_from_metadata(trajectory.metadata)
        if key is None:
            return 0.0
        stats = self._joint_action_controls.get(key)
        if stats is None or _control_feedback_updates(stats) <= 0:
            return 0.0
        return self.joint_action_objective_weight * self._score_control_value(
            self._joint_action_controls,
            key,
        )

    def _train_selection_priority_score(
        self,
        groups: Sequence[TrajectoryGroup],
    ) -> float:
        if self.train_selection_objective_weight <= 0.0 or not groups:
            return 0.0
        key = _train_selection_key(groups)
        stats = self._train_selection_controls.get(key)
        if stats is None or _control_feedback_updates(stats) <= 0:
            return 0.0
        return self.train_selection_objective_weight * self._score_control_value(
            self._train_selection_controls,
            key,
        )

    @staticmethod
    def _candidate_joint_action_key(
        *,
        arm_id: str,
        target_train_batch_groups: int | None,
        max_policy_lag: int | None,
        active_actor_count: int | None,
        rollout_admission_delay_ms: int | None,
        action_space_key: str | None = None,
    ) -> str | None:
        if (
            target_train_batch_groups is None
            or max_policy_lag is None
            or active_actor_count is None
            or rollout_admission_delay_ms is None
        ):
            return None
        return scheduling_action_key(
            arm_id=arm_id,
            target_train_batch_groups=target_train_batch_groups,
            max_policy_lag=max_policy_lag,
            active_actor_count=active_actor_count,
            admission_delay_ms=rollout_admission_delay_ms,
            action_space_key=action_space_key,
        )

    def _joint_action_scores_for_control_values(
        self,
        control_name: str,
        candidates: Sequence[int],
        *,
        target_train_batch_groups: int | None = None,
        max_policy_lag: int | None = None,
        active_actor_count: int | None = None,
        rollout_admission_delay_ms: int | None = None,
        action_space_key: str | None = None,
    ) -> dict[int, float]:
        if self.joint_action_objective_weight <= 0.0 or not candidates:
            return {}
        candidate_set = set(int(value) for value in candidates)
        action_space_key = _normalize_key_component(action_space_key)
        scores: dict[int, float] = {}
        fallback_scores: dict[int, float] = {}
        for key, stats in self._joint_action_controls.items():
            if _control_feedback_updates(stats) <= 0:
                continue
            parts = _joint_action_key_parts(key)
            fields = _joint_action_control_fields(key)
            if fields is None:
                continue
            candidate = fields.get(control_name)
            if candidate not in candidate_set:
                continue
            if (
                target_train_batch_groups is not None
                and fields.get("cadence") != target_train_batch_groups
            ):
                continue
            if (
                max_policy_lag is not None
                and fields.get("lag") != max_policy_lag
            ):
                continue
            if (
                active_actor_count is not None
                and fields.get("actors") != active_actor_count
            ):
                continue
            if (
                rollout_admission_delay_ms is not None
                and fields.get("admission_ms") != rollout_admission_delay_ms
            ):
                continue
            score = self.joint_action_objective_weight * self._score_control_value(
                self._joint_action_controls,
                key,
            )
            if self.control_exploration_bonus > 0.0:
                score = max(
                    -self.control_exploration_bonus,
                    min(self.control_exploration_bonus, score),
                )
            target_scores = scores
            if action_space_key is not None:
                historical_action_space_key = _normalize_key_component(
                    parts.get("action_space")
                )
                if historical_action_space_key != action_space_key:
                    target_scores = fallback_scores
            current = target_scores.get(candidate)
            if current is None or score > current:
                target_scores[candidate] = score
        return scores or fallback_scores

    def _estimated_rollout_dollar_seconds(
        self,
        arm_id: str,
        scenario: Scenario | None = None,
        codec: ActionCodec | None = None,
    ) -> float:
        stats = self._arms.get(arm_id)
        if stats is not None and stats.pulls > 0:
            if stats.dollar_seconds_ema > 0.0:
                return stats.dollar_seconds_ema
            return stats.total_dollar_seconds / stats.pulls

        scenario_id = (
            str(scenario.id) if scenario is not None else _arm_scenario_id(arm_id)
        )
        codec_key = (
            _codec_key(codec) if codec is not None else _arm_codec_key(arm_id)
        )
        estimates: list[float] = []

        scenario_cost = self._mean_observed_sample_cost(scenario_id=scenario_id)
        if scenario_cost > 0.0:
            estimates.append(scenario_cost)
        codec_cost = self._mean_observed_sample_cost(codec_key=codec_key)
        if codec_cost > 0.0:
            estimates.append(codec_cost)
        if not estimates:
            global_cost = self._mean_observed_sample_cost()
            if global_cost > 0.0:
                estimates.append(global_cost)
        if not estimates:
            return 0.0
        return sum(estimates) / len(estimates)

    def _unobserved_rollout_cost_penalty(
        self,
        arm_id: str,
        scenario: Scenario | None = None,
        codec: ActionCodec | None = None,
    ) -> float:
        stats = self._arms.get(arm_id)
        if stats is not None and stats.pulls > 0:
            return 0.0
        estimated_cost = self._estimated_rollout_dollar_seconds(
            arm_id,
            scenario,
            codec,
        )
        if estimated_cost <= 0.0:
            return 0.0
        baseline_cost = self._mean_observed_sample_cost()
        if baseline_cost <= 0.0:
            baseline_cost = estimated_cost
        return min(
            _UNOBSERVED_ARM_COST_PENALTY_CAP,
            estimated_cost / max(baseline_cost, 1e-12),
        )

    def _mean_observed_sample_cost(
        self,
        *,
        scenario_id: str | None = None,
        codec_key: str | None = None,
    ) -> float:
        total_cost = 0.0
        total_pulls = 0
        for arm_id, stats in self._arms.items():
            if stats.pulls <= 0:
                continue
            arm_scenario_id, arm_codec_key = _split_arm_id(arm_id)
            if scenario_id is not None and arm_scenario_id != scenario_id:
                continue
            if codec_key is not None and arm_codec_key != codec_key:
                continue
            total_cost += max(0.0, stats.total_dollar_seconds)
            total_pulls += stats.pulls
        if total_pulls <= 0:
            return 0.0
        return total_cost / total_pulls

    def _exploration_value(self, stats: ArmStats) -> float:
        if stats.pulls == 0:
            return 0.0
        effective_pulls = stats.pulls + stats.inflight
        return self.exploration_bonus * math.sqrt(
            math.log(self._total_pulls + self._total_inflight_rollouts() + 1)
            / effective_pulls
        )

    def _has_unaccepted_known_arm(self) -> bool:
        return any(stats.accepted == 0 for stats in self._arms.values())

    def _has_positive_objective_signal(self) -> bool:
        return (
            self._global_objective_ema > self.objective_threshold
            or self._train_objective_ema > self.objective_threshold
        )

    def _arm_value(self, stats: ArmStats) -> float:
        return self._raw_arm_value(stats) - self._confidence_penalty(stats)

    def _raw_arm_value(self, stats: ArmStats) -> float:
        unsafe_rate = stats.unsafe / stats.pulls if stats.pulls else 0.0
        objective = (
            self.train_objective_weight * stats.policy_improvement_objective_ema
            + self.rollout_objective_weight * stats.marginal_objective_ema
            + self.reward_efficiency_weight * stats.reward_efficiency_ema
        )
        return max(0.0, stats.action_quality_ema) * (
            objective - self.unsafe_penalty * unsafe_rate
        )

    def _confidence_penalty(self, stats: ArmStats) -> float:
        if self.confidence_penalty_weight <= 0.0:
            return 0.0
        observations = stats.objective_observations
        if observations <= 0:
            return 0.0
        if observations == 1:
            uncertainty = abs(stats.objective_mean)
        else:
            variance = max(0.0, stats.objective_m2 / (observations - 1))
            uncertainty = math.sqrt(variance)
        return (
            self.confidence_penalty_weight
            * uncertainty
            / math.sqrt(observations)
        )

    @staticmethod
    def _observe_objective_sample(stats: ArmStats, value: float) -> None:
        if not math.isfinite(value):
            return
        stats.objective_observations += 1
        delta = value - stats.objective_mean
        stats.objective_mean += delta / stats.objective_observations
        delta2 = value - stats.objective_mean
        stats.objective_m2 += delta * delta2

    def _normalized_reward_improvement(
        self,
        improvement: float,
        reward_scale: float,
    ) -> float:
        if self.reward_scale_normalization == "none":
            return improvement
        return improvement / max(reward_scale, 1e-12)

    @staticmethod
    def _arm_effective_reward_scale(stats: ArmStats, reward: float) -> float:
        if stats.pulls <= 0:
            return max(1.0, abs(reward))
        low = min(stats.min_effective_reward, reward)
        high = max(stats.max_effective_reward, reward)
        return max(1.0, high - low)

    @staticmethod
    def _observe_arm_effective_reward(stats: ArmStats, reward: float) -> None:
        if stats.pulls <= 1:
            stats.min_effective_reward = min(0.0, reward)
            stats.max_effective_reward = max(0.0, reward)
            return
        stats.min_effective_reward = min(stats.min_effective_reward, reward)
        stats.max_effective_reward = max(stats.max_effective_reward, reward)

    @staticmethod
    def _arm_train_reward_scale(stats: ArmStats, reward: float) -> float:
        if stats.train_reward_observations <= 0:
            return max(1.0, abs(reward))
        low = min(stats.min_train_reward, reward)
        high = max(stats.max_train_reward, reward)
        return max(1.0, high - low)

    @staticmethod
    def _observe_arm_train_reward(stats: ArmStats, reward: float) -> None:
        stats.train_reward_observations += 1
        if stats.train_reward_observations == 1:
            stats.min_train_reward = min(0.0, reward)
            stats.max_train_reward = max(0.0, reward)
            return
        stats.min_train_reward = min(stats.min_train_reward, reward)
        stats.max_train_reward = max(stats.max_train_reward, reward)

    def _estimate_stale_lost_reward_improving_experience(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        stale_cost: float,
        stale_experience: float,
    ) -> float:
        lost_experience = 0.0
        explicit_cost = _groups_sample_dollar_seconds(groups)
        extra_cost = max(0.0, stale_cost - explicit_cost)
        quality_total = sum(
            max(0.0, action_quality(trajectory))
            for group in groups
            for trajectory in group.trajectories
        )
        for group in groups:
            for trajectory in group.trajectories:
                quality = action_quality(trajectory)
                if quality <= 0.0:
                    continue
                arm_id = str(
                    trajectory.metadata.get("scheduler/arm_id", "unassigned")
                )
                stats = self._arms.get(arm_id)
                if stats is None:
                    continue
                objective = max(0.0, self._arm_value(stats))
                if objective <= 0.0:
                    continue
                trajectory_cost = _trajectory_sample_dollar_seconds(trajectory)
                if trajectory_cost <= 0.0:
                    if explicit_cost > 0.0 or stale_experience <= 0.0:
                        continue
                    trajectory_cost = stale_cost * quality / stale_experience
                elif extra_cost > 0.0 and quality_total > 0.0:
                    trajectory_cost += extra_cost * quality / quality_total
                lost_experience += objective * trajectory_cost
        return lost_experience

    def _credit_objective_to_arms(
        self,
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float = 0.0,
    ) -> None:
        arm_weights: dict[str, float] = {}
        arm_trajectories: dict[str, int] = {}
        for group in groups:
            for trajectory in group.trajectories:
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                quality = action_quality(trajectory)
                if quality <= 0.0:
                    weight = 0.0
                else:
                    weight = max(0.0, trajectory.reward * quality) or quality
                arm_weights[arm_id] = arm_weights.get(arm_id, 0.0) + weight
                arm_trajectories[arm_id] = arm_trajectories.get(arm_id, 0) + 1

        total_weight = sum(arm_weights.values())
        for arm_id, weight in arm_weights.items():
            stats = self._arms.setdefault(arm_id, ArmStats())
            credit = objective * weight / total_weight if total_weight > 0.0 else 0.0
            stale_credit = (
                stale_experience * weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if stale_feedback:
                stats.stale_updates += 1
                stats.stale_batches += 1
                stats.stale_trajectories += arm_trajectories.get(arm_id, 0)
                stats.stale_experience += stale_credit
                stats.total_stale_penalty_objective += credit
            else:
                stats.train_updates += 1
            stats.total_policy_improvement_objective += credit
            stats.policy_improvement_objective_ema = self._ema(
                stats.policy_improvement_objective_ema,
                credit,
                _arm_feedback_updates(stats),
            )
            self._observe_objective_sample(stats, credit)

    def _credit_train_objective_to_arms(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        reward: float,
        cost: float,
    ) -> tuple[float, float, float, dict[str, float], dict[str, float], float]:
        arm_weights = _arm_credit_weights(groups)
        arm_experience = _arm_useful_experience(groups)
        total_experience = sum(arm_experience.values())
        total_reward_improving_experience = 0.0
        total_control_reward_improving_experience = 0.0
        arm_objectives: dict[str, float] = {}

        for arm_id in arm_weights:
            stats = self._arms.setdefault(arm_id, ArmStats())
            experience = arm_experience.get(arm_id, 0.0)
            improvement = (
                max(0.0, reward - stats.last_train_reward)
                if experience > 0.0
                else 0.0
            )
            reward_scale = (
                self._arm_train_reward_scale(stats, reward)
                if experience > 0.0
                else 1.0
            )
            normalized_improvement = self._normalized_reward_improvement(
                improvement,
                reward_scale,
            )
            reward_improving_experience = improvement * experience
            control_reward_improving_experience = (
                normalized_improvement * experience
            )
            credit = control_reward_improving_experience / cost

            stats.train_updates += 1
            if experience > 0.0:
                self._observe_arm_train_reward(stats, reward)
                stats.last_train_reward = reward
                stats.last_train_reward_scale = reward_scale
            stats.last_train_reward_improvement = improvement
            stats.last_normalized_train_reward_improvement = (
                normalized_improvement
            )
            stats.total_reward_improving_experience += (
                reward_improving_experience
            )
            stats.total_normalized_reward_improving_experience += (
                control_reward_improving_experience
            )
            stats.total_policy_improvement_objective += credit
            stats.policy_improvement_objective_ema = self._ema(
                stats.policy_improvement_objective_ema,
                credit,
                _arm_feedback_updates(stats),
            )
            self._observe_objective_sample(stats, credit)
            arm_objectives[arm_id] = credit
            total_reward_improving_experience += reward_improving_experience
            total_control_reward_improving_experience += (
                control_reward_improving_experience
            )

        objective = total_control_reward_improving_experience / cost
        return (
            objective,
            total_experience,
            total_reward_improving_experience,
            arm_objectives,
            arm_weights,
            total_control_reward_improving_experience,
        )

    def _credit_train_objective_to_controls(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        objective: float | None,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
    ) -> None:
        if objective is not None:
            train_objective = sum(max(0.0, value) for value in arm_objectives.values())
            scale = objective / train_objective if train_objective > 0.0 else 0.0
            arm_objectives = {
                arm_id: max(0.0, value) * scale
                for arm_id, value in arm_objectives.items()
            }
        self._credit_train_objective_to_control_family(
            self._cadence_controls,
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
            keys=(
                "scheduler/active_target_train_batch_groups",
                "scheduler/target_train_batch_groups",
            ),
        )
        self._credit_train_objective_to_control_family(
            self._lag_controls,
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
            keys=(
                "scheduler/active_max_policy_lag",
                "scheduler/max_policy_lag",
            ),
        )
        self._credit_train_objective_to_control_family(
            self._admission_controls,
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
            keys=(
                "scheduler/active_rollout_admission_delay_ms",
                "scheduler/rollout_admission_delay_ms",
            ),
        )
        self._credit_train_objective_to_control_family(
            self._actor_count_controls,
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
            keys=(
                "scheduler/active_actor_count",
                "scheduler/actor_count",
            ),
        )
        self._credit_train_objective_to_joint_actions(
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
        )
        self._credit_train_objective_to_coverage_controls(
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
        )
        self._credit_train_objective_to_timing_responses(
            groups,
            arm_objectives=arm_objectives,
            arm_weights=arm_weights,
        )

    def _credit_train_objective_to_control_family(
        self,
        controls: dict[int, ControlStats],
        groups: Sequence[TrajectoryGroup],
        *,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
        keys: Sequence[str],
    ) -> None:
        value_credit: dict[int, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                value = _first_int_metadata(trajectory.metadata, keys)
                if value is None:
                    continue
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                total_arm_weight = arm_weights.get(arm_id, 0.0)
                if total_arm_weight <= 0.0:
                    continue
                trajectory_weight = _trajectory_credit_weight(trajectory)
                if trajectory_weight <= 0.0:
                    continue
                credit = (
                    arm_objectives.get(arm_id, 0.0)
                    * trajectory_weight
                    / total_arm_weight
                )
                value_credit[value] = value_credit.get(value, 0.0) + credit

        for value, credit in value_credit.items():
            stats = controls.setdefault(value, ControlStats())
            stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_train_objective_to_joint_actions(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
    ) -> None:
        key_credit: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                key = _joint_action_key_from_metadata(trajectory.metadata)
                if key is None:
                    continue
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                total_arm_weight = arm_weights.get(arm_id, 0.0)
                if total_arm_weight <= 0.0:
                    continue
                trajectory_weight = _trajectory_credit_weight(trajectory)
                if trajectory_weight <= 0.0:
                    continue
                credit = (
                    arm_objectives.get(arm_id, 0.0)
                    * trajectory_weight
                    / total_arm_weight
                )
                key_credit[key] = key_credit.get(key, 0.0) + credit

        for key, credit in key_credit.items():
            stats = self._joint_action_controls.setdefault(key, ControlStats())
            stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_train_objective_to_train_selection(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
    ) -> None:
        key_credit: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                key = _train_selection_key_from_metadata(trajectory.metadata)
                if key is None:
                    continue
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                total_arm_weight = arm_weights.get(arm_id, 0.0)
                if total_arm_weight <= 0.0:
                    continue
                trajectory_weight = _trajectory_credit_weight(trajectory)
                if trajectory_weight <= 0.0:
                    continue
                credit = (
                    arm_objectives.get(arm_id, 0.0)
                    * trajectory_weight
                    / total_arm_weight
                )
                key_credit[key] = key_credit.get(key, 0.0) + credit

        for key, credit in key_credit.items():
            stats = self._train_selection_controls.setdefault(key, ControlStats())
            stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_train_objective_to_coverage_controls(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
    ) -> None:
        key_credit: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                key = _coverage_control_key_from_metadata(trajectory.metadata)
                if key is None:
                    continue
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                total_arm_weight = arm_weights.get(arm_id, 0.0)
                if total_arm_weight <= 0.0:
                    continue
                trajectory_weight = _trajectory_credit_weight(trajectory)
                if trajectory_weight <= 0.0:
                    continue
                credit = (
                    arm_objectives.get(arm_id, 0.0)
                    * trajectory_weight
                    / total_arm_weight
                )
                key_credit[key] = key_credit.get(key, 0.0) + credit

        for key, credit in key_credit.items():
            stats = self._coverage_controls.setdefault(key, ControlStats())
            stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_train_objective_to_timing_responses(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
    ) -> None:
        key_credit: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                keys = _timing_response_keys_from_metadata(trajectory.metadata)
                if not keys:
                    continue
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                total_arm_weight = arm_weights.get(arm_id, 0.0)
                if total_arm_weight <= 0.0:
                    continue
                trajectory_weight = _trajectory_credit_weight(trajectory)
                if trajectory_weight <= 0.0:
                    continue
                credit = (
                    arm_objectives.get(arm_id, 0.0)
                    * trajectory_weight
                    / total_arm_weight
                )
                for key in keys:
                    key_credit[key] = key_credit.get(key, 0.0) + credit

        for key, credit in key_credit.items():
            stats = self._timing_response_controls.setdefault(key, ControlStats())
            stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _record_continuation_decision(
        self,
        *,
        policy_step: int,
        continue_training: bool,
        reason: str,
        pending_train_batches: int,
        train_queue_pressure: float,
    ) -> bool:
        key = _continuation_decision_key(
            continue_training=continue_training,
            reason=reason,
            pending_train_batches=pending_train_batches,
            train_queue_pressure=train_queue_pressure,
        )
        self._last_continuation_decision_continue = continue_training
        self._last_continuation_decision_key = key
        self._last_continuation_decision_reason = reason
        self._last_continuation_pending_train_batches = max(
            0,
            int(pending_train_batches),
        )
        pressure = (
            float(train_queue_pressure)
            if math.isfinite(float(train_queue_pressure))
            else 0.0
        )
        self._last_continuation_train_queue_pressure = max(0.0, pressure)

        if continue_training:
            if policy_step not in self._pending_continuation_decisions:
                stats = self._continuation_controls.setdefault(key, ControlStats())
                stats.decisions += 1
                self._pending_continuation_decisions[policy_step] = key
        else:
            marker = (policy_step, key)
            if marker not in self._recorded_continuation_stop_decisions:
                stats = self._continuation_controls.setdefault(key, ControlStats())
                stats.decisions += 1
                self._recorded_continuation_stop_decisions.add(marker)
        return continue_training

    def _credit_train_objective_to_continuation(
        self,
        *,
        policy_step: int,
        objective: float,
    ) -> None:
        key = self._pending_continuation_decisions.pop(policy_step, None)
        if key is None:
            return
        stats = self._continuation_controls.setdefault(key, ControlStats())
        stats.train_updates += 1
        stats.total_objective += objective
        stats.objective_ema = self._ema(
            stats.objective_ema,
            objective,
            _control_feedback_updates(stats),
        )

    def _credit_train_objective_to_actors(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        arm_objectives: Mapping[str, float],
        arm_weights: Mapping[str, float],
    ) -> None:
        actor_credit: dict[int, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                actor_id = _trajectory_actor_id(trajectory)
                if actor_id is None:
                    continue
                arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
                total_arm_weight = arm_weights.get(arm_id, 0.0)
                if total_arm_weight <= 0.0:
                    continue
                trajectory_weight = _trajectory_credit_weight(trajectory)
                if trajectory_weight <= 0.0:
                    continue
                credit = (
                    arm_objectives.get(arm_id, 0.0)
                    * trajectory_weight
                    / total_arm_weight
                )
                actor_credit[actor_id] = actor_credit.get(actor_id, 0.0) + credit

        for actor_id, credit in actor_credit.items():
            stats = self._actors.setdefault(actor_id, ActorStats())
            stats.train_updates += 1
            stats.total_objective += credit
            stats.total_train_objective += credit
            stats.train_objective_ema = self._ema(
                stats.train_objective_ema,
                credit,
                stats.train_updates,
            )
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _actor_feedback_updates(stats),
            )

    def _credit_objective_to_controls(
        self,
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float = 0.0,
    ) -> None:
        self._credit_objective_to_control_family(
            self._cadence_controls,
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
            keys=(
                "scheduler/active_target_train_batch_groups",
                "scheduler/target_train_batch_groups",
            ),
        )
        self._credit_objective_to_control_family(
            self._lag_controls,
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
            keys=(
                "scheduler/active_max_policy_lag",
                "scheduler/max_policy_lag",
            ),
        )
        self._credit_objective_to_control_family(
            self._admission_controls,
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
            keys=(
                "scheduler/active_rollout_admission_delay_ms",
                "scheduler/rollout_admission_delay_ms",
            ),
        )
        self._credit_objective_to_control_family(
            self._actor_count_controls,
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
            keys=(
                "scheduler/active_actor_count",
                "scheduler/actor_count",
            ),
        )
        self._credit_objective_to_joint_actions(
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
        )
        self._credit_objective_to_coverage_controls(
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
        )
        self._credit_objective_to_timing_responses(
            groups,
            objective,
            stale_feedback=stale_feedback,
            stale_experience=stale_experience,
        )

    def _credit_objective_to_joint_actions(
        self,
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float,
    ) -> None:
        key_weights: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                key = _joint_action_key_from_metadata(trajectory.metadata)
                if key is None:
                    continue
                quality = action_quality(trajectory)
                if quality <= 0.0:
                    weight = 0.0
                else:
                    weight = max(0.0, trajectory.reward * quality) or quality
                key_weights[key] = key_weights.get(key, 0.0) + weight

        total_weight = sum(key_weights.values())
        for key, weight in key_weights.items():
            stats = self._joint_action_controls.setdefault(key, ControlStats())
            credit = objective * weight / total_weight if total_weight > 0.0 else 0.0
            stale_credit = (
                stale_experience * weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if stale_feedback:
                stats.stale_updates += 1
                stats.stale_experience += stale_credit
                stats.total_stale_penalty_objective += credit
            else:
                stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_objective_to_coverage_controls(
        self,
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float,
    ) -> None:
        key_weights: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                key = _coverage_control_key_from_metadata(trajectory.metadata)
                if key is None:
                    continue
                quality = action_quality(trajectory)
                if quality <= 0.0:
                    weight = 0.0
                else:
                    weight = max(0.0, trajectory.reward * quality) or quality
                key_weights[key] = key_weights.get(key, 0.0) + weight

        total_weight = sum(key_weights.values())
        for key, weight in key_weights.items():
            stats = self._coverage_controls.setdefault(key, ControlStats())
            credit = objective * weight / total_weight if total_weight > 0.0 else 0.0
            stale_credit = (
                stale_experience * weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if stale_feedback:
                stats.stale_updates += 1
                stats.stale_experience += stale_credit
                stats.total_stale_penalty_objective += credit
            else:
                stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_objective_to_timing_responses(
        self,
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float,
    ) -> None:
        key_weights: dict[str, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                keys = _timing_response_keys_from_metadata(trajectory.metadata)
                if not keys:
                    continue
                quality = action_quality(trajectory)
                if quality <= 0.0:
                    weight = 0.0
                else:
                    weight = max(0.0, trajectory.reward * quality) or quality
                for key in keys:
                    key_weights[key] = key_weights.get(key, 0.0) + weight

        total_weight = sum(key_weights.values())
        for key, weight in key_weights.items():
            stats = self._timing_response_controls.setdefault(key, ControlStats())
            credit = objective * weight / total_weight if total_weight > 0.0 else 0.0
            stale_credit = (
                stale_experience * weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if stale_feedback:
                stats.stale_updates += 1
                stats.stale_experience += stale_credit
                stats.total_stale_penalty_objective += credit
            else:
                stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_rollout_objective_to_admission_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        self._credit_rollout_objective_to_control_family(
            self._admission_controls,
            trajectory,
            objective,
            keys=(
                "scheduler/active_rollout_admission_delay_ms",
                "scheduler/rollout_admission_delay_ms",
            ),
        )

    def _credit_rollout_objective_to_actor_count_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        self._credit_rollout_objective_to_control_family(
            self._actor_count_controls,
            trajectory,
            objective,
            keys=(
                "scheduler/active_actor_count",
                "scheduler/actor_count",
            ),
        )

    def _credit_rollout_objective_to_joint_action(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        key = _joint_action_key_from_metadata(trajectory.metadata)
        if key is None:
            return
        stats = self._joint_action_controls.setdefault(key, ControlStats())
        if stats.decisions <= stats.rollout_updates:
            stats.decisions += 1
        stats.rollout_updates += 1
        stats.total_objective += objective
        stats.objective_ema = self._ema(
            stats.objective_ema,
            objective,
            _control_feedback_updates(stats),
        )

    def _credit_rollout_objective_to_coverage_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        key = _coverage_control_key_from_metadata(trajectory.metadata)
        if key is None:
            return
        stats = self._coverage_controls.setdefault(key, ControlStats())
        if stats.decisions <= stats.rollout_updates:
            stats.decisions += 1
        stats.rollout_updates += 1
        stats.total_objective += objective
        stats.objective_ema = self._ema(
            stats.objective_ema,
            objective,
            _control_feedback_updates(stats),
        )

    def _credit_rollout_objective_to_timing_responses(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        for key in _timing_response_keys_from_metadata(trajectory.metadata):
            stats = self._timing_response_controls.setdefault(key, ControlStats())
            if stats.decisions <= stats.rollout_updates:
                stats.decisions += 1
            stats.rollout_updates += 1
            stats.total_objective += objective
            stats.objective_ema = self._ema(
                stats.objective_ema,
                objective,
                _control_feedback_updates(stats),
            )

    def _credit_rollout_objective_to_cadence_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        self._credit_rollout_objective_to_control_family(
            self._cadence_controls,
            trajectory,
            objective,
            keys=(
                "scheduler/active_target_train_batch_groups",
                "scheduler/target_train_batch_groups",
            ),
        )

    def _credit_rollout_objective_to_lag_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        self._credit_rollout_objective_to_control_family(
            self._lag_controls,
            trajectory,
            objective,
            keys=(
                "scheduler/active_max_policy_lag",
                "scheduler/max_policy_lag",
            ),
        )

    def _credit_rollout_objective_to_control_family(
        self,
        controls: dict[int, ControlStats],
        trajectory: Trajectory,
        objective: float,
        *,
        keys: Sequence[str],
    ) -> None:
        value = _first_int_metadata(
            trajectory.metadata,
            keys,
        )
        if value is None:
            return
        stats = controls.setdefault(value, ControlStats())
        stats.rollout_updates += 1
        stats.total_objective += objective
        stats.objective_ema = self._ema(
            stats.objective_ema,
            objective,
            _control_feedback_updates(stats),
        )

    def _credit_objective_to_control_family(
        self,
        controls: dict[int, ControlStats],
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float,
        keys: Sequence[str],
    ) -> None:
        value_weights: dict[int, float] = {}
        for group in groups:
            for trajectory in group.trajectories:
                value = _first_int_metadata(trajectory.metadata, keys)
                if value is None:
                    continue
                quality = action_quality(trajectory)
                if quality <= 0.0:
                    weight = 0.0
                else:
                    weight = max(0.0, trajectory.reward * quality) or quality
                value_weights[value] = value_weights.get(value, 0.0) + weight

        total_weight = sum(value_weights.values())
        for value, weight in value_weights.items():
            stats = controls.setdefault(value, ControlStats())
            credit = objective * weight / total_weight if total_weight > 0.0 else 0.0
            stale_credit = (
                stale_experience * weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if stale_feedback:
                stats.stale_updates += 1
                stats.stale_experience += stale_credit
                stats.total_stale_penalty_objective += credit
            else:
                stats.train_updates += 1
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _control_feedback_updates(stats),
            )

    def _credit_rollout_objective_to_actor(
        self,
        trajectory: Trajectory,
        objective: float,
        *,
        accepted: bool,
        rollout_cost: float,
        queue_wait_cost: float,
        admission_cost: float,
        quality: float,
        failure_modes: Sequence[str],
    ) -> None:
        actor_id = _trajectory_actor_id(trajectory)
        if actor_id is None:
            return
        stats = self._actors.setdefault(actor_id, ActorStats())
        if stats.inflight > 0:
            stats.inflight -= 1
        stats.pulls += 1
        if accepted:
            stats.accepted += 1
        if quality <= 0.0:
            stats.unsafe += 1
        if failure_modes:
            stats.failed_rollouts += 1
        stats.total_objective += objective
        stats.total_rollout_objective += objective
        stats.total_dollar_seconds += rollout_cost + queue_wait_cost + admission_cost
        stats.rollout_dollar_seconds += rollout_cost
        stats.queue_wait_dollar_seconds += queue_wait_cost
        stats.admission_dollar_seconds += admission_cost
        stats.action_units += trajectory.action_units
        stats.source_tokens += trajectory.token_count
        stats.rollout_objective_ema = self._ema(
            stats.rollout_objective_ema,
            objective,
            stats.pulls,
        )
        stats.objective_ema = self._ema(
            stats.objective_ema,
            objective,
            _actor_feedback_updates(stats),
        )

    def _credit_objective_to_actors(
        self,
        groups: Sequence[TrajectoryGroup],
        objective: float,
        *,
        stale_feedback: bool,
        stale_experience: float,
    ) -> None:
        actor_weights: dict[int, float] = {}
        actor_trajectories: dict[int, int] = {}
        for group in groups:
            for trajectory in group.trajectories:
                actor_id = _trajectory_actor_id(trajectory)
                if actor_id is None:
                    continue
                weight = _trajectory_credit_weight(trajectory)
                actor_weights[actor_id] = actor_weights.get(actor_id, 0.0) + weight
                actor_trajectories[actor_id] = (
                    actor_trajectories.get(actor_id, 0) + 1
                )

        total_weight = sum(actor_weights.values())
        for actor_id, weight in actor_weights.items():
            stats = self._actors.setdefault(actor_id, ActorStats())
            credit = objective * weight / total_weight if total_weight > 0.0 else 0.0
            stale_credit = (
                stale_experience * weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if stale_feedback:
                stats.stale_updates += 1
                stats.stale_batches += 1
                stats.stale_trajectories += actor_trajectories.get(actor_id, 0)
                stats.stale_experience += stale_credit
                stats.total_stale_penalty_objective += credit
            else:
                stats.train_updates += 1
                stats.total_train_objective += credit
                stats.train_objective_ema = self._ema(
                    stats.train_objective_ema,
                    credit,
                    stats.train_updates,
                )
            stats.total_objective += credit
            stats.objective_ema = self._ema(
                stats.objective_ema,
                credit,
                _actor_feedback_updates(stats),
            )

    def _control_candidates(
        self,
        *,
        min_value: int,
        configured: int,
        upper: int,
    ) -> tuple[int, ...]:
        values = {
            min_value,
            min(max(configured, min_value), upper),
            upper,
        }
        span = upper - min_value + 1
        if span <= self.max_control_candidate_values:
            values.update(range(min_value, upper + 1))
        elif self.max_control_candidate_values > 1:
            for index in range(self.max_control_candidate_values):
                fraction = index / (self.max_control_candidate_values - 1)
                values.add(round(min_value + (upper - min_value) * fraction))
        return tuple(sorted(value for value in values if min_value <= value <= upper))

    @staticmethod
    def _control_family_has_feedback(
        controls: Mapping[int, ControlStats],
        candidates: Sequence[int],
    ) -> bool:
        return any(
            _control_feedback_updates(controls[value]) > 0
            for value in candidates
            if value in controls
        )

    def _select_control_value(
        self,
        controls: dict[int, ControlStats],
        candidates: Sequence[int],
        *,
        preferred: int,
        joint_scores: Mapping[int, float] | None = None,
    ) -> int:
        if not candidates:
            return preferred
        candidate_values = tuple(candidates)
        if preferred not in candidate_values:
            preferred = candidate_values[0]
        normalized_joint_scores = {
            int(value): float(score)
            for value, score in (joint_scores or {}).items()
            if value in candidate_values and math.isfinite(float(score))
        }
        if all(
            controls.get(value) is None
            or (
                controls[value].decisions == 0
                and _control_feedback_updates(controls[value]) == 0
            )
            for value in candidate_values
        ) and not normalized_joint_scores:
            return preferred
        return max(
            candidate_values,
            key=lambda value: (
                self._score_control_value(controls, value)
                + normalized_joint_scores.get(value, 0.0),
                1 if value == preferred else 0,
                -abs(value - preferred),
                -value,
            ),
        )

    def _score_control_value(
        self,
        controls: Mapping[Any, ControlStats],
        value: Any,
    ) -> float:
        stats = controls.get(value)
        if stats is None:
            return 2.0 * self.control_exploration_bonus
        feedback_updates = _control_feedback_updates(stats)
        if feedback_updates <= 0:
            return self.control_exploration_bonus / math.sqrt(stats.decisions + 1)
        return stats.objective_ema + self._control_exploration_value(
            controls,
            stats,
        )

    def _control_exploration_value(
        self,
        controls: Mapping[Any, ControlStats],
        stats: ControlStats,
    ) -> float:
        if self.control_exploration_bonus <= 0.0:
            return 0.0
        feedback_updates = _control_feedback_updates(stats)
        if feedback_updates <= 0:
            return self.control_exploration_bonus / math.sqrt(stats.decisions + 1)
        total_updates = sum(
            _control_feedback_updates(candidate)
            for candidate in controls.values()
        )
        total_decisions = sum(candidate.decisions for candidate in controls.values())
        evidence = total_updates + total_decisions + 1
        return self.control_exploration_bonus * math.sqrt(
            math.log(evidence + 1) / feedback_updates
        )

    def _record_control_decision(
        self,
        controls: dict[int, ControlStats],
        value: int,
    ) -> int:
        controls.setdefault(value, ControlStats()).decisions += 1
        return value

    def _cancel_control_decision(
        self,
        controls: dict[Any, ControlStats],
        value: Any,
    ) -> None:
        stats = controls.get(value)
        if stats is None:
            return
        stats.decisions = max(0, stats.decisions - 1)
        if (
            stats.decisions == 0
            and _control_feedback_updates(stats) == 0
            and stats.total_objective == 0.0
            and stats.total_stale_penalty_objective == 0.0
            and stats.stale_experience == 0.0
        ):
            controls.pop(value, None)

    def _record_arm_decision(
        self,
        stats: ArmStats,
        *,
        reserved_rollout_dollar_seconds: float,
    ) -> None:
        stats.decisions += 1
        stats.inflight += 1
        stats.reserved_rollout_dollar_seconds += max(
            0.0,
            reserved_rollout_dollar_seconds,
        )
        self._total_decisions += 1

    def _record_joint_action_decision(self, key: str) -> None:
        self._joint_action_controls.setdefault(key, ControlStats()).decisions += 1

    def _record_coverage_decision(self, key: str) -> None:
        self._coverage_controls.setdefault(key, ControlStats()).decisions += 1

    def _record_timing_response_decision(
        self,
        *,
        knob: str,
        value: int,
        preference_reason: str,
        train_queue_pressure: float,
        pending_groups: int,
    ) -> str:
        key = _timing_response_key(
            knob=knob,
            value=value,
            preference_reason=preference_reason,
            train_queue_pressure=train_queue_pressure,
            pending_groups=pending_groups,
        )
        self._timing_response_controls.setdefault(key, ControlStats()).decisions += 1
        if knob == "cadence":
            self._last_cadence_response_key = key
            self._last_cadence_response_reason = preference_reason
        elif knob == "policy_lag":
            self._last_policy_lag_response_key = key
            self._last_policy_lag_response_reason = preference_reason
        return key

    def _total_inflight_rollouts(self) -> int:
        return sum(stats.inflight for stats in self._arms.values())

    def _reserved_inflight_rollout_dollar_seconds(self) -> float:
        return sum(
            max(0.0, stats.reserved_rollout_dollar_seconds)
            for stats in self._arms.values()
        )

    def _observe_accounted_train_objective(
        self,
        reward_improving_experience: float,
        *,
        raw_reward_improving_experience: float | None = None,
    ) -> float:
        accounted_dollar_seconds = self._accounted_dollar_seconds()
        interval_cost = max(
            0.0,
            accounted_dollar_seconds - self._previous_accounted_dollar_seconds,
        )
        self._previous_accounted_dollar_seconds = accounted_dollar_seconds
        objective = (
            reward_improving_experience / max(interval_cost, 1e-12)
            if interval_cost > 0.0
            else 0.0
        )
        self._last_accounted_objective = objective
        self._last_accounted_reward_improving_experience = (
            reward_improving_experience
            if raw_reward_improving_experience is None
            else raw_reward_improving_experience
        )
        self._last_accounted_control_reward_improving_experience = (
            reward_improving_experience
        )
        self._last_accounted_dollar_seconds = interval_cost
        self._accounted_objective_ema = self._ema(
            self._accounted_objective_ema,
            objective,
            self._train_observations,
        )
        return objective

    def _accounted_dollar_seconds(self) -> float:
        return (
            self._rollout_dollar_seconds
            + self._queue_wait_dollar_seconds
            + self._rollout_admission_dollar_seconds
            + self._train_dollar_seconds
            + self._stale_unobserved_sample_dollar_seconds
            + self._stale_additional_dollar_seconds
        )

    def _accounted_budget_exhausted(self) -> bool:
        if self.max_accounted_dollar_seconds is None:
            return False
        return (
            self._accounted_dollar_seconds()
            + self._reserved_inflight_rollout_dollar_seconds()
            >= self.max_accounted_dollar_seconds
        )

    def _ema(self, current: float, value: float, count: int) -> float:
        if count <= 1:
            return value
        return self.ema_alpha * value + (1 - self.ema_alpha) * current


def _arm_id(scenario: Scenario, codec: ActionCodec) -> str:
    return f"{scenario.id}|{_codec_key(codec)}"


def _split_arm_id(arm_id: str) -> tuple[str, str]:
    if "|" not in arm_id:
        return "", arm_id
    scenario_id, codec_key = arm_id.rsplit("|", 1)
    return scenario_id, codec_key


def _arm_scenario_id(arm_id: str) -> str:
    return _split_arm_id(arm_id)[0]


def _arm_codec_key(arm_id: str) -> str:
    return _split_arm_id(arm_id)[1]


def _codec_key(codec: ActionCodec) -> str:
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


def _safe_metric_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _actor_metric_key(actor_id: int) -> str:
    return _safe_metric_key(f"actor_{actor_id}")


def _train_selection_key(groups: Sequence[TrajectoryGroup]) -> str:
    trajectories = [
        trajectory
        for group in groups
        for trajectory in group.trajectories
    ]
    arm_ids = sorted(
        str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
        for trajectory in trajectories
    )
    joint_keys = sorted(
        key
        for key in (
            _joint_action_key_from_metadata(trajectory.metadata)
            for trajectory in trajectories
        )
        if key is not None
    )
    arm_component = "+".join(arm_ids) if arm_ids else "none"
    joint_component = "+".join(joint_keys) if joint_keys else "none"
    return (
        f"arms={arm_component}"
        f"|joints={joint_component}"
        f"|groups={len(groups)}"
        f"|trajectories={len(trajectories)}"
    )


def _train_selection_key_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    raw_key = metadata.get("scheduler/train_selection_key")
    if raw_key is None:
        raw_key = metadata.get("train_selection_key")
    if raw_key is None or isinstance(raw_key, bool):
        return None
    key = str(raw_key)
    return key if key else None


def _coverage_control_key(arm_id: str) -> str:
    normalized_arm = _normalize_key_component(arm_id) or "unassigned"
    return f"forced|arm={normalized_arm}"


def _coverage_control_key_from_metadata(
    metadata: Mapping[str, Any],
) -> str | None:
    raw_key = metadata.get("scheduler/coverage_control_key")
    if raw_key is None:
        raw_key = metadata.get("coverage_control_key")
    if raw_key is not None and not isinstance(raw_key, bool):
        key = str(raw_key)
        if key:
            return key

    coverage_forced = any(
        _state_bool(metadata.get(key), False)
        for key in (
            "coverage_forced",
            "scheduler/coverage_forced",
            "scheduler/decision/coverage_forced",
        )
    )
    if not coverage_forced:
        return None
    arm_id = metadata.get("scheduler/arm_id")
    if arm_id is None or isinstance(arm_id, bool):
        return None
    return _coverage_control_key(str(arm_id))


def _timing_response_key(
    *,
    knob: str,
    value: int,
    preference_reason: str,
    train_queue_pressure: float,
    pending_groups: int,
) -> str:
    return (
        f"control={_safe_metric_key(str(knob)) or 'unknown'}"
        f"|value={max(0, int(value))}"
        f"|preference={_safe_metric_key(str(preference_reason)) or 'unknown'}"
        f"|pressure={_queue_pressure_bucket(train_queue_pressure)}"
        f"|pending={_pending_batch_bucket(pending_groups)}"
    )


def _timing_response_keys_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    keys: list[str] = []
    for name in (
        "scheduler/cadence_response_key",
        "cadence_response_key",
        "scheduler/policy_lag_response_key",
        "policy_lag_response_key",
        "scheduler/batch_flush_response_key",
        "batch_flush_response_key",
    ):
        value = metadata.get(name)
        if value is None or isinstance(value, bool):
            continue
        key = str(value)
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _continuation_decision_key(
    *,
    continue_training: bool,
    reason: str,
    pending_train_batches: int,
    train_queue_pressure: float,
) -> str:
    action = "continue" if continue_training else "stop"
    return (
        f"action={action}"
        f"|reason={_continuation_reason_key(reason)}"
        f"|pending={_pending_batch_bucket(pending_train_batches)}"
        f"|pressure={_queue_pressure_bucket(train_queue_pressure)}"
    )


def _continuation_reason_key(reason: str) -> str:
    key = _safe_metric_key(str(reason))
    return key or "unknown"


def _pending_batch_bucket(pending_train_batches: int) -> str:
    pending = max(0, int(pending_train_batches))
    if pending == 0:
        return "0"
    if pending == 1:
        return "1"
    return "2plus"


def _queue_pressure_bucket(train_queue_pressure: float) -> str:
    pressure = (
        float(train_queue_pressure)
        if math.isfinite(float(train_queue_pressure))
        else 0.0
    )
    if pressure < 0.5:
        return "low"
    if pressure < 0.85:
        return "medium"
    return "high"


def _arm_feedback_updates(stats: ArmStats) -> int:
    return stats.train_updates + stats.stale_updates


def _control_feedback_updates(stats: ControlStats) -> int:
    return stats.rollout_updates + stats.train_updates + stats.stale_updates


def _actor_feedback_updates(stats: ActorStats) -> int:
    return stats.pulls + stats.train_updates + stats.stale_updates


def _seconds_to_milliseconds(seconds: float) -> int:
    if not math.isfinite(seconds):
        return 0
    return max(0, int(round(max(0.0, seconds) * 1000.0)))


def _objective_stddev(stats: ArmStats) -> float:
    if stats.objective_observations <= 1:
        return 0.0
    return math.sqrt(
        max(0.0, stats.objective_m2 / (stats.objective_observations - 1))
    )


def scheduler_checkpoint_metadata(scheduler: Any | None) -> dict[str, Any]:
    """Return checkpoint metadata for schedulers with snapshot support."""

    if scheduler is None:
        return {}
    state_dict = getattr(scheduler, "state_dict", None)
    if state_dict is None:
        return {}
    state = state_dict()
    if not isinstance(state, Mapping):
        return {}
    return {SCHEDULER_STATE_KEY: state}


def observe_stale_batch_feedback(
    scheduler: Any | None,
    *,
    groups: Sequence[TrajectoryGroup],
    policy_step: int,
    reason: str,
    additional_dollar_seconds: float = 0.0,
) -> bool:
    """Notify schedulers that support stale-batch feedback."""

    if scheduler is None:
        return False
    observer = getattr(scheduler, "observe_stale_batch", None)
    if observer is None:
        return False
    extra_cost = max(0.0, additional_dollar_seconds)
    if extra_cost > 0.0 and _call_accepts_keyword(
        observer,
        "additional_dollar_seconds",
    ):
        observer(
            groups=groups,
            policy_step=policy_step,
            reason=reason,
            additional_dollar_seconds=extra_cost,
        )
        return True
    observer(groups=groups, policy_step=policy_step, reason=reason)
    return True


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name == keyword
        for parameter in signature.parameters.values()
    )


def _first_int_metadata(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _joint_action_key_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    raw_key = metadata.get("scheduler/joint_action_key")
    if raw_key is None:
        raw_key = metadata.get("joint_action_key")
    if raw_key is None:
        raw_key = metadata.get("scheduler/scheduling_action_key")
    if raw_key is not None and not isinstance(raw_key, bool):
        key = str(raw_key)
        if key:
            return key

    arm_id = metadata.get("scheduler/arm_id")
    if arm_id is None or isinstance(arm_id, bool):
        return None
    cadence = _first_int_metadata(
        metadata,
        (
            "scheduler/active_target_train_batch_groups",
            "scheduler/target_train_batch_groups",
        ),
    )
    max_policy_lag = _first_int_metadata(
        metadata,
        (
            "scheduler/active_max_policy_lag",
            "scheduler/max_policy_lag",
        ),
    )
    active_actor_count = _first_int_metadata(
        metadata,
        ("scheduler/active_actor_count", "scheduler/actor_count"),
    )
    admission_delay_ms = _first_int_metadata(
        metadata,
        (
            "scheduler/active_rollout_admission_delay_ms",
            "scheduler/rollout_admission_delay_ms",
        ),
    )
    action_space_key = _first_text_metadata(
        metadata,
        (
            "scheduler/action_space_key",
            "scheduler/action_space_signature",
            "action_space_key",
        ),
    )
    if (
        cadence is None
        or max_policy_lag is None
        or active_actor_count is None
        or admission_delay_ms is None
    ):
        return None
    return scheduling_action_key(
        arm_id=str(arm_id),
        target_train_batch_groups=cadence,
        max_policy_lag=max_policy_lag,
        active_actor_count=active_actor_count,
        admission_delay_ms=admission_delay_ms,
        action_space_key=action_space_key,
    )


def _joint_action_control_fields(key: str) -> dict[str, int] | None:
    values = _joint_action_key_parts(key)
    cadence = _state_optional_int(values.get("cadence"), None)
    lag = _state_optional_int(values.get("lag"), None)
    actors = _state_optional_int(values.get("actors"), None)
    admission_ms = _state_optional_int(values.get("admission_ms"), None)
    if cadence is None or lag is None or actors is None or admission_ms is None:
        return None
    return {
        "cadence": cadence,
        "lag": lag,
        "actors": actors,
        "admission_ms": admission_ms,
    }


def _joint_action_key_parts(key: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in str(key).split("|"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        values[name] = value
    return values


def _first_text_metadata(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = metadata.get(key)
        normalized = _normalize_key_component(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_key_component(value: Any | None) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    key = safe_metric_key(str(value))
    return key or None


def _batch_staleness_state(
    groups: Sequence[TrajectoryGroup],
    *,
    policy_step: int,
    fallback_lag_limit: int | None,
) -> tuple[int, int | None, float]:
    trajectories = [
        trajectory
        for group in groups
        for trajectory in group.trajectories
    ]
    if not trajectories:
        return 0, fallback_lag_limit, 0.0

    policy_lag = max(
        0,
        max(policy_step - trajectory.policy_step for trajectory in trajectories),
    )
    limits: list[int] = []
    for trajectory in trajectories:
        limit = _first_int_metadata(
            trajectory.metadata,
            (
                "scheduler/active_max_policy_lag",
                "scheduler/max_policy_lag",
            ),
        )
        if limit is not None and limit >= 0:
            limits.append(limit)
    if fallback_lag_limit is not None and fallback_lag_limit >= 0:
        limits.append(fallback_lag_limit)
    lag_limit = min(limits) if limits else None
    if lag_limit is None:
        return policy_lag, None, 0.0
    if lag_limit <= 0:
        return policy_lag, lag_limit, 1.0
    return policy_lag, lag_limit, min(1.0, policy_lag / lag_limit)


def _dataclass_state(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _arm_stats_state(stats: ArmStats) -> dict[str, Any]:
    state = _dataclass_state(stats)
    state["inflight"] = 0
    state["reserved_rollout_dollar_seconds"] = 0.0
    return state


def _actor_stats_state(stats: ActorStats) -> dict[str, Any]:
    state = _dataclass_state(stats)
    state["inflight"] = 0
    return state


def _arm_credit_weights(groups: Sequence[TrajectoryGroup]) -> dict[str, float]:
    arm_weights: dict[str, float] = {}
    for group in groups:
        for trajectory in group.trajectories:
            arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
            arm_weights.setdefault(arm_id, 0.0)
            arm_weights[arm_id] += _trajectory_credit_weight(trajectory)
    return arm_weights


def _arm_useful_experience(groups: Sequence[TrajectoryGroup]) -> dict[str, float]:
    arm_experience: dict[str, float] = {}
    for group in groups:
        for trajectory in group.trajectories:
            quality = action_quality(trajectory)
            if quality <= 0.0:
                continue
            arm_id = str(trajectory.metadata.get("scheduler/arm_id", "unassigned"))
            arm_experience[arm_id] = arm_experience.get(arm_id, 0.0) + quality
    return arm_experience


def _trajectory_credit_weight(trajectory: Trajectory) -> float:
    quality = action_quality(trajectory)
    if quality <= 0.0:
        return 0.0
    return max(0.0, trajectory.reward * quality) or quality


def _trajectory_actor_id(trajectory: Trajectory) -> int | None:
    return _state_optional_int(trajectory.metadata.get("actor_id"), None)


def _mapping_state(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _arm_stats_from_state(value: Any) -> ArmStats:
    state = _mapping_state(value)
    default = ArmStats()
    return ArmStats(
        decisions=_state_int(state.get("decisions"), default.decisions),
        # In-flight reservations belong to a live process and should not resume.
        inflight=0,
        reserved_rollout_dollar_seconds=0.0,
        pulls=_state_int(state.get("pulls"), default.pulls),
        accepted=_state_int(state.get("accepted"), default.accepted),
        unsafe=_state_int(state.get("unsafe"), default.unsafe),
        failed_rollouts=_state_int(
            state.get("failed_rollouts"),
            default.failed_rollouts,
        ),
        failure_modes=_int_mapping_state(state.get("failure_modes")),
        reconstruction_observations=_state_int(
            state.get("reconstruction_observations"),
            default.reconstruction_observations,
        ),
        reconstruction_accuracy_ema=_state_float(
            state.get("reconstruction_accuracy_ema"),
            default.reconstruction_accuracy_ema,
        ),
        reconstruction_drift_ema=_state_float(
            state.get("reconstruction_drift_ema"),
            default.reconstruction_drift_ema,
        ),
        total_reconstruction_accuracy=_state_float(
            state.get("total_reconstruction_accuracy"),
            default.total_reconstruction_accuracy,
        ),
        min_reconstruction_accuracy=_state_float(
            state.get("min_reconstruction_accuracy"),
            default.min_reconstruction_accuracy,
        ),
        max_reconstruction_drift=_state_float(
            state.get("max_reconstruction_drift"),
            default.max_reconstruction_drift,
        ),
        train_updates=_state_int(
            state.get("train_updates"),
            default.train_updates,
        ),
        stale_updates=_state_int(
            state.get("stale_updates"),
            default.stale_updates,
        ),
        stale_batches=_state_int(
            state.get("stale_batches"),
            default.stale_batches,
        ),
        stale_trajectories=_state_int(
            state.get("stale_trajectories"),
            default.stale_trajectories,
        ),
        reward_ema=_state_float(state.get("reward_ema"), default.reward_ema),
        effective_reward_ema=_state_float(
            state.get("effective_reward_ema"),
            default.effective_reward_ema,
        ),
        min_effective_reward=_state_float(
            state.get("min_effective_reward"),
            default.min_effective_reward,
        ),
        max_effective_reward=_state_float(
            state.get("max_effective_reward"),
            default.max_effective_reward,
        ),
        last_reward_scale=_state_float(
            state.get("last_reward_scale"),
            default.last_reward_scale,
        ),
        last_normalized_positive_improvement=_state_float(
            state.get("last_normalized_positive_improvement"),
            default.last_normalized_positive_improvement,
        ),
        action_quality_ema=_state_float(
            state.get("action_quality_ema"),
            default.action_quality_ema,
        ),
        reward_efficiency_ema=_state_float(
            state.get("reward_efficiency_ema"),
            default.reward_efficiency_ema,
        ),
        marginal_objective_ema=_state_float(
            state.get("marginal_objective_ema"),
            default.marginal_objective_ema,
        ),
        policy_improvement_objective_ema=_state_float(
            state.get("policy_improvement_objective_ema"),
            default.policy_improvement_objective_ema,
        ),
        objective_observations=_state_int(
            state.get("objective_observations"),
            default.objective_observations,
        ),
        objective_mean=_state_float(
            state.get("objective_mean"),
            default.objective_mean,
        ),
        objective_m2=_state_float(
            state.get("objective_m2"),
            default.objective_m2,
        ),
        dollar_seconds_ema=_state_float(
            state.get("dollar_seconds_ema"),
            default.dollar_seconds_ema,
        ),
        train_reward_observations=_state_int(
            state.get("train_reward_observations"),
            default.train_reward_observations,
        ),
        min_train_reward=_state_float(
            state.get("min_train_reward"),
            default.min_train_reward,
        ),
        max_train_reward=_state_float(
            state.get("max_train_reward"),
            default.max_train_reward,
        ),
        last_train_reward=_state_float(
            state.get("last_train_reward"),
            default.last_train_reward,
        ),
        last_train_reward_scale=_state_float(
            state.get("last_train_reward_scale"),
            default.last_train_reward_scale,
        ),
        last_train_reward_improvement=_state_float(
            state.get("last_train_reward_improvement"),
            default.last_train_reward_improvement,
        ),
        last_normalized_train_reward_improvement=_state_float(
            state.get("last_normalized_train_reward_improvement"),
            default.last_normalized_train_reward_improvement,
        ),
        total_reward=_state_float(state.get("total_reward"), default.total_reward),
        total_effective_reward=_state_float(
            state.get("total_effective_reward"),
            default.total_effective_reward,
        ),
        total_positive_improvement=_state_float(
            state.get("total_positive_improvement"),
            default.total_positive_improvement,
        ),
        total_normalized_positive_improvement=_state_float(
            state.get("total_normalized_positive_improvement"),
            default.total_normalized_positive_improvement,
        ),
        total_reward_improving_experience=_state_float(
            state.get("total_reward_improving_experience"),
            default.total_reward_improving_experience,
        ),
        total_normalized_reward_improving_experience=_state_float(
            state.get("total_normalized_reward_improving_experience"),
            default.total_normalized_reward_improving_experience,
        ),
        total_policy_improvement_objective=_state_float(
            state.get("total_policy_improvement_objective"),
            default.total_policy_improvement_objective,
        ),
        total_stale_penalty_objective=_state_float(
            state.get("total_stale_penalty_objective"),
            default.total_stale_penalty_objective,
        ),
        total_dollar_seconds=_state_float(
            state.get("total_dollar_seconds"),
            default.total_dollar_seconds,
        ),
        rollout_dollar_seconds=_state_float(
            state.get("rollout_dollar_seconds"),
            max(
                0.0,
                _state_float(
                    state.get("total_dollar_seconds"),
                    default.total_dollar_seconds,
                )
                - _state_float(
                    state.get("queue_wait_dollar_seconds"),
                    default.queue_wait_dollar_seconds,
                ),
            ),
        ),
        queue_wait_dollar_seconds=_state_float(
            state.get("queue_wait_dollar_seconds"),
            default.queue_wait_dollar_seconds,
        ),
        admission_dollar_seconds=_state_float(
            state.get("admission_dollar_seconds"),
            default.admission_dollar_seconds,
        ),
        action_units=_state_int(
            state.get("action_units"),
            default.action_units,
        ),
        source_tokens=_state_int(
            state.get("source_tokens"),
            default.source_tokens,
        ),
        old_logprob_units=_state_int(
            state.get("old_logprob_units"),
            default.old_logprob_units,
        ),
        new_logprob_units=_state_int(
            state.get("new_logprob_units"),
            default.new_logprob_units,
        ),
        reference_logprob_units=_state_int(
            state.get("reference_logprob_units"),
            default.reference_logprob_units,
        ),
        old_new_logprob_pairs=_state_int(
            state.get("old_new_logprob_pairs"),
            default.old_new_logprob_pairs,
        ),
        old_reference_logprob_pairs=_state_int(
            state.get("old_reference_logprob_pairs"),
            default.old_reference_logprob_pairs,
        ),
        old_logprob_sum=_state_float(
            state.get("old_logprob_sum"),
            default.old_logprob_sum,
        ),
        new_logprob_sum=_state_float(
            state.get("new_logprob_sum"),
            default.new_logprob_sum,
        ),
        reference_logprob_sum=_state_float(
            state.get("reference_logprob_sum"),
            default.reference_logprob_sum,
        ),
        old_new_logprob_delta_sum=_state_float(
            state.get("old_new_logprob_delta_sum"),
            default.old_new_logprob_delta_sum,
        ),
        old_new_logprob_abs_delta_sum=_state_float(
            state.get("old_new_logprob_abs_delta_sum"),
            default.old_new_logprob_abs_delta_sum,
        ),
        old_reference_logprob_delta_sum=_state_float(
            state.get("old_reference_logprob_delta_sum"),
            default.old_reference_logprob_delta_sum,
        ),
        importance_ratio_sum=_state_float(
            state.get("importance_ratio_sum"),
            default.importance_ratio_sum,
        ),
        stale_experience=_state_float(
            state.get("stale_experience"),
            default.stale_experience,
        ),
    )


def _control_stats_from_state(value: Any) -> ControlStats:
    state = _mapping_state(value)
    default = ControlStats()
    return ControlStats(
        decisions=_state_int(state.get("decisions"), default.decisions),
        rollout_updates=_state_int(
            state.get("rollout_updates"),
            default.rollout_updates,
        ),
        train_updates=_state_int(
            state.get("train_updates"),
            default.train_updates,
        ),
        stale_updates=_state_int(
            state.get("stale_updates"),
            default.stale_updates,
        ),
        objective_ema=_state_float(
            state.get("objective_ema"),
            default.objective_ema,
        ),
        total_objective=_state_float(
            state.get("total_objective"),
            default.total_objective,
        ),
        total_stale_penalty_objective=_state_float(
            state.get("total_stale_penalty_objective"),
            default.total_stale_penalty_objective,
        ),
        stale_experience=_state_float(
            state.get("stale_experience"),
            default.stale_experience,
        ),
    )


def _control_family_from_state(value: Any) -> dict[int, ControlStats]:
    controls: dict[int, ControlStats] = {}
    for raw_key, raw_stats in _mapping_state(value).items():
        key = _state_optional_int(raw_key, None)
        if key is not None:
            controls[key] = _control_stats_from_state(raw_stats)
    return controls


def _string_control_family_from_state(value: Any) -> dict[str, ControlStats]:
    controls: dict[str, ControlStats] = {}
    for raw_key, raw_stats in _mapping_state(value).items():
        key = str(raw_key)
        if key:
            controls[key] = _control_stats_from_state(raw_stats)
    return controls


def _actor_stats_from_state(value: Any) -> ActorStats:
    state = _mapping_state(value)
    default = ActorStats()
    return ActorStats(
        decisions=_state_int(state.get("decisions"), default.decisions),
        inflight=0,
        pulls=_state_int(state.get("pulls"), default.pulls),
        accepted=_state_int(state.get("accepted"), default.accepted),
        unsafe=_state_int(state.get("unsafe"), default.unsafe),
        failed_rollouts=_state_int(
            state.get("failed_rollouts"),
            default.failed_rollouts,
        ),
        train_updates=_state_int(
            state.get("train_updates"),
            default.train_updates,
        ),
        stale_updates=_state_int(
            state.get("stale_updates"),
            default.stale_updates,
        ),
        stale_batches=_state_int(
            state.get("stale_batches"),
            default.stale_batches,
        ),
        stale_trajectories=_state_int(
            state.get("stale_trajectories"),
            default.stale_trajectories,
        ),
        objective_ema=_state_float(
            state.get("objective_ema"),
            default.objective_ema,
        ),
        rollout_objective_ema=_state_float(
            state.get("rollout_objective_ema"),
            default.rollout_objective_ema,
        ),
        train_objective_ema=_state_float(
            state.get("train_objective_ema"),
            default.train_objective_ema,
        ),
        total_objective=_state_float(
            state.get("total_objective"),
            default.total_objective,
        ),
        total_rollout_objective=_state_float(
            state.get("total_rollout_objective"),
            default.total_rollout_objective,
        ),
        total_train_objective=_state_float(
            state.get("total_train_objective"),
            default.total_train_objective,
        ),
        total_stale_penalty_objective=_state_float(
            state.get("total_stale_penalty_objective"),
            default.total_stale_penalty_objective,
        ),
        total_dollar_seconds=_state_float(
            state.get("total_dollar_seconds"),
            default.total_dollar_seconds,
        ),
        rollout_dollar_seconds=_state_float(
            state.get("rollout_dollar_seconds"),
            default.rollout_dollar_seconds,
        ),
        queue_wait_dollar_seconds=_state_float(
            state.get("queue_wait_dollar_seconds"),
            default.queue_wait_dollar_seconds,
        ),
        admission_dollar_seconds=_state_float(
            state.get("admission_dollar_seconds"),
            default.admission_dollar_seconds,
        ),
        stale_experience=_state_float(
            state.get("stale_experience"),
            default.stale_experience,
        ),
        action_units=_state_int(state.get("action_units"), default.action_units),
        source_tokens=_state_int(
            state.get("source_tokens"),
            default.source_tokens,
        ),
    )


def _actor_family_from_state(value: Any) -> dict[int, ActorStats]:
    actors: dict[int, ActorStats] = {}
    for raw_key, raw_stats in _mapping_state(value).items():
        key = _state_optional_int(raw_key, None)
        if key is not None:
            actors[key] = _actor_stats_from_state(raw_stats)
    return actors


def _state_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate


def _state_optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state_optional_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _state_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _state_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _int_mapping_state(value: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, raw_count in _mapping_state(value).items():
        count = _state_int(raw_count, 0)
        if count > 0:
            result[str(key)] = count
    return result


def _decision_to_state(decision: SchedulerDecision) -> dict[str, Any]:
    return {
        "arm_id": decision.arm_id,
        "target_train_batch_groups": decision.target_train_batch_groups,
        "max_policy_lag": decision.max_policy_lag,
        "metadata": _scalar_metadata(decision.metadata),
    }


def _decision_matches_snapshot_for_cancel(
    snapshot: Mapping[str, Any] | None,
    decision: SchedulerDecision,
) -> bool:
    if snapshot is None:
        return False
    state = _decision_to_state(decision)
    for key in ("arm_id", "target_train_batch_groups", "max_policy_lag"):
        if snapshot.get(key) != state.get(key):
            return False
    snapshot_metadata = _mapping_state(snapshot.get("metadata"))
    state_metadata = _mapping_state(state.get("metadata"))
    return all(
        state_metadata.get(key) == value
        for key, value in snapshot_metadata.items()
    )


def _decision_state_from_mapping(value: Any) -> dict[str, Any] | None:
    state = _mapping_state(value)
    arm_id = str(state.get("arm_id", ""))
    if not arm_id:
        return None
    return {
        "arm_id": arm_id,
        "target_train_batch_groups": _state_int(
            state.get("target_train_batch_groups"),
            0,
        ),
        "max_policy_lag": _state_int(state.get("max_policy_lag"), 0),
        "metadata": _scalar_metadata(_mapping_state(state.get("metadata"))),
    }


def _scalar_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (bool, int, str)):
            snapshot[str(key)] = value
        elif isinstance(value, float) and math.isfinite(value):
            snapshot[str(key)] = value
    return snapshot


def _useful_experience_count(groups: Sequence[TrajectoryGroup]) -> float:
    return sum(
        action_quality(trajectory)
        for group in groups
        for trajectory in group.trajectories
    )


def _groups_action_logprob_stats(
    groups: Sequence[TrajectoryGroup],
) -> ActionLogprobStats:
    action_units = 0
    old_logprob_units = 0
    new_logprob_units = 0
    reference_logprob_units = 0
    old_logprob_sum = 0.0
    new_logprob_sum = 0.0
    reference_logprob_sum = 0.0
    old_new_pairs = 0
    old_reference_pairs = 0
    old_new_logprob_delta_sum = 0.0
    old_new_logprob_abs_delta_sum = 0.0
    old_reference_logprob_delta_sum = 0.0
    importance_ratio_sum = 0.0
    for group in groups:
        for trajectory in group.trajectories:
            stats = action_logprob_stats(trajectory.actions)
            action_units += stats.action_units
            old_logprob_units += stats.old_logprob_units
            new_logprob_units += stats.new_logprob_units
            reference_logprob_units += stats.reference_logprob_units
            old_logprob_sum += stats.old_logprob_sum
            new_logprob_sum += stats.new_logprob_sum
            reference_logprob_sum += stats.reference_logprob_sum
            old_new_pairs += stats.old_new_pairs
            old_reference_pairs += stats.old_reference_pairs
            old_new_logprob_delta_sum += stats.old_new_logprob_delta_sum
            old_new_logprob_abs_delta_sum += (
                stats.old_new_logprob_abs_delta_sum
            )
            old_reference_logprob_delta_sum += (
                stats.old_reference_logprob_delta_sum
            )
            importance_ratio_sum += stats.importance_ratio_sum
    return ActionLogprobStats(
        action_units=action_units,
        old_logprob_units=old_logprob_units,
        new_logprob_units=new_logprob_units,
        reference_logprob_units=reference_logprob_units,
        old_logprob_sum=old_logprob_sum,
        new_logprob_sum=new_logprob_sum,
        reference_logprob_sum=reference_logprob_sum,
        old_new_pairs=old_new_pairs,
        old_reference_pairs=old_reference_pairs,
        old_new_logprob_delta_sum=old_new_logprob_delta_sum,
        old_new_logprob_abs_delta_sum=old_new_logprob_abs_delta_sum,
        old_reference_logprob_delta_sum=old_reference_logprob_delta_sum,
        importance_ratio_sum=importance_ratio_sum,
    )


def _groups_sample_dollar_seconds(groups: Sequence[TrajectoryGroup]) -> float:
    return sum(
        _trajectory_sample_dollar_seconds(trajectory)
        for group in groups
        for trajectory in group.trajectories
    )


def _groups_unobserved_sample_dollar_seconds(
    groups: Sequence[TrajectoryGroup],
) -> float:
    return sum(
        _trajectory_sample_dollar_seconds(trajectory)
        for group in groups
        for trajectory in group.trajectories
        if not _trajectory_rollout_observed(trajectory)
    )


def _trajectory_rollout_observed(trajectory: Trajectory) -> bool:
    return _state_bool(
        trajectory.metadata.get("scheduler/rollout_observed"),
        False,
    )


def _trajectory_sample_dollar_seconds(trajectory: Trajectory) -> float:
    explicit_total = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/dollar_seconds",),
    )
    if explicit_total is None:
        explicit_total = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/dollar_seconds",),
        )
    if explicit_total is not None:
        return explicit_total

    rollout_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("rollout/dollar_seconds",),
    )
    if rollout_cost is None:
        rollout_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("rollout/dollar_seconds",),
        )
    queue_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
    )
    if queue_cost is None:
        queue_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/actor_queue_wait_dollar_seconds", "queue_wait/dollar_seconds"),
        )
    admission_cost = _trajectory_admission_dollar_seconds(trajectory)
    return (rollout_cost or 0.0) + (queue_cost or 0.0) + admission_cost


def _trajectory_admission_dollar_seconds(trajectory: Trajectory) -> float:
    explicit_cost = _first_nonnegative_mapping_float(
        trajectory.metrics,
        ("cost/actor_admission_dollar_seconds", "admission/dollar_seconds"),
    )
    if explicit_cost is None:
        explicit_cost = _first_nonnegative_mapping_float(
            trajectory.metadata,
            ("cost/actor_admission_dollar_seconds", "admission/dollar_seconds"),
        )
    return explicit_cost or 0.0


def _trajectory_reserved_rollout_dollar_seconds(trajectory: Trajectory) -> float:
    explicit_cost = _first_nonnegative_mapping_float(
        trajectory.metadata,
        (
            "scheduler/decision/reserved_rollout_dollar_seconds",
            "scheduler/decision/estimated_rollout_dollar_seconds",
            "scheduler/decision/expected_rollout_dollar_seconds",
        ),
    )
    return explicit_cost or 0.0


def _decision_reserved_rollout_dollar_seconds(
    decision: SchedulerDecision,
) -> float:
    return max(
        0.0,
        _state_float(
            _mapping_state(decision.metadata).get("reserved_rollout_dollar_seconds"),
            0.0,
        ),
    )


def _first_nonnegative_mapping_float(
    values: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, float(value))
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return max(0.0, parsed)
    return None


def action_quality(trajectory: Trajectory) -> float:
    """Quality multiplier for action-level usefulness and safety feedback."""

    metadata = trajectory.metadata
    if trajectory.exception:
        return 0.0
    for key in ("action/safe", "reconstruction/safe", "verifier/passed"):
        if metadata.get(key) is False:
            return 0.0
    if _custom_failure_modes(metadata):
        return 0.0
    candidates = []
    for key in (
        "action/quality",
        "reconstruction/accuracy",
        "verifier/score",
    ):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            candidates.append(max(0.0, min(1.0, float(value))))
    if candidates:
        return min(candidates)
    return 1.0


def trajectory_reconstruction_accuracy(trajectory: Trajectory) -> float | None:
    """Return clamped reconstruction accuracy when rollout metadata provides it."""

    reconstruction_accuracy = _first_nonnegative_mapping_float(
        trajectory.metadata,
        ("reconstruction/accuracy",),
    )
    if reconstruction_accuracy is None:
        reconstruction_accuracy = _first_nonnegative_mapping_float(
            trajectory.metrics,
            ("reconstruction/accuracy",),
        )
    if reconstruction_accuracy is None:
        return None
    return min(1.0, max(0.0, reconstruction_accuracy))


def trajectory_failure_modes(
    trajectory: Trajectory,
    *,
    reconstruction_drift_threshold: float = 0.95,
) -> tuple[str, ...]:
    """Categorical action/rollout failure modes for scheduler credit."""

    metadata = trajectory.metadata
    modes: list[str] = []
    if trajectory.exception:
        modes.append("exception")
    if metadata.get("action/safe") is False:
        modes.append("action_unsafe")
    if metadata.get("reconstruction/safe") is False:
        modes.append("reconstruction_unsafe")
    if metadata.get("verifier/passed") is False:
        modes.append("verifier_failed")
    modes.extend(_custom_failure_modes(metadata))

    reconstruction_accuracy = trajectory_reconstruction_accuracy(trajectory)
    if (
        reconstruction_accuracy is not None
        and reconstruction_accuracy < reconstruction_drift_threshold
    ):
        modes.append("reconstruction_drift")
    return tuple(dict.fromkeys(modes))


def _custom_failure_modes(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    modes: list[str] = []
    for key in (
        "failure/mode",
        "failure/modes",
        "verifier/failure_mode",
        "verifier/failure_modes",
        "action/failure_mode",
        "action/failure_modes",
        "rollout/failure_mode",
        "rollout/failure_modes",
    ):
        modes.extend(_failure_modes_from_value(metadata.get(key)))
    return tuple(dict.fromkeys(modes))


def _failure_modes_from_value(value: Any) -> tuple[str, ...]:
    if value is None or value is False:
        return ()
    if isinstance(value, str):
        mode = _safe_metric_key(value.strip())
        return (mode,) if mode else ()
    if isinstance(value, (list, tuple, set)) and not isinstance(value, (str, bytes)):
        modes: list[str] = []
        for item in value:
            modes.extend(_failure_modes_from_value(item))
        return tuple(modes)
    return ()
