from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Protocol, Sequence

from .actions import ActionCodec, action_logprob_stats
from .types import Scenario, TrainResult, Trajectory, TrajectoryGroup, mean


SCHEDULER_STATE_KEY = "scheduler/state"


@dataclass(frozen=True)
class SchedulerDecision:
    """One closed-loop rollout/control decision."""

    scenario: Scenario
    action_codec: ActionCodec
    arm_id: str
    target_train_batch_groups: int
    max_policy_lag: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    ) -> SchedulerDecision:
        ...

    def target_train_batch_groups(
        self,
        *,
        configured: int,
        pending_groups: int,
        train_queue_pressure: float,
        policy_step: int,
    ) -> int:
        ...

    def max_policy_lag(
        self,
        *,
        configured: int,
        train_queue_pressure: float,
        policy_step: int,
    ) -> int:
        ...

    def rollout_admission_delay_s(
        self,
        *,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
    ) -> float:
        ...

    def active_actor_count(
        self,
        *,
        configured: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
    ) -> int:
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
    ) -> None:
        ...

    def score_train_groups(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        policy_step: int,
    ) -> float:
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
        confidence_penalty_weight: float = 0.0,
        control_exploration_bonus: float = 0.1,
        max_control_candidate_values: int = 8,
        min_rollout_coverage_fraction: float = 0.0,
        max_rollout_coverage_cost_fraction: float | None = None,
        min_train_steps: int = 1,
        roi_patience: int | None = None,
        min_train_objective: float = 0.0,
        continuation_objective: str = "train",
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
        if confidence_penalty_weight < 0:
            raise ValueError("confidence_penalty_weight must be non-negative")
        if control_exploration_bonus < 0:
            raise ValueError("control_exploration_bonus must be non-negative")
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
        self.confidence_penalty_weight = confidence_penalty_weight
        self.control_exploration_bonus = control_exploration_bonus
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
        self._last_train_batch_reward_improving_experience = 0.0
        self._last_train_batch_sample_dollar_seconds = 0.0
        self._last_train_batch_cost_normalized_priority = 0.0
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
        self._cadence_controls: dict[int, ControlStats] = {}
        self._lag_controls: dict[int, ControlStats] = {}
        self._admission_controls: dict[int, ControlStats] = {}
        self._actor_count_controls: dict[int, ControlStats] = {}
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
    ) -> SchedulerDecision:
        if not scenarios:
            raise ValueError("at least one scenario is required")
        if not action_codecs:
            raise ValueError("at least one action codec is required")

        arms = self._arm_candidates(scenarios, action_codecs)
        coverage_selection, coverage_cost_limited = self._coverage_candidate(arms)
        arm_ids = [candidate[0] for candidate in arms]
        if coverage_selection is None:
            arm_id, scenario, codec = max(
                arms,
                key=lambda arm: self._score_arm(arm[0]),
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
        coverage_cost_limit = self.max_rollout_coverage_cost_fraction or 0.0
        selected_stats = self._arms[arm_id]
        decision_score = self._score_arm(arm_id)
        objective_score = self._arm_value(selected_stats)
        exploration_score = self._exploration_value(selected_stats)
        self._record_arm_decision(selected_stats)
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
            target_train_batch_groups=self.target_train_batch_groups(
                configured=configured_train_batch_groups,
                pending_groups=0,
                train_queue_pressure=train_queue_pressure,
                policy_step=policy_step,
            ),
            max_policy_lag=self.max_policy_lag(
                configured=configured_max_policy_lag,
                train_queue_pressure=train_queue_pressure,
                policy_step=policy_step,
            ),
            metadata={
                "actor_id": actor_id,
                "policy_step": policy_step,
                "trajectory_queue_pressure": trajectory_queue_pressure,
                "train_queue_pressure": train_queue_pressure,
                "score": decision_score,
                "objective_score": objective_score,
                "exploration_score": exploration_score,
                "inflight_rollouts": selected_stats.inflight,
                "coverage_forced": coverage_forced,
                "coverage_target": coverage_target,
                "coverage_share": coverage_share,
                "coverage_deficit": coverage_deficit,
                "coverage_cost_share": coverage_cost_share,
                "coverage_cost_limit": coverage_cost_limit,
                "coverage_cost_limited": coverage_cost_limited,
                "expected_rollout_dollar_seconds": (
                    selected_stats.dollar_seconds_ema
                    if selected_stats.pulls
                    else 0.0
                ),
            },
        )
        self._last_decision = decision
        self._last_decision_snapshot = _decision_to_state(decision)
        return decision

    def target_train_batch_groups(
        self,
        *,
        configured: int,
        pending_groups: int,
        train_queue_pressure: float,
        policy_step: int,
    ) -> int:
        upper = self.max_train_batch_groups or configured
        upper = max(self.min_train_batch_groups, upper)
        configured = min(max(configured, self.min_train_batch_groups), upper)
        candidates = self._control_candidates(
            min_value=self.min_train_batch_groups,
            configured=configured,
            upper=upper,
        )
        if self._has_positive_objective_signal():
            preferred = self.min_train_batch_groups
        elif train_queue_pressure >= 0.75 or pending_groups >= upper:
            preferred = upper
        else:
            preferred = configured
        if (
            preferred == upper
            and not self._has_positive_objective_signal()
        ):
            return self._record_control_decision(self._cadence_controls, upper)
        return self._record_control_decision(
            self._cadence_controls,
            self._select_control_value(
                self._cadence_controls,
                candidates,
                preferred=preferred,
            ),
        )

    def max_policy_lag(
        self,
        *,
        configured: int,
        train_queue_pressure: float,
        policy_step: int,
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
        if self._has_unaccepted_known_arm():
            return self._record_control_decision(self._lag_controls, configured)
        if train_queue_pressure >= 0.75:
            preferred = self.min_policy_lag
        elif self._has_positive_objective_signal():
            preferred = self.min_policy_lag
        else:
            preferred = configured
        return self._record_control_decision(
            self._lag_controls,
            self._select_control_value(
                self._lag_controls,
                candidates,
                preferred=preferred,
            ),
        )

    def active_actor_count(
        self,
        *,
        configured: int,
        trajectory_queue_pressure: float,
        train_queue_pressure: float,
        policy_step: int,
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
            return self._record_control_decision(
                self._actor_count_controls,
                preferred,
            )
        if pressure >= 0.75 and not self._has_positive_objective_signal():
            preferred = self.min_actor_count
            return self._record_control_decision(
                self._actor_count_controls,
                preferred,
            )
        preferred = configured
        return self._record_control_decision(
            self._actor_count_controls,
            self._select_control_value(
                self._actor_count_controls,
                candidates,
                preferred=preferred,
            ),
        )

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
            stats.inflight -= 1
        rollout_cost = max(dollar_seconds, 1e-12)
        queue_wait_cost = max(0.0, queue_wait_dollar_seconds)
        admission_cost = _trajectory_admission_dollar_seconds(trajectory)
        cost = max(rollout_cost + queue_wait_cost + admission_cost, 1e-12)
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
        self._credit_rollout_objective_to_admission_control(
            trajectory,
            marginal_objective,
        )
        self._credit_rollout_objective_to_actor_count_control(
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
        selected_ms = self._select_control_value(
            self._admission_controls,
            self._control_candidates(
                min_value=0,
                configured=preferred_ms,
                upper=max_delay_ms,
            ),
            preferred=preferred_ms,
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
    ) -> None:
        """Record useful experience that was produced but never trained."""

        stale_trajectories = sum(
            len(group.trajectories)
            for group in groups
        )
        stale_experience = _useful_experience_count(groups)
        stale_cost = _groups_sample_dollar_seconds(groups)
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
        self._stale_sample_dollar_seconds += stale_cost
        self._last_stale_penalty_objective = penalty_objective
        self._last_stale_experience_count = stale_experience
        self._last_stale_lost_reward_improving_experience = penalty_experience
        self._last_stale_sample_dollar_seconds = stale_cost
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
        uncosted_base_priority = (
            arm_component + self.reward_efficiency_weight * raw_reward_component
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
        priority = base_priority + staleness_bonus
        self._last_train_batch_priority = priority
        self._last_train_batch_policy_lag = policy_lag
        self._last_train_batch_lag_limit = (
            lag_limit if lag_limit is not None else -1
        )
        self._last_train_batch_staleness_urgency = staleness_urgency
        self._last_train_batch_staleness_bonus = staleness_bonus
        self._last_train_batch_reward_improving_experience = (
            batch_reward_improving_experience
        )
        self._last_train_batch_sample_dollar_seconds = sample_dollar_seconds
        self._last_train_batch_cost_normalized_priority = base_priority
        return priority

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
            return False
        if self._accounted_budget_exhausted():
            self._stop_recommended = True
            return False
        if self.roi_patience is None:
            return True
        if policy_step < self.min_train_steps:
            return True
        if self._has_unaccepted_known_arm():
            return True
        if self._low_roi_train_steps >= self.roi_patience:
            self._stop_recommended = True
            return False
        return True

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
                "confidence_penalty_weight": self.confidence_penalty_weight,
                "control_exploration_bonus": self.control_exploration_bonus,
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
                "last_train_batch_reward_improving_experience": (
                    self._last_train_batch_reward_improving_experience
                ),
                "last_train_batch_sample_dollar_seconds": (
                    self._last_train_batch_sample_dollar_seconds
                ),
                "last_train_batch_cost_normalized_priority": (
                    self._last_train_batch_cost_normalized_priority
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
        budget_limit = self.max_accounted_dollar_seconds or 0.0
        budget_remaining = (
            max(0.0, budget_limit - accounted_dollar_seconds)
            if self.max_accounted_dollar_seconds is not None
            else 0.0
        )
        budget_fraction = (
            accounted_dollar_seconds / budget_limit
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
            "scheduler/budget/max_accounted_dollar_seconds": budget_limit,
            "scheduler/budget/accounted_dollar_seconds": accounted_dollar_seconds,
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
            "scheduler/last_train_batch_reward_improving_experience": (
                self._last_train_batch_reward_improving_experience
            ),
            "scheduler/last_train_batch_sample_dollar_seconds": (
                self._last_train_batch_sample_dollar_seconds
            ),
            "scheduler/last_train_batch_cost_normalized_priority": (
                self._last_train_batch_cost_normalized_priority
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
            "scheduler/weights/confidence_penalty": (
                self.confidence_penalty_weight
            ),
            "scheduler/weights/control_exploration": (
                self.control_exploration_bonus
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
            "scheduler/costs/total_dollar_seconds": accounted_dollar_seconds,
        }
        if self._last_decision_snapshot is not None:
            last_arm_id = str(self._last_decision_snapshot.get("arm_id", ""))
            metrics[
                f"scheduler/last_arm/{_safe_metric_key(last_arm_id)}"
            ] = 1.0
            metrics["scheduler/last_target_train_batch_groups"] = float(
                self._last_decision_snapshot.get("target_train_batch_groups", 0)
            )
            metrics["scheduler/last_max_policy_lag"] = float(
                self._last_decision_snapshot.get("max_policy_lag", 0)
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
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _control_feedback_updates(stats)
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
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _control_feedback_updates(stats)
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
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _control_feedback_updates(stats)
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
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/rollout_updates"] = float(stats.rollout_updates)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _control_feedback_updates(stats)
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
                and self._score_arm(arm_id) > self._score_arm(most_undercovered[0])
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

    def _score_arm(self, arm_id: str) -> float:
        stats = self._arms.setdefault(arm_id, ArmStats())
        if stats.pulls == 0:
            # Reserve unobserved arms before repeating in-flight work.
            return 1_000_000_000.0 - stats.inflight
        exploitation = self._arm_value(stats)
        exploration = self._exploration_value(stats)
        return exploitation + exploration

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
            self._credit_objective_to_controls(
                groups,
                objective,
                stale_feedback=False,
            )
            return
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

    def _credit_rollout_objective_to_admission_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        value = _first_int_metadata(
            trajectory.metadata,
            (
                "scheduler/active_rollout_admission_delay_ms",
                "scheduler/rollout_admission_delay_ms",
            ),
        )
        if value is None:
            return
        stats = self._admission_controls.setdefault(value, ControlStats())
        stats.rollout_updates += 1
        stats.total_objective += objective
        stats.objective_ema = self._ema(
            stats.objective_ema,
            objective,
            _control_feedback_updates(stats),
        )

    def _credit_rollout_objective_to_actor_count_control(
        self,
        trajectory: Trajectory,
        objective: float,
    ) -> None:
        value = _first_int_metadata(
            trajectory.metadata,
            (
                "scheduler/active_actor_count",
                "scheduler/actor_count",
            ),
        )
        if value is None:
            return
        stats = self._actor_count_controls.setdefault(value, ControlStats())
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

    def _select_control_value(
        self,
        controls: dict[int, ControlStats],
        candidates: Sequence[int],
        *,
        preferred: int,
    ) -> int:
        if not candidates:
            return preferred
        candidate_values = tuple(candidates)
        if preferred not in candidate_values:
            preferred = candidate_values[0]
        if all(
            controls.get(value) is None
            or (
                controls[value].decisions == 0
                and _control_feedback_updates(controls[value]) == 0
            )
            for value in candidate_values
        ):
            return preferred
        return max(
            candidate_values,
            key=lambda value: (
                self._score_control_value(controls, value),
                1 if value == preferred else 0,
                -abs(value - preferred),
                -value,
            ),
        )

    def _score_control_value(
        self,
        controls: Mapping[int, ControlStats],
        value: int,
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
        controls: Mapping[int, ControlStats],
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

    def _record_arm_decision(self, stats: ArmStats) -> None:
        stats.decisions += 1
        stats.inflight += 1
        self._total_decisions += 1

    def _total_inflight_rollouts(self) -> int:
        return sum(stats.inflight for stats in self._arms.values())

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
        )

    def _accounted_budget_exhausted(self) -> bool:
        if self.max_accounted_dollar_seconds is None:
            return False
        return self._accounted_dollar_seconds() >= self.max_accounted_dollar_seconds

    def _ema(self, current: float, value: float, count: int) -> float:
        if count <= 1:
            return value
        return self.ema_alpha * value + (1 - self.ema_alpha) * current


def _arm_id(scenario: Scenario, codec: ActionCodec) -> str:
    return f"{scenario.id}|{_codec_key(codec)}"


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
) -> bool:
    """Notify schedulers that support stale-batch feedback."""

    if scheduler is None:
        return False
    observer = getattr(scheduler, "observe_stale_batch", None)
    if observer is None:
        return False
    observer(groups=groups, policy_step=policy_step, reason=reason)
    return True


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


def _groups_sample_dollar_seconds(groups: Sequence[TrajectoryGroup]) -> float:
    return sum(
        _trajectory_sample_dollar_seconds(trajectory)
        for group in groups
        for trajectory in group.trajectories
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
