from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Protocol, Sequence

from .actions import ActionCodec
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
    pulls: int = 0
    accepted: int = 0
    unsafe: int = 0
    train_updates: int = 0
    stale_updates: int = 0
    stale_batches: int = 0
    stale_trajectories: int = 0
    reward_ema: float = 0.0
    effective_reward_ema: float = 0.0
    action_quality_ema: float = 1.0
    reward_efficiency_ema: float = 0.0
    marginal_objective_ema: float = 0.0
    policy_improvement_objective_ema: float = 0.0
    dollar_seconds_ema: float = 0.0
    total_reward: float = 0.0
    total_effective_reward: float = 0.0
    total_positive_improvement: float = 0.0
    total_policy_improvement_objective: float = 0.0
    total_stale_penalty_objective: float = 0.0
    total_dollar_seconds: float = 0.0
    rollout_dollar_seconds: float = 0.0
    queue_wait_dollar_seconds: float = 0.0
    stale_experience: float = 0.0


@dataclass
class ControlStats:
    decisions: int = 0
    train_updates: int = 0
    stale_updates: int = 0
    objective_ema: float = 0.0
    total_objective: float = 0.0
    total_stale_penalty_objective: float = 0.0
    stale_experience: float = 0.0


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
        ema_alpha: float = 0.25,
        exploration_bonus: float = 0.2,
        objective_threshold: float = 1e-6,
        unsafe_penalty: float = 1.0,
        rollout_objective_weight: float = 1.0,
        train_objective_weight: float = 1.0,
        reward_efficiency_weight: float = 0.0,
        stale_penalty_weight: float = 1.0,
        min_train_steps: int = 1,
        roi_patience: int | None = None,
        min_train_objective: float = 0.0,
    ) -> None:
        if min_train_batch_groups <= 0:
            raise ValueError("min_train_batch_groups must be positive")
        if min_policy_lag < 0:
            raise ValueError("min_policy_lag must be non-negative")
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
        if min_train_steps < 0:
            raise ValueError("min_train_steps must be non-negative")
        if roi_patience is not None and roi_patience <= 0:
            raise ValueError("roi_patience must be positive when set")
        self.min_train_batch_groups = min_train_batch_groups
        self.max_train_batch_groups = max_train_batch_groups
        self.min_policy_lag = min_policy_lag
        self.max_policy_lag_limit = max_policy_lag
        self.ema_alpha = ema_alpha
        self.exploration_bonus = exploration_bonus
        self.objective_threshold = objective_threshold
        self.unsafe_penalty = unsafe_penalty
        self.rollout_objective_weight = rollout_objective_weight
        self.train_objective_weight = train_objective_weight
        self.reward_efficiency_weight = reward_efficiency_weight
        self.stale_penalty_weight = stale_penalty_weight
        self.min_train_steps = min_train_steps
        self.roi_patience = roi_patience
        self.min_train_objective = min_train_objective
        self._arms: dict[str, ArmStats] = {}
        self._total_pulls = 0
        self._global_objective_ema = 0.0
        self._train_reward_ema = 0.0
        self._train_objective_ema = 0.0
        self._last_train_reward = 0.0
        self._last_train_objective = 0.0
        self._last_train_reward_improvement = 0.0
        self._last_train_experience_count = 0.0
        self._last_train_reward_improving_experience = 0.0
        self._last_stale_penalty_objective = 0.0
        self._last_stale_experience_count = 0.0
        self._last_stale_policy_step = -1
        self._last_stale_reason = ""
        self._last_decision: SchedulerDecision | None = None
        self._last_decision_snapshot: dict[str, Any] | None = None
        self._last_train_batch_priority = 0.0
        self._global_action_quality_ema = 1.0
        self._low_roi_train_steps = 0
        self._stop_recommended = False
        self._rollout_dollar_seconds = 0.0
        self._queue_wait_dollar_seconds = 0.0
        self._train_dollar_seconds = 0.0
        self._stale_batches = 0
        self._stale_trajectories = 0
        self._stale_experience = 0.0
        self._cadence_controls: dict[int, ControlStats] = {}
        self._lag_controls: dict[int, ControlStats] = {}

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
        arm_id, scenario, codec = max(arms, key=lambda arm: self._score_arm(arm[0]))
        selected_stats = self._arms[arm_id]
        objective_score = self._arm_value(selected_stats)
        exploration_score = self._exploration_value(selected_stats)
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
                "score": self._score_arm(arm_id),
                "objective_score": objective_score,
                "exploration_score": exploration_score,
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
        feedback_target = self._best_control_value(
            self._cadence_controls,
            candidates,
        )
        if self._has_positive_objective_signal() and feedback_target is not None:
            return self._record_control_decision(
                self._cadence_controls,
                feedback_target,
            )
        if self._has_positive_objective_signal():
            return self._record_control_decision(
                self._cadence_controls,
                self.min_train_batch_groups,
            )
        if train_queue_pressure >= 0.75:
            return self._record_control_decision(self._cadence_controls, upper)
        if pending_groups >= upper:
            return self._record_control_decision(self._cadence_controls, upper)
        return self._record_control_decision(self._cadence_controls, configured)

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
        feedback_lag = self._best_control_value(self._lag_controls, candidates)
        if self._has_positive_objective_signal() and feedback_lag is not None:
            return self._record_control_decision(
                self._lag_controls,
                feedback_lag,
            )
        if train_queue_pressure >= 0.75:
            return self._record_control_decision(
                self._lag_controls,
                self.min_policy_lag,
            )
        if self._has_positive_objective_signal():
            return self._record_control_decision(
                self._lag_controls,
                self.min_policy_lag,
            )
        return self._record_control_decision(self._lag_controls, configured)

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
        rollout_cost = max(dollar_seconds, 1e-12)
        queue_wait_cost = max(0.0, queue_wait_dollar_seconds)
        cost = max(rollout_cost + queue_wait_cost, 1e-12)
        self._rollout_dollar_seconds += rollout_cost
        self._queue_wait_dollar_seconds += queue_wait_cost
        quality = action_quality(trajectory)
        reward = trajectory.reward if accepted else 0.0
        effective_reward = reward * quality
        previous_reward = stats.effective_reward_ema if stats.pulls else 0.0
        positive_improvement = max(0.0, effective_reward - previous_reward)
        marginal_objective = positive_improvement / cost
        reward_efficiency = max(0.0, effective_reward) / cost

        stats.pulls += 1
        self._total_pulls += 1
        if accepted:
            stats.accepted += 1
        if quality <= 0.0:
            stats.unsafe += 1
        stats.total_reward += reward
        stats.total_effective_reward += effective_reward
        stats.total_positive_improvement += positive_improvement
        stats.total_dollar_seconds += cost
        stats.rollout_dollar_seconds += rollout_cost
        stats.queue_wait_dollar_seconds += queue_wait_cost
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
        stats.reward_efficiency_ema = self._ema(
            stats.reward_efficiency_ema,
            reward_efficiency,
            stats.pulls,
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
        improvement = max(0.0, reward - self._last_train_reward)
        experience_count = _useful_experience_count(groups)
        reward_improving_experience = improvement * experience_count
        objective = reward_improving_experience / cost
        self._last_train_reward = reward
        self._last_train_objective = objective
        self._last_train_reward_improvement = improvement
        self._last_train_experience_count = experience_count
        self._last_train_reward_improving_experience = reward_improving_experience
        self._credit_objective_to_arms(groups, objective, stale_feedback=False)
        self._credit_objective_to_controls(groups, objective, stale_feedback=False)
        if objective <= self.min_train_objective:
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
        penalty_objective = -(
            self.stale_penalty_weight
            * stale_experience
            / max(stale_cost, 1e-12)
        )

        self._stale_batches += 1
        self._stale_trajectories += stale_trajectories
        self._stale_experience += stale_experience
        self._last_stale_penalty_objective = penalty_objective
        self._last_stale_experience_count = stale_experience
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
                if stats is None or stats.accepted == 0:
                    arm_values.append(
                        max(0.0, trajectory.reward)
                        * action_quality(trajectory)
                    )
                else:
                    arm_values.append(self._arm_value(stats))
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
        priority = (
            arm_component + self.reward_efficiency_weight * raw_reward_component
        )
        self._last_train_batch_priority = priority
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
                "ema_alpha": self.ema_alpha,
                "exploration_bonus": self.exploration_bonus,
                "objective_threshold": self.objective_threshold,
                "unsafe_penalty": self.unsafe_penalty,
                "rollout_objective_weight": self.rollout_objective_weight,
                "train_objective_weight": self.train_objective_weight,
                "reward_efficiency_weight": self.reward_efficiency_weight,
                "stale_penalty_weight": self.stale_penalty_weight,
                "min_train_steps": self.min_train_steps,
                "roi_patience": self.roi_patience,
                "min_train_objective": self.min_train_objective,
            },
            "learning_state": {
                "total_pulls": self._total_pulls,
                "global_objective_ema": self._global_objective_ema,
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
                "last_stale_penalty_objective": (
                    self._last_stale_penalty_objective
                ),
                "last_stale_experience_count": self._last_stale_experience_count,
                "last_stale_policy_step": self._last_stale_policy_step,
                "last_stale_reason": self._last_stale_reason,
                "last_train_batch_priority": self._last_train_batch_priority,
                "global_action_quality_ema": self._global_action_quality_ema,
                "low_roi_train_steps": self._low_roi_train_steps,
                "stop_recommended": self._stop_recommended,
                "rollout_dollar_seconds": self._rollout_dollar_seconds,
                "queue_wait_dollar_seconds": self._queue_wait_dollar_seconds,
                "train_dollar_seconds": self._train_dollar_seconds,
                "stale_batches": self._stale_batches,
                "stale_trajectories": self._stale_trajectories,
                "stale_experience": self._stale_experience,
            },
            "arms": {
                arm_id: _dataclass_state(stats)
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

        learning_state = _mapping_state(state.get("learning_state"))
        total_pulls_default = sum(stats.pulls for stats in self._arms.values())
        self._total_pulls = _state_int(
            learning_state.get("total_pulls"),
            total_pulls_default,
        )
        self._global_objective_ema = _state_float(
            learning_state.get("global_objective_ema"),
            self._global_objective_ema,
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
        self._last_stale_penalty_objective = _state_float(
            learning_state.get("last_stale_penalty_objective"),
            self._last_stale_penalty_objective,
        )
        self._last_stale_experience_count = _state_float(
            learning_state.get("last_stale_experience_count"),
            self._last_stale_experience_count,
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
        self._last_decision = None
        self._last_decision_snapshot = _decision_state_from_mapping(
            state.get("last_decision")
        )

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "scheduler/total_rollout_decisions": float(self._total_pulls),
            "scheduler/global_marginal_objective_ema": self._global_objective_ema,
            "scheduler/global_action_quality_ema": self._global_action_quality_ema,
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
            "scheduler/stale_batches": float(self._stale_batches),
            "scheduler/stale_trajectories": float(self._stale_trajectories),
            "scheduler/stale_experience": self._stale_experience,
            "scheduler/stale_last_penalty_objective": (
                self._last_stale_penalty_objective
            ),
            "scheduler/stale_last_experience_count": (
                self._last_stale_experience_count
            ),
            "scheduler/stale_last_policy_step": float(self._last_stale_policy_step),
            "scheduler/last_train_batch_priority": self._last_train_batch_priority,
            "scheduler/low_roi_train_steps": float(self._low_roi_train_steps),
            "scheduler/stop_recommended": 1.0 if self._stop_recommended else 0.0,
            "scheduler/weights/rollout_objective": self.rollout_objective_weight,
            "scheduler/weights/train_objective": self.train_objective_weight,
            "scheduler/weights/reward_efficiency": self.reward_efficiency_weight,
            "scheduler/weights/stale_penalty": self.stale_penalty_weight,
            "scheduler/weights/unsafe_penalty": self.unsafe_penalty,
            "scheduler/costs/rollout_dollar_seconds": self._rollout_dollar_seconds,
            "scheduler/costs/queue_wait_dollar_seconds": (
                self._queue_wait_dollar_seconds
            ),
            "scheduler/costs/train_dollar_seconds": self._train_dollar_seconds,
            "scheduler/costs/total_dollar_seconds": (
                self._rollout_dollar_seconds
                + self._queue_wait_dollar_seconds
                + self._train_dollar_seconds
            ),
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
        for arm_id, stats in self._arms.items():
            prefix = f"scheduler/arm/{_safe_metric_key(arm_id)}"
            metrics[f"{prefix}/pulls"] = float(stats.pulls)
            metrics[f"{prefix}/accepted"] = float(stats.accepted)
            metrics[f"{prefix}/unsafe"] = float(stats.unsafe)
            metrics[f"{prefix}/unsafe_rate"] = (
                stats.unsafe / stats.pulls if stats.pulls else 0.0
            )
            metrics[f"{prefix}/reward_ema"] = stats.reward_ema
            metrics[f"{prefix}/effective_reward_ema"] = stats.effective_reward_ema
            metrics[f"{prefix}/action_quality_ema"] = stats.action_quality_ema
            metrics[f"{prefix}/marginal_objective_ema"] = (
                stats.marginal_objective_ema
            )
            metrics[f"{prefix}/policy_improvement_objective_ema"] = (
                stats.policy_improvement_objective_ema
            )
            metrics[f"{prefix}/reward_efficiency_ema"] = stats.reward_efficiency_ema
            metrics[f"{prefix}/objective_score"] = self._arm_value(stats)
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
            metrics[f"{prefix}/rollout_dollar_seconds"] = stats.rollout_dollar_seconds
            metrics[f"{prefix}/queue_wait_dollar_seconds"] = (
                stats.queue_wait_dollar_seconds
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
            metrics[f"{prefix}/total_positive_improvement"] = (
                stats.total_positive_improvement
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
        for value, stats in sorted(self._cadence_controls.items()):
            prefix = f"scheduler/control/cadence_{value}"
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _control_feedback_updates(stats)
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
            metrics[f"{prefix}/total_objective"] = stats.total_objective
            metrics[f"{prefix}/total_stale_penalty_objective"] = (
                stats.total_stale_penalty_objective
            )
            metrics[f"{prefix}/stale_experience"] = stats.stale_experience
        for value, stats in sorted(self._lag_controls.items()):
            prefix = f"scheduler/control/policy_lag_{value}"
            metrics[f"{prefix}/decisions"] = float(stats.decisions)
            metrics[f"{prefix}/train_updates"] = float(stats.train_updates)
            metrics[f"{prefix}/stale_updates"] = float(stats.stale_updates)
            metrics[f"{prefix}/feedback_updates"] = float(
                _control_feedback_updates(stats)
            )
            metrics[f"{prefix}/objective_ema"] = stats.objective_ema
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

    def _score_arm(self, arm_id: str) -> float:
        stats = self._arms.setdefault(arm_id, ArmStats())
        if stats.pulls == 0:
            # Explore each arm once in deterministic candidate order.
            return float("inf") - len(self._arms) * 1e-9
        exploitation = self._arm_value(stats)
        exploration = self._exploration_value(stats)
        return exploitation + exploration

    def _exploration_value(self, stats: ArmStats) -> float:
        if stats.pulls == 0:
            return 0.0
        return self.exploration_bonus * math.sqrt(
            math.log(self._total_pulls + 1) / stats.pulls
        )

    def _has_unaccepted_known_arm(self) -> bool:
        return any(stats.accepted == 0 for stats in self._arms.values())

    def _has_positive_objective_signal(self) -> bool:
        return (
            self._global_objective_ema > self.objective_threshold
            or self._train_objective_ema > self.objective_threshold
        )

    def _arm_value(self, stats: ArmStats) -> float:
        unsafe_rate = stats.unsafe / stats.pulls if stats.pulls else 0.0
        objective = (
            self.train_objective_weight * stats.policy_improvement_objective_ema
            + self.rollout_objective_weight * stats.marginal_objective_ema
            + self.reward_efficiency_weight * stats.reward_efficiency_ema
        )
        return max(0.0, stats.action_quality_ema) * (
            objective - self.unsafe_penalty * unsafe_rate
        )

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

    def _control_candidates(
        self,
        *,
        min_value: int,
        configured: int,
        upper: int,
    ) -> tuple[int, ...]:
        return tuple(sorted({min_value, configured, upper}))

    def _best_control_value(
        self,
        controls: dict[int, ControlStats],
        candidates: Sequence[int],
    ) -> int | None:
        observed = [
            (value, controls[value])
            for value in candidates
            if value in controls
            and _control_feedback_updates(controls[value]) > 0
            and controls[value].objective_ema > self.objective_threshold
        ]
        if not observed:
            return None
        return max(
            observed,
            key=lambda item: (item[1].objective_ema, item[1].total_objective),
        )[0]

    def _record_control_decision(
        self,
        controls: dict[int, ControlStats],
        value: int,
    ) -> int:
        controls.setdefault(value, ControlStats()).decisions += 1
        return value

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


def _arm_feedback_updates(stats: ArmStats) -> int:
    return stats.train_updates + stats.stale_updates


def _control_feedback_updates(stats: ControlStats) -> int:
    return stats.train_updates + stats.stale_updates


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


def _dataclass_state(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _mapping_state(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _arm_stats_from_state(value: Any) -> ArmStats:
    state = _mapping_state(value)
    default = ArmStats()
    return ArmStats(
        pulls=_state_int(state.get("pulls"), default.pulls),
        accepted=_state_int(state.get("accepted"), default.accepted),
        unsafe=_state_int(state.get("unsafe"), default.unsafe),
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
        dollar_seconds_ema=_state_float(
            state.get("dollar_seconds_ema"),
            default.dollar_seconds_ema,
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
    return (rollout_cost or 0.0) + (queue_cost or 0.0)


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
