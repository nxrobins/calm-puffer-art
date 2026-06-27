from __future__ import annotations

import json
import time
from collections.abc import Sequence

from .actions import (
    ActionCodec,
    ChunkActionCodec,
    TokenActionCodec,
    action_codec_key,
)
from .scheduler import ObjectiveScheduler, scheduling_action_key
from .types import Scenario, Trajectory


def run_scheduler_scalability_profile(
    *,
    scenario_count: int = 4,
    codec_count: int = 3,
    cadence_values: Sequence[int] = (1, 2),
    lag_values: Sequence[int] = (1, 2),
    actor_counts: Sequence[int] = (1, 4),
    admission_delay_ms_values: Sequence[int] = (0, 25),
    action_space_count: int = 2,
    observations_per_tuple: int = 1,
    selector_trials: int = 16,
) -> dict[str, float]:
    """Profile scheduler state growth as the joint action lattice expands.

    This is a deterministic local readiness probe, not a replacement for a real
    training benchmark. It answers the scalability question the local scaffold
    can answer today: how many scheduler keys, metrics, and checkpoint bytes are
    created as scenario/action/runtime/action-space combinations grow.
    """

    scenario_count = _positive_int(scenario_count, "scenario_count")
    codec_count = _positive_int(codec_count, "codec_count")
    action_space_count = _positive_int(action_space_count, "action_space_count")
    observations_per_tuple = _positive_int(
        observations_per_tuple,
        "observations_per_tuple",
    )
    selector_trials = max(0, int(selector_trials))

    cadences = _positive_values(cadence_values, "cadence_values")
    lags = _non_negative_values(lag_values, "lag_values")
    actors = _positive_values(actor_counts, "actor_counts")
    admission_delays = _non_negative_values(
        admission_delay_ms_values,
        "admission_delay_ms_values",
    )

    scenarios = tuple(Scenario(id=f"task_{index}") for index in range(scenario_count))
    codecs = _synthetic_codecs(codec_count)
    action_space_keys = tuple(
        f"ladder_{index}" for index in range(action_space_count)
    )

    scheduler = ObjectiveScheduler(
        min_train_batch_groups=min(cadences),
        max_train_batch_groups=max(cadences),
        min_policy_lag=min(lags),
        max_policy_lag=max(lags),
        min_actor_count=min(actors),
        max_actor_count=max(actors),
        ema_alpha=1.0,
        exploration_bonus=0.0,
        control_exploration_bonus=0.0,
        rollout_cadence_lag_control_weight=1.0,
        max_rollout_admission_delay_s=max(admission_delays) / 1000.0,
    )

    populate_started = time.perf_counter()
    observations = 0
    for repeat in range(observations_per_tuple):
        for scenario_index, scenario in enumerate(scenarios):
            for codec_index, codec in enumerate(codecs):
                arm_id = f"{scenario.id}|{action_codec_key(codec)}"
                for action_space_index, action_space_key in enumerate(
                    action_space_keys
                ):
                    for cadence in cadences:
                        for lag in lags:
                            for actor_count in actors:
                                for admission_delay_ms in admission_delays:
                                    scheduler.observe_rollout(
                                        _synthetic_trajectory(
                                            scenario=scenario,
                                            codec=codec,
                                            arm_id=arm_id,
                                            repeat=repeat,
                                            scenario_index=scenario_index,
                                            codec_index=codec_index,
                                            action_space_index=action_space_index,
                                            cadence=cadence,
                                            lag=lag,
                                            actor_count=actor_count,
                                            admission_delay_ms=admission_delay_ms,
                                            action_space_key=action_space_key,
                                        ),
                                        accepted=True,
                                        dollar_seconds=_synthetic_cost(
                                            scenario_index=scenario_index,
                                            codec_index=codec_index,
                                            cadence=cadence,
                                            lag=lag,
                                            actor_count=actor_count,
                                            admission_delay_ms=admission_delay_ms,
                                        ),
                                    )
                                    observations += 1
    populate_seconds = _elapsed_since(populate_started)

    select_started = time.perf_counter()
    for trial in range(selector_trials):
        decision = scheduler.select_rollout(
            scenarios=scenarios,
            action_codecs=codecs,
            actor_id=trial % max(actors),
            policy_step=trial,
            trajectory_queue_pressure=0.25,
            train_queue_pressure=0.5,
            configured_train_batch_groups=cadences[0],
            configured_max_policy_lag=lags[0],
            active_actor_count=actors[-1],
            rollout_admission_delay_ms=admission_delays[0],
            action_space_key=action_space_keys[trial % len(action_space_keys)],
        )
        scheduler.cancel_rollout_decision(decision)
    select_seconds = _elapsed_since(select_started)

    metrics_started = time.perf_counter()
    metrics = scheduler.metrics()
    metrics_seconds = _elapsed_since(metrics_started)

    state_started = time.perf_counter()
    state = scheduler.state_dict()
    state_dict_seconds = _elapsed_since(state_started)

    json_started = time.perf_counter()
    state_json = json.dumps(state, sort_keys=True, separators=(",", ":"))
    state_json_seconds = _elapsed_since(json_started)

    joint_action_keys = len(state.get("joint_action_controls", {}))
    runtime_control_contexts = state.get("runtime_control_contexts", {})
    runtime_control_keys = _count_runtime_control_keys(runtime_control_contexts)
    global_runtime_control_keys = sum(
        len(state.get(key, {}))
        for key in (
            "cadence_controls",
            "lag_controls",
            "actor_count_controls",
            "admission_controls",
        )
    )
    state_json_bytes = len(state_json.encode("utf-8"))
    expected_joint_action_keys = (
        len(scenarios)
        * len(codecs)
        * len(action_space_keys)
        * len(cadences)
        * len(lags)
        * len(actors)
        * len(admission_delays)
    )

    return {
        "scalability/scenarios": float(len(scenarios)),
        "scalability/codecs": float(len(codecs)),
        "scalability/action_spaces": float(len(action_space_keys)),
        "scalability/cadence_values": float(len(cadences)),
        "scalability/policy_lag_values": float(len(lags)),
        "scalability/actor_count_values": float(len(actors)),
        "scalability/admission_delay_values": float(len(admission_delays)),
        "scalability/observations": float(observations),
        "scalability/arms": float(len(state.get("arms", {}))),
        "scalability/expected_joint_action_keys": float(
            expected_joint_action_keys
        ),
        "scalability/joint_action_keys": float(joint_action_keys),
        "scalability/runtime_control_contexts": float(
            len(runtime_control_contexts)
        ),
        "scalability/runtime_control_keys": float(runtime_control_keys),
        "scalability/global_runtime_control_keys": float(
            global_runtime_control_keys
        ),
        "scalability/metrics_count": float(len(metrics)),
        "scalability/state_json_bytes": float(state_json_bytes),
        "scalability/state_bytes_per_joint_action_key": (
            state_json_bytes / joint_action_keys if joint_action_keys else 0.0
        ),
        "scalability/metrics_per_joint_action_key": (
            len(metrics) / joint_action_keys if joint_action_keys else 0.0
        ),
        "scalability/populate_seconds": populate_seconds,
        "scalability/metrics_seconds": metrics_seconds,
        "scalability/state_dict_seconds": state_dict_seconds,
        "scalability/state_json_seconds": state_json_seconds,
        "scalability/select_trials": float(selector_trials),
        "scalability/select_seconds": select_seconds,
        "scalability/select_decisions_per_second": (
            selector_trials / select_seconds if selector_trials else 0.0
        ),
    }


def _synthetic_codecs(count: int) -> tuple[ActionCodec, ...]:
    codecs: list[ActionCodec] = [TokenActionCodec()]
    for chunk_size in range(2, count + 1):
        codecs.append(ChunkActionCodec(chunk_size=chunk_size))
    return tuple(codecs[:count])


def _synthetic_trajectory(
    *,
    scenario: Scenario,
    codec: ActionCodec,
    arm_id: str,
    repeat: int,
    scenario_index: int,
    codec_index: int,
    action_space_index: int,
    cadence: int,
    lag: int,
    actor_count: int,
    admission_delay_ms: int,
    action_space_key: str,
) -> Trajectory:
    joint_action_key = scheduling_action_key(
        arm_id=arm_id,
        target_train_batch_groups=cadence,
        max_policy_lag=lag,
        active_actor_count=actor_count,
        admission_delay_ms=admission_delay_ms,
        action_space_key=action_space_key,
    )
    reward = (
        1.0
        + scenario_index * 0.05
        + codec_index * 0.03
        + action_space_index * 0.02
        + repeat * 0.01
    )
    cost = _synthetic_cost(
        scenario_index=scenario_index,
        codec_index=codec_index,
        cadence=cadence,
        lag=lag,
        actor_count=actor_count,
        admission_delay_ms=admission_delay_ms,
    )
    return Trajectory(
        scenario_id=scenario.id,
        policy_step=repeat,
        messages=[],
        actions=codec.encode("alpha beta gamma delta epsilon zeta"),
        reward=reward,
        metrics={"rollout/dollar_seconds": cost},
        metadata={
            "scheduler/arm_id": arm_id,
            "scheduler/active_target_train_batch_groups": cadence,
            "scheduler/active_max_policy_lag": lag,
            "scheduler/active_actor_count": actor_count,
            "scheduler/active_rollout_admission_delay_ms": admission_delay_ms,
            "scheduler/action_space_key": action_space_key,
            "scheduler/joint_action_key": joint_action_key,
        },
        duration_s=cost,
    )


def _synthetic_cost(
    *,
    scenario_index: int,
    codec_index: int,
    cadence: int,
    lag: int,
    actor_count: int,
    admission_delay_ms: int,
) -> float:
    return (
        1.0
        + scenario_index * 0.01
        + codec_index * 0.02
        + cadence * 0.001
        + lag * 0.002
        + actor_count * 0.0005
        + admission_delay_ms * 0.0001
    )


def _count_runtime_control_keys(runtime_control_contexts: object) -> int:
    if not isinstance(runtime_control_contexts, dict):
        return 0
    count = 0
    for families in runtime_control_contexts.values():
        if not isinstance(families, dict):
            continue
        for controls in families.values():
            if isinstance(controls, dict):
                count += len(controls)
    return count


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_values(values: Sequence[int], name: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed or parsed[0] <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return parsed


def _non_negative_values(values: Sequence[int], name: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed or parsed[0] < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return parsed


def _elapsed_since(started: float) -> float:
    return max(time.perf_counter() - started, 1e-12)
