from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


TELEMETRY_SCHEMA_VERSION = 1
CostProvenance = str


@dataclass(frozen=True)
class PricingConfig:
    """Optional monetary rates used without conflating missing prices with zero."""

    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None
    trainer_usd_per_hour: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("input_usd_per_million_tokens", self.input_usd_per_million_tokens),
            ("output_usd_per_million_tokens", self.output_usd_per_million_tokens),
            ("trainer_usd_per_hour", self.trainer_usd_per_hour),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def inference_priced(self) -> bool:
        return (
            self.input_usd_per_million_tokens is not None
            and self.output_usd_per_million_tokens is not None
        )

    @property
    def trainer_priced(self) -> bool:
        return self.trainer_usd_per_hour is not None

    def inference_cost(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        reported_usd: float | None = None,
        reported_provenance: CostProvenance = "measured",
    ) -> tuple[float | None, CostProvenance]:
        if reported_usd is not None:
            _require_non_negative_finite("reported_usd", reported_usd)
            return reported_usd, reported_provenance
        if not self.inference_priced:
            return None, "unavailable"
        assert self.input_usd_per_million_tokens is not None
        assert self.output_usd_per_million_tokens is not None
        value = (
            prompt_tokens * self.input_usd_per_million_tokens
            + completion_tokens * self.output_usd_per_million_tokens
        ) / 1_000_000.0
        return value, "estimated_from_token_rates"

    def trainer_cost(
        self,
        *,
        duration_s: float,
        reported_usd: float | None = None,
        reported_provenance: CostProvenance = "measured",
    ) -> tuple[float | None, CostProvenance]:
        if reported_usd is not None:
            _require_non_negative_finite("reported_usd", reported_usd)
            return reported_usd, reported_provenance
        if not self.trainer_priced:
            return None, "unavailable"
        assert self.trainer_usd_per_hour is not None
        return duration_s * self.trainer_usd_per_hour / 3600.0, (
            "estimated_from_wall_clock_rate"
        )

    def as_dict(self) -> dict[str, float | bool | None]:
        return {
            "input_usd_per_million_tokens": self.input_usd_per_million_tokens,
            "output_usd_per_million_tokens": self.output_usd_per_million_tokens,
            "trainer_usd_per_hour": self.trainer_usd_per_hour,
            "inference_priced": self.inference_priced,
            "trainer_priced": self.trainer_priced,
        }


class TelemetryLedger:
    """Append-only experiment evidence plus dependency-free monitoring summaries."""

    def __init__(
        self,
        *,
        run_id: str,
        pricing: PricingConfig | None = None,
        path: Path | None = None,
        echo_to_stderr: bool = False,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        self.run_id = run_id
        self.pricing = pricing or PricingConfig()
        self.path = path
        self.echo_to_stderr = echo_to_stderr
        self.started_at = time.perf_counter()
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            for event in load_telemetry_events(self.path):
                if event.get("run_id") == self.run_id:
                    self._events.append(event)
                    self._sequence = max(
                        self._sequence,
                        int(event.get("sequence", 0) or 0),
                    )

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    def emit(
        self,
        event: str,
        *,
        dimensions: Mapping[str, Any] | None = None,
        metrics: Mapping[str, int | float | bool | None] | None = None,
        attributes: Mapping[str, Any] | None = None,
        provenance: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        if not event:
            raise ValueError("event must not be empty")
        clean_metrics = _validated_metrics(metrics or {})
        payload: dict[str, Any]
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "sequence": self._sequence,
                "event": event,
                "run_id": self.run_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "timestamp_unix_s": round(time.time(), 6),
                "elapsed_s": round(
                    max(0.0, time.perf_counter() - self.started_at),
                    6,
                ),
                "dimensions": dict(dimensions or {}),
                "metrics": clean_metrics,
                "attributes": dict(attributes or {}),
                "provenance": dict(provenance or {}),
            }
            line = json.dumps(payload, sort_keys=True, allow_nan=False)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            if self.echo_to_stderr:
                import sys

                print(line, file=sys.stderr, flush=True)
            self._events.append(payload)
        return payload

    def record_inference(
        self,
        *,
        condition: str,
        seed: int,
        phase: str,
        task_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_s: float,
        attempts: int,
        reward: float,
        exact: bool,
        parsed: bool,
        stratum: str | None = None,
        policy_step: int | None = None,
        reported_api_usd: float | None = None,
        reported_cost_provenance: CostProvenance = "measured",
        attributes: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
            ("attempts", attempts),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        _require_non_negative_finite("latency_s", latency_s)
        _require_finite("reward", reward)
        api_usd, api_provenance = self.pricing.inference_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reported_usd=reported_api_usd,
            reported_provenance=reported_cost_provenance,
        )
        dimensions = {
            "condition": condition,
            "seed": seed,
            "phase": phase,
            "task_id": task_id,
        }
        if stratum is not None:
            dimensions["stratum"] = stratum
        if policy_step is not None:
            dimensions["policy_step"] = policy_step
        return self.emit(
            "inference_completed",
            dimensions=dimensions,
            metrics={
                "requests": 1,
                "attempts": attempts,
                "retries": max(0, attempts - 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_s": latency_s,
                "reward": reward,
                "exact": exact,
                "parsed": parsed,
                "api_usd": api_usd,
                "token_proxy_millions": total_tokens / 1_000_000.0,
            },
            attributes=attributes,
            provenance={
                "api_usd": api_provenance,
                "token_proxy_millions": "proxy",
                "reward": "verifier",
            },
        )

    def record_training_attempt(
        self,
        *,
        condition: str,
        seed: int,
        attempt: int,
        status: str,
        duration_s: float,
        initial_step: int,
        observed_step: int,
        trainer_metrics: Mapping[str, int | float | bool | None] | None = None,
        reported_trainer_usd: float | None = None,
        reported_cost_provenance: CostProvenance = "measured",
        error_type: str | None = None,
        error: str | None = None,
        artifact_name: str | None = None,
    ) -> Mapping[str, Any]:
        _require_non_negative_finite("duration_s", duration_s)
        trainer_usd, trainer_provenance = self.pricing.trainer_cost(
            duration_s=duration_s,
            reported_usd=reported_trainer_usd,
            reported_provenance=reported_cost_provenance,
        )
        metrics = {
            **dict(trainer_metrics or {}),
            "attempts": 1,
            "duration_s": duration_s,
            "trainer_usd": trainer_usd,
            "checkpoint_advance": max(0, observed_step - initial_step),
        }
        return self.emit(
            "training_attempt_finished",
            dimensions={
                "condition": condition,
                "seed": seed,
                "attempt": attempt,
                "status": status,
                "initial_step": initial_step,
                "observed_step": observed_step,
            },
            metrics=metrics,
            attributes={
                "error_type": error_type,
                "error": error,
                "artifact_name": artifact_name,
            },
            provenance={"trainer_usd": trainer_provenance},
        )

    def record_external_cost(
        self,
        *,
        condition: str,
        seed: int,
        phase: str,
        category: str,
        amount_usd: float | None,
        provenance: CostProvenance,
        proxy_value: float | None = None,
        proxy_unit: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if amount_usd is not None:
            _require_non_negative_finite("amount_usd", amount_usd)
        if proxy_value is not None:
            _require_non_negative_finite("proxy_value", proxy_value)
        if not category:
            raise ValueError("category must not be empty")
        return self.emit(
            "external_cost_observed",
            dimensions={
                "condition": condition,
                "seed": seed,
                "phase": phase,
                "category": category,
            },
            metrics={"amount_usd": amount_usd, "proxy_value": proxy_value},
            attributes={"proxy_unit": proxy_unit, **dict(attributes or {})},
            provenance={
                "amount_usd": provenance,
                "proxy_value": "proxy" if proxy_value is not None else "unavailable",
            },
        )

    def record_scheduler_decision(
        self,
        *,
        condition: str,
        seed: int,
        train_step: int,
        group_index: int,
        selected_stratum: str,
        task_id: str,
        estimated_cost: float | None = None,
        cost_provenance: CostProvenance = "proxy",
        attributes: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if estimated_cost is not None:
            _require_non_negative_finite("estimated_cost", estimated_cost)
        return self.emit(
            "scheduler_decision",
            dimensions={
                "condition": condition,
                "seed": seed,
                "train_step": train_step,
                "group_index": group_index,
                "selected_stratum": selected_stratum,
                "task_id": task_id,
            },
            metrics={"decisions": 1, "estimated_cost": estimated_cost},
            attributes=attributes,
            provenance={"estimated_cost": cost_provenance},
        )

    def record_condition_finished(
        self,
        *,
        condition: str,
        seed: int,
        status: str,
        wall_s: float,
        error_type: str | None = None,
        error: str | None = None,
    ) -> Mapping[str, Any]:
        _require_non_negative_finite("wall_s", wall_s)
        return self.emit(
            "condition_finished",
            dimensions={
                "condition": condition,
                "seed": seed,
                "status": status,
            },
            metrics={"wall_s": wall_s},
            attributes={"error_type": error_type, "error": error},
        )

    def summary(
        self,
        *,
        expected_inference_requests: int | None = None,
        stale_after_s: float = 600.0,
    ) -> dict[str, Any]:
        return summarize_telemetry(
            self._events,
            pricing=self.pricing,
            path=self.path,
            expected_inference_requests=expected_inference_requests,
            stale_after_s=stale_after_s,
        )


def load_telemetry_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid telemetry JSON at {path}:{line_number}"
            ) from exc
        if int(payload.get("schema_version", 0) or 0) != TELEMETRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported telemetry schema at {path}:{line_number}"
            )
        events.append(payload)
    return events


def summarize_telemetry(
    events: Sequence[Mapping[str, Any]],
    *,
    pricing: PricingConfig | None = None,
    path: Path | None = None,
    expected_inference_requests: int | None = None,
    stale_after_s: float = 600.0,
) -> dict[str, Any]:
    _require_non_negative_finite("stale_after_s", stale_after_s)
    pricing = pricing or PricingConfig()
    events = _events_with_pricing(events, pricing)
    run_started_events = [
        event for event in events if event.get("event") == "run_started"
    ]
    run_finished = any(event.get("event") == "run_finished" for event in events)
    run_contract = (
        dict(run_started_events[-1].get("attributes", {}))
        if run_started_events
        else {}
    )
    if expected_inference_requests is None:
        contract_minimum = run_contract.get("minimum_inference_requests")
        contract_maximum = run_contract.get("maximum_inference_requests")
        if (
            isinstance(contract_minimum, int)
            and contract_minimum == contract_maximum
        ):
            expected_inference_requests = contract_minimum
    last_event_age_s = (
        max(
            0.0,
            time.time()
            - max(float(event.get("timestamp_unix_s", 0.0) or 0.0) for event in events),
        )
        if events
        else 0.0
    )
    inference_events = [
        event for event in events if event.get("event") == "inference_completed"
    ]
    training_events = [
        event for event in events if event.get("event") == "training_attempt_finished"
    ]
    training_started_events = [
        event for event in events if event.get("event") == "training_attempt_started"
    ]
    decision_events = [
        event for event in events if event.get("event") == "scheduler_decision"
    ]
    external_cost_events = [
        event for event in events if event.get("event") == "external_cost_observed"
    ]
    condition_started_events = [
        event for event in events if event.get("event") == "condition_started"
    ]
    condition_events = [
        event for event in events if event.get("event") == "condition_finished"
    ]
    conditions = sorted(
        {
            str(event.get("dimensions", {}).get("condition"))
            for event in events
            if event.get("dimensions", {}).get("condition") is not None
        }
    )
    condition_summaries: dict[str, Any] = {}
    for condition in conditions:
        summary = _condition_summary(
            condition,
            inference_events=inference_events,
            training_started_events=training_started_events,
            training_events=training_events,
            decision_events=decision_events,
            external_cost_events=external_cost_events,
            condition_events=condition_events,
        )
        seeds = sorted(
            {
                int(event.get("dimensions", {}).get("seed"))
                for event in events
                if event.get("dimensions", {}).get("condition") == condition
                and isinstance(event.get("dimensions", {}).get("seed"), int)
            }
        )
        summary["by_seed"] = {
            str(seed): _condition_summary(
                condition,
                seed=seed,
                inference_events=inference_events,
                training_started_events=training_started_events,
                training_events=training_events,
                decision_events=decision_events,
                external_cost_events=external_cost_events,
                condition_events=condition_events,
            )
            for seed in seeds
        }
        condition_summaries[condition] = summary
    inference_priced = sum(
        event.get("metrics", {}).get("api_usd") is not None
        for event in inference_events
    )
    trainer_priced = sum(
        event.get("metrics", {}).get("trainer_usd") is not None
        for event in training_events
    )
    coverage = {
        "run_ids": sorted({str(event.get("run_id")) for event in events}),
        "sequence_issues": _sequence_issues(events),
        "expected_inference_requests": expected_inference_requests,
        "recorded_inference_requests": len(inference_events),
        "request_coverage": _coverage_ratio(
            len(inference_events),
            expected_inference_requests,
        ),
        "token_usage_coverage": _field_coverage(
            inference_events,
            "metrics",
            "total_tokens",
        ),
        "latency_coverage": _field_coverage(
            inference_events,
            "metrics",
            "latency_s",
        ),
        "performance_coverage": min(
            _field_coverage(inference_events, "metrics", "reward"),
            _field_coverage(inference_events, "metrics", "exact"),
            _field_coverage(inference_events, "metrics", "parsed"),
        ),
        "task_identity_coverage": _field_coverage(
            inference_events,
            "dimensions",
            "task_id",
        ),
        "inference_pricing_coverage": (
            inference_priced / len(inference_events) if inference_events else 1.0
        ),
        "trainer_pricing_coverage": (
            trainer_priced / len(training_events)
            if training_events
            else 1.0
        ),
        "external_cost_pricing_coverage": (
            sum(
                event.get("metrics", {}).get("amount_usd") is not None
                for event in external_cost_events
            )
            / len(external_cost_events)
            if external_cost_events
            else 1.0
        ),
        "training_attempts_started": len(training_started_events),
        "training_attempts_finished": len(training_events),
        "unfinished_training_attempts": _unfinished_lifecycles(
            training_started_events,
            training_events,
            keys=("condition", "seed", "attempt", "initial_step"),
        ),
        "conditions_started": len(condition_started_events),
        "conditions_finished": len(condition_events),
        "unfinished_conditions": _unfinished_lifecycles(
            condition_started_events,
            condition_events,
            keys=("condition", "seed"),
        ),
        "expected_condition_runs": run_contract.get("expected_condition_runs"),
        "condition_run_coverage": _coverage_ratio(
            len(condition_events),
            _optional_int(run_contract.get("expected_condition_runs")),
        ),
        "expected_training_updates": run_contract.get("expected_training_updates"),
        "successful_training_updates": len(
            [
                event
                for event in training_events
                if str(event.get("dimensions", {}).get("status"))
                in {"completed", "recovered"}
            ]
        ),
        "training_update_coverage": _coverage_ratio(
            len(
                [
                    event
                    for event in training_events
                    if str(event.get("dimensions", {}).get("status"))
                    in {"completed", "recovered"}
                ]
            ),
            _optional_int(run_contract.get("expected_training_updates")),
        ),
        "expected_scheduler_decisions": run_contract.get(
            "expected_scheduler_decisions"
        ),
        "scheduler_decision_coverage": _coverage_ratio(
            len(decision_events),
            _optional_int(run_contract.get("expected_scheduler_decisions")),
        ),
        "run_finished": run_finished,
        "last_event_age_s": last_event_age_s,
        "stale_after_s": stale_after_s,
        "scheduler_decisions_recorded": len(decision_events),
        "intermediate_heldout_evaluations": sum(
            str(event.get("dimensions", {}).get("phase", "")).startswith(
                "heldout_checkpoint"
            )
            for event in inference_events
        ),
    }
    points, frontier_basis = _condition_points(condition_summaries)
    alerts = _monitoring_alerts(
        condition_summaries,
        coverage=coverage,
        expected_inference_requests=expected_inference_requests,
        run_finished=run_finished,
        stale=last_event_age_s >= stale_after_s,
    )
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "path": str(path.resolve()) if path is not None else None,
        "events": len(events),
        "pricing": pricing.as_dict(),
        "coverage": coverage,
        "conditions": condition_summaries,
        "cost_performance": {
            "basis": frontier_basis,
            "point_estimates_only": True,
            "points": points,
            "pareto_frontier": pareto_frontier(points),
            "pairwise": _pairwise_condition_comparisons(condition_summaries),
        },
        "alerts": alerts,
        "healthy": not any(alert["severity"] == "error" for alert in alerts),
    }


def _events_with_pricing(
    events: Sequence[Mapping[str, Any]],
    pricing: PricingConfig,
) -> list[dict[str, Any]]:
    repriced: list[dict[str, Any]] = []
    for source in events:
        event = dict(source)
        metrics = dict(source.get("metrics", {}))
        provenance = dict(source.get("provenance", {}))
        if event.get("event") == "inference_completed" and metrics.get(
            "api_usd"
        ) is None:
            api_usd, api_provenance = pricing.inference_cost(
                prompt_tokens=int(metrics.get("prompt_tokens", 0) or 0),
                completion_tokens=int(metrics.get("completion_tokens", 0) or 0),
            )
            metrics["api_usd"] = api_usd
            if api_usd is not None:
                provenance["api_usd"] = f"offline_{api_provenance}"
        if event.get("event") == "training_attempt_finished" and metrics.get(
            "trainer_usd"
        ) is None:
            trainer_usd, trainer_provenance = pricing.trainer_cost(
                duration_s=float(metrics.get("duration_s", 0.0) or 0.0),
            )
            metrics["trainer_usd"] = trainer_usd
            if trainer_usd is not None:
                provenance["trainer_usd"] = f"offline_{trainer_provenance}"
        event["metrics"] = metrics
        event["provenance"] = provenance
        repriced.append(event)
    return repriced


def pareto_frontier(
    points: Sequence[Mapping[str, Any]],
    *,
    cost_key: str = "cost",
    performance_key: str = "performance",
) -> list[dict[str, Any]]:
    eligible = [
        dict(point)
        for point in points
        if _finite_or_none(point.get(cost_key)) is not None
        and _finite_or_none(point.get(performance_key)) is not None
    ]
    frontier: list[dict[str, Any]] = []
    for point in eligible:
        cost = float(point[cost_key])
        performance = float(point[performance_key])
        dominated = any(
            float(other[cost_key]) <= cost
            and float(other[performance_key]) >= performance
            and (
                float(other[cost_key]) < cost
                or float(other[performance_key]) > performance
            )
            for other in eligible
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: float(point[cost_key]))


def minimum_cost_to_target(
    points: Sequence[Mapping[str, Any]],
    *,
    target: float,
    cost_key: str = "cost",
    performance_key: str = "performance",
) -> float | None:
    qualifying = [
        float(point[cost_key])
        for point in points
        if _finite_or_none(point.get(cost_key)) is not None
        and _finite_or_none(point.get(performance_key)) is not None
        and float(point[performance_key]) >= target
    ]
    return min(qualifying) if qualifying else None


def _condition_summary(
    condition: str,
    *,
    seed: int | None = None,
    inference_events: Sequence[Mapping[str, Any]],
    training_started_events: Sequence[Mapping[str, Any]],
    training_events: Sequence[Mapping[str, Any]],
    decision_events: Sequence[Mapping[str, Any]],
    external_cost_events: Sequence[Mapping[str, Any]],
    condition_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def matches(event: Mapping[str, Any]) -> bool:
        dimensions = event.get("dimensions", {})
        return dimensions.get("condition") == condition and (
            seed is None or dimensions.get("seed") == seed
        )

    inference = [
        event
        for event in inference_events
        if matches(event)
    ]
    training = [
        event
        for event in training_events
        if matches(event)
    ]
    training_started = [
        event
        for event in training_started_events
        if matches(event)
    ]
    decisions = [
        event
        for event in decision_events
        if matches(event)
    ]
    external_costs = [
        event
        for event in external_cost_events
        if matches(event)
    ]
    finishes = [
        event
        for event in condition_events
        if matches(event)
    ]
    by_phase: dict[str, Any] = {}
    phases = sorted(
        {
            str(event.get("dimensions", {}).get("phase"))
            for event in inference
        }
    )
    for phase in phases:
        by_phase[phase] = _inference_rollup(
            [
                event
                for event in inference
                if event.get("dimensions", {}).get("phase") == phase
            ]
        )
    before = _combine_phase_rollups(by_phase, prefix="heldout_before")
    after = _combine_phase_rollups(by_phase, prefix="heldout_after")
    inference_rollup = _inference_rollup(inference)
    successful_training = [
        event
        for event in training
        if str(event.get("dimensions", {}).get("status"))
        in {"completed", "recovered"}
    ]
    inference_usd = _complete_sum(inference, "api_usd")
    trainer_usd = _complete_sum(training, "trainer_usd")
    external_usd = _complete_sum(external_costs, "amount_usd")
    total_usd = (
        inference_usd + trainer_usd + external_usd
        if inference_usd is not None
        and trainer_usd is not None
        and external_usd is not None
        else None
    )
    reward_delta = _metric_delta(after, before, "mean_reward")
    exact_delta = _metric_delta(after, before, "exact_accuracy")
    parse_delta = _metric_delta(after, before, "parse_rate")
    token_proxy = inference_rollup["total_tokens"] / 1_000_000.0
    wall_s = sum(
        float(event.get("metrics", {}).get("wall_s", 0.0) or 0.0)
        for event in finishes
    )
    allocation = Counter(
        str(event.get("dimensions", {}).get("selected_stratum"))
        for event in decisions
    )
    decision_attributes: dict[str, Counter[str]] = defaultdict(Counter)
    for event in decisions:
        for key, value in event.get("attributes", {}).items():
            if value is not None:
                decision_attributes[str(key)][str(value)] += 1
    external_by_category: dict[str, dict[str, float | None]] = {}
    for category in sorted(
        {
            str(event.get("dimensions", {}).get("category"))
            for event in external_costs
        }
    ):
        category_events = [
            event
            for event in external_costs
            if str(event.get("dimensions", {}).get("category")) == category
        ]
        external_by_category[category] = {
            "events": float(len(category_events)),
            "amount_usd": _complete_sum(category_events, "amount_usd"),
            "proxy_value": _complete_sum(category_events, "proxy_value"),
        }
    status_counts = Counter(
        str(event.get("dimensions", {}).get("status")) for event in finishes
    )
    return {
        "status_counts": dict(status_counts),
        "eligible_for_comparison": bool(status_counts)
        and set(status_counts) == {"completed"},
        "inference": inference_rollup,
        "phases": by_phase,
        "training": {
            "started_attempts": len(training_started),
            "attempts": len(training),
            "successful_updates": len(successful_training),
            "failed_attempts": len(training) - len(successful_training),
            "duration_s": sum(
                float(event.get("metrics", {}).get("duration_s", 0.0) or 0.0)
                for event in training
            ),
            "trainer_usd": trainer_usd,
        },
        "scheduler": {
            "decisions": len(decisions),
            "allocation": dict(allocation),
            "decision_attributes": {
                key: dict(counts) for key, counts in decision_attributes.items()
            },
            "max_allocation_fraction": (
                max(allocation.values()) / sum(allocation.values())
                if allocation
                else 0.0
            ),
        },
        "performance": {
            "heldout_mean_reward_delta": reward_delta,
            "heldout_exact_accuracy_delta": exact_delta,
            "heldout_parse_rate_delta": parse_delta,
        },
        "cost": {
            "input_tokens": inference_rollup["prompt_tokens"],
            "output_tokens": inference_rollup["completion_tokens"],
            "total_tokens": inference_rollup["total_tokens"],
            "token_proxy_millions": token_proxy,
            "inference_usd": inference_usd,
            "trainer_usd": trainer_usd,
            "external_usd": external_usd,
            "external_by_category": external_by_category,
            "total_usd": total_usd,
            "wall_s": wall_s,
        },
        "efficiency": {
            "reward_delta_per_million_tokens": (
                reward_delta / token_proxy
                if reward_delta is not None and token_proxy > 0.0
                else None
            ),
            "reward_delta_per_usd": (
                reward_delta / total_usd
                if reward_delta is not None
                and total_usd is not None
                and total_usd > 0.0
                else None
            ),
            "exact_delta_per_million_tokens": (
                exact_delta / token_proxy
                if exact_delta is not None and token_proxy > 0.0
                else None
            ),
        },
    }


def _inference_rollup(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(events)
    metrics = [event.get("metrics", {}) for event in events]
    rewards = [
        float(metric["reward"])
        for metric in metrics
        if metric.get("reward") is not None
    ]
    exact = [
        bool(metric["exact"])
        for metric in metrics
        if metric.get("exact") is not None
    ]
    parsed = [
        bool(metric["parsed"])
        for metric in metrics
        if metric.get("parsed") is not None
    ]
    return {
        "requests": count,
        "attempts": sum(int(metric.get("attempts", 0) or 0) for metric in metrics),
        "retries": sum(int(metric.get("retries", 0) or 0) for metric in metrics),
        "prompt_tokens": sum(
            int(metric.get("prompt_tokens", 0) or 0) for metric in metrics
        ),
        "completion_tokens": sum(
            int(metric.get("completion_tokens", 0) or 0) for metric in metrics
        ),
        "total_tokens": sum(
            int(metric.get("total_tokens", 0) or 0) for metric in metrics
        ),
        "latency_s": sum(
            float(metric.get("latency_s", 0.0) or 0.0) for metric in metrics
        ),
        "mean_reward": fmean(rewards) if rewards else None,
        "exact_accuracy": sum(exact) / len(exact) if exact else None,
        "parse_rate": sum(parsed) / len(parsed) if parsed else None,
        "api_usd": _complete_sum(events, "api_usd"),
    }


def _combine_phase_rollups(
    by_phase: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any] | None:
    phases = [value for key, value in by_phase.items() if key.startswith(prefix)]
    if not phases:
        return None
    requests = sum(int(phase["requests"]) for phase in phases)
    if requests == 0:
        return None
    return {
        "requests": requests,
        "mean_reward": _weighted_mean(phases, "mean_reward"),
        "exact_accuracy": _weighted_mean(phases, "exact_accuracy"),
        "parse_rate": _weighted_mean(phases, "parse_rate"),
    }


def _weighted_mean(phases: Sequence[Mapping[str, Any]], key: str) -> float | None:
    weighted = [
        (float(phase[key]), int(phase["requests"]))
        for phase in phases
        if phase.get(key) is not None and int(phase.get("requests", 0)) > 0
    ]
    total = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / total if total else None


def _metric_delta(
    after: Mapping[str, Any] | None,
    before: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if after is None or before is None:
        return None
    if after.get(key) is None or before.get(key) is None:
        return None
    return float(after[key]) - float(before[key])


def _complete_sum(
    events: Sequence[Mapping[str, Any]],
    metric: str,
) -> float | None:
    if not events:
        return 0.0
    values = [event.get("metrics", {}).get(metric) for event in events]
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def _condition_points(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    eligible_summaries = {
        condition: summary
        for condition, summary in summaries.items()
        if summary.get("eligible_for_comparison")
    }
    monetary_complete = bool(eligible_summaries) and all(
        summary.get("cost", {}).get("total_usd") is not None
        for summary in eligible_summaries.values()
    )
    basis = "total_usd" if monetary_complete else "token_proxy_millions"
    points = []
    for condition, summary in eligible_summaries.items():
        performance = summary.get("performance", {})
        cost = summary.get("cost", {})
        points.append(
            {
                "condition": condition,
                "cost": cost.get(basis),
                "performance": performance.get("heldout_mean_reward_delta"),
                "exact_accuracy_delta": performance.get(
                    "heldout_exact_accuracy_delta"
                ),
                "parse_rate_delta": performance.get("heldout_parse_rate_delta"),
                "wall_s": cost.get("wall_s"),
            }
        )
    return points, basis


def _pairwise_condition_comparisons(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    names = sorted(summaries)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = summaries[left_name]
            right = summaries[right_name]
            if not left.get("eligible_for_comparison") or not right.get(
                "eligible_for_comparison"
            ):
                continue
            comparisons[f"{left_name}_minus_{right_name}"] = {
                "reward_delta_difference": _difference(
                    left.get("performance", {}).get("heldout_mean_reward_delta"),
                    right.get("performance", {}).get("heldout_mean_reward_delta"),
                ),
                "total_token_difference": _difference(
                    left.get("cost", {}).get("total_tokens"),
                    right.get("cost", {}).get("total_tokens"),
                ),
                "output_token_difference": _difference(
                    left.get("cost", {}).get("output_tokens"),
                    right.get("cost", {}).get("output_tokens"),
                ),
                "wall_s_difference": _difference(
                    left.get("cost", {}).get("wall_s"),
                    right.get("cost", {}).get("wall_s"),
                ),
                "total_usd_difference": _difference(
                    left.get("cost", {}).get("total_usd"),
                    right.get("cost", {}).get("total_usd"),
                ),
            }
    return comparisons


def _monitoring_alerts(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    coverage: Mapping[str, Any],
    expected_inference_requests: int | None,
    run_finished: bool,
    stale: bool,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    contract_closed = run_finished or stale
    if len(coverage["run_ids"]) > 1:
        alerts.append(
            _alert(
                "mixed_run_ids",
                "error",
                "A telemetry summary must contain exactly one run ID.",
                run_ids=coverage["run_ids"],
            )
        )
    if coverage["sequence_issues"]:
        alerts.append(
            _alert(
                "telemetry_sequence_broken",
                "error",
                "Telemetry sequence numbers are missing, duplicated, or reordered.",
                issues=coverage["sequence_issues"],
            )
        )
    if coverage["inference_pricing_coverage"] < 1.0:
        alerts.append(
            _alert(
                "inference_pricing_incomplete",
                "warning",
                "Inference dollars are unavailable for some requests; "
                "token cost is a proxy.",
                coverage=coverage["inference_pricing_coverage"],
            )
        )
    if coverage["trainer_pricing_coverage"] < 1.0:
        alerts.append(
            _alert(
                "trainer_pricing_incomplete",
                "warning",
                "Trainer dollars are unavailable for some successful updates.",
                coverage=coverage["trainer_pricing_coverage"],
            )
        )
    if coverage["external_cost_pricing_coverage"] < 1.0:
        alerts.append(
            _alert(
                "external_cost_pricing_incomplete",
                "warning",
                "Some tool, evaluator, storage, or provider costs lack USD values.",
                coverage=coverage["external_cost_pricing_coverage"],
            )
        )
    if coverage["unfinished_training_attempts"]:
        lifecycle_error = run_finished or stale
        alerts.append(
            _alert(
                (
                    "unfinished_training_attempt"
                    if lifecycle_error
                    else "active_training_attempt"
                ),
                "error" if lifecycle_error else "info",
                (
                    "A trainer attempt started without a matching finish event."
                    if lifecycle_error
                    else "A trainer attempt is currently in progress."
                ),
                attempts=coverage["unfinished_training_attempts"],
            )
        )
    if coverage["unfinished_conditions"]:
        lifecycle_error = run_finished or stale
        alerts.append(
            _alert(
                "unfinished_condition" if lifecycle_error else "active_condition",
                "error" if lifecycle_error else "info",
                (
                    "A condition started without a matching finish event."
                    if lifecycle_error
                    else "A condition is currently in progress."
                ),
                conditions=coverage["unfinished_conditions"],
            )
        )
    if (
        contract_closed
        and
        expected_inference_requests is not None
        and coverage["recorded_inference_requests"] != expected_inference_requests
    ):
        alerts.append(
            _alert(
                "inference_request_coverage_mismatch",
                "error",
                "Recorded inference requests do not match the experiment contract.",
                expected=expected_inference_requests,
                recorded=coverage["recorded_inference_requests"],
            )
        )
    for label, coverage_key in (
        ("condition runs", "condition_run_coverage"),
        ("training updates", "training_update_coverage"),
        ("scheduler decisions", "scheduler_decision_coverage"),
    ):
        value = coverage.get(coverage_key)
        if contract_closed and value is not None and abs(float(value) - 1.0) > 1e-12:
            alerts.append(
                _alert(
                    f"{coverage_key}_mismatch",
                    "error",
                    f"Recorded {label} do not match the run contract.",
                    coverage=value,
                )
            )
    for condition, summary in summaries.items():
        failed_conditions = {
            status: count
            for status, count in summary.get("status_counts", {}).items()
            if status != "completed" and count
        }
        if failed_conditions:
            alerts.append(
                _alert(
                    "condition_failed",
                    "error",
                    "A failed condition is excluded from cost-performance comparisons.",
                    condition=condition,
                    status_counts=summary["status_counts"],
                )
            )
        performance = summary.get("performance", {})
        exact_delta = performance.get("heldout_exact_accuracy_delta")
        parse_delta = performance.get("heldout_parse_rate_delta")
        if (
            exact_delta is not None
            and parse_delta is not None
            and abs(float(exact_delta)) < 1e-12
            and float(parse_delta) >= 0.05
        ):
            alerts.append(
                _alert(
                    "format_gain_without_exact_gain",
                    "warning",
                    "Parseability improved materially without an exact-accuracy gain.",
                    condition=condition,
                    parse_rate_delta=parse_delta,
                    exact_accuracy_delta=exact_delta,
                )
            )
        scheduler = summary.get("scheduler", {})
        if float(scheduler.get("max_allocation_fraction", 0.0) or 0.0) > 0.5:
            alerts.append(
                _alert(
                    "scheduler_allocation_concentration",
                    "warning",
                    "More than half of scheduler decisions selected one stratum.",
                    condition=condition,
                    max_fraction=scheduler["max_allocation_fraction"],
                    allocation=scheduler["allocation"],
                )
            )
        retries = int(summary.get("inference", {}).get("retries", 0) or 0)
        failed_training = int(
            summary.get("training", {}).get("failed_attempts", 0) or 0
        )
        if retries or failed_training:
            alerts.append(
                _alert(
                    "reliability_overhead_observed",
                    "warning",
                    "Retries or failed training attempts added cost to the condition.",
                    condition=condition,
                    inference_retries=retries,
                    failed_training_attempts=failed_training,
                )
            )
    if coverage["intermediate_heldout_evaluations"] == 0:
        alerts.append(
            _alert(
                "cost_to_target_unobservable",
                "info",
                "No intermediate held-out evaluations were recorded, so "
                "cost-to-target cannot be measured.",
            )
        )
    return alerts


def _alert(code: str, severity: str, message: str, **evidence: Any) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def _coverage_ratio(recorded: int, expected: int | None) -> float | None:
    if expected is None:
        return None
    if expected == 0:
        return 1.0 if recorded == 0 else 0.0
    return recorded / expected


def _unfinished_lifecycles(
    started: Sequence[Mapping[str, Any]],
    finished: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    def identity(event: Mapping[str, Any]) -> tuple[Any, ...]:
        dimensions = event.get("dimensions", {})
        return tuple(dimensions.get(key) for key in keys)

    pending = Counter(identity(event) for event in started)
    pending.subtract(identity(event) for event in finished)
    return [
        dict(zip(keys, values, strict=True))
        for values, count in pending.items()
        for _ in range(max(0, count))
    ]


def _field_coverage(
    events: Sequence[Mapping[str, Any]],
    section: str,
    key: str,
) -> float:
    if not events:
        return 1.0
    return sum(event.get(section, {}).get(key) is not None for event in events) / len(
        events
    )


def _sequence_issues(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    previous: int | None = None
    for index, event in enumerate(events):
        sequence = event.get("sequence")
        if not isinstance(sequence, int):
            issues.append({"event_index": index, "sequence": sequence})
            continue
        expected = 1 if previous is None else previous + 1
        if sequence != expected:
            issues.append(
                {
                    "event_index": index,
                    "expected_sequence": expected,
                    "observed_sequence": sequence,
                }
            )
        previous = sequence
    return issues


def _difference(left: Any, right: Any) -> float | None:
    left_value = _finite_or_none(left)
    right_value = _finite_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return int(value) if isinstance(value, int) else None


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _validated_metrics(
    metrics: Mapping[str, int | float | bool | None],
) -> dict[str, int | float | bool | None]:
    clean: dict[str, int | float | bool | None] = {}
    for key, value in metrics.items():
        if value is not None and not isinstance(value, (int, float, bool)):
            raise TypeError(f"metric {key!r} must be numeric, boolean, or None")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")
        clean[str(key)] = value
    return clean


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_non_negative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
