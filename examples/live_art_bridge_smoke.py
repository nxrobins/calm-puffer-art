from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from calm_puffer_art import (
    ART_BACKEND_STATE_KEY,
    AdaptiveActionSpace,
    AsyncArtBackend,
    AsyncArtBackendConfig,
    ChunkActionCodec,
    ObjectiveScheduler,
    SCHEDULER_STATE_KEY,
    Scenario,
    TokenActionCodec,
    WeightBroadcastChannel,
)


@dataclass
class _StructuralTrainResult:
    step: int
    metrics: dict[str, float]
    checkpoint_path: str


class _StructuralDelegateBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.step = 0
        self.registered: list[Any] = []
        self.received_groups: list[Any] = []

    async def register(self, model: Any) -> None:
        self.registered.append(model)

    async def _get_step(self, model: Any) -> int:
        return self.step

    async def train(self, model: Any, trajectory_groups: Sequence[Any], **kwargs: Any):
        self.calls += 1
        self.step += 1
        self.received_groups.extend(trajectory_groups)
        rewards = [
            float(getattr(trajectory, "reward", 0.0) or 0.0)
            for group in trajectory_groups
            for trajectory in _group_trajectories(group)
        ]
        reward = sum(rewards) / len(rewards) if rewards else 0.0
        return _StructuralTrainResult(
            step=self.step,
            metrics={
                "train/reward": reward,
                "trainer/dollar_seconds": 0.5,
            },
            checkpoint_path=f".art/smoke/step_{self.step}",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test AsyncArtBackend against ART's public object surface.",
    )
    parser.add_argument(
        "--backend",
        choices=("structural", "serverless", "local"),
        default="structural",
    )
    parser.add_argument("--base-model", default="OpenPipe/Qwen3-0.6B")
    parser.add_argument("--project", default="calm-puffer-art-smoke")
    parser.add_argument("--model-name", default="smoke-agent")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--rollouts-per-group", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--cost-per-second-usd", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    art = _import_art()
    used_real_backend = args.backend != "structural"
    delegate_backend = _backend_for_mode(args.backend)
    model = _model_for_mode(
        art,
        mode=args.backend,
        name=args.model_name,
        project=args.project,
        base_model=args.base_model,
    )
    channel = WeightBroadcastChannel()
    updates = channel.subscribe()
    async_backend = AsyncArtBackend(
        backend=delegate_backend,
        config=AsyncArtBackendConfig(
            train_queue_capacity=max(2, args.groups),
            train_batch_groups=1,
            max_policy_lag=2,
            max_train_steps=max(1, args.groups),
            cost_per_second_usd=args.cost_per_second_usd,
        ),
        scheduler=ObjectiveScheduler(
            min_train_batch_groups=1,
            max_train_batch_groups=max(1, args.groups),
            min_policy_lag=1,
            max_policy_lag=2,
            min_actor_count=1,
            max_actor_count=max(1, args.rollouts_per_group),
            exploration_bonus=0.0,
            control_exploration_bonus=0.0,
            rollout_cadence_lag_control_weight=1.0,
        ),
        action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
        weight_channel=channel,
    )
    scenarios = (
        Scenario(id="art_smoke_easy", payload={"reward": 0.3}),
        Scenario(id="art_smoke_useful", payload={"reward": 1.0}),
    )
    codecs = (TokenActionCodec(), ChunkActionCodec(chunk_size=2))
    futures = []
    raw_groups: list[Any] = []

    try:
        await _register_model(model, async_backend, mode=args.backend)
        for group_index in range(args.groups):
            trajectories = []
            for rollout_index in range(args.rollouts_per_group):
                assignment = await async_backend.admit_and_select_rollout(
                    scenarios=scenarios,
                    action_codecs=codecs,
                    actor_id=rollout_index,
                    configured_actor_count=args.rollouts_per_group,
                    trajectory_queue_pressure=(
                        async_backend.ring.pending_batches
                        / async_backend.ring.capacity
                    ),
                    apply_delay=False,
                )
                if not assignment.admitted or assignment.decision is None:
                    break
                trajectory = await _build_trajectory(
                    art,
                    model=model,
                    mode=args.backend,
                    metadata=assignment.metadata,
                    group_index=group_index,
                    rollout_index=rollout_index,
                    reward=float(
                        assignment.decision.scenario.payload.get("reward", 0.0)
                    ),
                )
                trajectories.append(trajectory)
            if not trajectories:
                continue
            group = _make_trajectory_group(art, trajectories, group_index)
            raw_groups.append(group)
            futures.append(
                await async_backend.submit_group(
                    model,
                    group,
                    learning_rate=args.learning_rate,
                )
            )

        await async_backend.flush_pending_groups()
        if futures:
            await asyncio.gather(*futures)
        published = await _drain_updates(updates, expected=1)
        stats = async_backend.stats()
        update = published[-1] if published else None
        metadata = update.metadata if update is not None else {}
        local_groups = [
            item
            for batch in getattr(async_backend.ring, "_items", [])
            for item in getattr(batch, "groups", ())
        ]
        return {
            "used_real_art_package": True,
            "used_real_art_backend": used_real_backend,
            "backend": args.backend,
            "submitted_groups": int(stats.get("art_backend/submitted_groups", 0.0)),
            "completed_batches": int(stats.get("art_backend/completed_batches", 0.0)),
            "published_policy_updates": int(
                stats.get("art_backend/published_policy_updates", 0.0)
            ),
            "raw_art_group_preserved": _raw_group_preserved(
                raw_groups,
                delegate_backend,
            ),
            "raw_art_trajectory_preserved": _raw_trajectory_preserved(
                raw_groups,
                delegate_backend,
            ),
            "published_scheduler_state": SCHEDULER_STATE_KEY in metadata,
            "published_art_backend_state": ART_BACKEND_STATE_KEY in metadata,
            "metrics": stats,
            "retained_ring_groups": len(local_groups),
        }
    finally:
        await async_backend.close()


async def _register_model(
    model: Any,
    backend: AsyncArtBackend,
    *,
    mode: str,
) -> None:
    if mode == "structural":
        await backend.register(model)
        return
    register = getattr(model, "register", None)
    if register is None:
        raise RuntimeError(
            "Real ART backend mode requires TrainableModel.register(backend)"
        )
    try:
        result = register(backend)
        if inspect.isawaitable(result):
            await result
    except AttributeError as exc:
        missing = str(exc)
        raise RuntimeError(
            "ART model registration failed against AsyncArtBackend; "
            f"missing backend protocol member: {missing}"
        ) from exc


async def _build_trajectory(
    art: Any,
    *,
    model: Any,
    mode: str,
    metadata: Mapping[str, Any],
    group_index: int,
    rollout_index: int,
    reward: float,
) -> Any:
    user_message = _make_message(
        art,
        role="user",
        content=f"smoke group {group_index} rollout {rollout_index}",
    )
    choice = await _assistant_choice(
        art,
        model=model,
        mode=mode,
        prompt=user_message,
    )
    policy_step = int(float(metadata.get("scheduler/policy_step", 0)))
    trajectory_metadata = {
        **metadata,
        "scenario_id": str(metadata.get("scheduler/scenario_id", "art_smoke")),
        "smoke/group_index": group_index,
        "smoke/rollout_index": rollout_index,
    }
    metrics = {
        "rollout/dollar_seconds": 1.0 + rollout_index * 0.1,
        "duration": 0.01,
    }
    return _construct_art_object(
        art.Trajectory,
        positional=(
            [user_message, choice],
            reward,
        ),
        values={
            "messages_and_choices": [user_message, choice],
            "reward": reward,
            "initial_policy_version": policy_step,
            "final_policy_version": policy_step,
            "metrics": metrics,
            "metadata": trajectory_metadata,
        },
    )


async def _assistant_choice(
    art: Any,
    *,
    model: Any,
    mode: str,
    prompt: Any,
) -> Any:
    if mode == "structural":
        return _make_choice(
            art,
            _make_message(art, role="assistant", content="useful smoke answer"),
        )
    client_factory = getattr(model, "openai_client", None)
    if client_factory is None:
        raise RuntimeError("Real ART backend mode requires model.openai_client()")
    client = client_factory()
    inference_name = await _model_inference_name(model)
    completion = client.chat.completions.create(
        model=inference_name,
        messages=[{"role": "user", "content": _message_content(prompt)}],
    )
    if inspect.isawaitable(completion):
        completion = await completion
    return completion.choices[0]


async def _model_inference_name(model: Any) -> str:
    for name in ("get_inference_name", "inference_model_name"):
        value = getattr(model, name, None)
        if callable(value):
            resolved = value()
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if resolved is not None:
                return str(resolved)
        elif value is not None:
            return str(value)
    return str(getattr(model, "name", "model"))


def _message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _backend_for_mode(mode: str) -> Any:
    if mode == "structural":
        return _StructuralDelegateBackend()
    if mode == "serverless":
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required for --backend serverless")
        from art.serverless.backend import ServerlessBackend

        return ServerlessBackend()
    if mode == "local":
        try:
            from art.local import LocalBackend
        except ImportError:
            from art.local.backend import LocalBackend

        return LocalBackend()
    raise ValueError(f"unknown backend mode: {mode}")


def _model_for_mode(
    art: Any,
    *,
    mode: str,
    name: str,
    project: str,
    base_model: str,
) -> Any:
    if mode == "structural":
        return _construct_art_object(
            art.TrainableModel,
            positional=(name, project, base_model),
            values={
                "name": name,
                "project": project,
                "base_model": base_model,
                "base_model_name": base_model,
            },
        )
    return _construct_art_object(
        art.TrainableModel,
        positional=(name, project, base_model),
        values={
            "name": name,
            "project": project,
            "base_model": base_model,
            "base_model_name": base_model,
        },
    )


def _make_trajectory_group(
    art: Any,
    trajectories: Sequence[Any],
    group_index: int,
) -> Any:
    metadata = {"scenario_id": "art_smoke", "smoke/group_index": group_index}
    try:
        return _construct_art_object(
            art.TrajectoryGroup,
            positional=(list(trajectories),),
            values={"trajectories": list(trajectories), "metadata": metadata},
        )
    except Exception:
        for trajectory in trajectories:
            raw_metadata = getattr(trajectory, "metadata", None)
            if isinstance(raw_metadata, dict):
                raw_metadata.setdefault("scenario_id", "art_smoke")
        return list(trajectories)


def _make_message(art: Any, *, role: str, content: str) -> Any:
    message_cls = getattr(art, "Message", None)
    if message_cls is not None:
        try:
            return _construct_art_object(
                message_cls,
                positional=(role, content),
                values={"role": role, "content": content},
            )
        except Exception:
            pass
    return {"role": role, "content": content}


def _make_choice(art: Any, message: Any) -> Any:
    choice_cls = getattr(art, "Choice", None)
    if choice_cls is not None:
        try:
            return _construct_art_object(
                choice_cls,
                positional=(message,),
                values={"message": message},
            )
        except Exception:
            pass
    return {"message": message}


def _construct_art_object(
    cls: Any,
    *,
    positional: Sequence[Any] = (),
    values: Mapping[str, Any],
) -> Any:
    try:
        return cls(**dict(values))
    except TypeError:
        pass
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        kwargs = {
            key: value
            for key, value in values.items()
            if key in signature.parameters
        }
        if kwargs:
            try:
                return cls(**kwargs)
            except TypeError:
                pass
    try:
        return cls(*positional)
    except TypeError:
        if positional:
            return cls(positional[0])
        raise


def _raw_group_preserved(raw_groups: Sequence[Any], backend: Any) -> bool:
    received = getattr(backend, "received_groups", None)
    if received is None:
        return bool(raw_groups)
    expected = {id(group) for group in raw_groups}
    actual = {id(group) for group in received}
    return bool(expected and expected.intersection(actual))


def _raw_trajectory_preserved(raw_groups: Sequence[Any], backend: Any) -> bool:
    expected = {
        id(trajectory)
        for group in raw_groups
        for trajectory in _group_trajectories(group)
    }
    received = getattr(backend, "received_groups", None)
    if received is None:
        return bool(expected)
    actual = {
        id(trajectory)
        for group in received
        for trajectory in _group_trajectories(group)
    }
    return bool(expected and expected.intersection(actual))


def _group_trajectories(group: Any) -> list[Any]:
    trajectories = getattr(group, "trajectories", None)
    if trajectories is not None:
        return list(trajectories)
    try:
        return list(group)
    except TypeError:
        return []


async def _drain_updates(queue: asyncio.Queue[Any], *, expected: int) -> list[Any]:
    updates = []
    for _ in range(expected):
        try:
            updates.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return updates


def _import_art() -> Any:
    try:
        import art
    except ImportError as exc:
        raise RuntimeError(
            "The live ART bridge smoke requires the optional ART extra. "
            "Install with `pip install -e \".[art]\"`."
        ) from exc
    missing = [
        name
        for name in ("TrainableModel", "Trajectory", "TrajectoryGroup")
        if not hasattr(art, name)
    ]
    if missing:
        raise RuntimeError(f"ART package is missing required public classes: {missing}")
    return art


def _assert_output_contract(result: Mapping[str, Any]) -> None:
    metrics = result.get("metrics", {})
    required = (
        "art_backend/sample_dollar_seconds",
        "art_backend/trainer_dollar_seconds",
        "art_backend/accounted_dollar_seconds",
        "scheduler/joint_action/tuples",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"smoke metrics missing required keys: {missing}")
    if result.get("completed_batches", 0) <= 0:
        raise RuntimeError("smoke completed no train batches")
    if result.get("published_policy_updates", 0) <= 0:
        raise RuntimeError("smoke published no checkpoint updates")
    if not result.get("published_scheduler_state"):
        raise RuntimeError("published checkpoint missing scheduler/state")
    if not result.get("published_art_backend_state"):
        raise RuntimeError("published checkpoint missing art_backend/state")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = asyncio.run(run_smoke(args))
        _assert_output_contract(result)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"live ART bridge smoke failed: {exc}", file=sys.stderr)
        return 1
    output = {"ok": True, **result}
    if args.json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
