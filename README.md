# Calm Puffer ART

This repo is a runnable scaffold for an ART-shaped RL execution model:

- ART remains the control plane: user code defines scenarios, rollout workflows, trajectory groups, rewards, checkpoints, and trainable policies.
- Sample production becomes Puffer-like: many actors run continuously, push trajectories through bounded queues, and train against batches as soon as enough groups are ready.
- Action bandwidth becomes CALM-like: policies can emit action units above single tokens, including chunks, latent patches, command units, or reasoning-step units.

The north-star metric exposed by the runtime is `reward_improving_experience_per_dollar_second`, computed from reward improvement, accepted trainable trajectories, wall-clock time, and configured runtime cost. The run summary also reports rollout, trainer, queue-wait, wall-clock, and accounted dollar-second attribution so scheduler decisions can be audited by where the spend went.

## Why this shape

ART's current public training-loop docs describe a client/backend split where user rollouts create rewarded `Trajectory` objects, then grouped trajectories are sent to the backend; in the default shared-resource loop, inference is blocked while training executes. This scaffold keeps the trajectory/reward boundary but changes the runtime scheduler around it.

PufferLib's docs emphasize fast vectorized sample production through chunked environment buffers, independent rollout workers, static/backpressured memory movement, and throughput telemetry. This repo adapts those runtime ideas to language-agent workflows without depending on PufferLib internals.

CALM's README frames semantic bandwidth as predicting one continuous vector for a chunk of `K` tokens rather than one token at a time. This repo does not implement CALM training. It provides policy/action interfaces that let ART-style rollouts experiment with chunk, latent-patch, command, and compressed-reasoning decisions.

Primary references:

- ART: <https://github.com/OpenPipe/ART> and <https://art.openpipe.ai/fundamentals/training-loop>
- PufferLib: <https://github.com/PufferAI/PufferLib> and <https://puffer.ai/docs.html>
- CALM: <https://github.com/shaochenze/calm>

## Quick start

Run the deterministic example:

```powershell
$env:PYTHONPATH = "src"
python examples\counting_agent.py
```

Run the adaptive scheduler example:

```powershell
$env:PYTHONPATH = "src"
python examples\adaptive_scheduler_agent.py
```

Run the adaptive action-space example:

```powershell
$env:PYTHONPATH = "src"
python examples\adaptive_action_space_agent.py
```

Run the static-vs-objective ablation:

```powershell
$env:PYTHONPATH = "src"
python examples\objective_ablation.py
```

Run tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

Use an explicit checkpoint broadcast stream:

```python
channel = WeightBroadcastChannel()
updates = channel.subscribe()

summary = await ControlPlane(config).run(
    scenarios=scenarios,
    initial_policy=policy,
    trainer=trainer,
    workflow=rollout,
    action_codec=codec,
    weight_channel=channel,
)
```

Turn on the closed-loop scheduler with multiple action codecs:

```python
summary = await ControlPlane(config).run(
    scenarios=scenarios,
    initial_policy=policy,
    trainer=trainer,
    workflow=rollout,
    action_codecs=[TokenActionCodec(), ChunkActionCodec(chunk_size=4)],
    scheduler=ObjectiveScheduler(),
)
```

The scheduler explores `(scenario, action_codec)` arms, estimates marginal reward improvement per dollar-second, credits train-step policy improvement back to the arms and runtime controls that produced each consumed batch, and uses that signal to choose future rollouts, action granularity, train-batch priority, batch cadence, policy lag, and optional early stopping. Train feedback uses reward improvement multiplied by useful trajectory count, aligning it with the run-level north-star numerator. Raw reward efficiency is an explicit opt-in scoring weight; by default rollout selection and batch priority are driven by marginal rollout objective plus train-improvement objective. Rollouts can set `metrics["cost/dollar_seconds"]` or `metadata["cost/dollar_seconds"]` to account for API, token, tool, or GPU cost; otherwise runtime duration is multiplied by the configured infrastructure rate. Actor queue-wait cost is stamped onto trajectories as `cost/actor_queue_wait_dollar_seconds` and included in scheduler rollout cost, so backpressure lowers the marginal objective of the arms and runtime settings that caused it. Trainers can report `cost/dollar_seconds`, `train/dollar_seconds`, or `trainer/dollar_seconds` in `TrainResult.metrics` or metadata; that explicit cost feeds runtime telemetry and scheduler train-objective credit. It widens train batches under trainer pressure when no objective signal justifies tight updates, but credited cadence and lag settings can be reused when their observed objective is better. Rollouts can add `action/safe`, `action/quality`, `reconstruction/accuracy`, or `verifier/passed` metadata; unsafe actions receive zero effective reward and no train-improvement credit.

Gate checkpoint promotion when train reward is not enough:

```python
summary = await ControlPlane(config).run(
    scenarios=scenarios,
    initial_policy=policy,
    trainer=trainer,
    workflow=rollout,
    action_codecs=[TokenActionCodec()],
    scheduler=ObjectiveScheduler(),
    promotion_evaluator=MetricPromotionEvaluator(
        metric_key="eval/reward",
        min_delta=0.05,
        initial_score=baseline_eval_reward,
    ),
)
```

Without a `promotion_evaluator`, every train result is promoted, preserving the simple default. With one, each train result becomes a candidate: rejected candidates still count as train spend, but they do not advance the served policy step, do not append a checkpoint, and do not broadcast weights. `PromotionDecision` metadata records candidate score, baseline score, improvement, cost, and reason. Scheduler train credit uses the promotion-effective score under `promotion/score`, so rejected candidates do not create false positive policy-improvement credit merely because their trainer-local reward looked high. Built-in promotion evaluators snapshot their learned baseline under `promotion/state`, so resumed runs keep the same publish gate instead of resetting the acceptance threshold.

For held-out workflow evaluation, use `RolloutPromotionEvaluator`:

```python
promotion_evaluator = RolloutPromotionEvaluator(
    scenarios=heldout_scenarios,
    workflow=rollout,
    action_codec=TokenActionCodec(),
    min_delta=0.05,
    initial_score=baseline_eval_reward,
    cost_per_second_usd=runtime_cost_per_second,
)
```

It runs the candidate policy through the same ART-style rollout contract before publication, scores quality-adjusted reward, records evaluation failures, action units, source tokens, duration, and dollar-seconds, and promotes only when the held-out score improves enough. Held-out evaluation trajectories are also tagged with normal scheduler arm metadata and fed back through `observe_rollout()`, so eval successes, failures, action quality, and explicit `eval/dollar_seconds` costs can update future rollout/action choices.

Use `ObjectiveScheduler.state_dict()` and `ObjectiveScheduler.load_state_dict()` to persist the controller's learned arm statistics, runtime-control credit, budget counters, configuration, and scalar last-decision metadata alongside ART checkpoints. `ControlPlane` and `AsyncArtBackend` attach that snapshot under `scheduler/state` after train feedback is observed and before the checkpoint update is published. The snapshot intentionally excludes live `Scenario` and `ActionCodec` objects, so resumed runs should reconstruct those from user code and reload only the scheduler's numeric control memory.

To resume local control state, pass a `PolicySnapshot` as `initial_policy` and include the saved checkpoint metadata:

```python
snapshot = PolicySnapshot(
    step=saved_step,
    policy=loaded_policy,
    checkpoint_id=saved_checkpoint_id,
    created_at=saved_created_at,
    metadata=saved_checkpoint_metadata,
)

summary = await ControlPlane(config).run(
    scenarios=scenarios,
    initial_policy=snapshot,
    trainer=trainer,
    workflow=rollout,
    action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8),
    scheduler=ObjectiveScheduler(),
)
```

`ControlPlane` calls `restore_control_state()` before actors start, restoring `scheduler/state`, `action_space/state`, and `promotion/state` when compatible objects are provided. The initial checkpoint keeps its original step, so policy-lag checks continue from the resumed version instead of restarting at step 0. Promotion resume restores the evaluator's numeric control memory, such as best accepted score and threshold configuration; live rollout scenarios, workflows, and custom action codecs remain user-code objects supplied by the new run.

Let the action space promote larger chunks online:

```python
summary = await ControlPlane(config).run(
    scenarios=scenarios,
    initial_policy=policy,
    trainer=trainer,
    workflow=rollout,
    action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8),
    scheduler=ObjectiveScheduler(),
)
```

`AdaptiveActionSpace` starts with token and small chunk actions, then promotes larger `ChunkActionCodec` sizes only when scheduler metrics show positive objective signal, high action quality, and low unsafe rate. Promoted chunk sizes can also be disabled after enough bad evidence, so unsafe or non-improving high-bandwidth actions stop competing for rollout slots in the same run.

`AdaptiveActionSpace.state_dict()` and `load_state_dict()` persist active codecs, disabled codec keys, promotion/demotion counters, and action-space configuration. `ControlPlane` writes this snapshot under `action_space/state` in checkpoint metadata after applying scheduler feedback, so resumed runs keep the discovered semantic-bandwidth ladder instead of relearning chunk promotions from scratch. Built-in codecs are reconstructed directly; custom codecs remain user-code objects and are restored only when an equivalent codec is already present.

Adapt ART trajectory groups without depending on ART at import time:

```python
from calm_puffer_art import ArtBackendTrainer, art_groups_to_local

local_groups = art_groups_to_local(art_groups)
trainer = ArtBackendTrainer(backend=art_backend, model=art_model)
```

Converted groups retain the original ART group and trajectory objects in metadata, so an `ArtBackendTrainer` can delegate back to the real ART backend/loss implementation while the async runtime and scheduler use local telemetry. Untagged ART trajectories receive a scenario-scoped default scheduler arm such as `math|art`; user-supplied `scheduler/arm_id` metadata is preserved when present.

Wrap an ART-like backend in the bounded async substrate:

```python
from calm_puffer_art import AsyncArtBackend, AsyncArtBackendConfig

backend = AsyncArtBackend(
    backend=art_backend,
    config=AsyncArtBackendConfig(train_queue_capacity=3, max_policy_lag=2),
    scheduler=ObjectiveScheduler(),
)
await backend.register(art_model)
result = await backend.train(art_model, art_groups)
```

`AsyncArtBackend` exposes backend-shaped `register()`, `train()`, `_get_step()`, and `close()` methods, enqueues converted ART groups through the same fixed-capacity stale-aware train ring, observes train results through the scheduler, and publishes checkpoint updates. The scheduler controls train-batch cadence and the active stale-policy lag limit before each ring consume. Stale ART batches fail waiting callers and report lost useful experience back to the scheduler as negative objective feedback. The wrapper delegates the actual ART loss/checkpoint work to the supplied backend.

For no-stop-the-world submission, use `submit_train()`:

```python
future = await backend.submit_train(art_model, art_groups)
# Keep producing rollouts while the background trainer consumes the batch.
result = await future
```

`train()` is the compatibility wrapper that awaits the same future. `submit_train()` returns after bounded-ring admission, so callers pay backpressure only when the ring is full.

For scheduler-controlled batch cadence, submit individual ART trajectory groups:

```python
future = await backend.submit_group(art_model, art_group)
# Later, force a partial batch if cadence has not been reached:
await backend.flush_pending_groups()
result = await future
```

`submit_group()` accumulates compatible groups and flushes them into the train ring when the scheduler's target batch cadence is reached. The same scheduler can tighten `max_policy_lag` when useful train signal appears or the queue is pressured, causing over-stale queued ART batches to fail with `StaleArtBatchError` rather than training on obsolete experience. Those stale drops also debit the arms, cadence values, and lag values that produced the discarded batch.

## Core pieces

- `calm_puffer_art.types`: ART-like primitives for scenarios, action units, trajectories, trajectory groups, checkpoints, and run summaries.
- `calm_puffer_art.actions`: token, chunk, latent-patch, command, and reasoning-step codecs, plus the adaptive chunk promotion/demotion action space with checkpointable state.
- `calm_puffer_art.art_adapter`: dependency-free conversion between ART-shaped trajectory groups and local runtime groups, a delegating trainer wrapper, and a structural async ART backend wrapper.
- `calm_puffer_art.scheduler`: objective-driven rollout/action/train-priority/cadence/lag scheduler with action-quality, train-improvement, stale-drop, pressure feedback, and checkpointable control state.
- `calm_puffer_art.runtime`: async control plane with actor pools, bounded queues, background group assembly, priority-aware versioned train-batch rings, promotion-gated checkpoint broadcasts, stale-sample filtering, cost attribution, and telemetry.
- `examples/counting_agent.py`: a deterministic trainable toy policy whose reward improves over checkpoints.
- `examples/adaptive_scheduler_agent.py`: a deterministic closed-loop scheduler demo that learns which scenario/action-codec arm has better reward-per-cost signal.
- `examples/adaptive_action_space_agent.py`: a deterministic demo where objective feedback promotes a larger chunk action codec during the run.
- `examples/objective_ablation.py`: a deterministic static-vs-objective comparison that reports north-star lift from scheduler control.
- `docs/art_puffer_calm_synthesis.md`: cleaned integration plan for the ART backend and future optional CALM layer.

## Non-goals

This is not a fork of ART, PufferLib, or CALM. It does not train real LLM weights, implement GRPO, allocate CUDA vector buffers, or learn a continuous language autoencoder. It is a small, typed runtime seam that makes those integrations explicit and testable.
