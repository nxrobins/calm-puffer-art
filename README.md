# Calm Puffer ART

This repo is a runnable scaffold for an ART-shaped RL execution model:

- ART remains the control plane: user code defines scenarios, rollout workflows, trajectory groups, rewards, checkpoints, and trainable policies.
- Sample production becomes Puffer-like: many actors run continuously, push trajectories through bounded queues, and train against batches as soon as enough groups are ready.
- Action bandwidth becomes CALM-like: policies can emit action units above single tokens, including chunks, latent patches, command units, or reasoning-step units.

The north-star metric exposed by the runtime is `reward_improving_experience_per_dollar_second`, computed from reward improvement, accepted trainable trajectories, wall-clock time, and configured runtime cost. When checkpoint promotion is enabled, the summary also reports `published_policy_reward_improving_experience_per_dollar_second`, which counts only score improvements that actually produced a published checkpoint; rejected candidates remain in the cost denominator but not the useful policy-improvement numerator. The run summary also reports rollout, trainer, trainer-wait, actor-admission, queue-wait, wall-clock, accounted dollar-second attribution, projected accounted spend that includes reserved in-flight rollouts, throughput rates, drop/failure rates, and stage utilization, so scheduler decisions can be audited by where the spend went, what spend has already been admitted, and whether sample production or training is the bottleneck.

## Why this shape

ART's current public training-loop docs describe a client/backend split where user rollouts create rewarded `Trajectory` objects, then grouped trajectories are sent to the backend; in the default shared-resource loop, inference is blocked while training executes. This scaffold keeps the trajectory/reward boundary but changes the runtime scheduler around it.

PufferLib's docs emphasize fast vectorized sample production through chunked environment buffers, independent rollout workers, static/backpressured memory movement, and throughput telemetry. This repo adapts those runtime ideas to language-agent workflows without depending on PufferLib internals.

CALM's README frames semantic bandwidth as predicting one continuous vector for a chunk of `K` tokens rather than one token at a time. This repo does not implement CALM training. It provides policy/action interfaces that let ART-style rollouts experiment with chunk, latent-patch, command, and compressed-reasoning decisions, while `ActionUnit` records optional old/new/reference logprobs so higher-bandwidth actions can expose GRPO/CISPO-compatible probability evidence.

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

The scheduler explores `(scenario, action_codec)` arms, estimates marginal reward improvement per dollar-second, credits train-step policy improvement back to the arms and runtime controls that produced each consumed batch, and uses that signal to choose future rollouts, action granularity, train-batch priority, batch cadence, policy lag, active actor count, actor admission delay, and optional early stopping. Concurrent actor decisions reserve in-flight arms before rollout feedback arrives, so async sampling spreads across untried scenario/action pairs instead of stampeding the first zero-pull arm. If a local actor's selected reservation would exceed the hard accounted budget, the runtime cancels the unspent decision and stops the actor before invoking the rollout workflow. If shutdown cancels a local actor after reservation but before queue submission, the runtime records that cancelled assignment as zero-reward failed rollout feedback, releases the reservation, and debits the reserved or explicit rollout cost. When scenario, codec, or global sample-cost evidence exists, first-time exploration still happens before exploitation but cheaper estimated unobserved arms are tried first and the decision metadata reports the estimated rollout dollar-seconds. Under downstream queue saturation, the scheduler can reduce the active actor cap and delay actor admission before a rollout starts; local actors share each actor-cap choice across a pool sweep, and slots above the chosen cap yield without creating separate control probes, so only admitted rollout work remains available for objective credit. Positive objective signal reduces the initial delay preference so useful sampling keeps flowing, while low-signal saturation backs off before spending rollout/API/tool cost. Train feedback uses each arm's own previous train score as the improvement baseline, then multiplies positive improvement by useful trajectory count. This keeps the run-level north-star numerator from falsely crediting an arm just because the previous train step came from an easier or lower-reward workflow. For heterogeneous ART workflows whose reward scales are not directly comparable, `ObjectiveScheduler(reward_scale_normalization="arm_range")` preserves raw reward telemetry but divides rollout and train positive-improvement control credit by each arm's observed reward range before scoring future choices. Raw reward efficiency is an explicit opt-in scoring weight; by default rollout selection and batch priority are driven by marginal rollout objective plus train-improvement objective. When configured, confidence-aware scoring subtracts an uncertainty penalty from arms with sparse or high-variance objective samples, so noisy spikes do not automatically dominate rollout or batch-priority decisions. With `min_rollout_coverage_fraction`, the scheduler can reserve a bounded share of rollout decisions for under-covered arms after the initial sweep, preventing diagnostic workflows or action granularities from being permanently starved by the current best objective estimate. Set `max_rollout_coverage_cost_fraction` to skip that coverage override when the under-covered arm has already consumed too much of active arm rollout spend. Queued train batches are rescored by the scheduler at consume time, quality-adjusted against the actual trajectories in the batch, boosted or discounted by observed full scheduling-action tuple payoff when stamped, cost-normalized by explicit queued sample/API/tool dollar-seconds when present, boosted when useful batches approach the active policy-lag limit, and discounted when probability-accounted action units show high old/new logprob drift; that same drift signal can tighten train-batch cadence and the active policy-lag limit before more stale samples are admitted. Rollouts can set `metrics["cost/dollar_seconds"]` or `metadata["cost/dollar_seconds"]` for total sample/API/token/tool/GPU cost, or `rollout/dollar_seconds` for rollout-only cost. When total sample cost is present, the runtime subtracts separately stamped queue-wait and admission-delay cost before passing rollout cost to scheduler feedback, so the accounted denominator remains exact instead of double charging backpressure. Otherwise runtime duration is multiplied by the configured infrastructure rate and stamped as `rollout/dollar_seconds` before enqueue so train-batch priority sees the same sample denominator as rollout feedback. Actor queue-wait cost is stamped onto trajectories as `cost/actor_queue_wait_dollar_seconds` and included in scheduler rollout cost, so backpressure lowers the marginal objective of the arms and runtime settings that caused it. Scheduler admission-delay wall time is stamped onto trajectories as `cost/actor_admission_dollar_seconds`, included in per-arm rollout objective cost, and credited to `scheduler/control/admission_delay_ms_*` from rollout, train, and stale feedback. With `rollout_cadence_lag_control_weight > 0`, rollout feedback also credits the stamped cadence and policy-lag values under `scheduler/control/cadence_*` and `scheduler/control/policy_lag_*`, so those knobs can adapt during sample production before a train or stale event arrives. The scheduler also attributes rollout, train, stale, queue-wait, and admission cost back to individual actor slots under `scheduler/actor/*`, making active actor-count decisions auditable by which actor slots produced or wasted reward-improving experience. Trainers can report `cost/dollar_seconds`, `train/dollar_seconds`, or `trainer/dollar_seconds` in `TrainResult.metrics` or metadata; that explicit cost feeds runtime telemetry and scheduler train-objective credit. Time spent waiting for the next train batch is also charged into the candidate train denominator, so batch cadence pays for trainer idle time instead of treating it as free. Promotion evaluator cost is added to the scheduler's train-objective denominator for that candidate, so expensive publication gates lower marginal reward improvement per dollar-second instead of being treated as free. ROI patience can use the train-only objective or, with `continuation_objective="accounted"`, the same reward-improving numerator divided by rollout, queue-wait, actor-admission, trainer, trainer-wait, and promotion cost accumulated for that train interval. Set `max_accounted_dollar_seconds` to make that accounted denominator a hard continuation budget; once exhausted, the control plane stops before spending on another train step. Cadence, policy-lag, active-actor-count, and admission-delay choices are bounded control bandits: the configured or pressure-preferred value is tried first, pressure can still force an initial wider batch or lower actor cap when there is no positive signal, and then `control_exploration_bonus` plus rollout/train/stale credit selects the candidate value with the best observed reward-improving experience per dollar-second. Once cadence, actor-count, or policy-lag controls have rollout, stale, or train feedback, that feedback can override pressured widening, actor-cap backoff, or configured-lag protection before more stale or low-ROI work is admitted. Rollouts can add `action/safe`, `action/quality`, `reconstruction/accuracy`, `reconstruction/safe`, `verifier/passed`, `failure/mode`, `failure/modes`, `verifier/failure_mode`, or `verifier/failure_modes` metadata; unsafe or custom failed actions receive zero effective reward, no train-improvement credit, and negative train-batch priority when unsafe penalties are enabled. The scheduler also tracks failure modes such as verifier failure, reconstruction drift, and user-defined verifier modes, and the adaptive action space treats nonzero failure rate as a safety signal when promoting or retiring chunk codecs.

Runtime-control train credit defaults to `control_train_objective="accounted"`, so cadence, lag, actor-count, and admission-delay values learn from the train interval's reward-improving useful experience divided by rollout, queue, admission, trainer, trainer-wait, and promotion spend. In mixed batches, that accounted denominator is still distributed through each arm's actual train-improvement credit rather than raw trajectory reward, so a high-reward non-improving workflow cannot steal runtime-control credit from the lower-reward arm that moved the policy. Arm train credit still uses candidate train spend so rollout/action arms keep local policy-improvement attribution.

Every admitted rollout is also stamped with a `scheduler/joint_action_key` that combines the selected scenario/action arm, train cadence, policy-lag limit, active actor cap, and admission-delay bucket. The tuple decision is recorded when rollout work is selected and rolled back if the reservation is cancelled before spend, so `scheduler/joint_action/*/decisions` tracks scheduling actions rather than only completed samples. Once a tuple has feedback, `joint_action_objective_weight` adds matching tuple payoff to future rollout scoring and to the bounded cadence, lag, actor-cap, and admission-delay candidate scores when the relevant control context is known. `ObjectiveScheduler.metrics()` reports rollout, train, stale, score, and objective totals under `scheduler/joint_action/*`, so operators can audit the payoff of the full scheduling action tuple when individual knobs hide interactions.

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

Without a `promotion_evaluator`, every train result is promoted, preserving the simple default. With one, each train result becomes a candidate: rejected candidates still count as train spend, but they do not advance the served policy step, do not append a checkpoint, and do not broadcast weights. `PromotionDecision` metadata records candidate score, baseline score, improvement, cost, and reason. Scheduler train credit uses the promotion-effective score under `promotion/score` and divides by trainer plus non-itemized promotion-evaluation overhead, so rejected or expensive candidates do not create false positive policy-improvement credit merely because their trainer-local reward looked high. Built-in promotion evaluators snapshot their learned baseline under `promotion/state`, so resumed runs keep the same publish gate instead of resetting the acceptance threshold.

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

It runs the candidate policy through the same ART-style rollout contract before publication, scores quality-adjusted reward, records evaluation failures, action units, source tokens, duration, and dollar-seconds, and promotes only when the held-out score improves enough. Held-out evaluation trajectories are also tagged with normal scheduler arm metadata and fed back through `observe_rollout()`, so eval successes, failures, action quality, and explicit `eval/dollar_seconds` costs can update future rollout/action choices. Those itemized held-out rollout costs are not added a second time to scheduler train spend; only promotion overhead that is not already represented by evaluation trajectories is charged there. That rollout feedback can also run a promotion-only adaptive action-space refresh, allowing held-out semantic-bandwidth evidence to open the next chunk or latent-patch candidate without waiting for another train batch. Runtime telemetry separately reports published-policy reward-improving experience, so rejected candidates count as spend without pretending to improve the served policy.

Use `ObjectiveScheduler.state_dict()` and `ObjectiveScheduler.load_state_dict()` to persist the controller's learned arm statistics, runtime-control scores, budget counters, configuration, and scalar last-decision metadata alongside ART checkpoints. `ControlPlane` and `AsyncArtBackend` attach that snapshot under `scheduler/state` after train feedback is observed and before the checkpoint update is published. The snapshot intentionally excludes live `Scenario` and `ActionCodec` objects, so resumed runs should reconstruct those from user code and reload only the scheduler's numeric control memory.

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

`AdaptiveActionSpace` starts with token and small chunk actions, then promotes larger `ChunkActionCodec` sizes when scheduler metrics show at least `promotion_min_pulls` live observations, positive objective signal, high action quality, low unsafe rate, low reconstruction drift, observed semantic bandwidth from the current chunk arm, optional old/new/reference logprob coverage thresholds, optional source-token throughput per dollar-second advantage over the active lower-bandwidth parent, and enough reward-improving objective advantage over that parent when evidence exists. Rollout feedback can run a promotion-only refresh before the next train step, so high-throughput actors can try newly useful semantic bandwidth without waiting for a checkpoint update. With `promote_latent_patches=True`, the same evidence can introduce a deterministic `LatentPatchActionCodec` candidate for that chunk size, so the scheduler can test a CALM-like latent action unit inside the same run. Promoted chunk and latent-patch codecs can be disabled after train feedback, or after stale feedback when `demote_on_stale_feedback=True`, shows enough bad evidence, missing configured logprob coverage, or when the nearest smaller active chunk has better reward-improving objective per dollar-second or configured source-token throughput per dollar-second after enough pulls, so unsafe, drifty, stale-wasting, low-bandwidth, untrainable, cost-inefficient, or lower-ROI high-bandwidth actions stop competing for rollout slots. Retiring a promoted chunk also retires latent-patch candidates at that patch size or larger, keeping failed semantic-bandwidth branches out of future rollout selection.

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
from calm_puffer_art import AdaptiveActionSpace, AsyncArtBackend, AsyncArtBackendConfig, ObjectiveScheduler

backend = AsyncArtBackend(
    backend=art_backend,
    config=AsyncArtBackendConfig(train_queue_capacity=3, max_policy_lag=2),
    scheduler=ObjectiveScheduler(),
    action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8),
)
await backend.register(art_model)
result = await backend.train(art_model, art_groups)
```

`AsyncArtBackend` exposes backend-shaped `register()`, `train()`, `restore_control_state()`, `_get_step()`, and `close()` methods, enqueues converted ART groups through the same fixed-capacity stale-aware train ring, lets external ART producers ask the scheduler for rollout/action-codec decisions, observes submitted ART trajectories as rollout/sample evidence, observes train results through the scheduler, updates an optional adaptive action space from scheduler feedback, and publishes checkpoint updates. Submitted rollout evidence can run a promotion-only action-space refresh before the background trainer finishes, so external producers can see newly promoted chunk or latent-patch codecs through `select_rollout()` without waiting for train feedback. The scheduler controls train-batch cadence and the active stale-policy lag limit before each ring consume. Submitted ART trajectories contribute explicit `cost/dollar_seconds`, `rollout/dollar_seconds`, queue-wait, and admission cost to scheduler rollout/accounted-cost telemetry, exposed as `art_backend/sample_dollar_seconds`, so externally produced rollout/API/tool spend is not treated as free. Trainer wait for a ready ART batch is added to the scheduler's train-objective denominator and exposed in backend stats. Backend stats also include wall-clock throughput, submitted train-group cadence, trainer/sample/accounted dollar-seconds, published-policy reward-improving experience, and the attached scheduler/action-space metrics so ART producers can audit the same control objective without separately reaching into the scheduler. Because the bridge does not invent a promotion gate, every completed backend train result is still broadcast, but only positive published-score movement contributes to `art_backend/published_policy_reward_improving_experience`. Stale ART batches fail waiting callers, report estimated lost reward-improving experience back to the scheduler as negative objective feedback, and when configured run a demotion-only action-space refresh so stale-wasting action codecs can be retired; the bridge stamps active cadence and policy-lag metadata before async submission or pending-buffer admission, so stale feedback debits the runtime controls that admitted the doomed work. If a batch or pending group is already stale at submission, its sample spend is still counted but its rollout feedback is recorded as rejected so unusable high-reward trajectories do not improve an arm. Published backend checkpoints include `scheduler/state`, bridge accounting under `art_backend/state`, and, when supplied, `action_space/state`; pass the saved `PolicySnapshot`, `Checkpoint`, or checkpoint metadata to `backend.restore_control_state(...)` before restarting producers to reload scheduler/action-space/bridge memory and the policy step used by stale-lag checks. The wrapper delegates the actual ART loss/checkpoint work to the supplied backend.

For scheduler-chosen ART rollout work, ask the backend for a decision and merge the metadata into the ART trajectory you produce:

```python
from calm_puffer_art import Scenario

assignment = await backend.admit_and_select_rollout(
    scenarios=[Scenario(id="math")],
    actor_id=0,
    configured_actor_count=8,
    trajectory_queue_pressure=0.8,
)
if not assignment.admitted:
    return None
try:
    art_group = await produce_art_group(metadata=assignment.metadata)
except Exception as exc:
    backend.record_rollout_failure(assignment, exception=exc)
    raise
await backend.submit_group(art_model, art_group)
```

`admit_and_select_rollout()` applies the scheduler's continuation, budget, active actor-count, and pre-rollout admission-delay controls for external ART actor pools, then immediately selects and reserves a rollout arm when admitted. If the scheduler recommends stopping because ROI patience, `max_train_steps`, or projected `max_accounted_dollar_seconds` is exhausted, assignment returns `admitted=False` before the actor spends on another rollout. If the selected arm's estimated reservation would exceed the hard accounted budget, the bridge cancels that unspent scheduler decision, releases the reservation, and returns `admitted=False` instead of handing out work that cannot fit the objective envelope. When it sleeps, the delay cost is recorded once in scheduler admission telemetry and stamped into the returned metadata so the submitted trajectory can credit the chosen actor-count and admission-delay values. Once an assignment is admitted, the producer must either submit a trajectory carrying the returned metadata or call `record_rollout_failure(assignment, ...)`; the failure path releases the reservation, records zero-reward failed experience, and debits the reserved or explicit rollout cost into the same accounted-dollar-second objective. When an `AdaptiveActionSpace` is attached, selection reads its current codec set, so chunk or latent-patch codecs promoted from previous ART feedback become available to future ART rollout producers without restarting the backend. The lower-level `admit_rollout()` and `select_rollout()` methods remain available for custom producer loops, but the combined helper is the safer default for projected-budget accounting.

For no-stop-the-world submission, use `submit_train()`:

```python
future = await backend.submit_train(art_model, art_groups)
# Keep producing rollouts while the background trainer consumes the batch.
result = await future
```

`train()` is the compatibility wrapper that awaits the same future. `submit_train()` returns after bounded-ring admission, so callers pay backpressure only when the ring is full. With `AsyncArtBackendConfig(synchronous_fallback=True)`, the backend still receives the original ART groups inline and callers still receive the raw backend result, but the bridge now applies the same rollout/sample accounting, stale-policy rejection, train-feedback observation, action-space update, checkpoint metadata, and published-policy telemetry around that direct call.

For scheduler-controlled batch cadence, submit individual ART trajectory groups:

```python
future = await backend.submit_group(art_model, art_group)
# Later, force a partial batch if cadence has not been reached:
await backend.flush_pending_groups()
result = await future
```

`submit_group()` accumulates compatible groups and flushes them into the train ring when the scheduler's target batch cadence is reached. Each submitted non-stale group also gives the scheduler arm-level pulls, reward, semantic-bandwidth, safety/failure, and sample-cost evidence before train completion, while already-stale submissions are counted as rejected rollout pulls with the same sample spend; ART producers can feed the same action-granularity loop as the local actor runtime when they tag trajectories with scheduler arm metadata such as `task|chunk(chunk_size=2)`. Sample spend is counted when a group is submitted, not only after it reaches the train ring. The same scheduler can tighten `max_policy_lag` when useful train signal appears or the queue is pressured, causing over-stale queued ART batches and partial pending groups to fail with `StaleArtBatchError` rather than training on obsolete experience. Accepted pending groups carry the active cadence and lag metadata while waiting, so those stale drops also debit the arms, cadence values, and lag values that produced the discarded experience.

## Core pieces

- `calm_puffer_art.types`: ART-like primitives for scenarios, action units, trajectories, trajectory groups, checkpoints, and run summaries.
- `calm_puffer_art.actions`: token, chunk, latent-patch, command, and reasoning-step codecs, action logprob summaries, plus the adaptive chunk and latent-patch promotion/demotion action space with checkpointable state.
- `calm_puffer_art.art_adapter`: dependency-free conversion between ART-shaped trajectory groups and local runtime groups, a delegating trainer wrapper, and a structural async ART backend wrapper.
- `calm_puffer_art.scheduler`: objective-driven rollout/action/train-priority/actor-count/cadence/lag scheduler with action-quality, train-improvement, off-policy action-drift priority and lag control, optional per-arm reward-scale normalization, stale-drop, per-actor attribution, pressure feedback, exploratory runtime-control scoring, and checkpointable control state.
- `calm_puffer_art.runtime`: async control plane with actor pools, bounded queues, background group assembly, priority-aware versioned train-batch rings, promotion-gated checkpoint broadcasts, stale-sample filtering, cost attribution, and telemetry.
- `examples/counting_agent.py`: a deterministic trainable toy policy whose reward improves over checkpoints.
- `examples/adaptive_scheduler_agent.py`: a deterministic closed-loop scheduler demo that learns which scenario/action-codec arm has better reward-per-cost signal.
- `examples/adaptive_action_space_agent.py`: a deterministic demo where objective feedback promotes a larger chunk action codec during the run.
- `examples/objective_ablation.py`: deterministic static-vs-objective comparisons that report published-policy north-star lift from scheduler control and adaptive action-space control.
- `docs/art_puffer_calm_synthesis.md`: cleaned integration plan for the ART backend and future optional CALM layer.

## Non-goals

This is not a fork of ART, PufferLib, or CALM. It does not train real LLM weights, implement GRPO, allocate CUDA vector buffers, or learn a continuous language autoencoder. It is a small, typed runtime seam that makes those integrations explicit and testable.
