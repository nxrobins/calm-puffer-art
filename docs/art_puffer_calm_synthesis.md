# ART-Puffer-CALM Synthesis Notes

This is the cleaned integration plan distilled from the external architecture sketch. It is intentionally split into implemented runtime seams and speculative research work.

## North-Star Metric

The target is reward-improving agent experience per dollar-second, not raw tokens/sec.

That requires three separate levers:

1. ART remains the programmable trajectory and reward control plane.
2. Puffer-like runtime mechanics keep actors, batch assembly, and training moving with bounded buffers.
3. CALM-like action representations test whether one policy decision can carry more task progress than one token.

## Data Flow

```text
user rollout workflow
  -> rewarded Trajectory
  -> optional ART adapter conversion/preservation
  -> ObjectiveScheduler update
  -> same-scenario TrajectoryGroup
  -> ObjectiveScheduler batch priority
  -> VersionedTrajectoryBatch
  -> bounded train-batch ring
  -> highest-priority non-stale batch
  -> TrainerBackend.train(snapshot, groups)
  -> ObjectiveScheduler train update
  -> AdaptiveActionSpace promotion/stale-retirement update
  -> optional PromotionEvaluator publish gate
  -> PolicySnapshot checkpoint
  -> WeightBroadcastChannel update
```

The important invariant is that the user-defined rollout/reward layer does not need to know whether training is synchronous, overlapped, token-level, or chunk-level. The scheduler chooses those runtime and action-shape details around the ART-style trajectory contract.

## Closed-Loop Scheduler

The current `ObjectiveScheduler` is the first closed-loop controller:

- It treats each `(scenario, action_codec)` pair as an arm.
- It estimates marginal reward improvement per dollar-second for each arm.
- It explores untried arms, then prefers arms with better objective estimates.
- It reserves in-flight rollout decisions so concurrent actors explore distinct untried arms before feedback arrives.
- It cancels local rollout selections whose estimated reservation would exceed the hard accounted budget before invoking the workflow, and records later actor cancellation after rollout reservation as zero-reward failed experience with reserved or explicit rollout spend accounted instead of leaving projected budget stuck.
- It orders untried arms by estimated rollout dollar-seconds when related scenario, codec, or global sample-cost evidence exists, so first-time exploration is still guaranteed but early spend is not blind to cost.
- It can enforce an opt-in `min_rollout_coverage_fraction` after the initial sweep, temporarily selecting the most under-covered arm so diagnostic workflows and action granularities keep receiving bounded live evidence, while `max_rollout_coverage_cost_fraction` prevents that override from overspending on an already expensive arm. Forced coverage selections are stamped as coverage-control actions and credited from rollout, train, and stale feedback under `scheduler/coverage_control/*`, making the fairness override's marginal efficiency auditable.
- It gates the static actor pool with a learned active actor cap, so actor count becomes a runtime control without changing the user-facing rollout API; local actors share each cap choice across a pool sweep, rejected slots yield without separate probes, and saturation and low-ROI backoff remain initial preferences that actor-count feedback can override after admitted rollout work has scored the cap.
- It can delay actor admission before rollout under downstream queue saturation, then explore and reuse millisecond delay values based on rollout, train, and stale objective feedback.
- It stamps each admitted local or ART-bridge rollout with a joint scheduling-action key covering the selected scenario/action arm, cadence, policy-lag limit, actor cap, admission delay, and active action-space signature when an adaptive ladder is attached, records the tuple decision at selection time, rolls it back on unspent cancellation, then credits rollout, train, and stale payoff to that tuple under `scheduler/joint_action/*` and reuses exact tuple payoff during future rollout selection plus same-signature-first partial tuple payoff during runtime-control selection.
- It reports top-level joint-action aggregates for tuple count, decisions, feedback updates, positive-objective tuples, total objective, mean objective per decision, and mean objective per feedback update, so experiments can verify that scheduling actions are receiving payoff without knowing every tuple key in advance.
- It scores candidate train batches so ready samples with higher estimated arm, joint-scheduling-tuple, and matching train-selection objective value train first, cost-normalizing queued batches by explicit sample/API/tool dollar-seconds when present, while applying current trajectory quality so unsafe batches from historically good arms lose priority before training; the selected batch is recorded under `scheduler/train_selection/*`, credited from the same train-step objective, and reused in future batch priority, so queue-consume choices are payoff-bearing scheduler actions too.
- It rescores queued train batches at consume time and boosts positive-value batches as they approach the active policy-lag limit, reducing stale reward-improving experience waste before it happens.
- It can subtract a configurable confidence penalty from sparse or high-variance objective samples, so rollout and train-batch priority can prefer steadier marginal reward improvement per dollar-second over one-off spikes.
- It credits train-step reward-improving useful experience back to the scenario/action-codec arms that produced the consumed batch, using each arm's own previous train score as the baseline.
- It can enable `reward_scale_normalization="arm_range"` for heterogeneous workflows, preserving raw reward telemetry while scale-adjusting rollout and train positive-improvement credit by each arm's observed reward range before control decisions are scored.
- It credits train and stale objective feedback back to active actor-count, cadence, policy-lag, and actor-admission delay values, can opt cadence and policy-lag into rollout feedback with `rollout_cadence_lag_control_weight`, reports objective/exploration scores plus mean objective per decision and feedback update under `scheduler/control/*`, mirrors same-signature evidence under `scheduler/control_context/*`, and reuses action-space scoped runtime controls once that context has comparable feedback for at least two candidate values.
- It estimates lost reward-improving experience for stale train-ring drops from arm objective value and sample dollar-seconds, falling back to useful-experience count before value evidence exists; stale sample spend that never passed through rollout feedback is charged once into the accounted denominator instead of remaining stale telemetry only.
- It attributes rollout, train, stale, queue-wait, admission cost, semantic bandwidth, and objective back to individual actor slots under `scheduler/actor/*`, so actor-count control can be audited by marginal actor-slot contribution.
- It converts verifier and reconstruction metadata into effective reward, so unsafe high-bandwidth actions are demoted.
- It records verifier failures, user-defined verifier failure modes, reconstruction safety failures, reconstruction drift failure modes, and numeric reconstruction drift summaries as checkpointed scheduler evidence.
- It explores train-batch cadence candidates after the configured default, then tightens or widens cadence according to reward-improving experience per dollar-second.
- It widens train-batch cadence under trainer saturation before cadence feedback exists, then lets train, stale, and opt-in rollout objective feedback override pressured widening when a wider batch wastes reward-improving experience.
- It explores policy-lag candidates after the configured default, while preserving the configured allowance until known arms have accepted samples or lag feedback exists, then reuses lag values with stronger train, stale, or opt-in rollout objective credit.
- It records cadence, policy-lag, and partial-flush timing-response decisions under `scheduler/timing_response/*`, keyed by selected value, train-ring pressure bucket, pending-batch bucket, preference reason, and active action-space signature when present, so pressure widening, stale-risk tightening, off-policy tightening, protection, manual/compatibility flushes, and payoff reuse are auditable separately from the raw numeric cadence/lag value and from incompatible CALM ladders.
- It keeps the configured lag while known arms still lack accepted samples and no lag feedback exists, so exploration is not starved by stale-sample filtering before stale-waste evidence is available.
- It can stop training early when `roi_patience` is configured and either train-step objective or accounted interval objective stays below threshold, or when `max_accounted_dollar_seconds` exhausts the configured rollout/train/promotion spend envelope; continue/stop calls are recorded under `scheduler/continuation/*`, with continue decisions credited from the next train interval when they lead to spend and keyed by the active action-space signature when a CALM-style ladder is present. Both local actors and the ART bridge cancel unspent rollout selections that would exceed the projected accounted budget instead of leaking phantom reservations.
- It can feed an `AdaptiveActionSpace` that promotes larger chunk codecs and opt-in latent-patch candidates from rollout feedback when observed pulls, objective, quality, reconstruction drift, observed semantic-bandwidth, optional source-token throughput per dollar-second, optional old/new/reference logprob coverage, and active-parent objective-margin signals make higher-bandwidth actions worth trying, retires promoted codecs after train feedback or opt-in stale feedback shows enough bad objective, bandwidth, source-token throughput per dollar-second, quality, drift, failure-rate, safety, missing configured logprob coverage, lower-than-parent objective evidence, negative realized promotion-decision payoff, or optional negative realized source-token throughput payoff after enough post-decision observations, records each promotion or retirement under `action_space/decision/*` with target-vs-parent objective, estimated payoff, post-decision observations, realized payoff, and mean realized payoff per decision or observation, retires dependent latent patches when a chunk branch fails, snapshots that action-space state under `action_space/state`, and exposes a stable active/disabled ladder signature that becomes part of joint scheduling-action attribution.
- It makes raw reward efficiency an explicit scoring weight instead of a hidden default, so the default controller prioritizes marginal rollout and train-improvement objective.
- It snapshots and restores scheduler numeric control memory, including runtime-control scores and exploration configuration, through `state_dict()` / `load_state_dict()`, and checkpoint updates carry that state under `scheduler/state` after train feedback is credited.
- It snapshots adaptive action-space state under `action_space/state` and built-in promotion evaluator state plus promotion-decision payoff stats under `promotion/state`, preserving discovered semantic bandwidth, action-space decision payoff stats, promotion baselines, and promote/reject audit evidence across accepted checkpoints.
- It can resume local runs from a `PolicySnapshot` carrying checkpoint metadata, restoring scheduler/action-space/promotion control state before actor rollout begins and preserving the resumed policy step for staleness checks.
- The ART bridge can also restore scheduler/action-space control state and the bridge policy step from a saved `PolicySnapshot`, `Checkpoint`, or checkpoint metadata before external rollout producers restart, so the async substrate does not relearn scheduler memory or action-bandwidth promotions after process resume.
- The ART bridge exposes train-ring stats, wall-clock throughput, submitted train-group cadence, scheduler metrics, and action-space metrics through one `stats()` call, so external ART producers can audit the objective loop without depending on private scheduler state.
- The ART bridge also reports accepted action units, source tokens, and semantic bandwidth without requiring an attached scheduler, so static ART producer baselines and scheduler-controlled producer pools can be compared on action granularity.
- The ART bridge stale-checks scheduler-cadenced pending groups before they reach the train ring, so partial ART batches that become too old still fail callers, count sample spend, record rejected rollout feedback when already stale on submit, stamp active cadence/lag controls before rejection or buffering, debit stale feedback, and can run an opt-in demotion-only action-space refresh instead of hiding outside the backpressured queue.
- The ART bridge synchronous fallback still calls the supplied backend inline with original ART groups, but it now applies the same rollout/sample accounting, active-lag stale rejection, train feedback, action-space refresh, checkpoint metadata, and published-policy telemetry around that direct call.
- It accepts explicit trainer dollar-second metrics, so train-objective credit can reflect reported GPU/API spend instead of only wall-clock duration times a flat rate.
- It charges trainer wait for a ready batch into the train-objective denominator, so batch cadence pays for idle trainer time.
- It attributes actor queue-wait cost into scheduler rollout denominators, so backpressure is part of arm/control objective feedback rather than telemetry only.
- It attributes producer wait on full train-ring admission into scheduler train denominators for both the local runtime and ART bridge, and charges that same wait as stale-additional dollar-seconds when a batch is discarded before training, so backpressure reduces marginal improvement per dollar-second instead of only incrementing a queue counter.
- It stamps actor admission-delay cost onto trajectories, so pre-rollout backpressure avoidance is visible in accounted dollar-seconds and in the per-arm/control objective audit.
- It treats `cost/dollar_seconds` as total sample/API/tool cost and `rollout/dollar_seconds` as rollout-only cost; when total sample cost is present, local rollout feedback subtracts separately stamped queue-wait and admission-delay cost before scheduler accounting, and otherwise stamps inferred rollout cost before enqueue so train-batch priority reuses the same denominator as arm feedback.
- It can gate candidate checkpoints through a programmable `PromotionEvaluator`, including held-out workflow rollouts that feed back into scheduler arm evidence and promotion-only adaptive action-space refreshes, so train/eval spend is counted even when a candidate is rejected and scheduler credit follows the promotion-effective score rather than raw trainer-local reward. Promotion-evaluation overhead is included in the train-objective denominator for the candidate, while held-out rollout trajectory costs remain in rollout accounting and are not duplicated.
- It records checkpoint-publication promote/reject actions under `promotion/decision/*`, keyed by action and evaluator reason, so candidate improvement, actual published-policy reward-improving experience, and promotion-eval dollar-seconds are auditable as a control decision rather than only aggregate checkpoint telemetry.
- It reports a published-policy north-star companion that counts only positive promoted-checkpoint score improvement times useful promoted-batch experience, so rejected candidates remain spend without becoming useful policy improvement in run-level audits.
- It exposes local runtime throughput, drop/failure-rate, accounted-spend-rate, train-ring admission wait, and stage-utilization telemetry, so Puffer-like sample production can be audited by bottleneck and spend rate rather than only by final reward metrics.
- It defaults ROI patience to `continuation_objective="accounted"`, so stop/continue decisions divide reward-improving useful experience by rollout, queue, admission, train-ring admission wait, trainer, trainer-wait, and promotion cost accumulated for the train interval; `continuation_objective="train"` remains available for trainer-local compatibility.
- It defaults runtime-control train credit to `control_train_objective="accounted"`, so cadence, policy-lag, actor-count, and admission-delay choices are trained against the same accounted interval denominator rather than trainer spend alone, while mixed-batch credit still follows each arm's actual train-improvement contribution instead of raw trajectory reward.

This is still a local bandit controller, not the final supremum. It now has an opt-in reward-scale safeguard for heterogeneous workflows, but broader policy-level comparability still depends on users exposing meaningful reward scales, promotion gates, or evaluators for their domains.

## Bounded Staleness

A trajectory is stale by `S` versions if it was collected under policy step `n` and is consumed while the latest trainable policy is step `n + S`.

The runtime applies two filters:

- `TrajectoryGrouper` drops individual trajectories if `latest_step - trajectory.policy_step > max_policy_lag`.
- `TrajectoryRingBuffer` drops whole `VersionedTrajectoryBatch` objects if their oldest trajectory exceeds `max_policy_lag` by the time the trainer consumes them.

Dropped train batches also call a synchronous discard hook. Schedulers that implement `observe_stale_batch()` receive the discarded groups, policy step, and reason, so stale reward-improving experience becomes negative control feedback instead of only a counter.

Before that drop path fires, the train ring can ask the scheduler to rescore queued non-stale batches at the current policy step. `ObjectiveScheduler` uses the active lag limit stamped onto each batch to add stale-risk priority only to batches with positive estimated objective value, so near-stale useful experience can train ahead of lower-risk work without rewarding low-value stale churn.

This mirrors the useful part of the Puffer bridge sketch without making the current scaffold depend on ART internals or GPU resources.

## Why Not Paste The Full Bridge Yet

The downloaded bridge sketches the right interface, but a safe implementation needs a few corrections before it should become production code:

- Stale batch discard must not recursively await while holding a condition lock.
- Slot freeing must notify blocked producers.
- Batch policy version must come from the trajectories actually collected, not from the trainer's current version at flush time.
- Weight broadcast should be an explicit event stream, not just a mutable path on the backend.

Those corrections are now reflected in the local `TrajectoryRingBuffer`, `VersionedTrajectoryBatch`, and `WeightBroadcastChannel`.

The local `art_adapter` module now covers the safe part of that bridge: structural ART `Trajectory`/`TrajectoryGroup` conversion and raw-object preservation. It can hand preserved ART groups back to a supplied ART-like backend, so the scheduler can inspect rewards, versions, metrics, and messages without reimplementing ART's trainer.

`AsyncArtBackend` adds the backend-shaped lifecycle around that seam. It preserves ART's `train(model, trajectory_groups, **kwargs)` result path, but routes submitted groups through rollout-level scheduler observation, the bounded local train ring, train-selection decision recording, train feedback observation, stale-batch rejection, stale-waste feedback, optional action-space updates, and checkpoint broadcast before delegating the actual ART train call. Its `admit_rollout()` path gives external ART actor pools the same continuation, active-actor-count, and pre-rollout admission-delay controls as the local runtime, so exhausted ROI or accounted spend can stop rollout production before another external sample is created. Its `select_rollout()` path gives external ART producers the scheduler-selected scenario, action codec, cadence, lag controls, and current action-space signature, and `art_rollout_metadata()` converts that decision into plain ART trajectory metadata. The combined `admit_and_select_rollout()` path is the safer default for producer pools because it applies admission, reserves the selected rollout's estimated dollar-seconds, cancels that decision if the reservation would exceed the hard accounted budget, and otherwise returns merged ART trajectory metadata before the actor starts work. If that admitted rollout crashes or is abandoned before submission, `record_rollout_failure()` turns the assignment into zero-reward failed experience, releases the reservation, and accounts the reserved or explicit spend instead of leaving projected budget stuck. Its `submit_train()` path returns a future immediately after ring admission, allowing rollout producers to keep working while the background trainer consumes queued ART groups; if a bounded ring is full, the producer's admission wait is charged into the eventual train denominator and reported under `art_backend/train_ring_admission_wait_*`. Its synchronous fallback records the direct trainer input as the same `scheduler/train_selection/*` action before observing train payoff. Its `submit_group()` path records non-stale submitted ART samples as arm-level pulls with reward, cost, safety, and semantic-bandwidth evidence, records already-stale submissions as rejected rollout pulls with the same sample spend, applies promotion-only action-space updates before train feedback, accumulates individual ART `TrajectoryGroup` objects, and flushes them according to scheduler-selected batch cadence, while the trainer loop asks the scheduler for the active `max_policy_lag` before consuming queued ART batches. Explicit partial flushes from `flush_pending_groups()` and compatibility-triggered partial flushes both add a `control=batch_flush` timing-response key before the batch enters the ring, so short-run, shutdown, or model/kwargs boundary timing overrides receive the same train/stale payoff audit as cadence and lag choices. Submitted action units with old/new logprobs also affect queued train-batch priority, active cadence, and active lag selection: batches with high probability drift are discounted before the trainer consumes them, and the scheduler can tighten cadence and lag before more off-policy samples are admitted. Stale train-ring, synchronous, and pending-group rejections can run opt-in demotion-only action-space refreshes, so CALM-like semantic actions must remain trainable, not merely high-reward. Its stats surface reports bridge-local trainer/sample/accounted spend, published-policy reward-improving experience from actual broadcasts, and publish-all decision payoff under `art_backend/publication/decision/*`; its `art_backend/state` checkpoint metadata keeps published-score baselines and publication-decision stats across resume, so external ART producer loops can audit score-improving updates without adding a fake promotion gate. Untagged ART trajectories receive scenario-scoped scheduler arms such as `math|art`, so train credit does not collapse unrelated workflows into a single global arm.

## CALM Layer Boundary

The current default package provides lightweight action codecs:

- token actions
- fixed-size text chunks
- deterministic latent-patch stand-ins
- command units
- reasoning-step units
- an adaptive chunk and latent-patch promotion action space

The torch-backed CALM path should be optional and later. Before it is used for real policy optimization, it needs:

- a frozen pretrained autoencoder checkpoint;
- tokenizer-specific chunking and padding;
- reconstruction verification for the target domain;
- learned old/new chunk logprob producers that satisfy the explicit `ActionUnit` old/new/reference logprob contract used by scheduler metrics;
- a fallback path for code or tool calls when reconstruction fails.

For code-generation tasks, the conservative plan is `K=2` or `K=4`, syntax-aware chunking, and token-level fallback after verifier failure.

### Constraints & Fallbacks

The learned chunk path is deliberately bounded: CPU-only smoke, seed `1337`, max `1000` train steps, `30s` timeout, max `4096` input bytes, max `256` source tokens, max chunk size `8`, max latent dim `64`, exact `reconstruction_threshold=1.0`, and `max_unknown_tokens=0`. Learned actions may be emitted only when reconstruction is measured against original unpadded source tokens and every action has finite logprobs from explicit frozen `reference`, `old`, and `new` scorer snapshots.

Any violated bound or missing invariant must fail fast with a named `ValueError`, `AssertionError`, `TimeoutError`, or `NotImplementedError`; reconstruction drift, unknown tokens, or decode failure must return token fallback actions marked `action/fallback=True`, `reconstruction/safe=False`, and an explicit `failure/mode`, and those fallback actions must not count toward learned-chunk success, semantic-bandwidth, or logprob-coverage metrics.

## Implementation Phases

Phase 1: Runtime bridge

- Keep the local control plane dependency-free.
- Add structural ART group conversion while preserving raw ART objects for real backends.
- Add a structural async ART backend wrapper with register/submit/train/group-flush/close lifecycle.
- Exercise actor, grouper, train-ring, staleness, and broadcast behavior with deterministic tests.
- Add telemetry for queue wait, ring pressure, stale drops, reward delta, and checkpoint broadcasts.
- Attribute rollout, trainer, queue-wait, wall-clock, and accounted dollar-seconds separately.
- Add online objective feedback so rollout choice, action codec, cadence, and lag are no longer fixed constants.
- Consume high-priority train batches before lower-value ready batches while preserving bounded staleness.
- Use verifier/reconstruction feedback to penalize unsafe action granularities before they affect rollout selection or train-batch priority.
- Assign train-step policy-improvement credit back to the rollout/action arms that generated the consumed trajectories.
- Feed stale train-batch drops back into scheduler arm, cadence, and policy-lag objective memory as lost reward-improving experience.
- Gate checkpoint publication on programmable promotion decisions and feed the promotion-effective score into scheduler train credit.
- Make cadence pressure-aware so saturated trainers receive larger batches unless the objective signal justifies tighter updates.
- Promote larger chunk codecs online when smaller chunks have live pull evidence, positive objective signal, observed semantic bandwidth, and acceptable quality.
- Add ROI patience so the runtime stops spending after repeated low-value training steps instead of blindly exhausting `max_train_steps`.

Phase 2: ART adapter

- Package and verify the structural backend wrapper against live ART as a drop-in `art.Backend`.
- Keep `examples/live_art_bridge_smoke.py` green in structural mode with real ART classes when the optional `art` extra is installed, and use its `serverless` or `local` modes for manual real-backend checks.
- Preserve ART's existing trajectory schema and loss implementation.
- Treat async mode as opt-in and keep a synchronous fallback for debugging.

Phase 3: CALM action layer

- Add an optional `calm` extra for torch-backed encoders.
- Keep `examples/chunk_encoder_smoke.py --json` green as a smoke-only proof that a tiny learned chunk encoder can pass exact reconstruction and emit scheduler-compatible old/new/reference logprobs.
- Train or load a domain-specific autoencoder.
- Compare token-level, adaptive chunk-level, latent-patch, command-unit, and hybrid policies.

Phase 4: Measurement

- Keep the deterministic static-vs-objective ablations green as local proof that scheduler control, adaptive action-space control, the combined local closed-loop controller, and the ART-bridge external-producer path improve the north-star on controlled workloads, including learned actor-count, cadence, lag, chunk-granularity, joint-action payoff, and realized action-space payoff together.
- Benchmark stock ART, ART plus async runtime, and ART plus async runtime plus semantic actions; the deterministic `examples/objective_ablation.py` benchmark reports those three modes under common accounted north-star keys.
- Profile scheduler key growth as the joint action lattice expands; `examples/scalability_profile.py` reports arm count, joint scheduling-action keys, action-space-scoped runtime-control keys, metric count, checkpoint JSON bytes, and selector timing from deterministic synthetic feedback.
- Report reward-improving experience per dollar-second, not only throughput.
- Report throughput and utilization next to the objective so rollouts/sec gains are distinguishable from reward-improving experience gains.
- Compare both wall-clock infrastructure cost and accounted rollout/trainer/queue cost, including explicit rollout dollar-seconds for API, token, tool, or GPU spend when available.
- Include ablations for `max_policy_lag`, `train_queue_capacity`, actor count, and action codec; `run_control_dimension_ablation()` reports deterministic variants for those knobs as sensitivity telemetry rather than assuming every wider setting is universally better.
