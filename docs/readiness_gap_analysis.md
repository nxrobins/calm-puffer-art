# Readiness Gap Analysis

This is the seven-axis teardown for the ART-Puffer-CALM scaffold. It separates
what the repo has locally proven from what still needs a real training run.

## Summary

The strongest local investment is allocation efficiency: the scheduler already
treats rollout choice, action granularity, cadence, policy lag, actor count,
admission delay, train-batch selection, continuation, promotion, and action-space
promotion as payoff-bearing control actions. Cost accounting is broad, but still
depends on callers providing real GPU, API, token, and tool costs instead of a
flat wall-clock rate.

The binding constraint is integration. Until one real ART training job runs
through a real model, real environment, real inference workers, and real compute
cost telemetry, the project remains a control-plane scaffold with deterministic
local evidence, not an end-to-end reward-improving training system.

## Axis Status

| Axis | Current status | Gap | Next proof |
| --- | --- | --- | --- |
| Signal quality | Mostly exogenous. The runtime accepts rewards, verifiers, promotion evaluators, and failure metadata, but it does not make weak rewards strong. | Better task rewards, held-out verifiers, curriculum, and KL/logprob tuning. | A task where verifier quality predicts downstream policy improvement. |
| Semantic bandwidth | Locally smoke-proven. Token, chunk, latent-patch stand-ins, command units, reasoning-step units, adaptive action-space promotion, and a tiny optional torch learned chunk encoder are implemented. | The learned encoder is smoke-only: no production corpus, no real ART/vLLM loss integration, and no upstream CALM checkpoint. | Run the learned chunk path inside a real ART training job once the async bridge is live. |
| Allocation efficiency | Heavily invested. Joint-action payoff and scoped runtime-control feedback are implemented. | Risk of diminishing returns without real reward/cost signal; the scheduler can optimize only the evidence it receives. | A live run showing scheduler choices improve reward per dollar-second over fixed async ART. |
| Cost measurement fidelity | Broadly instrumented, but only as good as input costs. Runtime can account rollout, queue, admission, train-ring, trainer, stale, and promotion costs. | Placeholder wall-clock rates are not enough for real dollar-second optimization. | Integrate actual GPU rental, inference API, token, tool, and evaluator costs into trajectory and train metrics. |
| Learnability | Underdeveloped. Current control is local bandit-style learning with exploration bonuses and optional confidence penalties. | Joint-action spaces can become sparse and combinatorial. No sample-complexity model or adaptive exploration budget exists yet. | Stress profiles plus a policy for pruning, factorizing, or backing off from low-evidence joint keys. |
| Integration | Not proven. Structural ART adapters and async backend-shaped wrappers exist, but not a real OpenPipe ART training job in this repo. | Drop-in compatibility, checkpoint handoff, inference worker update, and real trainer semantics remain critical path. | One real ART job through `AsyncArtBackend`, with the same metrics the deterministic examples report. |
| Scalability | Now locally profiled. `run_scheduler_scalability_profile()` measures key growth, metrics count, state JSON bytes, and selector overhead as the joint-action lattice expands. | Deterministic synthetic profiles do not prove production throughput or memory behavior under long runs. | Track profile outputs over larger grids, then add pruning/factorization if state or selector cost grows too fast. |

## Current Priority

1. Integration proof: run one real ART training job through the async bridge.
2. Real cost input: replace flat cost-per-second placeholders with itemized GPU,
   API, token, tool, and evaluator costs.
3. Semantic-bandwidth integration: feed the optional learned encoder's explicit
   action logprob and reconstruction contract into a real training run.
4. Scalability guardrails: use the scheduler profile to detect when the current
   tabular joint-action controller needs pruning or factorization.

The current system is not the supremum of the trinity. It is a useful local
control scaffold whose next bottleneck is empirical: real model, real workflow,
real cost, and real improvement signal.
