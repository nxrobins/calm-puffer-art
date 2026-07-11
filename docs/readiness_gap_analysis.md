# Readiness Gap Analysis

This is the seven-axis teardown for the ART-Puffer-CALM scaffold. It separates
what the repo has proven from what still needs stronger live evidence.

## Summary

The strongest local investment is allocation efficiency: the scheduler already
treats rollout choice, action granularity, cadence, policy lag, actor count,
admission delay, train-batch selection, continuation, promotion, and action-space
promotion as payoff-bearing control actions. Cost accounting is broad, but still
depends on callers providing real GPU, API, token, and tool costs instead of a
flat wall-clock rate.

The first real ART checkpoint and a three-seed controlled run now prove the
serverless weight-update path. The binding constraints have moved to signal
quality, real monetary inputs, repeated budget sweeps, and real CALM integration.

## Axis Status

| Axis | Current status | Gap | Next proof |
| --- | --- | --- | --- |
| Signal quality | Mostly exogenous. The runtime accepts rewards, verifiers, promotion evaluators, and failure metadata, but it does not make weak rewards strong. | Better task rewards, held-out verifiers, curriculum, and KL/logprob tuning. | A task where verifier quality predicts downstream policy improvement. |
| Semantic bandwidth | Locally smoke-proven. Token, chunk, latent-patch stand-ins, command units, reasoning-step units, adaptive action-space promotion, and a tiny optional torch learned chunk encoder are implemented. | The learned encoder is smoke-only: no production corpus, no real ART/vLLM loss integration, and no upstream CALM checkpoint. | Run the learned chunk path inside a real ART training job once the async bridge is live. |
| Allocation efficiency | Heavily invested. Joint-action payoff and scoped runtime-control feedback are implemented. | Risk of diminishing returns without real reward/cost signal; the scheduler can optimize only the evidence it receives. | A live run showing scheduler choices improve reward per dollar-second over fixed async ART. |
| Cost measurement fidelity | Runtime accounting is joined by a versioned experiment evidence ledger covering raw performance, tokens, latency, retries, trainer lifecycle, scheduler allocation, pricing coverage, and Pareto views. Missing price is explicit rather than zero. | The valid live run supplied no authoritative inference or trainer rates, and trainer wall time is not the same as billed active GPU time. | Add provider-authoritative token, trainer, tool, and evaluator prices, then run repeated fixed-cost and fixed-quality sweeps. |
| Learnability | Underdeveloped. Current control is local bandit-style learning with exploration bonuses and optional confidence penalties. | Joint-action spaces can become sparse and combinatorial. No sample-complexity model or adaptive exploration budget exists yet. | Stress profiles plus a policy for pruning, factorizing, or backing off from low-evidence joint keys. |
| Integration | Proven for managed ART weight updates: one-step artifact persistence and eighteen controlled updates completed through the direct and async paths. | This does not yet prove long runs, production recovery, cost-to-target, or CALM actions inside real loss. | Run longer budget sweeps with intermediate held-out evaluation and the new telemetry ledger. |
| Scalability | Now locally profiled. `run_scheduler_scalability_profile()` measures key growth, metrics count, state JSON bytes, and selector overhead as the joint-action lattice expands. | Deterministic synthetic profiles do not prove production throughput or memory behavior under long runs. | Track profile outputs over larger grids, then add pruning/factorization if state or selector cost grows too fast. |

## Current Priority

1. Real cost input: replace flat cost-per-second placeholders with itemized GPU,
   API, token, tool, and evaluator costs.
2. Cost-performance sweeps: compare fixed-budget quality, fixed-quality cost,
   time-to-target, and the Pareto frontier across more seeds and checkpoints.
3. Semantic-bandwidth integration: feed the optional learned encoder's explicit
   action logprob and reconstruction contract into a real training run.
4. Scalability guardrails: use the scheduler profile to detect when the current
   tabular joint-action controller needs pruning or factorization.

The current system is not the supremum of the trinity. It is now a live ART
control scaffold whose next bottleneck is empirical: stronger tasks, real cost,
longer runs, and a real CALM contribution.
