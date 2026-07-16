# Experiment Telemetry And Monitoring

The experiment telemetry layer preserves performance, cost, latency,
reliability, and scheduler evidence without assuming in advance where the
system's advantage will appear. Raw observations remain available even when a
derived north-star changes.

## Evidence Contract

`TelemetryLedger` writes schema-versioned JSONL events and flushes every event.
The controlled ART harness records:

- run and condition lifecycle events;
- every inference completion, including phase, task, stratum, tokens, latency,
  attempts, verifier reward, exactness, and parseability;
- every managed-training start and finish, including failures, recovery,
  checkpoint advance, trainer metrics, duration, and cost provenance;
- every scheduler group decision and selected difficulty stratum; and
- the final monitoring summary inside the main experiment report.

Prompts, completions, credentials, and environment values are not written to
the evidence ledger. Task-level model content remains in the experiment report
where the harness already records it, not in the monitoring stream.

Each event contains:

| Field | Purpose |
| --- | --- |
| `schema_version` | Reject incompatible evidence instead of guessing |
| `sequence` | Detect missing or reordered events within a run |
| `run_id` | Separate runs when a file contains multiple ledgers |
| `event` | Lifecycle or measurement type |
| `timestamp_utc`, `elapsed_s` | Wall-clock and run-relative timing |
| `dimensions` | Condition, seed, phase, task, stratum, or checkpoint identity |
| `metrics` | Raw numeric and boolean observations |
| `attributes` | Non-metric status and error context |
| `provenance` | Measured, estimated, proxy, or unavailable cost semantics |

## Cost Semantics

Missing pricing is represented as `null`, never `$0`. Token counts remain
available as a labeled proxy. Explicit rates produce estimated monetary cost:

- input tokens times input USD per million;
- output tokens times output USD per million; and
- trainer wall time times trainer USD per hour.

Provider-reported costs can override estimates and retain their provenance.
Failed attempts remain in the denominator. A failed or timed-out trainer job is
not free merely because it produced no checkpoint.

The raw JSONL is immutable. If authoritative rates become available later,
`examples/telemetry_report.py` can reprice the run in memory without rewriting
the evidence.

## Derived Views

The summary reports raw condition totals and keeps performance separate from
cost before deriving:

- held-out reward, exact-accuracy, and parse-rate deltas;
- pooled and per-seed condition views so variance is not hidden by averages;
- input, output, and total tokens;
- inference, trainer, and total USD when fully priced;
- inference latency, condition wall time, retries, and failed trainer attempts;
- reward and exact delta per million tokens;
- reward delta per USD when full monetary coverage exists;
- per-seed and aggregate learning curves at each training checkpoint;
- cumulative learning cost, which excludes held-out measurement overhead;
- cumulative experiment cost, which includes that measurement overhead;
- minimum learning cost to each pre-registered reward target;
- scheduler allocation and concentration; and
- point-estimate cost-performance points, pairwise comparisons, and a Pareto
  frontier.

The frontier is explicitly labeled as point-estimate-only. Confidence
intervals and seed-paired inference remain in the controlled ablation report;
they must be considered before claiming one condition dominates another.

The controlled harness evaluates the held-out manifest after each non-final
training update by default and pre-registers reward targets before inference.
This makes cost-to-target observable without choosing thresholds after seeing
the result. If checkpoint evaluation is disabled, the monitor emits
`cost_to_target_unobservable` instead of fabricating a learning curve.

## Coverage And Alerts

The monitor reports coverage for expected requests, token usage, latency,
performance fields, inference pricing, trainer pricing, scheduler decisions,
and lifecycle completion. Error alerts make the experiment result unhealthy;
warnings preserve the run while preventing over-interpretation.

Current alerts include:

- missing, duplicated, reordered, or mixed-run evidence;
- request-count mismatch;
- unfinished condition or trainer lifecycle;
- failed conditions excluded from comparisons;
- missing inference or trainer pricing;
- retries and failed-training overhead;
- material parseability gain without exact-accuracy gain;
- scheduler allocation concentration above 50%; and
- unavailable cost-to-target evidence.

## Commands

The controlled ART harness creates a sibling `.telemetry.jsonl` file by
default:

```powershell
python examples\controlled_art_ablation.py --env-path .env --json
```

Pass an explicit path or echo each event to stderr for live collection:

```powershell
python examples\controlled_art_ablation.py --env-path .env --json --telemetry-path artifacts\run.jsonl --telemetry-echo
```

Summarize a completed or in-progress ledger:

```powershell
python examples\telemetry_report.py artifacts\run.jsonl --json
```

Recompute cost-to-target with explicit thresholds:

```powershell
python examples\telemetry_report.py artifacts\run.jsonl --json --performance-target 0.20 --performance-target 0.25
```

Apply prices learned after the run and fail automation on monitoring errors:

```powershell
python examples\telemetry_report.py artifacts\run.jsonl --json --fail-on-error --input-usd-per-million-tokens 1.0 --output-usd-per-million-tokens 4.0 --trainer-usd-per-hour 2.0
```

`--expected-inference-requests` turns the request contract into a hard coverage
check. If one JSONL file contains multiple runs, select one with `--run-id`.
`--stale-after-seconds` controls when an active lifecycle becomes a failed
stale lifecycle; its default is ten minutes.

## Remaining Gaps

The ledger makes missing evidence visible; it does not create unavailable
provider facts. Production claims still require authoritative model pricing,
trainer-active time or billed GPU cost, tool and evaluator cost, and repeated
budget sweeps. CALM-specific bandwidth savings also remain unobservable until
the learned action path changes real inference actions and optimizer loss.
