# Calm Puffer ART

An experimental control plane for language-agent RL.

The repo asks a specific systems question:

> Can arbitrary scored agent trajectories produce more useful policy improvement
> per dollar-second if rollout production is asynchronous and action units are
> larger than single tokens?

It combines three ideas without vendoring any of their upstream projects:

- **ART-shaped interface**: user code owns scenarios, rollouts, trajectory
  groups, rewards, checkpoints, and trainable policies.
- **Puffer-like runtime shape**: actors produce samples continuously through
  bounded queues while a trainer consumes ready trajectory batches.
- **CALM-like semantic bandwidth**: the scheduler can compare token actions
  against chunk, latent-patch, command, or reasoning-step action units.

The core package has **no runtime dependencies**.

## Current Status

This is a research scaffold, not a production trainer.

What works today:

- dependency-free async rollout/trainer control plane
- objective scheduler for `(scenario, action_codec)` arms
- bounded queues, stale-sample handling, backpressure, and cost telemetry
- adaptive action space for chunk promotion/demotion
- ART-compatible adapters that do not import ART unless the optional extra is installed
- deterministic local ablations and codegen verifier workloads
- optional torch chunk-encoder smoke
- optional Azure Foundry live Python-repair benchmark

What is intentionally out of scope for the core package:

- training real LLM weights
- implementing GRPO/CISPO losses
- managing CUDA/vLLM serving
- implementing PufferLib internals
- implementing upstream CALM continuous chunk training

## Install

From the repo root:

```powershell
py -m pip install -e .
```

Optional extras:

```powershell
py -m pip install -e ".[dev]"      # pytest only
py -m pip install -e ".[calm]"     # torch-backed chunk encoder smoke
py -m pip install -e ".[art]"      # real ART structural smoke
py -m pip install -e ".[foundry]"  # Azure Foundry live codegen benchmark
```

For direct example execution from a checkout:

```powershell
$env:PYTHONPATH = "src"
```

## First Runs

Start with the cheap local checks:

```powershell
$env:PYTHONPATH = "src"
python examples\python_codegen_showcase.py --json
python examples\objective_ablation.py
python examples\scalability_profile.py
python -m unittest discover -s tests
```

The most useful optional live check is the Azure Foundry budget race:

```powershell
$env:PYTHONPATH = "src"
python examples\azure_foundry_codegen_ablation.py --json --budget-race --budget-dollar-seconds 160 --env-path D:\topology-engine\.env --deployment gpt-5.5 --task-limit 17 --train-steps 512 --model-call-budget 256
```

That command makes live model calls. Keep credentials in a dotenv file with:

```text
COVENANT_AZURE_KEY=...
COVENANT_AZURE_ENDPOINT=...
COVENANT_AZURE_API_VERSION=...
```

## What To Run

| Goal | Command |
| --- | --- |
| Fast toy rollout/trainer smoke | `python examples\counting_agent.py` |
| Static vs objective scheduler | `python examples\objective_ablation.py` |
| Adaptive chunk action-space demo | `python examples\adaptive_action_space_agent.py` |
| Deterministic codegen semantic sweep | `python examples\codegen_semantic_sweep.py --json` |
| Three-condition local codegen showcase | `python examples\python_codegen_showcase.py --json` |
| Scheduler state-size and timing profile | `python examples\scalability_profile.py` |
| Torch learned chunk smoke | `python examples\chunk_encoder_smoke.py --json` |
| Real ART object compatibility smoke | `python examples\live_art_bridge_smoke.py --backend structural --json` |
| Live Azure Foundry train-step ablation | `python examples\azure_foundry_codegen_ablation.py --json --env-path D:\topology-engine\.env --deployment gpt-5.5` |
| Live Azure Foundry fixed-budget race | `python examples\azure_foundry_codegen_ablation.py --json --budget-race --budget-dollar-seconds 160 --env-path D:\topology-engine\.env --deployment gpt-5.5` |

## Architecture

The runtime centers on four objects:

```python
summary = await ControlPlane(config).run(
    scenarios=scenarios,
    initial_policy=policy,
    trainer=trainer,
    workflow=rollout,
    action_space=AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4),
    scheduler=ObjectiveScheduler(),
)
```

- `Scenario`: user-defined unit of work.
- `workflow`: async function that runs an agent and returns a scored
  `Trajectory`.
- `trainer`: consumes `TrajectoryGroup` batches and returns a `TrainResult`.
- `ObjectiveScheduler`: decides which scenario/action codec/runtime controls to
  spend on next.

The runtime reports both ordinary throughput and cost-accounted objective
metrics. The headline metric is:

```text
north_star/accounted_published_policy_reward_improving_experience_per_dollar_second
```

This only credits reward improvement that produced a published checkpoint. Failed,
unsafe, stale, rejected, or expensive samples remain in the denominator.

## Main Modules

| Module | Purpose |
| --- | --- |
| `calm_puffer_art.types` | Scenario, message, action, trajectory, checkpoint, train-result primitives |
| `calm_puffer_art.actions` | Token/chunk/latent/command/reasoning codecs and adaptive action-space state |
| `calm_puffer_art.runtime` | Async actor/trainer loop, queues, stale filtering, checkpoint broadcast, telemetry |
| `calm_puffer_art.scheduler` | Objective scheduler, budget accounting, arm stats, runtime-control attribution |
| `calm_puffer_art.art_adapter` | Dependency-free ART object conversion and async backend wrapper |
| `calm_puffer_art.chunk_encoder` | Optional torch learned chunk codec smoke |
| `calm_puffer_art.codegen_ablation` | Deterministic Python codegen verifier experiments |
| `calm_puffer_art.foundry_codegen` | Optional live Azure Foundry Python repair workload |
| `calm_puffer_art.objective_ablation` | Synthetic and torch-gated scheduler ablations |
| `calm_puffer_art.scalability` | Scheduler scale/readiness profiler |

## Action Units

An `ActionUnit` can represent a token, chunk, latent patch, command, or reasoning
step. Every action can carry:

- source-token count
- old/new/reference logprobs
- reconstruction metadata
- safety and verifier metadata
- arbitrary scheduler/accounting tags

This lets the scheduler compare action granularities by reward-improving value
per dollar-second, not just tokens/sec.

Built-in codecs:

- `TokenActionCodec`
- `ChunkActionCodec(chunk_size=K)`
- `LatentPatchActionCodec`
- `CommandActionCodec`
- `ReasoningStepCodec`

`AdaptiveActionSpace` starts from token plus a small chunk codec and can promote
or retire larger chunks when scheduler feedback shows useful, safe,
cost-effective semantic bandwidth.

## ART Integration

The package does not import ART at top level. Install `.[art]` only when you want
the real structural smoke:

```powershell
py -m pip install -e ".[art]"
$env:PYTHONPATH = "src"
python examples\live_art_bridge_smoke.py --backend structural --json
```

The bridge preserves raw ART group and trajectory objects in metadata, so the
control plane can use local scheduler telemetry while a real backend can still
receive ART-shaped objects.

Manual real-backend modes are available:

```powershell
python examples\live_art_bridge_smoke.py --backend serverless --json
python examples\live_art_bridge_smoke.py --backend local --json
```

Those modes can require credentials, GPU resources, and current ART backend
availability.

## Azure Foundry Workload

The live Foundry benchmark repairs embedded Python functions and verifies the
generated code in a separate timeout-bounded subprocess. It compares:

- `static_art`: fixed token-level round-robin baseline
- `scheduler_only`: objective scheduler with token actions
- `full_trinity`: objective scheduler plus token/chunk semantic action units

Train-step ablation:

```powershell
$env:PYTHONPATH = "src"
python examples\azure_foundry_codegen_ablation.py --json --env-path D:\topology-engine\.env --deployment gpt-5.5 --task-limit 17 --train-steps 160 --model-call-budget 256
```

Fixed-budget race:

```powershell
$env:PYTHONPATH = "src"
python examples\azure_foundry_codegen_ablation.py --json --budget-race --budget-dollar-seconds 160 --env-path D:\topology-engine\.env --deployment gpt-5.5 --task-limit 17 --train-steps 512 --model-call-budget 256
```

The fixed-budget race is the sharper test when asking whether the system wins on
both performance and cost. It stops every condition on the same accounted
dollar-second ceiling and reports:

- learned verified repairs
- accounted dollar-seconds
- learned repairs per dollar-second
- model calls
- verifier passes
- token/chunk pulls
- semantic bandwidth

Current empirical caveat: on the 17-task Foundry workload, full-trinity found a
qualified performance-plus-cost win versus `scheduler_only` at one sampled budget,
but did not beat the static round-robin baseline overall. The task bank still
saturates too easily for a broad performance claim.

## Metrics To Watch

Useful run-summary keys:

| Metric | Meaning |
| --- | --- |
| `north_star/accounted_published_policy_reward_improving_experience_per_dollar_second` | Published policy improvement per accounted spend |
| `costs/accounted_dollar_seconds` | Rollout, queue, trainer, stale, and promotion cost charged to the run |
| `actions/semantic_bandwidth_tokens_per_decision` | Average source tokens carried by each action decision |
| `scheduler/arm/*/pulls` | How often a scenario/action arm was sampled |
| `scheduler/arm/*/total_improvement_per_dollar_second` | Arm-level value signal |
| `action_space/active_codecs` | Count of currently active action codecs |
| `foundry/learned_solutions` | Verified repair tasks stored by the Foundry trainer |
| `foundry/codec/*/pulls` | Foundry workload pulls by action codec |

## Development

Run the default validation:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python examples\objective_ablation.py
python examples\scalability_profile.py
git diff --check
```

Optional checks:

```powershell
python examples\chunk_encoder_smoke.py --json
python examples\live_art_bridge_smoke.py --backend structural --json
python examples\azure_foundry_codegen_ablation.py --json --budget-race --budget-dollar-seconds 160 --env-path D:\topology-engine\.env --deployment gpt-5.5
```

The optional checks may need `torch`, `openpipe-art`, `openai`, credentials, or
cloud/GPU resources depending on the mode.

## Design Constraints

- Core imports stay dependency-free: `import calm_puffer_art` should not import
  `torch`, `openai`, `art`, `vllm`, `transformers`, or dataset packages.
- Optional integrations are behind extras and example-level imports.
- Rollout cost should be explicit when API/tool/GPU spend is involved.
- Verifier or reconstruction failures should produce zero useful credit, not
  silent positive scheduler signal.
- Runtime accounting should include queued, stale, backpressured, rejected, and
  promotion-evaluation spend.

## Docs

- `docs/architecture.md`: higher-level runtime architecture.
- `docs/art_puffer_calm_synthesis.md`: original ART/Puffer/CALM synthesis.
- `docs/readiness_gap_analysis.md`: gaps before a serious external integration.

## References

- ART: <https://github.com/OpenPipe/ART>
- PufferLib: <https://github.com/PufferAI/PufferLib>
- CALM: <https://github.com/shaochenze/calm>
