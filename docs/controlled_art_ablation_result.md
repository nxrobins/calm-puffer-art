# Controlled ART Ablation Result

Run date: 2026-07-11 UTC

This experiment compared no training, direct ART training, and ART training
through the adaptive async scheduler under a common manifest and fixed request
budget. It found a repeatable shaped-reward and response-format improvement
after three ART updates. It did not produce an exact answer, distinguish the
scheduler from direct ART, or test a real CALM treatment.

## Protocol

- Base model: `OpenPipe/Qwen3-14B-Instruct`
- W&B project: `calm-puffer-art-ablation`
- Run ID: `20260711-034333`
- Seeds: `101`, `202`, `303`
- Manifest seed: `20260711`
- Manifest fingerprint:
  `4d5c9334e4bc42b40ffe50e12fb6d572475a061cdff7eb162590b7f80eb1bfdf`
- Per seed: `12` train tasks and `50` held-out tasks across four difficulty
  strata
- Per trained condition: `3` updates, `4` groups per update, and `4` rollouts
  per group
- Learning rate: `5e-6`
- Total: `1,188` inference requests, `191,912` tokens, and `18` completed
  weight updates
- Managed-training retries in the valid run: `0`

The no-training arm repeated held-out evaluation against the same registered
base model to measure endpoint and evaluation drift. Training rollouts used
distinct deterministic request seeds; held-out requests were paired by seed.

## Aggregate Result

| Condition | Mean reward delta | Descriptive 95% CI | Exact delta | Parse-rate delta |
| --- | ---: | ---: | ---: | ---: |
| No training | `+0.0005` | `[-0.0144, +0.0154]` | `0.0` | `-0.7 pp` |
| Direct ART | `+0.0643` | `[-0.0222, +0.1508]` | `0.0` | `+12.7 pp` |
| Async scheduler | `+0.0657` | `[-0.0160, +0.1474]` | `0.0` | `+13.3 pp` |

Relative to the no-training drift, direct ART improved mean reward by `+0.0638`
and the scheduler arm by `+0.0652`. Their descriptive intervals still include
zero. Scheduler minus direct ART was only `+0.0014`, with a wide interval of
`[-0.1273, +0.1302]`; this run provides no evidence that scheduling improved
the outcome.

Every held-out exact-match score was `0/50` before and after training. The
positive result is therefore a shaped-reward result, not task mastery.

## Per-Seed Reward Deltas

| Seed | No training | Direct ART | Async scheduler |
| ---: | ---: | ---: | ---: |
| `101` | `-0.0064` | `+0.0929` | `+0.0348` |
| `202` | `+0.0044` | `+0.0744` | `+0.1003` |
| `303` | `+0.0035` | `+0.0255` | `+0.0620` |

The trained conditions improved shaped reward on all three seeds. The ordering
between direct and scheduled ART changed by seed.

## What Changed

Across the `150` held-out responses per condition, direct ART and scheduled ART
both raised parseable `FINAL=...` output from roughly `69-70%` to `82.7%`.
Completion tokens fell from `6,495` to `4,169` for direct ART and from `6,617`
to `4,168` for scheduled ART. The no-training arm stayed near `69%` parseability
and its completion tokens increased slightly.

Mean shaped reward among already parseable answers also rose from `0.2268` to
`0.2699` for direct ART and from `0.2292` to `0.2717` for scheduled ART. That
suggests the effect was not only formatting, but formatting and brevity account
for a substantial part of the aggregate gain.

The scheduler allocated its `36` groups as `23` hard, `7` medium, `3` easy, and
`3` challenge. This demonstrates adaptive allocation, but its outcome was
indistinguishable from balanced direct ART at this sample size.

## Invalid Attempts And Harness Fixes

An initial attempt reused one endpoint seed for every rollout in a group. That
collapsed several groups to identical rewards and made those conditions
invalid. The harness now derives a stable seed from experiment seed, task,
checkpoint, group, and rollout index, and tests both determinism and uniqueness.

A second attempt exposed an upstream training job that stopped emitting events
before checkpoint 3. The upstream client had no polling timeout. The harness
now bounds each managed call, checks the remote checkpoint before retrying the
same trajectories, and rejects unexpected multi-checkpoint jumps. The valid
run did not invoke this recovery path.

## Interpretation

The repository now has evidence that repeated ART updates can change held-out
behavior in the intended direction under a fixed live budget. The strongest
observed behavior is better adherence to the required answer format, shorter
responses, and somewhat better modular proximity among parsed answers.

This is encouraging engineering and early empirical evidence, not a research
claim. Three seeds give very wide intervals, exact accuracy remains zero, and
the scheduler has not beaten direct ART. A stronger next experiment needs a
task family with nonzero base exact accuracy, more seeds, a reward that reports
format compliance separately from correctness, and enough updates to test a
learning curve.

CALM was excluded. The current learned chunk codec is reconstruction-smoke-only
and does not alter ART inference actions, policy logprobs, or optimizer loss.
Calling this an ART-plus-CALM ablation would overstate what the code runs today.
