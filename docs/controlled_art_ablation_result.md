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

## Cost And Runtime Result

The performance comparison was nearly flat between the two trained conditions,
but their resource use was not identical:

| Measurement | Direct ART | Async scheduler | Scheduler difference |
| --- | ---: | ---: | ---: |
| Requests | `444` | `444` | `0.0%` |
| Total tokens | `71,402` | `69,432` | `-2.8%` |
| Training-inference tokens | `23,016` | `20,925` | `-9.1%` |
| Prompt tokens | `55,866` | `56,202` | `+0.6%` |
| Completion tokens | `15,536` | `13,230` | `-14.8%` |
| Condition wall time | `474.1 s` | `483.4 s` | `+1.9%` |

At essentially the same shaped-reward delta, the scheduler used fewer training
and completion tokens but took slightly longer wall-clock time. This is a
potential efficiency result, not yet a monetary-cost result. Input, output, and
trainer rates were not supplied, so dollars are unknown rather than zero.

These figures were reconstructed from the raw experiment report after the run.
The instrumented repeat below emits the versioned JSONL evidence ledger described in
[`telemetry.md`](telemetry.md), including cost provenance, coverage, lifecycle
failures, efficiency views, and point-estimate Pareto analysis.

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
learning curve. It should sweep multiple budgets and report both performance at
fixed cost and cost to fixed performance rather than choosing one outcome in
advance.

CALM was excluded. The current learned chunk codec is reconstruction-smoke-only
and does not alter ART inference actions, policy logprobs, or optimizer loss.
Calling this an ART-plus-CALM ablation would overstate what the code runs today.

## Fully Instrumented Repeat

A follow-up on the same date repeated the protocol with held-out evaluation at
every checkpoint and pre-registered mean-reward targets of `0.20`, `0.225`, and
`0.25`. The primary run was `20260711-055038`. Direct ART seed `202` and async
scheduler seed `303` each exhausted two five-minute managed-training attempts.
Targeted, predeclared recovery runs `20260711-062014` and `20260711-062201`
repeated only those failed condition-seed cells. Both recoveries completed
without a retry.

The consolidated efficacy view uses completed cells from the primary run and
the corresponding recovery cell where the primary cell failed. It does not
silently erase failed work: the reliability cost is reported separately below.

| Condition | Mean reward delta | Descriptive 95% CI | Final mean reward | Parse-rate delta |
| --- | ---: | ---: | ---: | ---: |
| No training | `+0.0025` | `[-0.0076, +0.0127]` | `0.1610` | `+2.0 pp` |
| Direct ART | `+0.0748` | `[+0.0330, +0.1166]` | `0.2310` | `+16.0 pp` |
| Async scheduler | `+0.0925` | `[+0.0422, +0.1427]` | `0.2478` | `+18.7 pp` |

Scheduler minus direct ART was `+0.0176`, with a descriptive paired interval
of `[-0.0722, +0.1074]`. The trained arms again improved shaped reward while
exact accuracy remained zero. This remains evidence of behavioral movement,
not task mastery or a statistically established scheduler advantage.

### Learning Curve

| Checkpoint | Direct mean reward | Direct learning tokens | Scheduler mean reward | Scheduler learning tokens |
| --- | ---: | ---: | ---: | ---: |
| Before training | `0.1562` | `0` | `0.1554` | `0` |
| Step 1 | `0.2254` | `2,612` | `0.2226` | `2,612` |
| Step 2 | `0.2084` | `5,101` | `0.2475` | `4,767` |
| Step 3 | `0.2310` | `7,713` | `0.2478` | `6,975` |

The curve changes the interpretation. Most of the initial gain arrived after
one update. Direct ART then regressed at step 2 and partially recovered, while
the scheduler improved materially at step 2 and was flat at step 3. More
training was not monotonically better in either condition.

At the aggregate-mean level, both arms reached `0.20` after `2,612` learning
tokens. Direct ART reached `0.225` at that point; the scheduler reached it after
`4,767` learning tokens. Neither aggregate mean reached `0.25`.

The per-seed attainment view is less brittle than an aggregate threshold:

| Target | Direct ART | Async scheduler |
| --- | ---: | ---: |
| `0.20` | `3/3` seeds | `3/3` seeds |
| `0.225` | `2/3` seeds | `3/3` seeds |
| `0.25` | `1/3` seeds | `2/3` seeds |

### Cost And Reliability

| Clean completed-cell measurement | Direct ART | Async scheduler | Scheduler difference |
| --- | ---: | ---: | ---: |
| Requests | `744` | `744` | `0.0%` |
| Total experiment tokens | `118,074` | `114,140` | `-3.3%` |
| Learning-inference tokens | `23,139` | `20,925` | `-9.6%` |
| Successful-cell wall time | `258.5 s` | `260.7 s` | `+0.8%` |

The scheduler again used fewer tokens, especially in the learning phase, at
similar successful-run wall time. This is the most consistent positive result
across the two experiments. It is still a token-efficiency result rather than
a dollar-cost result because authoritative inference and trainer rates were
not supplied.

The complete campaign consumed `2,052` inference requests and `325,147` tokens,
including failed work. The two abandoned cells account for `264` requests,
`41,856` tokens, four failed training attempts, and `1,200.3 s` of managed-call
time before recovery. One failure occurred in each trained condition, so this
run does not establish a reliability difference between them. It does show
that backend completion variance can outweigh the observed token savings and
must remain part of any product-level cost objective.

This repeat closes the missing learning-curve and cost-to-target evidence gap.
The next rigor gap is larger: use more seeds, nonzero exact accuracy, multiple
training budgets, authoritative prices, and an actual CALM treatment that
changes inference or optimizer behavior.
