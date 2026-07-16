# First Real ART Weight-Update Result

Run date: 2026-07-11 UTC

This experiment produced the first verified remote weight update through the
project's ART bridge. It proves that verifier-scored trajectories can reach a
managed ART trainer and produce a persisted LoRA checkpoint. It did not improve
the small held-out slice, so it is an integration milestone rather than an
algorithmic result.

## Result

- Base model: `OpenPipe/Qwen3-14B-Instruct`
- Project: `calm-puffer-art-proof`
- Model: `weight-proof-20260711-024935`
- Model ID: `87bd0b68-8bcd-443a-abb3-a766f2d83c5d`
- Checkpoint: step `0 -> 1`
- Artifact: `nxrobins-supremum/calm-puffer-art-proof/weight-proof-20260711-024935:step1`
- Verified non-uniform reward groups: `4/4`
- Training trajectories: `16`
- Training exact accuracy: `1/16` (`6.25%`)
- Training mean verifier reward: `0.3094`
- Held-out exact accuracy: `0/4 -> 0/4`
- Held-out mean verifier reward: `0.1935 -> 0.1911`
- Held-out exact-accuracy delta: `0.0`
- Held-out mean-reward delta: `-0.0024`

ART's progress output reported one gradient step with `grad_norm=1.92`,
`loss=-0.128`, and `probs_corr=0.999`. The persisted step-1 artifact is the
binding evidence that the update completed.

## Usage And Cost

The successful attempt used `3,541` inference tokens: `509` for the baseline
held-out pass, `2,523` for training trajectories, and `509` for post-update
evaluation. A preceding managed-training heartbeat failure used another `3,449`
inference tokens and produced no checkpoint, for `6,990` tokens across both
attempts.

The report records monetary rates as zero because a model-specific inference
rate for `OpenPipe/Qwen3-14B-Instruct` was not supplied. This means monetary
cost is unknown, not proven to be zero. W&B says managed training is free during
the public preview and failed training jobs are not charged for GPU time; token
inference and artifact storage remain account-dependent.

## Failures Found

1. The first managed training job failed after about 20 minutes with
   `Training job failed: activity Heartbeat timeout`. Registration, baseline
   evaluation, and all four reward-varied training groups had succeeded.
2. The retry completed its gradient step and persisted step 1, then the client
   raised `No module named 'wandb'` while ART recorded provenance. The `art`
   optional dependency now installs `wandb`, and the proof harness can recover
   an already-published checkpoint without training again.
3. Because the provenance exception occurred before `AsyncArtBackend` received
   the train result, the saved control-plane metrics show a failed batch and no
   publication. The recovered report labels those metrics as a pre-recovery
   failure snapshot rather than presenting them as successful bridge telemetry.

## Interpretation

This run closes the question of whether the repository can drive a real ART
weight update: it can. It does not show that one update improves generalization,
that the scheduler improves ART, or that the CALM-style action layer contributes
anything yet. The held-out set is too small, and its result was flat on exact
accuracy and slightly negative on graded reward.

The next research gate is a repeated, fixed-budget ablation with larger train
and held-out manifests. The bridge must also complete its normal publication
path with the corrected dependency before scheduler attribution can be treated
as end-to-end evidence.
