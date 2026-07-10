# Real ART Weight-Update Proof

`examples/real_art_weight_update.py` is the first binding test of the ART
integration. It registers a current `art.TrainableModel` through
`AsyncArtBackend`, evaluates a fixed held-out set, samples verifier-scored
training groups, submits one serverless ART update, and evaluates the resulting
checkpoint on the same held-out set.

The script deliberately reports three different claims:

1. `checkpoint_advanced`: ART returned a later checkpoint step.
2. `weight_update_evidence`: the checkpoint advanced, an artifact was named,
   and at least one submitted group had non-uniform verifier reward.
3. `heldout_improved`: exact held-out accuracy increased after the update.

The first two do not imply the third. A single successful run proves that the
training path is real; it is not evidence that the combined method is better
than a baseline.

## Preflight

Install the current supported ART line and inspect readiness without making
network calls:

```bash
python -m pip install -e ".[art]"
python examples/real_art_weight_update.py --preflight --json
```

The serverless backend requires `WANDB_API_KEY`. Keep it in the environment;
the script reports only whether it is present.

## Live Run

Create an ignored `.env` file containing `WANDB_API_KEY=...`, then run:

```bash
python examples/real_art_weight_update.py \
  --env-path .env \
  --input-usd-per-million-tokens 0 \
  --output-usd-per-million-tokens 0 \
  --trainer-usd-per-hour 0 \
  --json
```

Replace the zero rates with the rates that apply to the account. The report
always includes raw request, token, and wall-time usage. Monetary values are
explicit estimates from the supplied rates; the script does not label unknown
provider charges as measured cost.

The generated JSON report is written under `artifacts/` by default, including
when a post-registration phase fails. It includes
the ART artifact identity, trainer metrics, per-task verifier results,
control-plane accounting, and the exact pricing assumptions used.

## Research Gate

Do not turn one successful report into a paper claim. The next gate is a seeded,
fixed-budget comparison of at least these conditions:

- base model with no update
- direct ART update without adaptive controls
- ART plus the scheduler
- ART plus scheduler and CALM-style action bandwidth

Each condition needs repeated seeds, fixed train and held-out task manifests,
quality and safety metrics, total cost, and confidence intervals.
