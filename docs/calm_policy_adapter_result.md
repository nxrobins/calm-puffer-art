# State-Conditioned CALM Policy Adapter Result

Run date: 2026-07-11 UTC

This proof connects the eligible code-domain chunk-size-2 checkpoint to a
state-conditioned latent action policy and ART 0.5.18's actual Torch loss.
It closes the synthetic-scorer gap in the local optimization contract: old,
new, and reference logprobs now come from policy snapshots conditioned on an
external state vector and score the same sampled latent action.

It does not yet use hidden states from a serving language model, and ART's
managed serverless backend does not expose a custom latent-action tensor hook.

## Protocol

- Codec: code-domain chunk size `2`, latent dimension `32`
- State dimension: `32`
- State source: deterministic context features for each code chunk
- Policy: Gaussian latent head with a `64`-unit hidden layer
- Behavior/reference snapshots: frozen copies of the fitted policy
- Actions: `48` across `8` same-domain sequences
- Loss: `art.loss.loss_fn` from OpenPipe ART `0.5.18`, PPO mode
- Update: one Adam step on the current policy only

ART's token loss expects old logprobs and advantages shifted relative to model
outputs. `build_art_chunk_loss_batch()` preserves that contract by placing each
chunk's behavior logprob, advantage, mask, weight, and group id at the target
position while current/reference policy scores occupy the preceding prediction
position. Variable action sequences are padded and masked.

## Result

| Measurement | Result |
| --- | ---: |
| Exact reconstructed actions | `48/48` |
| Behavior logprob coverage | `1.0` |
| Current logprob coverage | `1.0` |
| Reference logprob coverage | `1.0` |
| Policy fit MSE | `16.1725 -> 0.000055` |
| ART policy loss | `-0.2499` |
| Gradient norm | `15,140.99` |
| Current policy state changed | Yes |

The large gradient is expected from the deliberately narrow Gaussian behavior
distribution and proves that the tested loss is active rather than detached.
It is not a recommended production optimizer setting.

## Interpretation

The project now has an executable chunk-action optimization seam, not merely
post-hoc chunk metadata. The behavior policy samples a latent conditioned on
state, and current/reference policies evaluate that same action. ART's own loss
then produces gradients in the current policy head.

The remaining boundary is architectural rather than mathematical. Serverless
ART trains tokenized `messages_and_choices` and does not accept these custom
state, latent, and chunk-logprob tensors. The next proof therefore needs a local
open-weight model whose hidden states feed `StateConditionedChunkPolicy`, plus a
custom ART backend that owns optimizer and checkpoint publication for the base
model, chunk head, and codec together.

## Command

Run the domain checkpoint proof first, then the policy proof:

```powershell
python examples\code_domain_chunk_codec.py --json
python examples\state_conditioned_chunk_policy.py --json
```

The policy proof is persisted under `artifacts/calm_policy_adapter/report.json`.
