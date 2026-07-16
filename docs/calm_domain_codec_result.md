# Offline CALM Domain Codec Result

Run date: 2026-07-11 UTC

This proof moved the optional learned chunk path beyond its synthetic four-token
smoke corpus. It trained bounded codecs on Python arithmetic repair fragments,
saved complete checkpoints, reloaded them with strict state and identity
validation, and evaluated reconstruction and fallback behavior on held-out
sequences.

It is an offline representation and persistence proof. It is not a native CALM
policy, does not produce serving-LLM policy probabilities, and is not connected
to ART's GRPO/CISPO loss.

## Protocol

- Domain: small Python code-repair fragments
- Tokenizer: bounded whitespace vocabulary, hash `aa63d64143b770d5`
- Train examples: `8`, containing `96` source tokens
- Holdout examples: `4`, containing `96` source tokens
- Holdout design: unseen full-sequence recombinations of chunks observed during
  training, not unseen-chunk or distribution-shift generalization
- Candidates: chunk sizes `2` and `4`, latent dimension `32`
- Reconstruction gate: exact `1.0`
- Fallback: token actions on unknown tokens or reconstruction drift
- Checkpoints: atomic Torch save, weights-only load, strict state loading, and
  recomputed encoder/scorer/vocabulary identity verification

## Result

| Candidate | Raw train reconstruction | Raw holdout reconstruction | Holdout fallback | Effective held-out bandwidth | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chunk 2 | `1.0000` | `1.0000` | `0/4` | `2.0` tokens/decision | Yes |
| Chunk 4 | Below `1.0` | Below `1.0` | Nonzero | Below `4.0` tokens/decision | No |

Chunk size 2 reached exact reconstruction after `25` autoencoder steps. Its
`86,332`-byte checkpoint survived round-trip loading with identical model,
scorer, vocabulary, and codec identities. Every learned action retained complete
offline old/new/reference scorer coverage. An unknown-symbol probe failed closed
to token actions.

Chunk size 4 exhausted `1,000` autoencoder steps without reaching exact
reconstruction. The local reference run reached `86.46%` raw reconstruction and
fell back on every held-out example; Torch `2.13` reconstructed more examples,
but still emitted reconstruction-drift fallbacks and failed the exact gate.
Kernel-level numeric differences can change the fallback count, while the
stable eligibility decision remains rejection. Its apparent four-token action
width therefore cannot be treated as reliable semantic bandwidth.

## Interpretation

The positive result is narrower and more useful than making both candidates
pass artificially. The checkpoint lifecycle and safety gate now work on a
domain-shaped corpus, and the system has selected one representation candidate
for the next experiment. It also demonstrates why semantic width alone is not a
benefit: the wider candidate loses all compression once correctness-preserving
fallback is applied.

The next milestone is a `CalmPolicyAdapter` that conditions chunk distributions
on the serving model's hidden state. It must emit genuine behavior, updated, and
reference policy logprobs for the sampled latent action, preserve chunk
boundaries in ART trajectories, and feed those values into an actual ART loss.
Until that bridge exists, the scheduler may inspect offline reconstruction and
bandwidth evidence but must not call these scorer values policy probabilities.

## Command

```powershell
python examples\code_domain_chunk_codec.py --json
```

Generated checkpoints and `report.json` are written under
`artifacts/calm_domain_codec/`, which is ignored by Git.
