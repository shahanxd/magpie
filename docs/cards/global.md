# Robustness card -- Tiny Aya Global (vendor, M1)

*A deployer-facing summary. Numbers are content-validated with the HarmBench classifier unless
marked "lexicon"; full provenance in [`docs/RESULTS.md`](../RESULTS.md). Model:
`CohereLabs/tiny-aya-global`, 3.35B, cohere2, open weights -- the vendor's safest Tiny Aya variant.*

**One line.** Behaves almost identically to Fire across the ladder: robust to a direct ask, holds
against templates, shallow to prefill, and broken by weight edits. "Safest variant" does **not**
mean more expensive to break.

## Cost to break (English)

| Attacker level | Capability | ASR | ~cost (log₁₀ FLOPs) |
|---|---|---|---|
| L0 direct | query only | **0.07** (HarmBench) | 12.9 |
| L1 templates | query + public jailbreaks | **0.29** (HarmBench) | 13.1 |
| L2 prefill | controls the decoder | **0.16** (HarmBench; lexicon 1.00, 1 token → 0.89) | 12.6 |
| L3 abliteration | edits the weights | **0.68** (HarmBench) | 13.5 |

## Over-refusal (benign prompts)

13% at L0, falling to ~5% under jailbreak templates -- the same robustness/over-refusal trade as
Fire.

## Deployer guidance

- **Fire ≈ Global at L0/L1/L3**, so picking the "safer" vendor label buys ~no extra cost-to-break
  there; the real separation is attacker *capability* (L0→L3), not model choice. The one exception
  is **prefill**, where Global is content-safer (0.16 vs Fire's 0.31) -- but both are fully
  bypassed by the *lexicon* (1.00), so this edge is invisible without a content judge and should
  not be leaned on.
- The prefill and weight-edit cautions from the Fire card apply unchanged: guard the decoder,
  assume weight access breaks the model.
- Global was evaluated in English; the per-form (Hinglish/Devanagari) and best-of-N studies were
  run on Fire. Treat Fire's Devanagari gap as the likely Global behaviour pending a matched run.
