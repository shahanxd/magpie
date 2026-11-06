# Robustness card — Tiny Aya Fire + Circuit-Breakers/RR (hardened, M2)

*A deployer-facing summary. Numbers are content-validated with the HarmBench classifier unless
marked "lexicon"; full provenance in [`docs/RESULTS.md`](../RESULTS.md). Model: Tiny Aya Fire
hardened with SafeTune M2 (Circuit Breakers / Representation Rerouting), 150 steps, LoRA on 2
layers, merged. Weights not published.*

**One line.** A plausible-looking defense that **shifts no survival curve**: the loss drops and L0
is clean, but the hardened model costs exactly as much to break as vendor Fire at every level. This
is the honest negative the cost framing is built to surface.

## Cost to break vs vendor Fire (paired, same prompts)

| Attacker level | Fire | fire_rr | Δ |
|---|---|---|---|
| L0 direct | 0.21 lexicon | 0.21 lexicon | none |
| L1 templates | **0.30** HarmBench | **0.30** HarmBench | none |
| L2 prefill | **0.31** HarmBench | **0.31** HarmBench | none |
| L2 prefill curve | k=1 0.80, k=2 0.98, k≥4 1.00 | **identical** | none |

The checkpoint is genuinely the hardened one — its completions differ prompt-by-prompt from Fire's,
so the merged weights loaded — it simply does not cost more to break.

## Verdict for a deployer

- **Do not ship a defense on its training loss.** RR here drove loss 5.0→0.50 and left L0 clean,
  yet moved no attacker-cost curve. Validate a defense by re-attacking it and measuring the
  survival curve, not by its loss or its L0 refusal rate.
- **Representation rerouting did not touch prefill at all** (safety-depth curve byte-identical) —
  a shallow-safety attack needs a depth-targeted defense (Qi et al.), not RR.
- A heavier RR schedule (more steps, more layers) or a different method (CAST) is required before
  any hardening claim; this lightweight pass is not sufficient.

> Companion negative: **M3 RepNoise** on the same model *collapsed* it (all outputs degenerate),
> which the collapse guard flagged automatically — a defense that zeroes ASR by destroying utility.
> Neither lightweight defense is deployable as-is.
