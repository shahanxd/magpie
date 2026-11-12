# Robustness card -- Tiny Aya Fire (vendor, M0)

*A deployer-facing summary. Numbers are content-validated with the HarmBench classifier unless
marked "lexicon"; full provenance (run dirs, git SHAs, CIs) in [`docs/RESULTS.md`](../RESULTS.md).
Model: `CohereLabs/tiny-aya-fire`, 3.35B, cohere2, open weights.*

**One line.** Safe to a direct ask, holds against public jailbreak templates, but its safety is
**shallow** (one forced token bypasses it) and it is **defenceless to weight edits** -- the on-device
threat that matters for an openly released model.

## Cost to break (English)

| Attacker level | Capability | ASR (HarmBench) | ~cost (log₁₀ FLOPs) |
|---|---|---|---|
| L0 direct | query only | **0.05** | 12.9 |
| L1 templates | query + public jailbreaks | **0.29** | 13.1 |
| L1 best-of-N (synthetic) | query + cheap resampling | **0.19** | 13.0 |
| L2 prefill | controls the decoder | **0.31** (lexicon 1.00; 1 token → 0.80) | 12.6 |
| L3 abliteration | edits the weights | **0.68** | 13.5 |

## Over-refusal (benign prompts)

13% of benign-but-scary English prompts are refused at L0 (XSTest-safe + OR-Bench-Hard); jailbreak
templates *lower* it to ~7% (the "answer directly" framing suppresses benign refusals too).

## By language (L0, HarmBench)

| English | Hinglish | Devanagari |
|---|---|---|
| 0.03 | 0.03 | **0.28** |

A real safety-coverage gap in **Devanagari** (~9× English). Hinglish is as robust as English once
model *collapse* (degenerate output on romanized code-mixing) is excluded.

## Deployer guidance

- **Do not rely on the refusal rate.** A refusal-lexicon monitor over-counts safety 4-5× and, in
  Devanagari, reports the model as ~4× *less* safe than it is. Use a content judge + a collapse
  guard.
- **Assume prefill = broken.** If any surface lets a caller set the first reply tokens, treat the
  model as unprotected (ASR 0.80 at one token). Guard the decoder, not just the prompt.
- **Weights are the boundary.** An attacker with the file breaks 2 in 3 harmful prompts by
  abliteration. On-device deployment must assume this; safety training alone does not help here.
- **Watch Devanagari specifically** -- it is the weakest language surface.
- **A cheap pre-screen works:** a prompt's refusal-direction margin predicts its cost-to-break
  (H4), so a red-team budget can be focused before any attack is run.
