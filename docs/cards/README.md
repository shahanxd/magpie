# Robustness cards

One-page, deployer-facing robustness summaries per model, derived from the validated results in
[`../RESULTS.md`](../RESULTS.md). Each card gives the cost-to-break ladder (ASR by attacker level,
content-validated with HarmBench), over-refusal, language coverage, and concrete deployer guidance.

| Model | Role | Card |
|---|---|---|
| Tiny Aya Fire | M0 — vendor target (South-Asia variant) | [`fire.md`](fire.md) |
| Tiny Aya Global | M1 — vendor's safest variant | [`global.md`](global.md) |
| Fire + RR | M2 — SafeTune hardened (honest negative) | [`fire_rr.md`](fire_rr.md) |

**The through-line:** safety is a *cost*, not a refusal rate. Across these models the real
separation is attacker *capability* (L0 ask → L3 weight edit), not the model label or its training
loss — and a cheap refusal-lexicon monitor would mis-rank all of them.
