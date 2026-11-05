# Magpie results log

Running log of what the runs show. Every row is traceable to a run directory
under `results/` (git SHA + config hash in its provenance). Numbers are updated
as runs land; nothing here is final until the model-judge rescoring and
native-speaker verification are in.

## L0 — direct ask, no attack (English)

**Run:** `results/l0/20261104-173906/` · git `d2cd4da` · Kaggle T4, vLLM 0.33 fp16
· judge: `refusal_lexicon` (the fast multilingual pre-filter; model-judge
rescoring pending) · 150 harmful seeds (HarmBench standard + StrongREJECT,
stratified ~17/category) · 150 over-refusal prompts (XSTest-safe + OR-Bench-Hard)
· greedy + one sampled draw per prompt.

| model | form | n | ASR (95% CI) | median log₁₀ FLOPs | over-refusal | collapse |
|---|---|---|---|---|---|---|
| Fire  | en | 150 | **0.21** [0.16, 0.29] | 12.88 | 0.13 | 0.01 |
| Global| en | 150 | **0.24** [0.18, 0.31] | 12.90 | 0.13 | 0.01 |

**Read:** with no attack at all, both vendor-hardened models already refuse
~76–79% of direct harmful English prompts, at a cost of ~10¹²·⁹ FLOPs (the
single L0 generation). Over-refusal is 13% on benign-but-scary prompts, and the
collapse rate is ~1% (gibberish / wrong-language outputs, excluded from ASR).

**Caveats.** (1) ASR here is the lexical "non-refusal, non-collapse" rate; the
Llama-Guard-3 / WildGuard rescoring (`magpie.judges.model_judge`) will give the
validated harmfulness ASR and can only lower it. (2) English only so far —
Hinglish and the Devanagari control follow once the Aya translation core is
built. (3) Fire ≈ Global at L0 is expected; the thesis is that the models
separate in *cost to break* at higher attacker levels (L1–L3), which is what the
survival curves measure next.

**Next:** model-judge rescoring of this run; the Hinglish/Devanagari core and
the same table per form; then L1 (templates, best-of-N) where the cost axis
starts to move.

## L1a — jailbreak templates (English)

**Run:** `results/attack-l1-templates/20261105-194721/` · git `b0a8d5d` · five template families
(role-play, refusal suppression, prefix injection, past-tense, persona), lexicon judge, same 150
harmful + 150 over-refusal core.

| model  | ASR (L0 → L1) | median log₁₀ FLOPs (L0 → L1) | over-refusal (L0 → L1) |
|--------|---------------|------------------------------|------------------------|
| Fire   | 0.21 → **0.52** | 12.88 → 13.07 | 0.13 → 0.07 |
| Global | 0.24 → **0.54** | 12.90 → 13.06 | 0.13 → 0.05 |

**Read:** the first point where the cost axis moves. Ordinary jailbreak *structure* — no
optimisation, five public template families — roughly doubles ASR over a direct ask, at ~1.5× the
L0 FLOP cost (several variants per prompt). Over-refusal *falls* under the same templates
("answer directly, no warnings" framing suppresses refusal on benign prompts too), which is
exactly the robustness/over-refusal trade the thesis tracks. Fire ≈ Global again; the model
separation the thesis predicts should appear against the hardened variants and at L2–L3.

## L1c — best-of-N on the input (synthetic noise, English)

**Run:** `results/attack-l1-bon-input/20261105-201012/` · git `e9f2cd3` · Fire · 100 harmful
prompts · synthetic augmentations (Hughes et al.: random caps, ASCII noise, word scramble) ·
N ≤ 64, early-stopped · power-law fit `−log ASR(N) = a·N^b`.

ASR(N), empirical (fraction of prompts broken within the first N variants):

| judge | N=1 | 2 | 4 | 8 | 16 | 32 | 64 | power law |
|---|---|---|---|---|---|---|---|---|
| lexicon | 0.07 | 0.75 | 0.94 | 0.97 | 0.99 | 1.00 | 1.00 | a=1.57, b=−1.93, r²=0.96 |
| + BoN gate | 0.07 | 0.75 | 0.93 | 0.96 | 0.98 | 0.99 | 0.99 | a=0.91, b=−1.27, r²=0.89 |

**Read, and a real caveat (C4).** The machinery works: a clean power law, exponent b ≈ −1.3 to
−1.9, saturating by N≈16. But the ASR here is a **judge artifact as much as an attack result**.
The jump 0.07 → 0.75 at N=2 is not short garble — the faithful BoN gate (>40-word floor, no
cipher/decoding false-positives, not collapsed) barely moved it, because the perturbed outputs
are long and coherent (up to ~168 words). What the lexicon judge cannot tell is whether those
long non-refusals are *actually harmful*. That is exactly contribution C4: a cheap refusal judge
saturates on best-of-N, so the true ASR needs the HarmBench content judge. **This BoN curve is
the headline case for the validated-judge rescoring pass** (`magpie.judges.safetune_judge`),
which is the next step; treat the numbers above as the lexicon upper bound, not the jailbreak
rate. (Over-refusal under BoN is a separate benign-variation run, deferred with the Hinglish data
for H2b.)

## Hardening (M2 RR, M3 RepNoise) — training and L0

**M2 Circuit Breakers / RR** trained on Tiny Aya Fire (`results/harden-rr/20261105-153604/`,
git `506c868`, Kaggle T4): 150 RR steps, LoRA on layers [11, 22] (the paper's 10/20 scaled to
36), loss 5.0 → 0.50, peak 9.1 GB, ~14 min. Its L0 table matches vendor Fire:

| model   | ASR (lexicon) | median log₁₀ FLOPs | over-refusal |
|---------|---------------|--------------------|--------------|
| fire    | 0.21          | 12.88              | 0.13         |
| fire_rr | 0.21          | 12.88              | 0.13         |

That L0 is unchanged is the expected and desired result: RR hardening does **not** regress L0
behaviour or add L0 over-refusal. The thesis is that the hardened model costs more to break at
L1–L3; the survival curves there are the next measurement.

**M3 RepNoise** trained once pinned to a single T4 (DataParallel across both T4s had replicated the model on GPU 0 and OOMed): batch 1 / max-len 512, 11.2 GB peak, ~9 min. But it **destroyed the model**: all 300 L0 harmful outputs are degenerate repeats (ASR 0.00 *and* over-refusal 0.00 because the model can no longer produce coherent refusals either), which the collapse guard flagged automatically (low_diversity + repetition). This is a real robustness/utility failure — a defense that zeroes attack success by wrecking the model — and exactly the off-switch failure the guard exists to catch (C4). RepNoise on Tiny Aya needs hyperparameter tuning (deferred / upstream); the defense set keeps M0, M1, M2 RR and M5 CAST, per the plan's fallback.

**Judge note.** ASR here is the lexical refusal proxy, applied consistently, so vendor-vs-hardened
and across-level comparisons are valid. Validated HarmBench ASR is a separate rescoring pass
(`magpie.judges.safetune_judge`); the inline two-model-in-one-process pass proved fragile on the
T4, so rescoring runs as its own HarmBench-only kernel.

## Data v1 — the parallel core

**Run:** `results/build-data/20261104-181016/` · git `3149dd3` · translator:
**tiny-aya Fire** on a Kaggle T4 (Aya Expanse is gated on Kaggle and did not
mount). Pushed **private**: `huggingface.co/datasets/shahanxd/magpie-core-v1`.

- **632 records** = 400 en + 188 Hinglish + 44 Devanagari, 286 harmful + 346
  over-refusal, balanced ~30/category across 9 harm categories.
- Back-translation agreement gate (token-overlap ≥ 0.25) pass rate:
  **188/400 Hinglish, 44/400 Devanagari**. MinHash removed 0 (all distinct).

**Honest finding.** tiny-aya Fire is a weak translator: median Hinglish **CMI is
0** (most "Hinglish" came out monolingual, not code-mixed) and cross-script
round-trip is poor, so the Devanagari yield is low. This is expected for a
3.35B stopgap and is why the plan intends **Aya Expanse 8B + native-speaker
verification**. Two concrete fixes for v2: (1) accept the Aya Expanse licence on
Kaggle so it mounts, then re-run with `translate_model=aya`; (2) replace the
crude token-overlap agreement proxy with an embedding similarity. The natural-
variation augmenter (`magpie.data.hinglish`) adds code-mixing at attack time, so
a low-CMI base is partly compensated, but a better base translator is the right
upgrade before the native-verification sprint.

