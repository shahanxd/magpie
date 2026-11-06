# Reproducing Magpie

Two layers of reproduction: the **CPU core** (regenerates every figure and aggregate from the
committed `results/` tables, no GPU, no network) and the **full GPU pipeline** (the Kaggle T4 runs
that produced those tables). Everything is free: Kaggle T4 quota and open models only.

## CPU core (seconds, offline)

```bash
make install          # one-time: .venv with CPU torch + extras
make reproduce-core   # CPU tests, then rebuild docs/figures/hero_ladder.png from results/
```

`make reproduce-core` runs the offline test suite and re-derives the hero figure (and the ladder
points) from the committed, no-text result tables — so the headline visual is reproducible without
a GPU.

## Full GPU pipeline (Kaggle T4, ~a few GPU-hours total)

Set Kaggle credentials in the environment (`KAGGLE_USERNAME`, `KAGGLE_KEY`), accept the Tiny Aya and
Aya-Expanse licences on Kaggle once, then run, in order. Each command pushes a private kernel,
polls, and pulls no-text aggregates into `results/`; harmful completions stay in the kernel/gated
dataset.

```bash
PY="python -m magpie.runners.kaggle"

# 0. Smoke (sanity: fp16 stability, vLLM vs HF)           — no quota-heavy work
$PY smoke

# 1. Data: the Aya parallel core (en/hing/hi-Deva) → gated HF dataset
$PY run build_data --set translate_model=aya

# 2. English attacker ladder on the vendor models
$PY run l0                                                  # L0 direct + over-refusal
$PY run attack --set attack=l1_templates                   # L1 templates
$PY run attack --set attack=l1_bon_input \
     --set 'attack_params={"source":"synthetic","n":64,"sigma":0.4}'   # L1 best-of-N
$PY run attack --set attack=l2_prefill --set n_benign=0     # L2 prefill / safety depth
$PY run ablation                                           # L3a abliteration (weight access)

# 3. H1 per form (same prompts, en vs hing vs hi-Deva) and C2 natural best-of-N
$PY run attack --set attack=l0_direct --set core_dataset=<user>/magpie-core-v2 \
     --set 'forms=["en","hing","hi-Deva"]'
$PY run attack --set attack=l1_bon_input --set core_dataset=<user>/magpie-core-v2 \
     --set 'forms=["hing"]' --set 'attack_params={"source":"natural","n":64}'

# 4. Defenses: harden, then re-attack paired vs vendor Fire
$PY run harden --set method=rr
$PY run attack --set attack=l1_templates --set hardened_from=<user>/magpie-harden-rr
$PY run attack --set attack=l2_prefill   --set hardened_from=<user>/magpie-harden-rr --set n_benign=0

# 5. Mechanistic audit (H4): per-prompt refusal margins
$PY run mech

# 6. Validate every attack-level ASR with the HarmBench content judge
$PY rescore results/attack-l1-templates/<ts> results/attack-l2-prefill/<ts> ...

# 7. Rebuild the hero figure from the pulled tables
python scripts/make_hero.py
```

**Notes.** Run only one `rescore` at a time (they share one kernel slug). The Cox H4 join runs
locally from the pulled margins + per-prompt break depths (`magpie.analysis.cox`). Attacked and
abliterated weights are never published — the hardened `fire_rr` checkpoint stays in its kernel
output and is attached to the re-attack kernel by reference, never pulled. Every run writes its
git SHA and config into `results/<job>/<ts>/out/`.
