# Magpie

**Robustness, priced as attacker cost.** Magpie measures how much effort it takes to break a
hardened, on-device multilingual model, Cohere **Tiny Aya Fire** (3.35B). It measures in
English and in **Hinglish**, the romanized, code-mixed Hindi–English that hundreds of millions
of people actually type. Every number is reported next to **over-refusal**. Magpie then tests
whether cheap white-box measurements *predict* that price before any attack is run.

> Refusal rate is the wrong safety number for an open-weight model on a phone: anyone holding
> the phone holds the weights. The question that matters is *what it costs to break it*, at each
> attacker capability level.

Lexsi AI Safety & Alignment Hackathon, Track 2a (red-teaming and robustness evaluation).
Full plan: [`docs/PLAN.md`](docs/PLAN.md) · decisions and conventions: [`docs/CONTEXT.md`](docs/CONTEXT.md).

## How it works

- **One cost axis.** Every attack attempt is priced in FLOPs (forward ≈ 2·P·tokens,
  training ≈ 6·P·tokens, P = 3.35e9), queries, and wall-clock on reference hardware (Kaggle T4,
  an RTX 3050 laptop). Input-space and weight-space attacks land on the same x-axis.
  This lives in [`magpie/costs.py`](magpie/costs.py).
- **Survival analysis.** A prompt survives until a judge marks a *valid* success; unbroken
  prompts are censored. Outputs are Kaplan–Meier curves of P(still refused) against log-FLOPs,
  median cost-to-break with bootstrap CIs, and a Cox regression with white-box covariates.
- **Attacker ladder.** L0 natural → L1 black-box → L2 white-box input → L3 weight edits. Every
  attack implements one interface ([`magpie/attacks/base.py`](magpie/attacks/base.py)), so cost
  logging and survival bookkeeping are automatic.
- **Built on the Lexsi stack.** CuratorKIT builds the datasets, SafeTune provides the hardening,
  attacks and judges, CircuitKIT finds the refusal circuits, and AuditKIT keeps the audit record.

## Layout

```
magpie/
  costs.py    attacker-cost model (FLOPs / queries / wall-clock)
  types.py    Attempt / Outcome / PromptRecord schema
  config.py   run configs, seeding, provenance (git SHA + config hash)
  serve/      vLLM / HF transformers / echo behind one generate()
  attacks/    Attack interface + runner (L0–L3 attacks land here)
  data/       dataset builders, Hinglish tools (variants, CMI), loaders
  judges/     refusal lexicon, local judges, translate-then-judge, calibration
  mech/       refusal margins, safety depth, refusal rank, circuits
  analysis/   survival curves, Cox models, BoN power laws, figures
  runners/    Kaggle GPU kernels, HF Hub publishing
  defenses/   SafeTune hardening configs (M2–M6)
configs/  notebooks/  space/  tests/  docs/
```

## Quickstart

```bash
make install   # .venv with Python 3.12, CPU torch, dev + test extras
make test      # CPU suite: tiny random cohere2 model, no GPU, no network
magpie cost --prompt-tokens 400 --output-tokens 256 --n-samples 64
```

GPU work runs on Kaggle (T4) through `magpie/runners/kaggle.py`, which reads credentials from
the environment and never from code. Tiny Aya is attached from the Kaggle Models mirror
(`cohere-labs/tiny-aya`).

## Reproducibility

- fixed seeds everywhere
- every result row carries a git SHA and a config hash ([`magpie/config.py`](magpie/config.py))
- hypotheses are pre-registered in `docs/PREREGISTRATION.md` before the main runs

## Responsible red-teaming

- We release prompts, scores and statistics.
- Harmful completions are **gated** on Hugging Face and never committed to this repository.
- Attacked or abliterated weights are **not** published.
- Material findings go to Cohere Labs and Lexsi before publication.
- Tiny Aya is used under CC-BY-NC-4.0 and the Cohere Labs Acceptable Use Policy (non-commercial
  safety research).

## Team

| Role | Modules |
|---|---|
| R1 Attacks & infra | `attacks`, `serve`, `runners`, `defenses` |
| R2 Data, languages & judges | `data`, `judges` |
| R3 Mechanisms, analysis & story | `mech`, `analysis`, article, Space |
