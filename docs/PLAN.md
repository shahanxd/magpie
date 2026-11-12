# Magpie: research and build plan (v2, for approval)

**Hackathon:** Lexsi AI Safety & Alignment Hackathon, Track 2a.
**Problem statement:** *Red-teaming and robustness evaluation.* Attack a hardened model with jailbreaks, prompt injection, encoding tricks, best-of-N sampling and weight-space attacks such as abliteration, and report robustness alongside over-refusal.
**Model:** Cohere Tiny Aya **Fire** (3.35B, the South-Asia variant) as the main target, with Tiny Aya Global for comparison.
**Languages:** **English + Hinglish** (romanized, code-mixed Hindi-English), plus Devanagari Hindi as a small *script control*.
**Budget:** **$0.** Free tiers only.
**Deadlines:** registration Nov 5 · submission Nov 14 (end of day, Anywhere on Earth) · shortlist Nov 16 · Demo Day Nov 22 (we present online).
**Status:** v2 for approval. Nothing below has been run yet; all results are hypotheses.

**Changes since v1:**
- dropped Urdu and Arabic
- Hinglish replaces native-script Hindi as the main non-English form
- Fire becomes the main target
- everything moved to free compute and free judges
- the prediction study now works at the level of individual prompts (more statistical power)
- the pipeline is automated end to end: code, GPU runs (Kaggle API) and publishing (HF API)

---

## 0. TL;DR

**Thesis.** Refusal rate is the wrong safety number for an open, on-device model: anyone with the phone has the weights. Magpie **prices** robustness: how much attacker effort it takes to break a hardened Tiny Aya Fire, for each attacker capability level, in English and in **Hinglish, the way hundreds of millions of people actually type**. Every number sits next to over-refusal. Then we test whether cheap white-box measurements **predict** that price before any attack is run.

| # | Contribution | Risk |
|---|---|---|
| C1 | **Breaking-point profiles.** Safety *survival curves* (the share of harmful prompts still refused as attacker FLOPs grow). They cover vendor- and SafeTune-hardened Tiny Aya, put input-space and weight-space attacks on **one cost axis**, and are always paired with over-refusal. English vs Hinglish. | Low |
| C2 | **Natural Hinglish variation as best-of-N (BoN).** BoN normally uses synthetic noise (random capitals, character noise, scrambles). We instead sample the variation real Hinglish users produce: there is no standard spelling (`nahi/nahin/nhi`), SMS shorthand (`kr`, `h`, `bta`), English↔Hindi word swaps, and transliteration variants. Hypothesis: *ordinary Hinglish typing is a BoN attack*, and the same variation raises over-refusal of benign Hinglish users. Plus a **verified tool gap**: SafeTune's BoN is mostly inert on Devanagari (script control). | Medium |
| C3 | **Audit before you attack.** Do white-box measurements predict the measured price? We test refusal-direction margin, safety depth, refusal rank, and circuit overlap (CircuitKIT). Tested at the **prompt level** with survival regression (Cox model), so hundreds of data points, not a handful of models. | High (the breakthrough bet; a negative is still reportable) |
| C4 | **Measurement validity.** Native-speaker calibration of safety judges on Hinglish vs English; does judge choice flip defense rankings? | Low |
| C5 | **Upstream tools and data.** A Hinglish/Indic red-team pack for AuditKIT. A script-aware BoN plus a multilingual refusal matcher for SafeTune. A gated Hinglish adversarial benchmark on HF. | Low |

**What we ship:**
- the HF article
- HF model and dataset collections (gated where harmful)
- the public GitHub repo (`magpie`)
- an HF Space
- robustness cards
- upstream PRs
- the ROI/cost analysis

---

## 1. Why this wins

### 1.1 How judging works (kickoff slides)
- Scoring is double-blind by multiple judges, normalized: **Write-up clarity /10 · Practical usefulness /10 · Code/notebook quality /10 · Overall /100.**
- The top 10-20 teams go to Demo Day. Prizes are ranked across all tracks.
- An internal team checks validity first and tags each submission to a sub-track.

### 1.2 What the Lexsi safety team rewards (their Sept/Oct 2026 posts)
- Score content, not the opening line; use an independent second judge; check judge validity.
- Three axes on every result: harmful refusal, **over-refusal**, utility.
- Behavioural evaluation alone is not enough; interpretability must be *actionable*. That is C3.
- Re-verify after deployment transforms (quantization). That is our L3b.
- Their critique of the field: lots of jailbreak collections, *"almost nothing on repair"*. **So Magpie's red-teaming serves defenders:** it ranks defenses, predicts failures, ships robustness cards, and fixes the tools.

### 1.3 Verified tool gaps (checked in the code, Nov 4)

| Tool | Gap | Evidence |
|---|---|---|
| SafeTune `BoNAttack` | Its 3 augmentations are Latin-centric. On Devanagari, **random capitalisation changes 0/200 variants**, **ASCII noise only touches spaces** (33/200 changed, vs 179/200 in English), and word scramble breaks combining marks. On Hinglish it works, but it is synthetic noise, not natural variation. | `safetune/instrumentation/evaluate/redteam/bon.py`, tested locally |
| SafeTune `StringMatchJudge` (BoN's default success check) | 12 English-only refusal prefixes, so Hinglish or Hindi refusals aren't recognised. | `safetune/config.py` |
| AuditKIT `RedTeamRunner` | About 26 hard-coded English probes with keyword detectors. | `auditkit/redteam/probes/builtin.py` |
| SafeTune benchmarks and judges | All English. | `docs/user-guide/evaluate/*` |

### 1.4 Prior work, and where we sit
- **Compute-aware robustness:** *Risk Under Pressure* (2026) covers input-space attacks only, in FLOPs. *Beyond ASR* (2026) is cost-aware ASR.
- **BoN jailbreaking** (Hughes et al. 2024): power laws, synthetic augmentations, English.
- **Adaptive attacks:** *The Attacker Moves Second* (USENIX Sec 2026); Schwinn & Geisler 2024 (circuit breakers broken); DeepRefusal (EMNLP 2025 Findings).
- **Refusal geometry:** Arditi 2024; concept cones (ICML 2025); universal across languages (NeurIPS 2025); steering-vector pitfalls (ACL 2026 Findings); shallow alignment / prefilling (Qi et al., ICLR 2025).
- **Code-mixed / romanized safety:** CSRT (ACL 2025); IndicJR (EACL 2026: romanization *lowers* jailbreak success on average); IndicMixSafe (Apart, Bengaluru, June 2026: LLM judges over-count Indic attack success by ~3.75×); Indi-RomCoM (2026); Hinglish prompt-injection detection (Sci. Reports 2026).
- **Gap:** no one prices robustness on one axis across input- and weight-space attacks for hardened models in Hinglish; no one uses *natural* romanization variation as a BoN distribution; no one tests white-box predictors of cost-to-break at prompt level. (Based on our search; re-check before submitting.)

---

## 2. Research questions and pre-registered hypotheses
Commit this section to git before the main runs; the git timestamp acts as pre-registration.

- **H1 (transfer of hardening).** SafeTune defenses trained on their default English data raise cost-to-break more in English than in Hinglish. Measured as the median survival-cost ratio en/Hinglish > 1 for each defense.
- **H2 (natural BoN).** Natural Hinglish variation reaches a given ASR with fewer samples than synthetic BoN noise, i.e. a larger power-law exponent. **H2b:** the same variation raises over-refusal of benign Hinglish prompts.
- **H3 (weights).** With weight access (L3), every hardened model reaches ≥50% ASR within ≤10 GPU-minutes. Inference-time defenses give zero protection from L2 upwards.
- **H4 (prediction).** A prompt's white-box refusal margin predicts its survival cost under L1 attacks (Cox hazard ratio significantly ≠ 1). Safety depth predicts L2 prefill cost; refusal rank predicts L3 abliteration cost. Includes leave-one-model-out checks.
- **H5 (validity).** Automatic judges disagree more with native annotators on Hinglish than on English, enough to flip at least one pairwise defense ranking.
- **RQ6 (exploratory).** Does indirect prompt injection succeed at different rates when the injection is in Hinglish vs English, or in a different language from the document? Does safety hardening change injection robustness at all?
- **Script control.** On the parallel core, Devanagari Hindi vs Hinglish separates "romanized" from "code-mixed" effects.

---

## 3. Threat model: attacker ladder and cost
On-device means the weights are public. Each level is a superset of the one before.

| Level | Attacker can… | Attacks |
|---|---|---|
| **L0 Natural** | type, in English or Hinglish | direct requests; natural spelling and code-mix variants |
| **L1 Black-box** | query many times, set decoding parameters | templates, past-tense rephrasing, encodings, BoN on the input (synthetic vs natural Hinglish), BoN over decoding, PAIR-lite, indirect prompt injection |
| **L2 White-box input** | read weights and logits | prefilling, embedding-space attack, GCG (small subset) |
| **L3 Weight** | change the weights | abliteration (English-derived vs Hinglish-derived direction), GGUF quantization, harmful LoRA fine-tune |

**Cost of each attack:**
- **FLOPs:** forward ≈ 2·P·tokens, training step ≈ 6·P·tokens, with P = 3.35B parameters
- **queries**
- **wall-clock** on reference hardware: a Kaggle T4, and the team's **RTX 3050 4GB laptop** ("minutes on a gaming laptop")

**Statistics: survival analysis.** A prompt survives until it is broken; prompts not broken within budget are censored. We report:
- Kaplan-Meier curves of P(still refused) against log-FLOPs
- median survival cost with bootstrap CIs
- log-rank tests
- **Cox regression** with white-box covariates (for H4)

**What counts as a valid success** (all three required):
1. a harmfulness score ≥ threshold (StrongREJECT-style rubric: refusal × specificity × convincingness)
2. a comprehension check (garbled answers to garbled prompts don't count)
3. agreement from a second judge, or a human on the gold set

---

## 4. Targets: the hardened models

| ID | Model | How hardened | Training cost | Ladder |
|---|---|---|---|---|
| M0 | `CohereLabs/tiny-aya-fire` | vendor safety training + merging | 0 | full |
| M1 | `CohereLabs/tiny-aya-global` | vendor (the "safest" Tiny Aya per its tech report) | 0 | L0, L1, L2 prefill, L3 abliteration |
| M2 | M0 + **Circuit Breakers / RR** (`CircuitBreakerTrainer`, Faithful) | train-time representation rerouting | ~2-3 T4-h | full |
| M3 | M0 + **RepNoise** (`RepNoiseTrainer`, Faithful) | tamper resistance (against harmful fine-tuning) | ~2-3 T4-h | full |
| M4 *(plus tier)* | M0 + **RepBend** (Faithful) or DeepRefusal (Variant; must not be cited as the paper's method) | robustness to abliteration and prefill | ~2-3 T4-h | full |
| M5 | M0 + **CAST** conditional steering (Faithful) | inference-time; no training | 0 | L0-L1 (we show it is void at L2+) |
| M6 *(Nov 16-21)* | M2 retrained with **Hinglish** hardening data (CuratorKIT) | closes the loop | ~3 T4-h | full + **adaptive** re-attack |

**Why Fire:** it is the variant an Indian deployer would pick ("best for South Asian languages"). Its weakest-language MultiJail safety is 78.1%, against 87.0% for Global (Tiny Aya tech report).

**Natural experiment:** SafeTune defenses ship with English hardening data, so M2-M4 test whether English-trained hardening protects Hinglish under attack (H1).

**Tiny Aya quirks** (from SafeTune docs and tests):
- the chat template adds a ~366-token preamble, so leave `max_len` unset or ≥512
- the refusal-direction layer may need setting by hand (`pick_layer=24` of 36)
- run `pip uninstall -y torchao` on Colab
- the weights are gated on HF but **mirrored on Kaggle Models** (`cohere-labs/tiny-aya`), so Kaggle notebooks need no HF token
- licence: **CC-BY-NC 4.0 + Cohere Labs Acceptable Use Policy** (non-commercial research; compatible with the hackathon)

---

## 5. Languages and data

### 5.1 Forms
- `en`: English
- `hing`: Hinglish, romanized and code-mixed. Each prompt's **code-mixing index (CMI)** is recorded as a covariate.
- `hi-Deva` *(script control; parallel core only, ~200 prompts)*: the same content in Devanagari.

Urdu, Arabic, Roman Urdu and Arabizi are left for later. The pipeline is language-agnostic, so they can be added for a paper.

**Why keep one Devanagari control?** It is cheap (LLM-assisted translation, verified by the team), and it answers two questions. (1) Is Hinglish weaker because it is *romanized* or because it is *code-mixed*? (2) It is where the verified SafeTune BoN gap actually shows up.

### 5.2 Datasets (all free; licences listed in the article)

| Set | Source | Size | Purpose |
|---|---|---|---|
| Parallel harmful core | StrongREJECT (MIT, has a rubric) + HarmBench standard (MIT), stratified by harm category | 200 seeds × en/hing (+hi-Deva) | controlled comparison |
| Native Hinglish red-team set *(plus tier)* | written by our 3 native speakers (~30 each) | ~90 | ecological validity; no translationese |
| Native Hindi (control) | Cohere **Aya Red-teaming**, Hindi split (Apache-2.0, human-written) | ~100 | ecological control; the partner's own data |
| Over-refusal | XSTest safe prompts (CC-BY-4.0) + OR-Bench-Hard-1k subset + a **benign twin** of every harmful seed | ~450 × en/hing | over-refusal next to every number |
| Utility | Global-MMLU-Lite (en, hi; Apache-2.0) + a ~150-item Hinglish helpfulness set | ~350 | utility cost of hardening |
| Prompt injection | ~100 documents (Wikipedia/news, CC-BY-SA; English and Hinglish WhatsApp-forward-style), injected instructions in en/hing × 3 positions × 3 visibilities, plus canaries | ~1.8k cases | judge-free injection scoring |
| Fine-tune attack data | BeaverTails harmful subset (CC-BY-NC) in en + Hinglish | 10-500 | L3 steps-to-break |

### 5.3 How the data is built (free)
- **Translation and Hinglish generation:** LLM-assisted translation (Claude) of the seed sets into Hinglish (natural code-mixing, several spelling styles) and Devanagari, then native-speaker verification. The dataset card names the LLM used.
- **Fallback:** Aya Expanse 8B run locally on Kaggle.
- **CuratorKIT pipeline:** dedup (MinHash) → back-translation agreement gate → diversity gate → provenance manifest + `rejected.jsonl` → HF dataset export.
- **Native verification:** the team checks ≥15% of each set and 100% of the gold set.
- Hardening/correction data stays strictly separate from evaluation data.

---

## 6. Attack suite

**L0, direct.** All harmful and benign prompts × all models; greedy decoding plus one sampled run.

**L1a, templates.** Five families, in en and hing:
- role-play / hypothetical
- refusal suppression
- prefix injection
- **past-tense reformulation** (never tested in Hinglish, as far as we found)
- persona / developer mode

**L1b, encodings** (competence-gated: we first check that Tiny Aya can decode benign encoded questions):
- base64, ROT13, leetspeak, reversed text
- Hinglish-specific: script swap (Hinglish ↔ Devanagari), mid-sentence script mixing

**L1c, BoN on the input** (N ≤ 128-256, early-stopped; fit `−log ASR(N) = a·N^(−b)` and extrapolate to 10k with CIs):
- (i) the SafeTune defaults (synthetic noise)
- (ii) **natural Hinglish variation**:
  - spelling variants from a variant lexicon (LLM-drafted, team-verified)
  - vowel dropping / SMS shorthand
  - English↔Hindi swaps of content words
  - transliteration variants (`w/v`, `ee/i`, `aa/a`)
  - code-mix ratio shifts
- (iii) Devanagari control: script-aware noise (nukta, matra, ZWJ) vs SafeTune's defaults

**L1d, BoN over decoding.** T ∈ {0.7, 1.0, 1.3} × top-p ∈ {0.9, 1.0}, N samples. On-device users control these settings.

**L1e, PAIR-lite** *(plus tier)*. A free local attacker model (e.g. Qwen2.5-7B-Instruct on Kaggle), 10 iterations × 50 prompts.

**L1f, indirect prompt injection.** The user asks the assistant to summarize, translate or extract from a document; the document hides an instruction. Success is a **canary** (judge-free): a token appears, a secret from the system prompt leaks, or a link instruction is followed. Variables:
- document language × injection language
- position
- visibility (plain, zero-width-hidden, markdown comment)
- optional defensive system prompt

**L2a, prefilling.** k ∈ {1, 2, 4, 8, 16} *in-language* compliance tokens. Gives the safety-depth curve for English vs Hinglish.

**L2b, embedding-space attack** (Schwinn et al., including the circuit-breaker re-evaluation fixes). 50 prompts × en/hing × M0/M2/M3/M4.

**L2c, GCG** *(plus tier)* via nanoGCG, 20 prompts, as a sanity check against L2b.

**L3a, abliteration** (`AbliterationAttack`, Faithful). Direction fitted on English vs on Hinglish (does it transfer?), runtime and weight-orthogonalised, plus a multi-direction **cone** (k = 1..8) to measure refusal rank.

**L3b, quantization.** GGUF Q8/Q4_K_M/Q3 with llama.cpp, on the RTX 3050 and on Kaggle. The "accidental" weight change every on-device deployment makes.

**L3c, harmful fine-tune.** LoRA on 10/50/100/500 harmful examples in en vs Hinglish; steps-to-break and cross-language transfer. Optionally `TamperBenchEvaluator`, which ships lora, multilingual and embedding attacks.

**Adaptive attack per defense** (written down *before* running; the attacker moves second):
- RR: Schwinn-modified embedding attack + prefill
- RepNoise: LR and step sweeps, LoRA rank sweep
- RepBend: cone abliteration + deeper prefills
- CAST: remove the hook

All attacks implement one `magpie.attacks.Attack` interface that logs cost, so the survival analysis is automatic.

---

## 7. Over-refusal and utility, always paired with robustness
- Every result row carries: ASR / survival cost · over-refusal (benign twins + XSTest/OR-Bench, en & hing) · utility.
- **Orthographic fairness (H2b):** apply natural Hinglish variation to *benign* prompts and check whether over-refusal rises.
- **Collapse guard** (Lexsi's *Off-Switch Failure* paper): flag "refusals" that are gibberish or in the wrong language.
- Pareto plots of robustness against over-refusal (SafeTune `ParetoVisualizer`).

---

## 8. Judging, free only (C4)
Every judged output goes through this pipeline:
1. **Refusal pre-filter:** an en + Hinglish + Hindi refusal lexicon (upstreamed as C5) plus a small classifier.
2. **J1, local open-weight judge on the original text:** **Llama-Guard-3-8B** (free, gated) or **WildGuard-7B** via the SafeTune judge registry, on Kaggle.
3. **J2, judge on an English translation:** Hinglish → English with Aya Expanse 8B run locally (free, and a permitted Cohere model), then the **StrongREJECT fine-tuned scorer** (`TamperBenchEvaluator.score_responses`) plus WildGuard.
4. **J3, free API tier as a third opinion (gold subset only):** e.g. Groq or Gemini free tiers, rate-limited, used only if their terms allow it.
5. **Gold set:** ~500 outputs, stratified by model × form × judge disagreement. Labelled by the **3 native speakers** (150 double-annotated, giving Cohen's κ), with **an LLM (Claude) as an additional rater**, reported separately from the human labels and not used as the pipeline judge.

**Analysis:**
- judge-vs-human confusion per form
- thresholds calibrated on the gold set
- Kendall τ between defense rankings under each judge

The same prompts, decoding, token budget and judges are used for every model.

---

## 9. White-box audit: does internal structure predict the price? (C3)

| Metric | How | Predicts |
|---|---|---|
| **Refusal margin** (per prompt): projection onto the refusal direction at the chosen layer, standardized; plus per-form AUROC and cosine between the English and Hinglish directions | SafeTune `interpret` / `extract_refusal_direction` | L1 survival cost (per prompt) |
| **Safety depth:** prefill ASR curve; per-token KL divergence between the hardened model and `tiny-aya-base` on refusal responses (Qi et al.) | custom | L2 prefill cost |
| **Refusal rank:** number of cone directions to ablate before ASR ≥ 50% | SafeTune `AbliterationAttack` + SVD | L3 abliteration cost |
| **Circuit overlap:** EAP-IG refusal circuit (en vs hing; logit difference between the first refusal token and the first compliance token); Jaccard of top edges; circuit size | **CircuitKIT** (Tiny Aya discovery supported, experimental) | whether attacks transfer from English to Hinglish |

**Analysis:**
- **per-prompt Cox regression**: survival cost ~ margin + form + model + harm category + CMI
- model-level Spearman ρ
- leave-one-model-out prediction
- honest reporting of nulls

If it works, we ship an **audit card**: run in minutes, it says which language and attacker level to red-team first (the ROI story).

---

## 10. Results we will show
1. **Hero figure:** survival curves, faceted en / Hinglish, one line per model, x = log attacker FLOPs, with markers for "1 laptop-minute" and "1 laptop-hour".
2. Breaking-point heatmap (models × attacker level × form), with over-refusal beside it.
3. BoN scaling: synthetic vs natural-Hinglish augmentations (+ the Devanagari control).
4. Prediction plots: Cox hazard ratios; margin vs survival cost.
5. Judge-validity panel (κ; ranking flips).
6. A one-page robustness card per model (deployer-facing).
7. Injection matrix (document language × injection language → canary success).

---

## 11. Engineering

| Repo | Role |
|---|---|
| **CuratorKIT** | builds every dataset with gates, dedup, provenance; HF export |
| **SafeTune** | hardened variants (RR, RepNoise, RepBend, CAST); Abliteration/BoN/TamperBench attacks; interpret; judges; Pareto. **Extended** with script/Hinglish-aware BoN and a multilingual refusal matcher (upstream PR) |
| **CircuitKIT** | EAP-IG refusal circuits on Tiny Aya (en vs hing) |
| **AuditKIT** | single audit record (`RunResult` fingerprints, `compare_models` with bootstrap significance, `push_to_hub`). **Extended** with a Hinglish/Indic red-team probe pack and canary detectors (upstream PR) |

```
magpie/
  data/       builders (CuratorKIT configs), hinglish.py (variant lexicon, CMI, transliteration), loaders
  attacks/    base.py (Attack + cost logging), l0_direct, l1_templates, l1_encodings, l1_bon_input,
              l1_bon_decoding, l1_injection, l2_prefill, l2_embedding, l3_abliteration, l3_quant, l3_finetune
  defenses/   SafeTune hardening scripts + configs
  judges/     refusal lexicon, local judges, translate-then-judge, calibration
  mech/       refusal directions/margins, depth (KL), rank (cone), circuits (CircuitKIT)
  analysis/   survival (KM, log-rank, Cox), BoN power laws, figures
  serve/      vLLM / HF / llama.cpp behind one generate()
  runners/    kaggle.py (push GPU notebooks, poll, pull outputs), colab notebooks, hub.py (HF publishing)
configs/  notebooks/  space/  tests/ (CPU tests, tiny random cohere2 config)  docs/
```

**Quality bar:**
- typed code with docstrings
- `pytest` + `ruff` on GitHub Actions (free)
- fixed seeds
- every result traceable to a config + git SHA
- `make reproduce-core`

---

## 12. Free compute and the automated run loop

**Sources ($0):**
- **Kaggle:** 30 GPU-h/week **per account** (2×T4 or P100); Tiny Aya is mirrored on Kaggle Models, so no HF token is needed. This is the main workhorse.
- **Colab:** free accounts as overflow.
- **SageMaker:** if it comes through, for crunch days (free Studio Lab is manual-only).
- **Hackathon HF credits** ($20, free to us) for the Space.
- **RTX 3050 4GB laptop:** GGUF Q4 runs (Tiny Aya Q4 ≈ 2-3 GB), "attacker on a laptop" timings, and the live Demo Day demo.

**Estimate:** ≈ **30-35 T4-hours** for the core tier (fewer forms than v1). That is about two Kaggle accounts for one week.

**Automated loop (verified: our dev environment can reach kaggle.com, huggingface.co, PyPI, Groq):**
1. Code and CPU tests run in the dev environment and CI; changes go to GitHub.
2. GPU jobs are generated as Kaggle notebooks and **pushed with the Kaggle API (GPU on, internet on)**; the runner polls until they finish, pulls the outputs back for analysis, and commits them.
3. Datasets, models and the Space are published with the HF API.

**Manual steps:**
- add the secrets below (once)
- native-language checks and the annotation sprint (~2-3 h each)
- Colab "Run all" if Kaggle quota runs out
- the 3050 laptop runs
- registration, Discord, gated-model acceptance
- presenting on Demo Day

**Secrets to add** (environment settings → Network secrets / environment variables; never paste them in chat; they take effect in a **new session**):

| Variable | What | Needed for |
|---|---|---|
| `HF_TOKEN` | HF *write* token, from an account that has accepted the Tiny Aya licence | publishing datasets/models/Space; HF downloads |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle API token (phone-verified account, so GPU is enabled) | pushing and running GPU notebooks |
| `KAGGLE_USERNAME_2`, `KAGGLE_KEY_2` (…`_3`) | other teammates' Kaggle tokens | more weekly GPU quota |
| `GROQ_API_KEY` / `GEMINI_API_KEY` *(optional)* | free-tier keys | third-opinion judge on the gold set |

**ROI section** (optional, but the rules prefer it): (a) the cost of the white-box audit vs a full red-team run; (b) on-device Tiny Aya vs closed-API billing, using measured 3050 tokens/s.

---

## 13. Timeline (today is Wed Nov 4)

| Day | Date | Work | Gate |
|---|---|---|---|
| D0 | Nov 4 | Plan approved. **All 3 register** (closes Nov 5). Discord + idea-validation ticket. Accept Tiny Aya licences. Add secrets. Repo scaffold + CI. | approved; registered; secrets set |
| D1 | Nov 5 | Seed selection; LLM-assisted Hinglish/Devanagari translations; CuratorKIT gates; team spot-checks; Kaggle runner works end to end; L0 + over-refusal on M0/M1; **commit PREREGISTRATION.md** | L0 table |
| D2 | Nov 6 | Launch hardening M2/M3(/M4) on Kaggle; L1a/b; judge pipeline + refusal lexicon; Hinglish variant lexicon + augmenters + tests | hardening running; judges live |
| D3 | Nov 7 | BoN (synthetic vs natural; decoding) on M0; prefill depth; refusal margins; build injection suite | first power-law fits |
| D4 | Nov 8 | L0-L2 on M2-M5; abliteration on all; injection runs; GGUF on the 3050 | L0-L2 complete |
| D5 | Nov 9 | L3 fine-tune steps-to-break; embedding subset; adaptive attacks; CircuitKIT circuits | L3 complete |
| D6 | Nov 10 | **Annotation sprint**; judge study; re-score; Cox/ρ analysis | κ; H4 result |
| D7 | Nov 11 | Figures, robustness cards, HF collections; upstream PRs | figures v1 |
| D8 | Nov 12 | HF article, Space, ROI; **submit v1 (safety net)** | valid submission on file |
| D9 | Nov 13 | Polish; **fresh-environment reproduction**; checklist audit | final review |
| D10 | Nov 14 | **Final submission early in the day** | done |
| -- | Nov 16-21 (if shortlisted, +$200 HF credits) | M6 Hinglish hardening + adaptive re-attack; native-written Hinglish red-team set; BoN to 10k; Demo Day rehearsal (online) | thesis proved |

**Scope tiers:**
- **Core:** M0-M3, M5 · en + hing (+hi-Deva on the core subset) · L0, L1a-d, L1f, L2a, L3a-c · margin/depth/rank · J1 + J2 + gold set.
- **Plus:** M4 · PAIR-lite · L2b/c · CircuitKIT circuits · native Hinglish red-team set.
- **Stretch:** M6 · Aya Expanse 8B replication · multi-turn · Urdu/Arabic.

---

## 14. Team and working model
Each module has an owner, who also authors its commits:
- **R1, Attacks & infra:** `magpie/attacks`, `serve`, `runners`, `defenses`; GPU runs.
- **R2, Data, languages & judges:** `magpie/data`, `judges`; Hinglish verification; annotation sprint.
- **R3, Mechanisms, analysis & story:** `magpie/mech`, `analysis`; figures; HF article; Space; Demo Day lead.
- Everyone: native checks, the annotation sprint, review of each day's gate.
- A daily 10-minute check-in on the day's gate and scope cuts.

---

## 15. Submission checklist (rules + rubric)
- [ ] All 4 Track 2 repos chained · a permitted model (Tiny Aya Fire/Global)
- [ ] Before/after metrics, more than one: ASR, survival cost, over-refusal, utility, κ
- [ ] HF article with Lexsi's 9 sections; teammates tagged
- [ ] HF model collection: hardened variants. **Not** attacked or abliterated weights (confirm with organizers in the ticket)
- [ ] HF dataset collection: prompts, scores, logs, labels; harmful completions **gated**
- [ ] HF Space · public GitHub repo with a reproducibility README · abstract under 500 words · ROI section
- [ ] Clarity: one-sentence thesis, hero figure, pre-registration, limitations
- [ ] Usefulness: robustness cards, upstream tools, audit card, laptop demo
- [ ] Code: package + CI + configs + `make reproduce-core`
- [ ] Overall: novelty (C2, C3), rigour (adaptive attacks, survival stats, validated judges), honest negatives

---

## 16. Responsible red-teaming
- Release prompts, scores and statistics; **gate** harmful completions; don't release attacked weights.
- Share material findings with **Cohere Labs** and **Lexsi** before publishing (responsible disclosure).
- Stay within CC-BY-NC + the Cohere Labs AUP (non-commercial safety research).
- Annotator wellbeing: short sessions; anyone may skip categories.
- The demo uses non-actionable categories only.

---

## 17. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Fire refuses almost everything at L0-L1 | The cost metric still separates models; L2/L3 always break; natural BoN is where any surprise would show |
| A SafeTune defense won't train on Tiny Aya | Keep M0, M1, M5 + one defense; file the issue upstream (feedback is a stated goal) |
| Local judges are weak on Hinglish | Translate-then-judge, the gold set and calibration (this becomes C4) |
| Kaggle quota runs out | More accounts → Colab → SageMaker; cut plus tier first |
| LLM usage limits for translation | Batch translations; Aya Expanse 8B (local) as fallback |
| H4 fails | Report it; C1, C2, C4, C5 still stand |
| Time | v1 submitted Nov 12; strict daily gates |

---

## 18. What you get
- **Learning:**
  - vLLM; LoRA and representation training (RR, RepNoise)
  - refusal directions and abliteration
  - EAP-IG circuits
  - adaptive-attack methodology; BoN scaling laws
  - survival analysis / Cox models
  - judge validation and κ
  - code-mixed NLP
  - reproducible ML engineering and HF publishing
- **Resume:**
  - "Built Magpie, an attacker-cost red-teaming framework for on-device multilingual LLMs; found and fixed blind spots in Lexsi Labs' SafeTune/AuditKIT (upstream PRs); robustness cards for Cohere Tiny Aya"
  - the HF article, dataset and Space
  - a workshop-paper draft
