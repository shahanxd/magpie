# Magpie: project context and conventions

Read this together with `docs/PLAN.md` (plan v2) before doing any work. It records what has been decided, what has been checked, and what happens next.

**Status (Nov 4, 2026):** plan v2 **approved**; build phase **through the D1 gate**. The full framework (cost model, vLLM/HF/echo serving, L0–L1 attacks, multilingual judge + collapse guard + model judges, Hinglish tools, survival analysis, Kaggle runner, data loaders, HF publishing) is committed with CPU CI green. The day-1 smoke test passed on a Kaggle T4; `docs/PREREGISTRATION.md` is committed. **First L0 table** (Fire vs Global, en) and **data-v1 parallel core** (private HF dataset) are done — see §3 for verified numbers, `docs/RESULTS.md` for the tables and findings, and §6 for what's next.

---

## 1. Decisions (final unless the team reopens them)

| Topic | Decision |
|---|---|
| Hackathon / track | Lexsi AI Safety & Alignment Hackathon, **Track 2a: Red-teaming and robustness evaluation** |
| Thesis | Price robustness as attacker cost (survival curves over FLOPs), always next to over-refusal; test whether white-box metrics predict that cost. See PLAN §0. |
| Main target | `CohereLabs/tiny-aya-fire`. Comparison: `CohereLabs/tiny-aya-global`. Base for KL/depth: `CohereLabs/tiny-aya-base`. |
| Languages / forms | `en`, `hing` (Hinglish: romanized, code-mixed; record CMI), and `hi-Deva` as a **script control on the parallel core only** (~200 prompts). Urdu and Arabic are deferred. |
| Budget | **$0.** No paid APIs or compute. Free tiers only. |
| GPU | **Kaggle is primary** (API-driven, T4 ×2). Colab is manual overflow. SageMaker may come later. The team's RTX 3050 4 GB laptop runs GGUF/llama.cpp jobs and the Demo Day demo. |
| Judges | Local open-weight: Llama-Guard-3-8B, WildGuard-7B, StrongREJECT scorer (via SafeTune). Translate-then-judge with Aya Expanse 8B. Human gold set from 3 native speakers, plus an LLM rater reported separately. Optional Groq/Gemini free tiers only for a third opinion. |
| Translations | LLM-assisted (Claude), verified by native speakers. The dataset card states this. |
| Demo Day | Online |
| Name | **Magpie** (not all caps) |

## 2. Environment and access
- GitHub: this repo, push access from the dev environment.
- Secrets (environment variables): `HF_TOKEN` (write), `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`, `KAGGLE_KEY`.
  - The Kaggle credential is the **new-style single API token**, stored under both names. Try `KAGGLE_API_TOKEN` first and fall back to `KAGGLE_KEY`.
  - Only the owner's Kaggle account is set for now. Ask the team for `KAGGLE_USERNAME_2/_3` + tokens when the GPU quota gets tight (expected around Nov 7–8).
- Gated HF access requested: tiny-aya-fire, -global, -base, aya-expanse-8b, meta-llama/Llama-Guard-3-8B, allenai/wildguard (Meta/AllenAI may still be pending). Check before relying on them.
- Network from the dev environment, verified Nov 4: kaggle.com, huggingface.co, pypi.org, api.groq.com, openrouter.ai reachable.
- Tiny Aya is mirrored on **Kaggle Models** (`cohere-labs/tiny-aya`, 5 variations). Attach it as a model source in Kaggle notebooks, so no HF token is needed there.
- Licences: Tiny Aya is **CC-BY-NC-4.0 + Cohere Labs Acceptable Use Policy** (non-commercial research; fine for us).

## 3. Verified facts to build on (checked in code, Nov 4)
- **SafeTune `BoNAttack` augmentations are Latin-centric** (`src/safetune/instrumentation/evaluate/redteam/bon.py`). Out of 200 variants:

  | Language | random capitalisation | ASCII noise | word scramble |
  |---|---|---|---|
  | en | 200 | 179 | 184 |
  | hi-Deva | 0 | 33 (spaces only) | 189 (breaks matras) |
  | ur | 0 | 36 | 189 |
  | ar | 0 | 28 | 182 |

- **SafeTune `StringMatchJudge`** (BoN's default success check): 12 English-only refusal prefixes (`src/safetune/config.py: ADVBENCH_REFUSAL_PREFIXES`).
- **AuditKIT red-team**: about 26 hard-coded English probes + keyword detectors (`src/auditkit/redteam/probes/builtin.py`).
- **SafeTune benchmarks/judges are English-only.** Attacks available: `AbliterationAttack`, `BoNAttack`, `TamperBenchEvaluator` (lora/multilingual/embedding/refusal-ablation/GCG), `ParetoVisualizer`.
- **Tiny Aya quirks:**
  - the chat template adds a ~366-token preamble (`max_len` unset or ≥512)
  - refusal-direction layer: set `pick_layer=24` (of 36) if the sweep picks a bad layer
  - Colab: `pip uninstall -y torchao`
  - 36 layers, d_mlp 11008, cohere2 architecture
- **CircuitKIT** supports Tiny Aya discovery (EAP, EAP-IG, …); experimental and gated.
- **T4 has no bf16**, so run fp16 and check for numerical issues on day 1. Don't use the P100 (recent vLLM doesn't support it).

### Verified in the build environment (Nov 4, from runs, not memory)
- **Tiny Aya Fire config (from HF):** `cohere2`, 36 layers, hidden 2048, d_mlp 11008, 16 heads / 4 KV, head_dim 128, vocab 262144, embeddings tied → **P ≈ 3.349B** (matches the 3.35B tech-report figure; `magpie.costs` is built on this). Chat template adds a **~370-token** system preamble; generation EOS is `<|END_OF_TURN_TOKEN|>`, pad id 0.
- **Smoke test on Kaggle T4 (fp16), both backends green in one run:** vLLM 0.33 **208 tok/s** batch (load ~160 s, CUDA-graph capture slow on Turing), HF transformers **123 tok/s** batch / ~18 tok/s single. **fp16 is numerically stable**: logits finite, max layer activation ~680 (×96 headroom under 65504), KL = 0.0000 and top-1 match vs an fp32 CPU reference, 0 degenerate outputs on en+Hinglish. Caveat: `torch.cuda.is_bf16_supported()` returns **True** on the T4 (emulated on Turing) — ignore it, use fp16.
- **Kaggle env:** 2×T4 (15 GB each, sm_75, driver CUDA 13.0), system torch 2.11.0+cu130, Python 3.13, 32.9 GB RAM, ~21 GB free disk. vLLM 0.33 (pins torch 2.14) installs into an isolated venv in ~60 s; the HF fallback runs in system Python (it has `accelerate`, which the venv lacks). The model mounts at `/kaggle/input/models/cohere-labs/tiny-aya/transformers/<variant>/1`.
- **Kaggle GPU quota:** 30 h/week on the owner's account, unused as of Nov 4. (More accounts wire in via `KAGGLE_*_2/_3`.)
- **HF gated access (owner's token):** Fire, Global, Base, Aya-Expanse-8B, **Llama-Guard-3-8B**, WildGuard, HarmBench classifiers and the StrongREJECT scorer are **accessible**; Llama-Guard-3-1B and shieldgemma-2b are **still gated** (request if needed).
- **Lexsi repos (PyPI):** safetune 0.1.12, auditkit 1.1.3, curatorkit 0.1.22, circuitkit 0.2.0; all install on Python 3.12. SafeTune's BoN augmentations and the 12-prefix English-only `StringMatchJudge` gap are confirmed in the 0.1.12 source.
- **Seed sources (public, loaded live):** HarmBench standard **200** (GitHub CSV), StrongREJECT **313** (GitHub CSV), XSTest-safe **250**, OR-Bench-Hard **1k** (both HF, public). The walledai mirrors are gated; we use the canonical sources. A stratified core of 200 harmful seeds spreads ~22 per category across 9 unified categories.

## 4. Hackathon facts
- Registration closes Nov 5 · submission **Nov 14, end of day (Anywhere on Earth)** (plan: v1 on Nov 12) · shortlist Nov 16 · prove-thesis window Nov 16–21 (+$200 HF credits) · Demo Day Nov 22.
- Rubric: write-up clarity /10, practical usefulness /10, code quality /10, overall /100. Double-blind, multiple judges.
- Rules:
  - chain ≥2 Track 2 repos (we use all 4)
  - permitted Cohere model
  - all artifacts on HF as collections
  - public demo or repo
  - more than one before/after metric
  - HF article with the organizers' 9-section checklist; teammates tagged
  - ROI section preferred
- Only the latest submission counts.
- To ask organizers (Discord ticket): whether attacked/abliterated weights must be published (our plan: don't publish them, describe them instead); the double-blind vs teammate-tagging question.

## 5. Conventions
- **Commits are authored by the module owner** (PLAN §14). Messages are plain imperative English, with no trailers. Until teammates send their details, commits are authored as the owner:

  | Role | Modules | GitHub | Commit email |
  |---|---|---|---|
  | R1 Attacks & infra | `attacks`, `serve`, `runners`, `defenses` | Harsh | dahiyaharsh2005@gmail.com |
  | R2 Data, languages & judges | `data`, `judges` | iHamza14 | 189707312+iHamza14@users.noreply.github.com |
  | R3 Mechanisms, analysis & story | `mech`, `analysis`, article, Space | shahanxd | 137555891+shahanxd@users.noreply.github.com |

  Harsh commits with his primary email (`dahiyaharsh2005@gmail.com`); the other two use their GitHub noreply addresses. Per-commit identity is set with `git -c user.name=… -c user.email=… commit …`. The email must be a verified address (or the GitHub noreply address) on that teammate's account. Commit messages stay plain imperative English with **no trailers**.
- Work on short-lived branches per stage; merge to `main`.
- Status updates to the team are short and plain.
- Never put secrets in code, notebooks, logs or chat. Read them from the environment.
- Gate harmful completions on HF; never publish attacked weights.

## 6. Build-phase progress

**Done (committed, CPU CI green):**
- Scaffold: `pyproject.toml` (Python 3.12), the PLAN §11 package layout, `ruff` + `pytest`, GitHub Actions CPU CI, a `Makefile`, CPU tests on a tiny random cohere2 model.
- Cost model (`magpie.costs`): FLOPs / queries / wall-clock on the real Tiny Aya Fire shape.
- Serving (`magpie.serve`): one `generate()` over vLLM / HF / a scripted echo backend.
- Attacks (`magpie.attacks`): the `Attack` interface + runner (cost logging, survival bookkeeping), L0 direct, L1 templates (en+hing), L1 input BoN (synthetic / natural-Hinglish / Devanagari control), L1 decoding BoN.
- Judges (`magpie.judges`): multilingual refusal matcher + collapse guard + the `CompositeJudge` validity gate.
- Hinglish (`magpie.data.hinglish`): CMI, romanization/shorthand variants, the natural-variation augmenter (C2).
- Analysis (`magpie.analysis`): KM survival, bootstrap median, log-rank, Wilson ASR, BoN power-law fit, and result-table aggregation.
- Runner (`magpie.runners.kaggle`): build/push/poll/pull private T4 kernels; the smoke job (**passed**) and the L0 job.
- Data (`magpie.data`): seed loaders, harm taxonomy, stratification, parallel-core assembly.
- `docs/PREREGISTRATION.md` committed before the runs.

**Done (runs landed, committed):**
- **First L0 table** on Fire/Global (en): `results/l0/20261104-173906/` — ASR 0.21 (Fire) / 0.24 (Global), over-refusal 0.13, collapse ~1% (lexicon judge). See `docs/RESULTS.md`.
- **Data v1 parallel core**: `results/build-data/20261104-181016/` — 632 records (400 en + 188 hing + 44 hi-Deva), pushed **private** to `huggingface.co/datasets/shahanxd/magpie-core-v1`.

**Remaining (next):**
1. **Model-judge rescoring** of L0 with Llama-Guard-3-8B / WildGuard / the StrongREJECT scorer, to replace the lexicon pre-filter in the headline ASR (the code is in `magpie.judges.model_judge`).
2. **Better translations** for the core: accept the **Aya Expanse 8B** licence on Kaggle so it mounts (it is gated; v1 used tiny-aya Fire, which gave low-CMI Hinglish and a low back-translation pass rate), re-run `build_data` with `translate_model=aya`, replace the token-overlap agreement proxy with embedding similarity, then the native-verification sprint. Then run L0 per form (en vs hing vs hi-Deva).
3. Hardening (M2 RR / M3 RepNoise / M5 CAST) on Kaggle via SafeTune; L1a/b; prefill depth; refusal margins.
4. BoN power laws (synthetic vs natural vs Devanagari), abliteration, injection suite, then the Cox prediction study (H4).

**Known environment facts for GPU jobs:** the vLLM venv on Kaggle must also carry `accelerate`, `datasets`, `curatorkit`, `sentencepiece`, `tiktoken` (now installed by `install_vllm_venv`); release vLLM between models with `VLLMBackend.close()`; attach a Kaggle model source with its version (`.../8b/5`); Aya Expanse (`cohereforai/aya-expanse`) is gated on Kaggle and needs manual licence acceptance before it will mount.

## 7. Open items
- Discord idea-validation ticket (owner).
- Teammates' Kaggle tokens when the single 30 h/week quota gets tight (`KAGGLE_*_2/_3`).
- Request Llama-Guard-3-1B / shieldgemma-2b access only if a lighter judge is wanted; the 8B judges are already accessible.
- 3050 laptop: GGUF runs are done by the team (the dev environment can't reach the laptop).
- Native-speaker verification + annotation sprint (team, §8 of the plan).
