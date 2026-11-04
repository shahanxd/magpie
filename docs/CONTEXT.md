# Magpie: project context and conventions

Read this together with `docs/PLAN.md` (plan v2) before doing any work. It records what has been decided, what has been checked, and what happens next.

**Status (Nov 4, 2026):** plan v2 **approved**. Build phase starts now.

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
  | R1 Attacks & infra | `attacks`, `serve`, `runners`, `defenses` | _tbd_ | _tbd_ |
  | R2 Data, languages & judges | `data`, `judges` | _tbd_ | _tbd_ |
  | R3 Mechanisms, analysis & story | `mech`, `analysis`, article, Space | _tbd_ | _tbd_ |
  | Owner (until filled) | all | shahanxd | shahanxd@gmail.com |

  Use per-commit identity: `git -c user.name=… -c user.email=… commit …`. The email must be a verified address (or the GitHub noreply address) on that teammate's account.
- Work on short-lived branches per stage; merge to `main`.
- Status updates to the team are short and plain.
- Never put secrets in code, notebooks, logs or chat. Read them from the environment.
- Gate harmful completions on HF; never publish attacked weights.

## 6. Next steps (build phase, D1 = Nov 5)
1. **Scaffold:**
   - `pyproject.toml` (Python 3.12, required by SafeTune)
   - the package layout from PLAN §11
   - `ruff` + `pytest`, GitHub Actions CPU CI, a `Makefile`
   - CPU tests use a tiny random cohere2 config, as SafeTune's tests do
2. **Kaggle runner** (`magpie/runners/kaggle.py`):
   - auth from env, supporting both token styles
   - build kernel metadata: GPU T4×2, internet on, model source `cohere-labs/tiny-aya`
   - push, poll status, download outputs into `results/`
   - **Smoke test:** load tiny-aya-fire in fp16 on a T4; generate 5 en + 5 hing prompts with vLLM (fall back to HF); log tokens/s and check fp16 stability.
3. **Data v1:**
   - pick 200 harmful seeds (StrongREJECT + HarmBench standard, stratified by category)
   - over-refusal sets: XSTest safe + OR-Bench-Hard subset + a benign twin per seed
   - translate to `hing` (several spelling styles) and `hi-Deva`; compute CMI
   - CuratorKIT gates + provenance
   - push as a **private** HF dataset (gating decided later)
4. **Commit `docs/PREREGISTRATION.md`** (hypotheses from PLAN §2) **before** the main runs.
5. L0 + over-refusal on Fire and Global; judge pipeline v0; first L0 table.

## 7. Open items
- Teammates' names, GitHub handles, commit emails, and who takes R1/R2/R3.
- Discord idea-validation ticket.
- Teammates' Kaggle tokens when needed.
- 3050 laptop: GGUF runs are done by the team (the dev environment can't reach the laptop).
