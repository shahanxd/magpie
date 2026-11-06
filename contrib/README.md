# Upstream contributions (C5)

Ready-to-submit contributions back to the Lexsi Track-2 repos. They are self-contained (no
dependency on `magpie`) so they drop straight into the target repo. This session can't open PRs to
those repos, so apply and submit them yourself — commands below.

| # | Target repo | Gap it fixes | File here |
|---|---|---|---|
| 1 | **SafeTune** | `StringMatchJudge` recognises only 12 English refusal prefixes, so a Hinglish/Hindi refusal is scored as a successful jailbreak (over-counts ASR on non-English). | `safetune_multilingual_refusal.py` |
| 2 | **SafeTune** | `BoNAttack` augmentations are synthetic ASCII/caps/scramble noise, mostly inert on Devanagari and unlike how Hinglish users really type. | see "BoN" below |
| 3 | **AuditKIT** | The red-team probe pack is ~26 English probes; nothing for Indic languages or over-refusal in them. | `auditkit_indic_redteam.py` |

Find each repo's GitHub URL from its PyPI page (Homepage/Source):
https://pypi.org/project/safetune/ · https://pypi.org/project/auditkit/

## How to submit each one

```bash
# 1. Fork the repo on GitHub (top-right "Fork"), then:
git clone https://github.com/<you>/<repo>.git && cd <repo>
git checkout -b magpie-<short-name>
# 2. Copy the contrib file into the path noted in its header, wire it in (its
#    docstring shows the ~3-line hook), run the repo's tests.
# 3. Commit and push, then open a PR to the upstream repo.
git add -A && git commit -m "<title below>" && git push -u origin HEAD
```

### 1 — SafeTune multilingual refusal matcher
- **Where:** beside `src/safetune/` judges; extend/replace the `StringMatchJudge` success check
  (`ADVBENCH_REFUSAL_PREFIXES` in `src/safetune/config.py`).
- **Hook:** `MultilingualRefusalMatcher().is_refusal(text, lang)` — treat a detected refusal as a
  *failed* jailbreak. The file's docstring has the exact 3-line change.
- **PR title:** `Detect non-English (Hinglish/Hindi) refusals in the string-match judge`
- **PR body:** English-only refusal prefixes over-count attack success on multilingual models: a
  Hinglish or Devanagari refusal is counted as a jailbreak. This adds Hinglish + Devanagari refusal
  cues and a Unicode-script check, with tests. Measured impact (Magpie, Tiny Aya Fire, L0
  Devanagari): the English-only matcher reports ASR 0.86 where the content judge says 0.28 — 73% of
  its "successes" are Devanagari refusals it could not read.

### 2 — SafeTune script-aware / natural-Hinglish BoN
The implementation is Magpie's `magpie/data/hinglish.py` (`HinglishAugmenter.variants` and
`natural_bon_variants`): natural respelling (`nahi/nahin/nhi`), SMS shorthand, en↔hi word swaps and
transliteration — the variation real Hinglish users produce, plus a Devanagari-aware path. Lift
those two functions (they're dependency-free) into SafeTune's BoN augmentation list.
- **PR title:** `Add script-aware (Hinglish/Devanagari) best-of-N augmentations`
- **PR body:** Current BoN noise (random caps/ASCII/scramble) is mostly inert on Devanagari and
  unlike real code-mixed typing. This adds natural-respelling + transliteration augmentations.
  Magpie finding: natural Hinglish respelling does *not* scale like synthetic noise (flat vs
  b≈−1.9), so the two measure different robustness — both are worth having.

### 3 — AuditKIT Indic red-team pack
- **Where:** a new probe module beside `src/auditkit/redteam/probes/builtin.py`.
- **Ships benign / over-refusal probes only** (in English, Hinglish, Devanagari) plus a refusal
  detector; the *harmful* probes come from the gated HF dataset, not from this file, per
  responsible-disclosure.
- **PR title:** `Add an Indic (Hinglish/Devanagari) red-team + over-refusal probe pack`
- **PR body:** Extends the English-only probe set with parallel Indic probes and an over-refusal
  set, so a deployer can measure refusal *and* over-refusal on the languages Tiny Aya targets.

---

*These mirror implementations committed in the Magpie repo (`magpie/judges/refusal.py`,
`magpie/data/hinglish.py`); the files here are standalone copies for upstreaming.*
