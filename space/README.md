---
title: Magpie — Pricing Robustness
emoji: 🐦
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: true
license: cc-by-nc-4.0
short_description: Red-teaming Tiny Aya as attacker cost, in English + Hinglish
tags:
  - red-teaming
  - ai-safety
  - robustness
  - hinglish
  - multilingual
  - lexsi-hackathon
---

# 🐦 Magpie — pricing the robustness of an open, on-device safety model

**For an open model that ships on the phone, the attacker owns the weights — so the right safety
number is not the refusal rate, but how much attacker effort a break costs, at each capability
level, in the languages people actually use.**

Magpie red-teams Cohere Labs' **Tiny Aya** (Fire/Global) across an attacker ladder — just ask →
public jailbreaks → decoder prefill → weight edits — in English, Hinglish and Devanagari, judges
every attempt with a content classifier, and tests whether a cheap white-box measurement predicts
the cost of a break. Lexsi Hackathon, Track 2a.

**Open the 📄 Article tab for the full write-up**; the other tabs are interactive views of the
results: the cost ladder, the judge-validity headline, the hardening negative, the H4 audit, and
per-model robustness cards.

- Code & reproducibility: https://github.com/shahanxd/magpie
- Dataset (gated): https://huggingface.co/datasets/shahanxd/magpie-core-v1
- Hardened model: https://huggingface.co/shahanxd/fire_rr

*A static page (no runtime, runs no models) rendering committed, HarmBench-validated results.
Rebuild it from the repo with `python space/build_space_data.py`; an interactive Gradio version
of the same views is in the repo (`python space/app.py`).*
