---
title: Magpie — Pricing Robustness
emoji: 🐦
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# Magpie — pricing the robustness of an open, on-device safety model

An interactive view of the Magpie red-teaming study of Cohere Labs' Tiny Aya (Lexsi Hackathon,
Track 2a): the attacker-cost ladder, the judge-validity headline, the honest hardening negative,
the white-box audit (H4), and per-model robustness cards.

**This Space is self-contained and CPU-only** — it reads `data.json` and `hero_ladder.png` (both in
this folder), which are generated from the committed results by `space/build_space_data.py` in the
[GitHub repo](https://github.com/shahanxd/magpie). It runs no models.

## To deploy

1. Create a Gradio Space: https://huggingface.co/new-space (SDK: **Gradio**).
2. Upload the three files in this folder: `app.py`, `requirements.txt`, `data.json`, and the image
   `hero_ladder.png` (and this README). That's it — the Space builds and runs.

To refresh the numbers after new runs: from the repo root, `python space/build_space_data.py`, then
re-upload `data.json` and `hero_ladder.png`.
