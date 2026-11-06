#!/usr/bin/env python
"""Generate the Space's self-contained data bundle from the committed results.

Run from the repo root (needs the `magpie` package and the `results/` tables):

    python space/build_space_data.py

Writes `space/data.json` and copies the hero figure + robustness cards into
`space/`. The Space app (`space/app.py`) then reads only those files, so it runs
on Hugging Face with no dependency on the repo or the full results tree.

The ladder points are re-derived from the committed HarmBench tables; the C4 /
H1 / hardening panels mirror the numbers in `docs/RESULTS.md` (with run
provenance), kept here as one explicit, auditable dict.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from magpie.analysis.figures import build_hero

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


def _ladder_points() -> list[dict]:
    tables = sorted(
        glob.glob(str(ROOT / "results/rescore/*/out/table_harmbench.json"))
        + glob.glob(str(ROOT / "results/attack-l2-prefill/*/out/table.json"))
    )
    info = build_hero(tables, SPACE / "hero_ladder.png")  # also (re)writes the PNG
    # Keep only validated (HarmBench) points for the demo's ladder.
    return [p for p in info["points"] if p.get("judge") == "harmbench"]


# Curated panels — these mirror docs/RESULTS.md (validated with HarmBench).
JUDGE_FLIP_EN = {
    "caption": "Same attempts, two judges. The cheap refusal lexicon over-counts and flips the "
               "attack ranking; only a content judge tells a long non-refusal from a real jailbreak.",
    "rows": [
        {"attack": "L1 templates", "lexicon": 0.52, "harmbench": 0.29},
        {"attack": "L1 best-of-N", "lexicon": 1.00, "harmbench": 0.19},
    ],
}
JUDGE_FLIP_LANG = {
    "caption": "Identical prompts, three forms. The lexicon says Devanagari is 4x English and "
               "Hinglish is safest; the content judge says the opposite in order and a third in size.",
    "rows": [
        {"form": "English", "lexicon": 0.21, "harmbench": 0.03},
        {"form": "Devanagari", "lexicon": 0.86, "harmbench": 0.28},
        {"form": "Hinglish", "lexicon": 0.09, "harmbench": 0.03},
    ],
}
HARDENING = {
    "caption": "A lightweight Circuit-Breakers/RR pass (fire_rr) re-attacked paired against vendor "
               "Fire. It raises attacker cost at no level — an honest negative.",
    "rows": [
        {"level": "L1 templates", "fire": 0.30, "fire_rr": 0.30},
        {"level": "L2 prefill", "fire": 0.31, "fire_rr": 0.31},
    ],
}
H4 = {
    "caption": "A prompt's attack-free refusal-direction margin predicts how expensive it is to "
               "break (Cox regression; hazard ratio < 1 means a higher margin survives longer).",
    "rows": [
        {"attack": "best-of-N", "hazard_ratio": 0.72, "ci": "[0.58, 0.89]", "p": "0.003"},
        {"attack": "templates", "hazard_ratio": 0.60, "ci": "—", "p": "2e-6"},
    ],
}

LEVEL_LABELS = {
    "l0_direct": "L0 direct",
    "l1_templates": "L1 templates",
    "l1_bon_input": "L1 best-of-N",
    "l2_prefill": "L2 prefill",
    "l3_abliteration": "L3 abliteration",
}


def main() -> None:
    SPACE.mkdir(exist_ok=True)
    points = _ladder_points()
    for p in points:
        p["level_label"] = LEVEL_LABELS.get(p["attack"], p["attack"])
    # Copy the deployer cards in as plain markdown.
    cards = {}
    for name in ("fire", "global", "fire_rr"):
        src = ROOT / "docs" / "cards" / f"{name}.md"
        if src.is_file():
            cards[name] = src.read_text(encoding="utf-8")
    data = {
        "thesis": "For an open, on-device model the attacker owns the weights, so safety is not a "
                  "refusal rate — it is the attacker cost to break the model, at each capability "
                  "level, in the languages people actually use. Magpie prices that, pairs every "
                  "number with over-refusal, and tests whether a cheap white-box margin predicts "
                  "the price.",
        "ladder": points,
        "judge_flip_en": JUDGE_FLIP_EN,
        "judge_flip_lang": JUDGE_FLIP_LANG,
        "hardening": HARDENING,
        "h4": H4,
        "cards": cards,
        "links": {
            "code": "https://github.com/shahanxd/magpie",
            "dataset": "https://huggingface.co/datasets/shahanxd/magpie-core-v2",
        },
        "source": "Generated from committed results by space/build_space_data.py; numbers mirror "
                  "docs/RESULTS.md (HarmBench-validated).",
    }
    (SPACE / "data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {SPACE/'data.json'} ({len(points)} ladder points, {len(cards)} cards) "
          f"and {SPACE/'hero_ladder.png'}")


if __name__ == "__main__":
    main()
