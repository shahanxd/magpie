#!/usr/bin/env python
"""Regenerate the hero cost-ladder figure from the committed result tables.

    python scripts/make_hero.py

Prefers HarmBench-validated tables; falls back to the lexicon table per level.
"""
from __future__ import annotations

import glob
import json

from magpie.analysis.figures import build_hero

TABLES = (
    glob.glob("results/rescore/*/out/table_harmbench.json")
    + glob.glob("results/attack-l2-prefill/*/out/table.json")
)

if __name__ == "__main__":
    info = build_hero(sorted(TABLES), "docs/figures/hero_ladder.png")
    print(json.dumps(info, indent=2))
