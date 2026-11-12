"""Figures: the attacker-cost ladder (the hero figure) and the BoN / depth curves.

Reads the committed result tables and draws the C1 story -- validated ASR as the
attacker climbs the ladder (L0 → L1 → L2 → L3), per model -- plus the best-of-N
power law and the prefill safety-depth curve. matplotlib only (the ``analysis``
extra); headless Agg backend so it runs in CI and in a kernel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LadderPoint:
    model: str
    attack: str
    level: str
    asr: float
    lo: float
    hi: float
    judge: str


# Order attacks along the cost axis for the hero figure.
LEVEL_ORDER = {"l0_direct": 0, "l1_templates": 1, "l1_bon_input": 2,
               "l2_prefill": 3, "l3_abliteration": 4}
PRETTY = {"l0_direct": "L0\ndirect", "l1_templates": "L1\ntemplates",
          "l1_bon_input": "L1\nbest-of-N", "l2_prefill": "L2\nprefill",
          "l3_abliteration": "L3\nabliteration"}


def load_table(path: str | Path) -> list[LadderPoint]:
    """Read a ``table*.json`` into ladder points (judge inferred from the name)."""
    d = json.loads(Path(path).read_text())
    judge = "harmbench" if "harmbench" in d.get("name", "").lower() else "lexicon"
    return [LadderPoint(r["model"], r["attack"], r.get("level", ""), r["asr"],
                        r.get("asr_lo", r["asr"]), r.get("asr_hi", r["asr"]), judge)
            for r in d["rows"]]


def plot_cost_ladder(points: list[LadderPoint], out_png: str | Path,
                     title: str = "Magpie -- attacker-cost ladder (Tiny Aya, English)") -> Path:
    """Draw ASR vs attacker level, one line per model, with CI bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[LadderPoint]] = {}
    for p in points:
        if p.attack in LEVEL_ORDER:
            by_model.setdefault(p.model, []).append(p)
    xs_all = sorted({LEVEL_ORDER[p.attack] for p in points if p.attack in LEVEL_ORDER})
    labels = [PRETTY[a] for a in sorted(LEVEL_ORDER, key=LEVEL_ORDER.get) if LEVEL_ORDER[a] in xs_all]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, pts in sorted(by_model.items()):
        pts = sorted(pts, key=lambda p: LEVEL_ORDER[p.attack])
        xs = [LEVEL_ORDER[p.attack] for p in pts]
        ys = [p.asr for p in pts]
        lo = [p.asr - p.lo for p in pts]
        hi = [p.hi - p.asr for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=model, linewidth=2)
    ax.set_xticks(xs_all)
    ax.set_xticklabels(labels)
    ax.set_ylabel("attack success rate (validated / HarmBench where available)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("attacker capability →")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="model")
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_hero(table_paths: list[str | Path], out_png: str | Path) -> dict:
    """Merge several tables (prefer HarmBench rows per model/attack) and plot the ladder."""
    best: dict[tuple[str, str], LadderPoint] = {}
    for path in table_paths:
        if not Path(path).is_file():
            continue
        for p in load_table(path):
            key = (p.model, p.attack)
            cur = best.get(key)
            if cur is None or (p.judge == "harmbench" and cur.judge != "harmbench"):
                best[key] = p
    points = list(best.values())
    plot_cost_ladder(points, out_png)
    return {"n_points": len(points),
            "points": [{"model": p.model, "attack": p.attack, "asr": p.asr, "judge": p.judge}
                       for p in sorted(points, key=lambda p: (p.model, LEVEL_ORDER[p.attack]))]}
