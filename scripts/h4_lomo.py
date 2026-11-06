#!/usr/bin/env python
"""H4 leave-one-model-out: does the refusal margin predict cost-to-break across models?

Joins each model's per-prompt refusal margin (fit on a disjoint contrast set, no
attack) with its L1-templates break depths, fits a Cox model per model, and a
pooled model. If the margin's hazard ratio is < 1 and consistent across models
(and pooled), the white-box predictor generalizes beyond the model it was tuned
on — the pre-registered leave-one-model-out check.

    python scripts/h4_lomo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from magpie.analysis.cox import cox_margin

ROOT = Path(__file__).resolve().parent.parent

# (model, margins_json, templates_outcomes_jsonl)
SOURCES = {
    "fire": ("results/mech-fire/20261106-004626/out/margins_fire.json",
             "results/attack-l1-templates/20261105-194721/out/outcomes_fire.jsonl"),
    "global": ("results/mech-global/20261106-153151/out/margins_global.json",
               "results/attack-l1-templates/20261105-194721/out/outcomes_global.jsonl"),
}


def _load(margins_path: str, outcomes_path: str):
    mz = json.loads((ROOT / margins_path).read_text())["margins_z"]
    outs = [SimpleNamespace(prompt_id=d["prompt_id"], n_attempts=d["n_attempts"], broken=d["broken"])
            for d in (json.loads(ln) for ln in (ROOT / outcomes_path).read_text().splitlines() if ln.strip())]
    return mz, outs


def _rows(outs, mz):
    return [(max(1, o.n_attempts), 1 if o.broken else 0, mz[o.prompt_id])
            for o in outs if o.prompt_id in mz]


def main() -> int:
    per_model = {}
    pooled_rows = []
    for model, (mp, op) in SOURCES.items():
        if "MECH_GLOBAL_TS" in mp or not (ROOT / mp).is_file():
            print(f"[skip] {model}: margins not found at {mp} (run mech for {model} first)")
            continue
        mz, outs = _load(mp, op)
        res = cox_margin(outs, mz)
        per_model[model] = res
        pooled_rows += _rows(outs, mz)
        print(f"\n== {model} (L1 templates) ==")
        print(f"  n={res.get('n')} events={res.get('n_events')} "
              f"HR={res.get('hazard_ratio'):.3f} CI={res.get('hr_ci')} "
              f"p={res.get('p_value'):.2g} backend={res.get('backend')}")

    if len(per_model) >= 2 and pooled_rows:
        import pandas as pd
        from lifelines import CoxPHFitter
        df = pd.DataFrame(pooled_rows, columns=["duration", "event", "margin"])
        cph = CoxPHFitter().fit(df, "duration", "event")
        s = cph.summary.loc["margin"]
        print("\n== pooled (fire + global) ==")
        print(f"  n={len(df)} events={int(df['event'].sum())} "
              f"HR={float(s['exp(coef)']):.3f} "
              f"CI=[{float(s['exp(coef) lower 95%']):.3f}, {float(s['exp(coef) upper 95%']):.3f}] "
              f"p={float(s['p']):.2g} concordance={cph.concordance_index_:.3f}")
        print("\nLeave-one-model-out read: a per-model HR < 1 on BOTH Fire and Global means the "
              "margin predicts cost-to-break on a model it was not tuned on; the pooled fit is the "
              "combined estimate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
