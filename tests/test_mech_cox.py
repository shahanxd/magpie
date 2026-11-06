"""Refusal-margin standardisation and the Cox H4 test (with a synthetic signal)."""

import random

from magpie.analysis.cox import cox_margin
from magpie.mech.margins import standardize
from magpie.types import Outcome


def test_standardize_zero_mean_unit_sd():
    z = standardize({"a": 1.0, "b": 2.0, "c": 3.0})
    assert abs(sum(z.values())) < 1e-9
    assert abs(z["c"] - 1.0) < 1e-9 and abs(z["a"] + 1.0) < 1e-9


def _outcome(pid, n_attempts, broken):
    return Outcome(prompt_id=pid, model="m", attack="l1_bon_input", form="en", broken=broken,
                   cost_to_break=0.0, queries_to_break=0, n_attempts=n_attempts)


def test_cox_recovers_margin_effect():
    # Build prompts where a higher margin => later break (higher n_attempts).
    rng = random.Random(0)
    outs, margins = [], {}
    for i in range(120):
        m = rng.gauss(0, 1)
        # break time increases with margin; all break within budget (event=1)
        depth = max(1, min(64, int(round(2 ** (1 + m) + rng.random()))))
        pid = f"p{i}"
        outs.append(_outcome(pid, depth, broken=depth < 64))
        margins[pid] = m
    z = standardize(margins)
    res = cox_margin(outs, z)
    assert res["backend"] in ("lifelines", "univariate")
    # Higher margin -> harder to break -> hazard ratio < 1.
    assert res["hazard_ratio"] < 1.0
    assert res["p_value"] < 0.05


def test_cox_handles_too_few_events():
    outs = [_outcome(f"p{i}", 64, False) for i in range(10)]
    res = cox_margin(outs, {f"p{i}": float(i) for i in range(10)})
    assert res["backend"] == "none"
