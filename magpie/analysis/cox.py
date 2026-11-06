"""Cox regression: does the refusal margin predict cost-to-break? (H4)

The pre-registered test (PLAN §9): per-prompt survival-cost regressed on the
white-box refusal margin. A hazard ratio for the margin significantly below 1
means a higher margin (further into refusal space) lowers the hazard of being
broken — i.e. the cheap measurement predicts robustness, the C3 bet. A null is
reported as such.

The duration is the attacker effort to break a prompt (best-of-N draws or
prefill depth, i.e. ``n_attempts`` at the breaking step); the event is whether
it broke within budget (censored otherwise). Uses lifelines when available
(the reference path); a dependency-free fallback computes the univariate Cox
partial-likelihood estimate for the single margin covariate so the test runs
and is unit-tested without lifelines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from magpie.types import Outcome


def _rows(outcomes: Sequence[Outcome], margins_z: dict[str, float]):
    """(duration, event, margin) per prompt that has a margin; duration = n_attempts."""
    rows = []
    for o in outcomes:
        if o.prompt_id in margins_z:
            rows.append((max(1, o.n_attempts), 1 if o.broken else 0, margins_z[o.prompt_id]))
    return rows


def cox_margin(outcomes: Sequence[Outcome], margins_z: dict[str, float],
               extra: dict[str, dict[str, float]] | None = None) -> dict:
    """Fit a Cox model for the margin covariate; return HR, CI, p, n, backend.

    ``extra`` maps covariate name -> {prompt_id: value} for additional
    standardized covariates (e.g. CMI); used only on the lifelines path.
    """
    rows = _rows(outcomes, margins_z)
    n, n_events = len(rows), sum(r[1] for r in rows)
    if n_events < 3 or len({r[2] for r in rows}) < 2:
        return {"backend": "none", "n": n, "n_events": n_events,
                "note": "too few events or no margin variation"}
    try:
        return _cox_lifelines(outcomes, margins_z, extra, n, n_events)
    except ImportError:
        return _cox_univariate(rows, n, n_events)


def _cox_lifelines(outcomes, margins_z, extra, n, n_events) -> dict:
    import pandas as pd
    from lifelines import CoxPHFitter

    extra = extra or {}
    data = []
    for o in outcomes:
        if o.prompt_id not in margins_z:
            continue
        row = {"duration": max(1, o.n_attempts), "event": 1 if o.broken else 0,
               "margin": margins_z[o.prompt_id]}
        for name, m in extra.items():
            if o.prompt_id in m:
                row[name] = m[o.prompt_id]
        data.append(row)
    df = pd.DataFrame(data).dropna()
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    s = cph.summary.loc["margin"]
    return {"backend": "lifelines", "n": int(len(df)), "n_events": int(df["event"].sum()),
            "hazard_ratio": float(s["exp(coef)"]), "coef": float(s["coef"]),
            "p_value": float(s["p"]),
            "hr_ci": [float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])],
            "concordance": float(cph.concordance_index_),
            "covariates": ["margin", *(extra or {})]}


def _cox_univariate(rows, n, n_events) -> dict:
    """Breslow-tie Cox partial-likelihood for one covariate, by Newton–Raphson."""
    beta = 0.0
    for _ in range(50):
        num_g = num_h = 0.0  # gradient, hessian over event rows
        for dur_i, ev_i, x_i in rows:
            if not ev_i:
                continue
            risk = [(x, math.exp(beta * x)) for dur, _e, x in rows
                    if dur >= dur_i]  # at-risk set at this event time
            w = sum(e for _x, e in risk)
            wx = sum(x * e for x, e in risk)
            wxx = sum(x * x * e for x, e in risk)
            num_g += x_i - wx / w
            num_h += (wx / w) ** 2 - wxx / w
        if abs(num_h) < 1e-12:
            break
        step = num_g / num_h
        beta -= step
        if abs(step) < 1e-8:
            break
    # Variance from the final Hessian (information = -hessian).
    info = -num_h if num_h != 0 else 1e-12
    se = math.sqrt(1.0 / info) if info > 0 else float("inf")
    z = beta / se if se not in (0.0, float("inf")) else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"backend": "univariate", "n": n, "n_events": n_events,
            "hazard_ratio": math.exp(beta), "coef": beta, "p_value": p,
            "hr_ci": [math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)]
            if se != float("inf") else None,
            "covariates": ["margin"]}
