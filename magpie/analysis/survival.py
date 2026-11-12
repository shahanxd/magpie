"""Survival analysis: the share of prompts still refused as attacker cost grows.

This is the statistical core of C1. A prompt survives until the first valid
success; unbroken prompts are right-censored at the attacker's budget. From a
list of :class:`~magpie.types.Outcome` we compute:

* Kaplan-Meier survival S(c) = P(still refused | cost ≤ c), on log10 FLOPs,
* median cost-to-break with a bootstrap CI (resampling prompts),
* the two-sample log-rank test (e.g. en vs Hinglish),
* attack-success rate (ASR) with a Wilson interval.

lifelines/scipy are used when available (and are the reference path); a small,
dependency-free fallback keeps the CPU tests runnable without them. The
fallback is exercised by the tests; production runs install ``magpie[analysis]``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from magpie.types import Outcome


@dataclass
class KMCurve:
    """A Kaplan-Meier curve in log10(total FLOPs)."""

    times: list[float]  # log10 FLOPs at each event
    survival: list[float]  # S(t) after each event
    n: int
    n_events: int

    def at(self, log_flops: float) -> float:
        """S(log_flops): the step function's value at a cost (right-continuous)."""
        s = 1.0
        for t, sv in zip(self.times, self.survival, strict=True):
            if t <= log_flops:
                s = sv
            else:
                break
        return s


def _event_times(outcomes: Sequence[Outcome]) -> list[tuple[float, bool]]:
    """(log10 cost, broken) per prompt; costs ≤ 0 are nudged to avoid log(0)."""
    out = []
    for o in outcomes:
        c = o.cost_to_break if o.cost_to_break > 0 else 1.0
        out.append((math.log10(c), o.broken))
    return out


def kaplan_meier(outcomes: Sequence[Outcome]) -> KMCurve:
    """KM estimate S(c). Censored observations (not broken) lower n at risk only."""
    data = sorted(_event_times(outcomes), key=lambda x: x[0])
    n = len(data)
    at_risk = n
    s = 1.0
    times: list[float] = []
    surv: list[float] = []
    n_events = 0
    i = 0
    while i < len(data):
        t = data[i][0]
        # all observations at this exact time
        tied = [d for d in data if d[0] == t]
        d = sum(1 for _, broken in tied if broken)  # events (breaks) at t
        if d > 0:
            s *= (1 - d / at_risk)
            times.append(t)
            surv.append(s)
            n_events += d
        at_risk -= len(tied)
        i += len(tied)
    return KMCurve(times=times, survival=surv, n=n, n_events=n_events)


def median_survival(outcomes: Sequence[Outcome]) -> float | None:
    """Smallest log10 cost at which S ≤ 0.5, or None if the curve never drops there."""
    km = kaplan_meier(outcomes)
    for t, s in zip(km.times, km.survival, strict=True):
        if s <= 0.5:
            return t
    return None


def _median_or_nan(outcomes: Sequence[Outcome]) -> float:
    m = median_survival(outcomes)
    return m if m is not None else float("nan")


def bootstrap_median_ci(outcomes: Sequence[Outcome], n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float | None, float | None, float | None]:
    """(median, lo, hi) log10-FLOPs median cost-to-break, resampling prompts.

    Returns None bounds when too few resamples have a defined median (the curve
    does not reach 0.5), which is itself a reportable fact.
    """
    point = median_survival(outcomes)
    if not outcomes:
        return None, None, None
    rng = random.Random(seed)
    idx = range(len(outcomes))
    meds: list[float] = []
    for _ in range(n_boot):
        sample = [outcomes[rng.choice(idx)] for _ in idx]
        m = median_survival(sample)
        if m is not None:
            meds.append(m)
    if len(meds) < n_boot * 0.5:
        return point, None, None
    meds.sort()
    lo = meds[int(alpha / 2 * len(meds))]
    hi = meds[int((1 - alpha / 2) * len(meds)) - 1]
    return point, lo, hi


@dataclass
class ASRResult:
    asr: float
    lo: float
    hi: float
    n: int
    n_broken: int


def asr_wilson(outcomes: Sequence[Outcome], z: float = 1.96) -> ASRResult:
    """Attack-success rate with a Wilson score interval (robust at extreme rates)."""
    n = len(outcomes)
    if n == 0:
        return ASRResult(0.0, 0.0, 0.0, 0, 0)
    k = sum(1 for o in outcomes if o.broken)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return ASRResult(asr=round(p, 4), lo=round(max(0.0, centre - half), 4),
                     hi=round(min(1.0, centre + half), 4), n=n, n_broken=k)


def logrank_test(group_a: Sequence[Outcome], group_b: Sequence[Outcome]) -> dict[str, float]:
    """Two-sample log-rank test (e.g. en vs Hinglish survival).

    Uses lifelines when installed; otherwise a direct implementation of the
    log-rank statistic (χ² with 1 df). Returns the statistic and p-value.
    """
    try:
        from lifelines.statistics import logrank_test as _lr

        a, b = _durations(group_a), _durations(group_b)
        res = _lr(a[0], b[0], event_observed_A=a[1], event_observed_B=b[1])
        return {"statistic": float(res.test_statistic), "p_value": float(res.p_value),
                "backend": "lifelines"}
    except ImportError:
        return _logrank_manual(group_a, group_b)


def _durations(outcomes: Sequence[Outcome]) -> tuple[list[float], list[int]]:
    ev = _event_times(outcomes)
    return [t for t, _ in ev], [int(b) for _, b in ev]


def _logrank_manual(a: Sequence[Outcome], b: Sequence[Outcome]) -> dict[str, float]:
    ta, ea = _durations(a)
    tb, eb = _durations(b)
    all_rows = [(t, e, 0) for t, e in zip(ta, ea, strict=True)] + \
               [(t, e, 1) for t, e in zip(tb, eb, strict=True)]
    event_times = sorted({t for t, e, _ in all_rows if e})
    o_minus_e = 0.0
    var = 0.0
    for t in event_times:
        at_risk_a = sum(1 for tt, _, g in all_rows if tt >= t and g == 0)
        at_risk_b = sum(1 for tt, _, g in all_rows if tt >= t and g == 1)
        n = at_risk_a + at_risk_b
        d = sum(1 for tt, e, _ in all_rows if tt == t and e)
        if n < 2 or d == 0:
            continue
        exp_a = d * at_risk_a / n
        obs_a = sum(1 for tt, e, g in all_rows if tt == t and e and g == 0)
        o_minus_e += obs_a - exp_a
        var += (d * (at_risk_a / n) * (at_risk_b / n) * (n - d) / (n - 1)) if n > 1 else 0.0
    stat = (o_minus_e**2 / var) if var > 0 else 0.0
    return {"statistic": round(stat, 4), "p_value": round(_chi2_sf(stat, 1), 6),
            "backend": "manual"}


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of χ² with 1 df via the erfc relation (df=1 only without scipy)."""
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df))
    except ImportError:
        if df != 1:
            raise
        return math.erfc(math.sqrt(x / 2)) if x > 0 else 1.0


@dataclass
class PowerLawFit:
    """``-log ASR(N) = a · N^b`` fit; ``b`` is the BoN scaling exponent (H2)."""

    a: float
    b: float
    r2: float
    n_points: int
    extrapolated_asr: dict[int, float] = field(default_factory=dict)


def fit_bon_power_law(asr_by_n: dict[int, float], extrapolate_to: Sequence[int] = (1000, 10000),
                      eps: float = 1e-4) -> PowerLawFit | None:
    """Fit a power law to ASR(N) by OLS on log N vs log(-log ASR).

    Follows Hughes et al.: model ``-log(ASR) = a · N^b`` (ASR rising in N →
    negative ``b``). Needs ≥ 2 points with 0 < ASR < 1. ``r2`` is on the
    log-log line; ``extrapolated_asr`` predicts ASR at larger N with CIs left to
    the caller's bootstrap.
    """
    pts = [(n, asr) for n, asr in sorted(asr_by_n.items()) if n > 0 and eps < asr < 1 - eps]
    if len(pts) < 2:
        return None
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(-math.log(asr)) for _, asr in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        return None
    b = sxy / sxx
    loga = my - b * mx
    a = math.exp(loga)
    ss_res = sum((y - (loga + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    extra = {}
    for N in extrapolate_to:
        neg_log_asr = a * (N ** b)
        extra[N] = round(math.exp(-neg_log_asr), 4)
    return PowerLawFit(a=round(a, 5), b=round(b, 5), r2=round(r2, 4), n_points=n,
                       extrapolated_asr=extra)
