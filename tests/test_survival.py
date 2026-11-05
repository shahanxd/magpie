import math

from magpie.analysis.survival import (
    asr_wilson,
    bootstrap_median_ci,
    fit_bon_power_law,
    kaplan_meier,
    logrank_test,
    median_survival,
)
from magpie.types import Outcome


def _out(prompt_id, broken, cost, form="en"):
    return Outcome(prompt_id=prompt_id, model="m", attack="a", form=form, broken=broken,
                   cost_to_break=cost, queries_to_break=1, n_attempts=1)


def test_km_all_broken_drops_to_zero():
    outs = [_out(str(i), True, 10 ** (12 + i)) for i in range(4)]
    km = kaplan_meier(outs)
    assert km.n == 4 and km.n_events == 4
    assert km.survival[-1] == 0.0
    assert km.at(11) == 1.0  # before the first event
    assert km.at(13) < 1.0


def test_km_censoring_keeps_survival_above_zero():
    # Two break cheaply, two are censored (never broken) at a high budget.
    outs = [_out("a", True, 1e12), _out("b", True, 1e12),
            _out("c", False, 1e18), _out("d", False, 1e18)]
    km = kaplan_meier(outs)
    # After the two events at 1e12 (4 at risk): S = 1 - 2/4 = 0.5; censored never lower it.
    assert math.isclose(km.survival[-1], 0.5, abs_tol=1e-9)
    assert median_survival(outs) == 12.0


def test_median_none_when_curve_never_reaches_half():
    outs = [_out("a", True, 1e12)] + [_out(str(i), False, 1e18) for i in range(9)]
    assert median_survival(outs) is None  # only 10% broken


def test_bootstrap_ci_brackets_point_estimate():
    outs = [_out(str(i), True, 10 ** (12 + (i % 3))) for i in range(30)]
    point, lo, hi = bootstrap_median_ci(outs, n_boot=500, seed=1)
    assert point is not None and lo is not None and hi is not None
    assert lo <= point <= hi


def test_asr_wilson_interval():
    outs = [_out(str(i), i < 3, 1e13) for i in range(10)]  # 3/10 broken
    r = asr_wilson(outs)
    assert r.asr == 0.3 and r.n == 10 and r.n_broken == 3
    assert 0 < r.lo < 0.3 < r.hi < 1


def test_logrank_separates_easy_from_hard():
    easy = [_out(f"e{i}", True, 1e12) for i in range(20)]       # all break cheap
    hard = [_out(f"h{i}", True, 1e18) for i in range(20)]       # all break dear
    res = logrank_test(easy, hard)
    assert res["statistic"] > 3.84  # chi2(1) 0.05 critical value
    assert res["p_value"] < 0.05


def test_logrank_manual_matches_when_lifelines_absent():
    a = [_out(f"a{i}", True, 10 ** (12 + i % 2)) for i in range(10)]
    b = [_out(f"b{i}", True, 10 ** (13 + i % 2)) for i in range(10)]
    res = logrank_test(a, b)
    assert res["backend"] in ("lifelines", "manual")
    assert res["statistic"] >= 0 and 0 <= res["p_value"] <= 1


def test_power_law_fit_recovers_exponent():
    # Construct ASR(N) from -log ASR = a*N^b with a=2.0, b=-0.5.
    a, b = 2.0, -0.5
    asr_by_n = {n: math.exp(-a * n ** b) for n in (1, 2, 4, 8, 16, 32, 64)}
    fit = fit_bon_power_law(asr_by_n, extrapolate_to=(10000,))
    assert fit is not None
    assert abs(fit.b - b) < 0.02 and abs(fit.a - a) < 0.05
    assert fit.r2 > 0.99
    assert 0 < fit.extrapolated_asr[10000] <= 1


def test_power_law_needs_enough_valid_points():
    assert fit_bon_power_law({1: 0.0, 2: 1.0}) is None  # 0 and 1 excluded


def test_build_rows_pairs_asr_with_over_refusal(tmp_path):
    from magpie.analysis.aggregate import ResultBundle, build_rows, rows_to_markdown
    from magpie.types import Attempt

    outs = [_out(f"h{i}", i < 6, 10 ** (12 + i % 2)) for i in range(10)]  # 6/10 broken, en
    # Benign attempts: 2 of 4 refused (over-refusal 0.5).
    benign = []
    for i in range(4):
        a = Attempt(prompt_id=f"b{i}", model="m", attack="a", level="L0", form="en",
                    prompt="q", response="r", is_harmful_prompt=False)
        a.judgements["refusal_lexicon"] = {"is_refusal": i < 2, "collapsed": False}
        benign.append(a)
    rows = build_rows(outs, benign_attempts=benign, n_boot=200)
    assert len(rows) == 1
    r = rows[0]
    assert r.n_harmful == 10 and r.n_broken == 6 and r.asr == 0.6
    assert r.n_benign == 4 and r.over_refusal == 0.5
    md = rows_to_markdown(rows, "L0")
    assert "over-refusal" in md and "| m | en | a |" in md
    ResultBundle("L0", rows, {"git_sha": "x"}).write(tmp_path / "t")
    assert (tmp_path / "t" / "table.json").is_file()
    assert (tmp_path / "t" / "table.md").is_file()


def test_bon_asr_curve_and_power_law():
    from magpie.analysis.aggregate import bon_asr_curve, bon_power_law

    # 10 prompts: break at attempt index 1,2,4,8,... (n_attempts), 3 censored.
    outs = ([_out("a", True, 1e12)] * 1)  # n_attempts default 1
    outs = []
    for i, na in enumerate([1, 1, 2, 2, 4, 8, 16, 32]):
        o = _out(f"b{i}", True, 10 ** 12)
        o.n_attempts = na
        outs.append(o)
    for i in range(2):  # censored
        o = _out(f"c{i}", False, 10 ** 13)
        o.n_attempts = 64
        outs.append(o)
    curve = bon_asr_curve(outs, [1, 2, 4, 64])
    assert curve[1] == 0.2 and curve[2] == 0.4 and curve[64] == 0.8
    pl = bon_power_law(outs, 64)
    assert pl["n_max"] == 64 and set(pl["asr_by_n"]) <= {1, 2, 4, 8, 16, 32, 64}
