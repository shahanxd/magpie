"""Figure assembly: HarmBench rows preferred, ladder plot renders."""

import json

import pytest

from magpie.analysis.figures import LadderPoint, build_hero, load_table, plot_cost_ladder


def test_load_table_infers_judge(tmp_path):
    p = tmp_path / "table_harmbench.json"
    p.write_text(json.dumps({"name": "L0 (harmbench judge)",
                             "rows": [{"model": "fire", "attack": "l0_direct", "asr": 0.05,
                                       "asr_lo": 0.02, "asr_hi": 0.09}]}))
    pts = load_table(p)
    assert pts[0].judge == "harmbench" and pts[0].asr == 0.05


def test_build_hero_prefers_harmbench_and_renders(tmp_path):
    pytest.importorskip("matplotlib")
    lex = tmp_path / "table.json"
    lex.write_text(json.dumps({"name": "l0_direct (refusal_lexicon judge)",
                               "rows": [{"model": "fire", "attack": "l0_direct", "asr": 0.21,
                                         "asr_lo": 0.16, "asr_hi": 0.29}]}))
    hb = tmp_path / "table_harmbench.json"
    hb.write_text(json.dumps({"name": "L0 (harmbench judge)",
                              "rows": [{"model": "fire", "attack": "l0_direct", "asr": 0.05,
                                        "asr_lo": 0.02, "asr_hi": 0.09},
                                       {"model": "fire", "attack": "l3_abliteration", "asr": 0.68,
                                        "asr_lo": 0.60, "asr_hi": 0.76}]}))
    out = tmp_path / "hero.png"
    info = build_hero([lex, hb], out)
    assert out.is_file() and out.stat().st_size > 1000
    l0 = [p for p in info["points"] if p["attack"] == "l0_direct"][0]
    assert l0["judge"] == "harmbench" and l0["asr"] == 0.05  # HarmBench won over lexicon


def test_plot_handles_single_model(tmp_path):
    pytest.importorskip("matplotlib")
    pts = [LadderPoint("fire", "l0_direct", "L0", 0.05, 0.02, 0.09, "harmbench"),
           LadderPoint("fire", "l3_abliteration", "L3", 0.68, 0.60, 0.76, "harmbench")]
    assert plot_cost_ladder(pts, tmp_path / "x.png").is_file()
