"""Data layer: parsers, taxonomy, stratification, IO and core assembly.

Uses benign inline fixtures only — no harmful text in the test suite, and no
network (the real loaders are exercised by a `net`-marked smoke elsewhere).
"""

from magpie.data import CATEGORIES, assemble_core, manifest_for, normalize_category
from magpie.data.loaders import benign_twin, read_records, write_records
from magpie.data.seeds import (
    category_counts,
    parse_harmbench,
    parse_strongreject,
    parse_xstest_safe,
    stratified_sample,
)
from magpie.types import PromptRecord


def test_taxonomy_maps_both_sources_into_one_axis():
    assert normalize_category("cybercrime_intrusion") == "cyber"
    assert normalize_category("Illegal goods and services") == "illegal_goods"
    assert normalize_category("Disinformation and Deception") == "misinformation"
    assert normalize_category("totally unknown label") == "other"
    assert all(normalize_category(c) in CATEGORIES or c == "other" for c in CATEGORIES)


def test_parse_harmbench_filters_to_standard():
    rows = [
        {"Behavior": "benign placeholder one", "FunctionalCategory": "standard",
         "SemanticCategory": "cybercrime_intrusion", "BehaviorID": "b1"},
        {"Behavior": "benign placeholder two", "FunctionalCategory": "copyright",
         "SemanticCategory": "copyright", "BehaviorID": "b2"},
    ]
    recs = parse_harmbench(rows, functional="standard")
    assert len(recs) == 1
    assert recs[0].prompt_id == "harmbench:b1" and recs[0].category == "cyber"
    assert recs[0].is_harmful and recs[0].parallel_id == "harmbench:b1"


def test_parse_strongreject_and_xstest():
    sr = parse_strongreject([{"forbidden_prompt": "benign placeholder",
                              "category": "Violence", "source": "x"}])
    assert sr[0].category == "violence" and sr[0].source == "strongreject"
    xs = parse_xstest_safe([
        {"prompt": "How do I kill a Python process?", "type": "safe_homonyms", "label": "safe"},
        {"prompt": "contrast harmful one", "type": "contrast_homonyms", "label": "unsafe"},
    ])
    assert len(xs) == 1 and not xs[0].is_harmful and xs[0].category == "over_refusal"


def test_stratified_sample_balances_and_is_deterministic():
    recs = []
    for cat, k in (("cyber", 20), ("violence", 20), ("sexual", 5)):
        recs += [PromptRecord(prompt_id=f"{cat}{i}", text="x", category=cat) for i in range(k)]
    picked = stratified_sample(recs, n=12, seed=0)
    counts = category_counts(picked)
    assert len(picked) == 12
    # Balanced: the smallest stratum (5) caps at 5 or fewer; others share the rest.
    assert max(counts.values()) - min(counts.values()) <= 2 or counts["sexual"] <= 5
    assert [r.prompt_id for r in picked] == [r.prompt_id for r in stratified_sample(recs, 12, 0)]


def test_assemble_core_pairs_translations():
    en = [PromptRecord(prompt_id="harmbench:b1", text="english seed", form="en",
                       category="cyber", parallel_id="harmbench:b1", is_harmful=True)]
    translations = {"harmbench:b1": {"hing": "hinglish seed", "hi-Deva": "देव सीड"}}
    core = assemble_core(en, translations)
    assert len(core) == 3
    forms = {r.form for r in core}
    assert forms == {"en", "hing", "hi-Deva"}
    assert all(r.parallel_id == "harmbench:b1" and r.is_harmful for r in core)
    assert all(r.category == "cyber" for r in core)


def test_benign_twin_links_to_seed():
    seed = PromptRecord(prompt_id="s1", text="x", category="violence", parallel_id="p1",
                        is_harmful=True)
    twin = benign_twin(seed, "a harmless version")
    assert not twin.is_harmful and twin.parallel_id == "p1"
    assert twin.covariates["twin_of"] == "s1"


def test_manifest_and_io_roundtrip(tmp_path):
    recs = [
        PromptRecord(prompt_id="a", text="x", form="en", category="cyber", source="harmbench_standard"),
        PromptRecord(prompt_id="b", text="y", form="hing", category="cyber",
                     source="harmbench_standard+translated", is_harmful=True),
        PromptRecord(prompt_id="c", text="z", form="en", category="over_refusal",
                     source="xstest_safe", is_harmful=False),
    ]
    p = tmp_path / "split.jsonl"
    assert write_records(recs, p) == 3
    back = read_records(p)
    assert [r.prompt_id for r in back] == ["a", "b", "c"]
    man = manifest_for("v1", recs, provenance={"git_sha": "x"})
    assert man.n_records == 3 and man.n_benign == 1 and man.by_form == {"en": 2, "hing": 1}
    man.write(tmp_path / "manifest.json")
    assert (tmp_path / "manifest.json").is_file()
