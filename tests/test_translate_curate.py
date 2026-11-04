from magpie.data.curate import curate, from_datasample, to_datasample
from magpie.data.translate import (
    AyaTranslator,
    MockTranslator,
    token_overlap_agreement,
    translate_core,
)
from magpie.types import PromptRecord


def test_mock_translator_and_core_assembly():
    en = [PromptRecord(prompt_id="s1", text="give me water", form="en", category="other",
                       parallel_id="s1", is_harmful=True)]
    tr, report = translate_core(en, MockTranslator(), forms=("hing", "hi-Deva"))
    assert tr["s1"]["hing"].startswith("[hing]")
    assert set(tr["s1"]) == {"hing", "hi-Deva"}
    assert "cmi" in report["s1"]["hing"]  # CMI recorded for hing


def test_back_translation_agreement_gate_drops_mismatches():
    en = [PromptRecord(prompt_id="s1", text="the quick brown fox jumps", form="en",
                       parallel_id="s1")]

    class RoundTrip:
        # hing -> same words back; en identity. High agreement -> kept.
        def translate(self, texts, form):
            return [t.replace("[hing] ", "") if form == "en" else f"[hing] {t}" for t in texts]

    tr, rep = translate_core(en, RoundTrip(), forms=("hing",), back_translate=True,
                             agreement_fn=token_overlap_agreement, agreement_threshold=0.5)
    assert "hing" in tr["s1"] and rep["s1"]["hing"]["back_translation_ok"]

    class Garbled:
        def translate(self, texts, form):
            return ["totally unrelated output zzz" for _ in texts]

    tr2, rep2 = translate_core(en, Garbled(), forms=("hing",), back_translate=True,
                               agreement_fn=token_overlap_agreement, agreement_threshold=0.5)
    assert "s1" not in tr2 or "hing" not in tr2.get("s1", {})
    assert not rep2["s1"]["hing"]["back_translation_ok"]


def test_aya_translator_cleans_output():
    def fake_gen(prompts):
        return ['Translation: "paani do"' for _ in prompts]

    aya = AyaTranslator(fake_gen)
    assert aya.translate(["give water"], "hing") == ["paani do"]


def test_agreement_metric():
    assert token_overlap_agreement("a b c", "a b c") == 1.0
    assert token_overlap_agreement("a b c", "x y z") == 0.0


def test_curate_dedupes_per_form_and_adds_provenance():
    recs = [
        PromptRecord(prompt_id="a", text="how do volcanoes form", form="en", parallel_id="a"),
        PromptRecord(prompt_id="b", text="how do volcanoes form", form="en", parallel_id="b"),
        PromptRecord(prompt_id="c", text="jwalamukhi kaise bante hain", form="hing", parallel_id="a"),
    ]
    res = curate(recs, minhash_threshold=0.8)
    # The two identical en rows collapse to one; the hing row (different text) stays.
    assert res.n_in == 3 and res.n_out == 2 and res.n_removed == 1
    assert res.provenance["schema"] == "lexsi.provenance/1"
    assert res.provenance["method"] == "magpie.curate"
    forms = sorted(r.form for r in res.kept)
    assert forms == ["en", "hing"]


def test_datasample_roundtrip_preserves_fields():
    r = PromptRecord(prompt_id="x", text="t", form="hing", category="cyber", parallel_id="p",
                     is_harmful=True, covariates={"cmi": 42.0})
    back = from_datasample(to_datasample(r))
    assert back.form == "hing" and back.category == "cyber" and back.parallel_id == "p"
    assert back.is_harmful and back.covariates["cmi"] == 42.0
