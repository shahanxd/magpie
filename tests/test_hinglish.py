from magpie.data.hinglish import (
    HinglishAugmenter,
    cmi,
    natural_bon_variants,
    romanization_variants,
    tag_token,
)


def test_cmi_monolingual_is_zero():
    assert cmi("This is a perfectly ordinary English sentence about cooking.").cmi == 0.0
    # Unambiguous Hindi words -> recognised as monolingual Hindi (no en homographs).
    assert cmi("ghar aur khana dost").cmi == 0.0


def test_cmi_ignores_english_hindi_homographs():
    # "is/in/the/to" are English here, not romanized Hindi; a naive lexicon would
    # wrongly flag this plain-English sentence as code-mixed.
    assert not cmi("the answer is in the book we read").is_code_mixed


def test_cmi_detects_code_mixing():
    r = cmi("main office ja raha hun but thoda late ho jaunga")
    assert r.is_code_mixed and r.cmi > 0
    assert r.counts["hi"] > 0 and r.counts["en"] > 0


def test_cmi_is_balanced_at_fifty_for_even_mix():
    # 2 hi + 2 en language tokens, evenly split -> CMI = 100*(1-2/4) = 50.
    r = cmi("ghar office dost work")
    assert r.n_lang_tokens == 4
    assert r.cmi == 50.0


def test_tag_token():
    assert tag_token("nahi") == "hi"
    assert tag_token("computer") == "en"
    assert tag_token("123") == "univ"
    assert tag_token("दोस्त") == "hi"  # Devanagari


def test_romanization_variants_cover_known_spellings():
    v = romanization_variants("nahi")
    assert "nahi" == v[0]
    assert {"nahin", "nhi"} & set(v)
    assert "acha" in romanization_variants("accha")  # geminate collapse
    assert romanization_variants("photo")[0] == "photo"  # ph->f swap available
    assert "foto" in romanization_variants("photo")


def test_natural_variants_are_distinct_deterministic_and_plausible():
    text = "bhai mujhe nahi pata ye kaise karna hai"
    a = natural_bon_variants(text, n=12, seed=0)
    b = natural_bon_variants(text, n=12, seed=0)
    assert a == b  # deterministic
    assert a[0] == text  # original first
    assert len(set(a)) == len(a)  # distinct
    assert len(a) > 3  # this text admits several rewrites
    # Every variant keeps roughly the same length (natural, not shredded).
    assert all(0.5 < len(v) / len(text) < 1.6 for v in a)


def test_augmenter_can_shift_code_mix_ratio():
    aug = HinglishAugmenter()
    variants = aug.variants("please give me water and food at home", n=25, seed=3)
    joined = " ".join(variants).lower()
    # At least one English content word should have been swapped to Hinglish.
    assert any(h in joined for h in ("paani", "khana", "ghar"))


def test_variants_fall_back_gracefully_on_unmixable_text():
    # Pure English with no lexicon hits yields few rewrites but never crashes.
    out = natural_bon_variants("xyzzy qwerty", n=10, seed=1)
    assert out[0] == "xyzzy qwerty"
