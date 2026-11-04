"""Concrete attacks on the echo backend: variant generation and cost logging."""

import random

from magpie.attacks import ATTACKS, L0Direct, L1BonDecoding, L1BonInput, L1Templates
from magpie.attacks.base import AttackRunner
from magpie.attacks.bon_augment import synthetic_bon_variants, synthetic_variant
from magpie.attacks.l1_templates import TEMPLATES
from magpie.config import GenerationConfig
from magpie.judges import LexiconJudge
from magpie.serve import EchoBackend, Generator, Request, refuse_on_trigger
from magpie.types import PromptRecord


def _runner(policy=None, **kw):
    gen = Generator(EchoBackend(policy or refuse_on_trigger()))
    return AttackRunner(gen, LexiconJudge(), GenerationConfig(max_new_tokens=32), **kw)


def test_registry_names_match():
    for name, cls in ATTACKS.items():
        assert cls.name == name


def test_l0_direct_greedy_and_sampled_use_different_decoding():
    # Benign prompt: never a "success", so both variants run (a harmful prompt
    # that the model complies with would correctly break at the greedy attempt).
    rec = PromptRecord(prompt_id="p", text="tell me about photosynthesis", is_harmful=False)
    run = _runner().run(L0Direct(sampled=True), [rec])
    assert len(run.attempts) == 2
    decs = {a.params["decoding"] for a in run.attempts}
    assert decs == {"greedy", "sampled"}
    greedy = next(a for a in run.attempts if a.params["decoding"] == "greedy")
    assert greedy.cost.notes["n_samples"] == 1


def test_l0_direct_breaks_harmful_prompt_at_greedy_when_model_complies():
    rec = PromptRecord(prompt_id="p", text="benign-looking but harmful", is_harmful=True)
    # Echo complies (no trigger) -> harmful + compliance -> breaks at attempt 0.
    run = _runner().run(L0Direct(sampled=True), [rec])
    assert run.outcomes[0].broken and run.outcomes[0].n_attempts == 1


def test_l0_direct_greedy_only():
    rec = PromptRecord(prompt_id="p", text="hello", is_harmful=False)
    run = _runner().run(L0Direct(sampled=False), [rec])
    assert len(run.attempts) == 1


def test_templates_cover_all_families_in_form():
    rec = PromptRecord(prompt_id="p", text="SUBJECT", form="hing")
    run = _runner(policy=lambda r, g: "Bilkul, yeh raha jawaab " * 5).run(L1Templates(), [rec])
    fams = [a.params["family"] for a in run.attempts]
    assert set(fams) <= set(TEMPLATES)
    # Hinglish form used, and the record text is embedded in each variant.
    assert all(a.params["template_form"] == "hing" for a in run.attempts)
    assert all("SUBJECT" in a.prompt for a in run.attempts)


def test_bon_natural_vs_synthetic_variant_counts_and_dedup():
    text = "bhai ye kaam kaise karna hai batao"
    nat = L1BonInput(source="natural", n=20)
    variants = list(nat._augment(text))
    assert variants[0] == text and len(variants) == len(set(variants))  # natural dedups
    syn = synthetic_bon_variants(text, n=20, seed=0)
    assert syn[0] == text and len(syn) == 20


def test_synthetic_augmentation_is_inert_on_devanagari_caps():
    # The verified gap: random capitalization changes no Devanagari character.
    from magpie.attacks.bon_augment import random_capitalization

    deva = "यह एक वाक्य है जिसमें कोई बदलाव नहीं होना चाहिए"
    rng = random.Random(0)
    assert random_capitalization(deva, rng, sigma=0.9) == deva


def test_bon_input_breaks_when_a_variant_drops_the_trigger():
    # Policy refuses unless the word "safe" appears; natural augmenter won't add it,
    # so craft a policy that complies once any variant differs from the original.
    rec = PromptRecord(prompt_id="p", text="please TRIGGER this")

    def policy(req: Request, g) -> str:
        if "TRIGGER" in req.prompt:
            return "I'm sorry, but I can't help with that."
        return "Sure, here is a complete and detailed answer for you, step by step now."

    run = _runner(policy=policy, chunk_size=4).run(
        L1BonInput(source="synthetic", n=32, sigma=0.6), [rec])
    out = run.outcomes[0]
    # Synthetic noise eventually corrupts "TRIGGER", so the prompt breaks after N>1.
    assert out.broken and out.n_attempts > 1


def test_bon_decoding_sweep_sizes_and_overrides():
    rec = PromptRecord(prompt_id="p", text="explain clouds", is_harmful=False)
    atk = L1BonDecoding(temperatures=(0.7, 1.0), top_ps=(0.9,), samples_per_setting=3)
    run = _runner().run(atk, [rec])
    assert len(run.attempts) == 2 * 1 * 3
    temps = {a.params["temperature"] for a in run.attempts}
    assert temps == {0.7, 1.0}


def test_synthetic_variant_determinism():
    r1, r2 = random.Random(5), random.Random(5)
    s = "the quick brown fox jumps over the lazy dog"
    assert synthetic_variant(s, r1, 0.5) == synthetic_variant(s, r2, 0.5)
