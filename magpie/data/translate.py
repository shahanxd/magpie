"""Translate the English core into Hinglish and Devanagari (PLAN §5.3).

The parallel core is built by LLM-assisted translation, then native-speaker
verification. The free, in-house path uses **Aya Expanse 8B** (a permitted
Cohere model) run locally on Kaggle, so no paid API and no data leaves the
kernel. Translation of harmful seeds happens in-kernel and its output lives
only in the gated dataset, never in the repo.

Design:

* a :class:`Translator` protocol (``translate(texts, target_form) -> list[str]``),
* :class:`AyaTranslator` (GPU; vLLM or HF) with per-form instructions —
  ``hing`` asks for natural romanized code-mixing (the way people type), and a
  **back-translation agreement gate** (translate back to English and keep only
  pairs whose meaning round-trips) that CuratorKIT records as provenance,
* deterministic mock translators for CPU tests.

Hinglish *spelling variation* is not added here: translation yields one
canonical Hinglish per seed; the natural-variation augmenter
(:mod:`magpie.data.hinglish`) generates the best-of-N variants at attack time.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from magpie.data.hinglish import cmi
from magpie.types import PromptRecord

FORM_INSTRUCTIONS: dict[str, str] = {
    "hing": (
        "Translate the following English text into Hinglish: natural, romanized "
        "Hindi-English code-mixing written in the Latin script, the way people "
        "actually type in everyday messages (do not use Devanagari). Preserve the "
        "exact meaning and intent. Output only the translation, nothing else."
    ),
    "hi-Deva": (
        "Translate the following English text into Hindi written in the Devanagari "
        "script. Preserve the exact meaning and intent. Output only the "
        "translation, nothing else."
    ),
    "en": "Repeat the following English text exactly. Output only the text.",
}


class Translator(Protocol):
    def translate(self, texts: Sequence[str], target_form: str) -> list[str]:
        ...


@dataclass
class MockTranslator:
    """Deterministic, offline translator for tests: tags text with the form."""

    def translate(self, texts: Sequence[str], target_form: str) -> list[str]:
        return [f"[{target_form}] {t}" for t in texts]


def _clean_output(text: str) -> str:
    """Strip quotes / "Translation:" preambles an instruct model may add."""
    t = (text or "").strip()
    t = re.sub(r"^(translation|hinglish|hindi)\s*[:\-]\s*", "", t, flags=re.IGNORECASE).strip()
    if len(t) >= 2 and t[0] in "\"'“" and t[-1] in "\"'”":
        t = t[1:-1].strip()
    return t


class AyaTranslator:
    """Aya Expanse 8B translator. ``generate_fn`` is injected; GPU loader via ``from_pretrained``."""

    def __init__(self, generate_fn: Callable[[Sequence[str]], list[str]],
                 max_new_tokens: int = 512) -> None:
        self._generate = generate_fn
        self.max_new_tokens = max_new_tokens

    def translate(self, texts: Sequence[str], target_form: str) -> list[str]:
        if target_form not in FORM_INSTRUCTIONS:
            raise ValueError(f"no instruction for form {target_form!r}")
        instr = FORM_INSTRUCTIONS[target_form]
        prompts = [f"{instr}\n\nText:\n{t}" for t in texts]
        return [_clean_output(o) for o in self._generate(prompts)]

    @classmethod
    def from_pretrained(cls, model_dir: str = "CohereLabs/aya-expanse-8b",
                        backend: str = "vllm", max_new_tokens: int = 512) -> AyaTranslator:
        from magpie.config import GenerationConfig
        from magpie.serve import Request, load_backend

        be = load_backend(backend, model_dir, dtype="float16", max_model_len=2048)
        gen = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)

        def generate(prompts: Sequence[str]) -> list[str]:
            comps = be.generate([Request(prompt=p) for p in prompts], gen)
            return [c.text for c in comps]

        return cls(generate, max_new_tokens)


def translate_core(
    en_records: Sequence[PromptRecord],
    translator: Translator,
    forms: Sequence[str] = ("hing", "hi-Deva"),
    back_translate: bool = False,
    agreement_fn: Callable[[str, str], float] | None = None,
    agreement_threshold: float = 0.5,
) -> tuple[dict[str, dict[str, str]], dict[str, dict]]:
    """Translate each en record into ``forms``; return (translations, report).

    ``translations[parallel_id][form] -> text`` feeds
    :func:`magpie.data.loaders.assemble_core`. When ``back_translate`` and an
    ``agreement_fn`` are given, each translation is translated back to English
    and scored; those below ``agreement_threshold`` are dropped and recorded in
    the report (the back-translation agreement gate). CMI is recorded for hing.
    """
    translations: dict[str, dict[str, str]] = {}
    report: dict[str, dict] = {}
    texts = [r.text for r in en_records]
    for form in forms:
        outs = translator.translate(texts, form)
        back = translator.translate(outs, "en") if back_translate else [None] * len(outs)
        for r, tr, bt in zip(en_records, outs, back, strict=True):
            info: dict = {"form": form, "ok": True}
            if form == "hing":
                info["cmi"] = cmi(tr).cmi
            if back_translate and agreement_fn is not None:
                score = agreement_fn(r.text, bt or "")
                info["agreement"] = round(score, 3)
                info["back_translation_ok"] = score >= agreement_threshold
                if score < agreement_threshold:
                    info["ok"] = False
            report.setdefault(r.parallel_id, {})[form] = info
            if info["ok"] and tr:
                translations.setdefault(r.parallel_id, {})[form] = tr
    return translations, report


def token_overlap_agreement(a: str, b: str) -> float:
    """A cheap back-translation agreement proxy: Jaccard over lowercased word sets.

    A stand-in for an embedding similarity (the real gate uses sentence
    embeddings); good enough to catch a translation that round-trips to
    something unrelated, and dependency-free for tests.
    """
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
