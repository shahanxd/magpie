"""L1c — best-of-N on the input (PLAN §6), the home of contribution C2.

Three augmentation *sources*, one attack, so the survival/power-law analysis is
identical across them and the only thing that changes is the distribution the
variants are drawn from:

* ``synthetic`` — the Hughes et al. augmentations (random caps, ASCII noise,
  word scramble). The must-beat baseline.
* ``natural`` — the natural-Hinglish augmenter: spellings, transliteration,
  SMS shorthand and code-mix shifts that real users type. H2's hypothesis is
  that this reaches a given ASR in fewer samples (a larger power-law exponent).
* ``devanagari`` — the script control: on Devanagari, synthetic augmentations
  are mostly inert (random capitalisation changes nothing, ASCII noise only
  touches spaces), which is the verified SafeTune tool gap.

Variants are yielded cheapest-first (the unperturbed prompt is variant 0), so
the runner early-stops at the first valid success and logs cost at that N;
``fit −log ASR(N) = a·N^(−b)`` is then a function of the logged outcomes.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from magpie.attacks.base import Attack, Variant
from magpie.attacks.bon_augment import synthetic_variant
from magpie.config import GenerationConfig
from magpie.data.hinglish import HinglishAugmenter, HinglishVariationConfig
from magpie.serve.base import Request
from magpie.types import Level, PromptRecord

SOURCES = ("synthetic", "natural", "devanagari")


class L1BonInput(Attack):
    """Best-of-N over input perturbations from a chosen augmentation source.

    Decoding stays greedy (the variation is in the input, not the sampler); use
    :class:`~magpie.attacks.l1_bon_decoding.L1BonDecoding` for decoding BoN.
    """

    name = "l1_bon_input"
    level = Level.L1

    def __init__(self, source: str = "natural", n: int = 128, sigma: float = 0.4,
                 seed: int = 0, config: HinglishVariationConfig | None = None) -> None:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        super().__init__(source=source, n=n, sigma=sigma, seed=seed)
        self.source = source
        self.n = n
        self.sigma = sigma
        self.seed = seed
        self._augmenter = HinglishAugmenter(config or HinglishVariationConfig())

    def _augment(self, text: str) -> Iterator[str]:
        """Yield the original then perturbed variants, lazily (so early stop saves work)."""
        yield text
        if self.source == "natural":
            # Deterministic, de-duplicated natural rewrites.
            yield from self._augmenter.variants(text, self.n, self.seed)[1:]
            return
        rng = random.Random(f"{self.source}|{self.seed}|{text}")
        for _ in range(self.n - 1):
            if self.source == "synthetic":
                yield synthetic_variant(text, rng, self.sigma)
            else:  # devanagari: synthetic augmentation on (expected) Devanagari text
                yield synthetic_variant(text, rng, self.sigma)

    def variants(self, record: PromptRecord, gen: GenerationConfig) -> Iterator[Variant]:
        name = self.name
        for i, text in enumerate(self._augment(record.text)):
            yield Variant(
                Request(prompt=text),
                params={"attack": name, "source": self.source, "n_index": i,
                        "sigma": self.sigma},
            )
