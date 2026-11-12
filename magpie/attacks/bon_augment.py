"""Synthetic best-of-N augmentations (Hughes et al. 2024), faithful to the paper.

These are the *synthetic* baseline that Magpie's natural-Hinglish augmenter
(``magpie.data.hinglish``) is compared against in H2. The three text
augmentations and their ``sigma``-derived probabilities are reimplemented here
from the authors' ``run_text_bon.py`` (and match SafeTune's ``BoNAttack``):

* word scrambling -- interior letters of words > 3 chars, p = sigma**0.5
* random capitalization -- per letter case flip, p = sigma**0.5
* ASCII noising -- per printable char ±1 code point, p = sigma**3

All three are applied, in that order, to every variant; diversity comes from
the per-character sampling, not from choosing a subset. Keeping our own copy
avoids importing SafeTune's heavy stack just to build a prompt string; the
verified-gap note is why this matters for Devanagari (the plan's C2 control).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def word_scramble(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    p = sigma ** 0.5
    out = []
    for word in text.split():
        if len(word) > 3 and rng.random() < p:
            mid = list(word[1:-1])
            rng.shuffle(mid)
            out.append(word[0] + "".join(mid) + word[-1])
        else:
            out.append(word)
    return " ".join(out)


def random_capitalization(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    p = sigma ** 0.5
    out = []
    for c in text:
        if c.isalpha() and rng.random() < p:
            if "a" <= c <= "z":
                out.append(chr(ord(c) - 32))
            elif "A" <= c <= "Z":
                out.append(chr(ord(c) + 32))
            else:
                out.append(c)  # non-ASCII letter: no ASCII case flip (the verified gap)
        else:
            out.append(c)
    return "".join(out)


def ascii_noise(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    p = sigma ** 3
    out = []
    for c in text:
        if c.isprintable() and rng.random() < p:
            shifted = ord(c) + rng.choice((-1, 1))
            out.append(chr(shifted) if 32 <= shifted <= 126 else c)
        else:
            out.append(c)
    return "".join(out)


SYNTHETIC_AUGMENTATIONS = (word_scramble, random_capitalization, ascii_noise)


@dataclass
class SyntheticBoNConfig:
    sigma: float = 0.4
    seed: int = 0


def synthetic_variant(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    """Apply all three augmentations in order (the paper's composition)."""
    out = text
    for aug in SYNTHETIC_AUGMENTATIONS:
        out = aug(out, rng, sigma)
    return out


def synthetic_bon_variants(text: str, n: int, seed: int = 0, sigma: float = 0.4) -> list[str]:
    """``n`` synthetic BoN variants of ``text`` (the original first).

    Unlike the natural augmenter, variants are *not* de-duplicated: the paper's
    power law is over draws, and at high sigma distinct draws are the norm.
    """
    rng = random.Random(f"{seed}|{text}")
    return [text] + [synthetic_variant(text, rng, sigma) for _ in range(max(0, n - 1))]
