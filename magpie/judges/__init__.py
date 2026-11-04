"""Judging (PLAN §8): refusal lexicon, collapse guard, local judges, the validity gate.

The multilingual refusal matcher and collapse guard (:mod:`magpie.judges.refusal`)
fix a verified SafeTune gap — English-only refusal matching that over-counts
non-English attack success. :class:`LexiconJudge` is the CPU pre-filter;
:class:`CompositeJudge` is the second-judge validity gate.
"""

from magpie.judges.base import CompositeJudge, LexiconJudge
from magpie.judges.refusal import (
    RefusalMatcher,
    RefusalResult,
    detect_collapse,
    devanagari_fraction,
)

__all__ = [
    "LexiconJudge",
    "CompositeJudge",
    "RefusalMatcher",
    "RefusalResult",
    "detect_collapse",
    "devanagari_fraction",
]
