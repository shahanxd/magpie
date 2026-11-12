"""Judging (PLAN §8): refusal lexicon, collapse guard, local judges, the validity gate.

The multilingual refusal matcher and collapse guard (:mod:`magpie.judges.refusal`)
fix a verified SafeTune gap -- English-only refusal matching that over-counts
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

from magpie.judges.model_judge import (  # noqa: E402
    JUDGE_MODELS,
    ModelJudge,
    ModelJudgeConfig,
    parse_llama_guard,
    parse_wildguard,
)

__all__ += ["ModelJudge", "ModelJudgeConfig", "JUDGE_MODELS", "parse_llama_guard",
            "parse_wildguard"]

from magpie.judges.safetune_judge import SafeTuneJudge  # noqa: E402

__all__ += ["SafeTuneJudge"]

from magpie.judges.bon_gate import BoNGate  # noqa: E402

__all__ += ["BoNGate"]

from magpie.judges.harmbench import HarmBenchJudge  # noqa: E402

__all__ += ["HarmBenchJudge"]
