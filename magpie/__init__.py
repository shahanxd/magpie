"""Magpie -- pricing robustness as attacker cost.

Magpie measures *how much attacker effort* it takes to break a hardened,
on-device multilingual LLM (Cohere Tiny Aya Fire), in English and in Hinglish,
and reports that price next to over-refusal. The top-level package stays light:
importing :mod:`magpie` pulls in no ML stack. The cost model, the Hinglish
tools, the refusal matcher and the survival analysis all work on a plain
CPU-only install; torch / transformers / vLLM / lifelines are lazy-imported
behind the extras that need them.
"""

from __future__ import annotations

__version__ = "0.1.0"

from magpie.costs import TINY_AYA_FIRE, AttackCost, CostModel
from magpie.types import Attempt, Form, Language, Level, Outcome, Verdict

__all__ = [
    "__version__",
    "AttackCost",
    "CostModel",
    "TINY_AYA_FIRE",
    "Attempt",
    "Outcome",
    "Verdict",
    "Form",
    "Language",
    "Level",
]
