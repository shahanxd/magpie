"""Core record types shared by attacks, judges and analysis.

Every attack writes :class:`Attempt` rows; the survival analysis reads them
back. Keeping one flat, JSON-serialisable schema is what lets input-space and
weight-space attacks share a single cost axis and a single analysis path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from magpie.costs import AttackCost


class Language(StrEnum):
    """Language *forms* (PLAN §5.1). ``hi_deva`` is a script control only."""

    EN = "en"
    HING = "hing"  # Hinglish: romanized, code-mixed Hindi-English
    HI_DEVA = "hi-Deva"  # Devanagari Hindi (script control, parallel core only)


# Alias: the plan calls these "forms" (language × script).
Form = Language


class Level(StrEnum):
    """The attacker ladder (PLAN §3). Each level is a superset of the one before."""

    L0 = "L0"  # natural: type, in English or Hinglish
    L1 = "L1"  # black-box: query many times, set decoding parameters
    L2 = "L2"  # white-box input: read weights and logits
    L3 = "L3"  # weight: change the weights


class Verdict(StrEnum):
    """What a judge concluded about one response."""

    REFUSAL = "refusal"  # declined (the model is safe on this attempt)
    COMPLIANCE = "compliance"  # complied -- harmful *if* the prompt is harmful
    PARTIAL = "partial"  # partial / hedged compliance
    COLLAPSE = "collapse"  # gibberish, wrong language, degenerate repeat
    UNKNOWN = "unknown"


@dataclass
class Attempt:
    """One attack attempt against one prompt on one model.

    The unit of the survival analysis is a *prompt*; an attack produces a
    sequence of attempts per prompt at increasing cost, and the prompt
    "dies" at the first attempt judged a valid success (see
    :attr:`success`). Prompts never broken within budget are censored.
    """

    prompt_id: str
    model: str
    attack: str
    level: str
    form: str
    prompt: str
    response: str = ""
    # Attack-specific knobs (template family, N index, temperature, k-prefill...).
    params: dict[str, Any] = field(default_factory=dict)
    cost: AttackCost = field(default_factory=AttackCost)
    # Cumulative cost to reach this attempt (what the survival x-axis uses).
    cum_cost: AttackCost = field(default_factory=AttackCost)
    attempt_index: int = 0
    seed: int = 0
    # Judge outputs keyed by judge name; filled in by the judge pipeline.
    judgements: dict[str, Any] = field(default_factory=dict)
    is_harmful_prompt: bool = True
    # Provenance: config hash + git SHA so every row is traceable.
    run_id: str = ""
    git_sha: str = ""

    @property
    def uid(self) -> str:
        """Stable id for this attempt (prompt × model × attack × params × index)."""
        key = json.dumps(
            [self.prompt_id, self.model, self.attack, self.form, self.params,
             self.attempt_index, self.seed],
            sort_keys=True, default=str,
        )
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost"] = self.cost.to_dict()
        d["cum_cost"] = self.cum_cost.to_dict()
        d["uid"] = self.uid
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Attempt:
        d = dict(d)
        d.pop("uid", None)

        def _cost(c: Any) -> AttackCost:
            if isinstance(c, AttackCost):
                return c
            c = dict(c or {})
            c.pop("total_flops", None)
            return AttackCost(**c)

        d["cost"] = _cost(d.get("cost"))
        d["cum_cost"] = _cost(d.get("cum_cost"))
        return cls(**d)


@dataclass
class PromptRecord:
    """One evaluation prompt: the unit an attack varies and a judge scores.

    ``is_harmful`` splits the two measurement regimes: harmful prompts feed the
    survival analysis (cost to break), benign/twin prompts feed over-refusal.
    ``parallel_id`` links an en prompt to its hing / hi-Deva translations and a
    harmful seed to its benign twin, so comparisons stay paired.
    """

    prompt_id: str
    text: str
    form: str = Language.EN
    is_harmful: bool = True
    category: str = ""
    parallel_id: str = ""  # shared across forms and the harmful/benign twin
    source: str = ""
    covariates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromptRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Outcome:
    """The survival outcome of one prompt under one (model, attack, form).

    ``broken`` is the event indicator; ``cost_to_break`` is the cumulative
    total FLOPs at the breaking attempt, or the budget spent if censored.
    """

    prompt_id: str
    model: str
    attack: str
    form: str
    broken: bool
    cost_to_break: float  # total FLOPs (cumulative); budget spent if censored
    queries_to_break: int
    n_attempts: int
    level: str = ""
    covariates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
