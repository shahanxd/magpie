"""L0 — the natural attacker: just type the prompt.

The floor of the ladder (PLAN §6). No perturbation, no templates. We send the
prompt greedy, and optionally one sampled draw, because an on-device user who
re-rolls the temperature is still "L0 natural". The cost is one (or two)
forward generations; most hardened prompts should survive L0, which is exactly
what makes the higher levels' cost the interesting number.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from magpie.attacks.base import Attack, Variant
from magpie.config import GenerationConfig
from magpie.serve.base import Request
from magpie.types import Level, PromptRecord


class L0Direct(Attack):
    """Send the prompt as-is: greedy, then (optionally) one sampled draw."""

    name = "l0_direct"
    level = Level.L0

    def __init__(self, sampled: bool = True, temperature: float = 1.0, top_p: float = 1.0,
                 seed: int = 0) -> None:
        super().__init__(sampled=sampled, temperature=temperature, top_p=top_p, seed=seed)
        self.sampled = sampled
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def variants(self, record: PromptRecord, gen: GenerationConfig) -> Iterator[Variant]:
        yield Variant(Request(prompt=record.text), params={"decoding": "greedy"},
                      gen_override=replace(gen, temperature=0.0))
        if self.sampled:
            yield Variant(
                Request(prompt=record.text),
                params={"decoding": "sampled", "temperature": self.temperature,
                        "top_p": self.top_p},
                gen_override=replace(gen, temperature=self.temperature, top_p=self.top_p,
                                     seed=self.seed),
            )
