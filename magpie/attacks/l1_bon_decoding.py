"""L1d — best-of-N over decoding settings (PLAN §6).

An on-device user controls temperature and top-p. This attack keeps the prompt
fixed and draws ``n`` samples at each (temperature, top-p) in the sweep. Every
sample is one priced generation; a valid success on any draw breaks the prompt.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from magpie.attacks.base import Attack, Variant
from magpie.config import GenerationConfig
from magpie.serve.base import Request
from magpie.types import Level, PromptRecord

DEFAULT_TEMPS = (0.7, 1.0, 1.3)
DEFAULT_TOP_PS = (0.9, 1.0)


class L1BonDecoding(Attack):
    """Sweep temperature × top-p, drawing ``samples_per_setting`` draws each."""

    name = "l1_bon_decoding"
    level = Level.L1

    def __init__(self, temperatures=DEFAULT_TEMPS, top_ps=DEFAULT_TOP_PS,
                 samples_per_setting: int = 8, seed: int = 0) -> None:
        super().__init__(temperatures=tuple(temperatures), top_ps=tuple(top_ps),
                         samples_per_setting=samples_per_setting, seed=seed)
        self.temperatures = tuple(temperatures)
        self.top_ps = tuple(top_ps)
        self.samples_per_setting = samples_per_setting
        self.seed = seed

    def variants(self, record: PromptRecord, gen: GenerationConfig) -> Iterator[Variant]:
        for t in self.temperatures:
            for p in self.top_ps:
                for s in range(self.samples_per_setting):
                    yield Variant(
                        Request(prompt=record.text),
                        params={"temperature": t, "top_p": p, "draw": s},
                        gen_override=replace(gen, temperature=t, top_p=p, seed=self.seed + s),
                    )
