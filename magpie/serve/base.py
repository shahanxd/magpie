"""One ``generate()`` behind every backend (vLLM, HF transformers, echo).

All backends share one prompt-formatting path: the *model's own chat
template* (via its HF tokenizer) renders the conversation, then an optional
``assistant_prefix`` is appended verbatim. That keeps prompts byte-identical
across vLLM and HF, which matters because Tiny Aya's template injects a
~370-token system preamble and the cost model counts those tokens.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from magpie.config import GenerationConfig
from magpie.costs import AttackCost, CostModel


@dataclass
class Request:
    """One generation request.

    ``prompt`` is the user turn. ``system`` overrides the model's default
    preamble when set (``None`` keeps the vendor default, which is what an
    on-device user gets). ``assistant_prefix`` forces the start of the reply
    (L2a prefilling). ``messages`` overrides everything with a full chat.
    """

    prompt: str
    system: str | None = None
    assistant_prefix: str = ""
    messages: list[dict[str, str]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        if self.messages is not None:
            return list(self.messages)
        msgs: list[dict[str, str]] = []
        if self.system is not None:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.prompt})
        return msgs


@dataclass
class Completion:
    """One generated response plus the token accounting the cost model needs."""

    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    latency_s: float = 0.0
    # The assistant prefix is part of the response an attacker sees.
    assistant_prefix: str = ""

    @property
    def full_text(self) -> str:
        return self.assistant_prefix + self.text


@runtime_checkable
class Backend(Protocol):
    """What every serving backend implements."""

    name: str

    def generate(self, requests: Sequence[Request], gen: GenerationConfig) -> list[Completion]:
        ...


def render_chat(tokenizer: Any, request: Request) -> str:
    """Render a request with the model's chat template, plus any assistant prefix."""
    text = tokenizer.apply_chat_template(
        request.to_messages(), tokenize=False, add_generation_prompt=True
    )
    return text + request.assistant_prefix


class Generator:
    """A backend plus the cost model: generate and price in one call.

    ``generate`` returns completions; ``generate_priced`` also returns the
    :class:`~magpie.costs.AttackCost` of each completion so attacks can log
    cost without knowing which backend ran.
    """

    def __init__(self, backend: Backend, cost_model: CostModel | None = None) -> None:
        self.backend = backend
        self.cost_model = cost_model or CostModel()

    @property
    def name(self) -> str:
        return self.backend.name

    def generate(self, requests: Sequence[Request], gen: GenerationConfig) -> list[Completion]:
        if not requests:
            return []
        t0 = time.perf_counter()
        out = self.backend.generate(list(requests), gen)
        if len(out) != len(requests):
            raise RuntimeError(
                f"backend {self.backend.name!r} returned {len(out)} completions "
                f"for {len(requests)} requests"
            )
        elapsed = time.perf_counter() - t0
        # Backends that cannot time per-request get an even share of the batch.
        for c in out:
            if c.latency_s == 0.0:
                c.latency_s = elapsed / len(out)
        return out

    def price(self, completion: Completion) -> AttackCost:
        cost = self.cost_model.generation(
            prompt_tokens=completion.prompt_tokens,
            output_tokens=completion.output_tokens,
            n_samples=1,
        )
        # Measured wall-clock beats the throughput estimate when we have it.
        if completion.latency_s > 0:
            cost = AttackCost(
                flops=cost.flops, queries=cost.queries, wall_clock_s=completion.latency_s,
                train_flops=cost.train_flops, gpu=cost.gpu, notes=cost.notes,
            )
        return cost

    def generate_priced(
        self, requests: Sequence[Request], gen: GenerationConfig
    ) -> list[tuple[Completion, AttackCost]]:
        comps = self.generate(requests, gen)
        return [(c, self.price(c)) for c in comps]
