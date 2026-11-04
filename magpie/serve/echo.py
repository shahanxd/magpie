"""A deterministic, dependency-free backend for CPU tests and dry runs.

``EchoBackend`` never loads a model. It answers from a scripted policy so the
attack and judge plumbing can be tested end to end on CI: by default it
refuses anything containing a configured trigger word and complies otherwise,
which is enough to exercise survival bookkeeping, early stopping and cost
accounting without a GPU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from magpie.config import GenerationConfig
from magpie.serve.base import Completion, Request

Policy = Callable[[Request, GenerationConfig], str]


def _approx_tokens(text: str) -> int:
    # ~4 chars/token is the usual rule of thumb; good enough for a dry run.
    return max(1, len(text) // 4)


def refuse_on_trigger(triggers: Sequence[str] = ("TRIGGER",)) -> Policy:
    """A policy that refuses when any trigger appears, complies otherwise."""
    lowered = [t.lower() for t in triggers]

    def policy(req: Request, gen: GenerationConfig) -> str:
        text = (req.assistant_prefix + " " + req.prompt).lower()
        if any(t in text for t in lowered) and not req.assistant_prefix:
            return "I'm sorry, but I can't help with that."
        return "Sure, here is a general, high-level overview. " * 3

    return policy


class EchoBackend:
    """Scripted backend: ``policy(request, gen) -> response text``.

    Sampling (``temperature > 0``) is emulated by hashing the request with the
    seed, so a policy can depend on ``gen.seed`` deterministically.
    """

    name = "echo"

    # Tiny Aya's template preamble is ~370 tokens; emulate it so cost numbers
    # in dry runs have the right order of magnitude.
    PREAMBLE_TOKENS = 370

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or refuse_on_trigger()

    def generate(self, requests: Sequence[Request], gen: GenerationConfig) -> list[Completion]:
        out: list[Completion] = []
        for req in requests:
            text = self.policy(req, gen)
            words = text.split()
            budget = max(1, gen.max_new_tokens)
            if len(words) > budget:
                text = " ".join(words[:budget])
            prompt_tokens = self.PREAMBLE_TOKENS + _approx_tokens(req.prompt)
            out.append(
                Completion(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=_approx_tokens(text),
                    finish_reason="stop",
                    assistant_prefix=req.assistant_prefix,
                )
            )
        return out

    @staticmethod
    def sample_key(req: Request, gen: GenerationConfig) -> float:
        """A deterministic pseudo-random number in [0, 1) for (request, seed)."""
        h = hashlib.sha256(f"{req.prompt}|{gen.seed}|{gen.temperature}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF
