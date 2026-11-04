"""Serving: vLLM / HF transformers / echo behind one ``generate()``.

``load_backend("vllm", ...)`` falls back to HF when vLLM can't be imported or
fails to initialise (the plan's "vLLM, fall back to HF" rule), and records
which backend actually ran.
"""

from __future__ import annotations

import logging
from typing import Any

from magpie.serve.base import Backend, Completion, Generator, Request, render_chat
from magpie.serve.echo import EchoBackend, refuse_on_trigger

log = logging.getLogger(__name__)


def load_backend(kind: str, model: str | None = None, **kwargs: Any) -> Backend:
    """Build a backend by name: ``"vllm"`` (falls back to ``"hf"``), ``"hf"``, ``"echo"``."""
    if kind == "echo":
        return EchoBackend(**kwargs)
    if model is None:
        raise ValueError(f"backend {kind!r} needs a model id or path")
    if kind == "vllm":
        try:
            from magpie.serve.vllm_backend import VLLMBackend

            return VLLMBackend(model, **kwargs)
        except Exception as exc:  # ImportError, CUDA init errors, unsupported arch...
            log.warning("vLLM unavailable (%s: %s); falling back to HF transformers",
                        type(exc).__name__, exc)
            kind = "hf"
            kwargs = {k: v for k, v in kwargs.items() if k in ("dtype", "device", "batch_size")}
    if kind == "hf":
        from magpie.serve.hf import HFBackend

        return HFBackend(model, **kwargs)
    raise ValueError(f"unknown backend {kind!r}; expected vllm | hf | echo")


__all__ = [
    "Backend",
    "Completion",
    "EchoBackend",
    "Generator",
    "Request",
    "load_backend",
    "refuse_on_trigger",
    "render_chat",
]
