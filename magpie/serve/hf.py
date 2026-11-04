"""HF transformers backend: batched greedy / sampled generation.

Used for CPU tests (with the tiny random cohere2 fixture), as the fallback on
Kaggle when vLLM is unavailable, and wherever white-box access is needed
(prefilling, activations). torch / transformers are imported lazily.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from magpie.config import GenerationConfig
from magpie.serve.base import Completion, Request, render_chat


def resolve_dtype(name: str, device: str) -> Any:
    import torch

    if device == "cpu":
        return torch.float32  # fp16 matmuls are slow / unsupported on many CPUs
    return {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}[name]


class HFBackend:
    """Generate with ``transformers``. Accepts a model id/path or a loaded model."""

    name = "hf"

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        dtype: str = "float16",
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        if isinstance(model, str):
            self.model_id = model
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=resolve_dtype(dtype, device), device_map=device
            )
        else:
            if tokenizer is None:
                raise ValueError("pass `tokenizer` together with a loaded model")
            self.model_id = getattr(getattr(model, "config", None), "_name_or_path", "model")
            self.model, self.tokenizer = model, tokenizer
        self.model.eval()
        # Left padding so every row's continuation starts at the same column.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _eos_ids(self) -> list[int]:
        ids: set[int] = set()
        for t in (self.tokenizer.eos_token_id, getattr(self.model.generation_config,
                                                        "eos_token_id", None)):
            if isinstance(t, int):
                ids.add(t)
            elif isinstance(t, list | tuple):
                ids.update(int(x) for x in t)
        return sorted(ids)

    def generate(self, requests: Sequence[Request], gen: GenerationConfig) -> list[Completion]:
        import torch

        out: list[Completion] = []
        eos = self._eos_ids()
        for start in range(0, len(requests), self.batch_size):
            batch = list(requests[start : start + self.batch_size])
            texts = [render_chat(self.tokenizer, r) for r in batch]
            # The chat template already carries BOS; don't add another.
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": gen.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": eos or None,
            }
            if gen.greedy:
                kwargs["do_sample"] = False
            else:
                torch.manual_seed(gen.seed)
                kwargs.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p)
            t0 = time.perf_counter()
            with torch.no_grad():
                seqs = self.model.generate(**enc, **kwargs)
            dt = time.perf_counter() - t0
            new = seqs[:, enc["input_ids"].shape[1]:]
            for i, req in enumerate(batch):
                row = new[i]
                n_out = int((row != self.tokenizer.pad_token_id).sum().item())
                hit_eos = bool(eos) and any(int(t) in eos for t in row.tolist())
                text = self.tokenizer.decode(row, skip_special_tokens=True)
                out.append(Completion(
                    text=text,
                    prompt_tokens=int(enc["attention_mask"][i].sum().item()),
                    output_tokens=n_out,
                    finish_reason="stop" if hit_eos else "length",
                    latency_s=dt / len(batch),
                    assistant_prefix=req.assistant_prefix,
                ))
        return out
