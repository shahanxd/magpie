"""vLLM backend for high-throughput GPU generation on Kaggle T4s.

Prompts are rendered with the HF tokenizer's chat template (same path as the
HF backend) and passed to vLLM as token ids, so vLLM never re-tokenises or
adds a second BOS. On pre-Ampere GPUs (T4 = sm_75) we pin the Triton attention
backend, which supports Turing; Tiny Aya's 4096-token sliding window is wider
than our context cap, so every layer behaves as full attention here.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

from magpie.config import GenerationConfig
from magpie.serve.base import Completion, Request, render_chat


def _configure_env_for_gpu() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        import torch

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8:  # Turing (T4) and older: no FlashAttention 2
                os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    except Exception:  # pragma: no cover - best effort only
        pass


class VLLMBackend:
    """Generate with vLLM. ``model`` is an HF id or a local path."""

    name = "vllm"

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.88,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        seed: int = 0,
        **llm_kwargs: Any,
    ) -> None:
        _configure_env_for_gpu()
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            seed=seed,
            **llm_kwargs,
        )
        self.max_model_len = max_model_len

    def generate(self, requests: Sequence[Request], gen: GenerationConfig) -> list[Completion]:
        from vllm import SamplingParams

        prompts: list[dict[str, Any]] = []
        for r in requests:
            ids = self.tokenizer.encode(render_chat(self.tokenizer, r), add_special_tokens=False)
            budget = self.max_model_len - gen.max_new_tokens
            if len(ids) > budget:  # left-truncate the user content, keep the reply room
                ids = ids[-budget:]
            prompts.append({"prompt_token_ids": ids})
        params = SamplingParams(
            temperature=gen.temperature,
            top_p=gen.top_p if not gen.greedy else 1.0,
            max_tokens=gen.max_new_tokens,
            seed=None if gen.greedy else gen.seed,
        )
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        dt = time.perf_counter() - t0
        result: list[Completion] = []
        for req, p, o in zip(requests, prompts, outs, strict=True):
            c = o.outputs[0]
            result.append(Completion(
                text=c.text,
                prompt_tokens=len(p["prompt_token_ids"]),
                output_tokens=len(c.token_ids),
                finish_reason=c.finish_reason or "stop",
                # vLLM batches everything; report the batch's mean latency per request.
                latency_s=dt / max(1, len(requests)),
                assistant_prefix=req.assistant_prefix,
            ))
        return result

    def close(self) -> None:
        """Release the GPU so another model can be loaded in the same process.

        vLLM holds device memory in engine/worker state that ``del`` alone does
        not free; the documented teardown tears down the (single-GPU) model
        parallel state first. Best-effort and import-guarded across vLLM
        versions — callers that need a hard guarantee run one model per process.
        """
        import gc

        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001 - version-dependent; fall through to del
            pass
        for attr in ("llm_engine", "engine"):
            try:
                delattr(self.llm, attr)
            except Exception:  # noqa: BLE001
                pass
        try:
            del self.llm
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
