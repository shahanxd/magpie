"""Smoke-test worker: one backend per process, so a vLLM crash can't take HF down.

Run as ``python -m magpie.runners.jobs.smoke_worker --backend {vllm,hf} --model DIR --out DIR``.
Measures load time and decode throughput on benign en + Hinglish prompts and,
for the HF backend, checks fp16 numerics against an fp32 CPU reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from magpie.config import GenerationConfig
from magpie.runners.jobs.common import write_json
from magpie.serve.base import Request

# Benign prompts only: the smoke test checks the plumbing, not safety.
SMOKE_PROMPTS: list[tuple[str, str]] = [
    ("en", "Explain how photosynthesis works in three sentences."),
    ("en", "Give me three practical tips to save money on groceries."),
    ("en", "Write a short, friendly message wishing a colleague a happy birthday."),
    ("en", "What is the difference between weather and climate?"),
    ("en", "Suggest a simple 20-minute workout I can do at home."),
    ("hing", "Mujhe ek simple dal tadka recipe batao, step by step."),
    ("hing", "Exam ke liye padhai kaise karun taaki focus bana rahe? 3 tips do."),
    ("hing", "Mere dost ka birthday hai, usko ek funny sa message likh do."),
    ("hing", "Monsoon mein ghar ko seelan se kaise bachayein?"),
    ("hing", "Ek beginner ko Python seekhna kahan se shuru karna chahiye?"),
]


def _degeneracy(text: str) -> dict[str, Any]:
    words = text.split()
    letters = sum(c.isalpha() for c in text)
    return {
        "chars": len(text),
        "alpha_frac": round(letters / max(1, len(text)), 3),
        "distinct_word_frac": round(len(set(words)) / max(1, len(words)), 3),
        "looks_degenerate": len(words) < 3 or (len(words) > 20 and len(set(words)) / len(words) < 0.2),
    }


def _measure(backend: Any, max_new_tokens: int) -> dict[str, Any]:
    reqs = [Request(prompt=p) for _, p in SMOKE_PROMPTS]
    gen = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)
    backend.generate([Request(prompt="Hi")], GenerationConfig(max_new_tokens=8))  # warm-up

    t0 = time.perf_counter()
    batch = backend.generate(reqs, gen)
    batch_s = time.perf_counter() - t0
    out_tok = sum(c.output_tokens for c in batch)

    single: list[float] = []
    for r in reqs[:3]:
        t1 = time.perf_counter()
        (c,) = backend.generate([r], gen)
        single.append(c.output_tokens / max(1e-6, time.perf_counter() - t1))

    rows = []
    for (form, prompt), c in zip(SMOKE_PROMPTS, batch, strict=True):
        rows.append({"form": form, "prompt": prompt, "response": c.text,
                     "prompt_tokens": c.prompt_tokens, "output_tokens": c.output_tokens,
                     "finish_reason": c.finish_reason, **_degeneracy(c.text)})
    return {
        "batch_size": len(reqs),
        "batch_seconds": round(batch_s, 2),
        "batch_output_tokens": out_tok,
        "batch_tokens_per_s": round(out_tok / max(1e-6, batch_s), 1),
        "single_stream_tokens_per_s": round(sum(single) / len(single), 1),
        "mean_prompt_tokens": round(sum(c.prompt_tokens for c in batch) / len(batch), 1),
        "n_degenerate": sum(r["looks_degenerate"] for r in rows),
        "rows": rows,
    }


def _fp16_check(model_dir: str, tokenizer: Any, model16: Any) -> dict[str, Any]:
    """fp16-on-GPU vs fp32-on-CPU: logits finite, activation headroom, KL, top-k."""
    import torch
    from transformers import AutoModelForCausalLM

    from magpie.serve.base import render_chat

    fp16_max = 65504.0
    layers = model16.model.layers
    peaks = [0.0] * len(layers)
    nonfinite = [False] * len(layers)

    def make_hook(i: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.detach().float()
            peaks[i] = max(peaks[i], float(hf.abs().max()))
            nonfinite[i] = nonfinite[i] or not bool(torch.isfinite(hf).all())
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    logits16: list[Any] = []
    all_finite = True
    max_logit = 0.0
    try:
        with torch.no_grad():
            for _, prompt in SMOKE_PROMPTS:
                text = render_chat(tokenizer, Request(prompt=prompt))
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
                    model16.device)
                lg = model16(**ids).logits[0, -1].float()
                all_finite = all_finite and bool(torch.isfinite(lg).all())
                max_logit = max(max_logit, float(lg.abs().max()))
                logits16.append(lg.cpu())
    finally:
        for h in handles:
            h.remove()

    # fp32 reference on CPU for one en and one hing prompt (13.4 GB of RAM, ~1 min each).
    ref_idx = [0, 5]
    t0 = time.perf_counter()
    model32 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                                   device_map="cpu").eval()
    comps = []
    with torch.no_grad():
        for i in ref_idx:
            text = render_chat(tokenizer, Request(prompt=SMOKE_PROMPTS[i][1]))
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            lg32 = model32(**ids).logits[0, -1].float()
            p32 = torch.log_softmax(lg32, -1)
            p16 = torch.log_softmax(logits16[i], -1)
            kl = float(torch.sum(p32.exp() * (p32 - p16)))
            top5_32 = set(torch.topk(lg32, 5).indices.tolist())
            top5_16 = set(torch.topk(logits16[i], 5).indices.tolist())
            comps.append({"form": SMOKE_PROMPTS[i][0], "kl_fp32_fp16": kl,
                          "top1_match": int(lg32.argmax()) == int(logits16[i].argmax()),
                          "top5_overlap": len(top5_32 & top5_16) / 5})
    del model32
    peak = max(peaks) if peaks else 0.0
    return {
        "logits_all_finite": all_finite,
        "max_abs_logit": round(max_logit, 2),
        "layer_peak_abs_activation": [round(p, 1) for p in peaks],
        "max_layer_peak": round(peak, 1),
        "fp16_headroom": round(fp16_max / max(peak, 1e-6), 1),  # >2 is comfortable
        "any_layer_nonfinite": any(nonfinite),
        "fp32_reference": comps,
        "fp32_reference_seconds": round(time.perf_counter() - t0, 1),
        "verdict": "ok" if (all_finite and not any(nonfinite) and peak < fp16_max / 2
                            and all(c["kl_fp32_fp16"] < 0.05 and c["top1_match"]
                                    for c in comps)) else "check",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--fp16-check", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu only for local tests")
    args = ap.parse_args(argv)
    out = Path(args.out)
    res: dict[str, Any] = {"backend": args.backend, "model": args.model, "dtype": "float16"}
    t0 = time.perf_counter()
    try:
        if args.backend == "vllm":
            from magpie.serve.vllm_backend import VLLMBackend

            try:
                backend = VLLMBackend(args.model, dtype="float16", max_model_len=2048)
            except Exception as exc:  # CUDA-graph capture can fail on old GPUs
                res["cuda_graph_error"] = f"{type(exc).__name__}: {exc}"[:2000]
                backend = VLLMBackend(args.model, dtype="float16", max_model_len=2048,
                                      enforce_eager=True)
                res["enforce_eager"] = True
        else:
            from magpie.serve.hf import HFBackend

            backend = HFBackend(args.model, dtype="float16", device=args.device, batch_size=10)
        res["load_seconds"] = round(time.perf_counter() - t0, 1)
        res.update(_measure(backend, args.max_new_tokens))
        if args.backend == "hf" and args.fp16_check:
            res["fp16_check"] = _fp16_check(args.model, backend.tokenizer, backend.model)
        res["ok"] = True
    except Exception as exc:  # report, the orchestrator decides what it means
        import traceback

        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        res["traceback"] = traceback.format_exc()[-8000:]
    rows = res.pop("rows", [])
    write_json(out / f"{args.backend}.json", res)
    with open(out / f"{args.backend}_generations.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: res.get(k) for k in ("backend", "ok", "load_seconds",
                                              "batch_tokens_per_s", "error")}), flush=True)
    return 0 if res["ok"] else 1



if __name__ == "__main__":
    raise SystemExit(main())
