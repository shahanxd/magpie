"""CLI: harden a model with M2 (RR) or M3 (RepNoise) and save the merged checkpoint.

    python -m magpie.defenses.train --method rr --model DIR --out DIR [--n 256 --steps 150]

Runs in its own process inside a kernel so the GPU is fully released afterwards.
Loads the base model in fp16 on one T4 with gradient checkpointing, pins
SafeTune to fp16, trains, and writes ``train_status.json`` (time, steps, loss
trace summary, peak GPU memory) next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Reduce fragmentation OOMs on the 15 GB T4 (RepNoise holds harmful + noise
# activations alongside the retain batch). Set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Train on ONE T4. SafeTune's HF Trainer otherwise wraps the model in
# DataParallel across both GPUs, which replicates it on GPU 0 and OOMs; the
# 3.35B model fits one T4 at a small batch, and our RR loop is single-GPU too.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rr", "repnoise"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256, help="hardening rows per set")
    ap.add_argument("--steps", type=int, default=150, help="RR optimisation steps")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from magpie.defenses.harden import (
        RRConfig,
        configure_safetune,
        default_rr_data,
        train_circuit_breakers,
        train_repnoise,
    )

    fp16 = args.device.startswith("cuda")
    configure_safetune(dtype="float16" if fp16 else "float32", max_len=args.max_len, n=args.n,
                       seed=args.seed, device=args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if fp16 else torch.float32, device_map=args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out = Path(args.out)
    status: dict = {"method": args.method, "model": args.model, "n": args.n,
                    "batch_size": args.batch_size, "lr": args.lr, "max_len": args.max_len}
    t0 = time.time()
    if args.method == "rr":
        retain, harmful_set = default_rr_data(tok, n=args.n)
        cfg = RRConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, fp16=fp16,
                       seed=args.seed, grad_checkpointing=False)  # already enabled above
        res = train_circuit_breakers(model, tok, retain, harmful_set, out, cfg)
        status.update(steps=res.steps, target_layers=res.target_layers,
                      loss_first=res.losses[0] if res.losses else None,
                      loss_last=res.losses[-1] if res.losses else None,
                      n_retain=len(retain), n_cb=len(harmful_set))
    else:
        path = train_repnoise(model, tok, out, n=args.n, batch_size=args.batch_size,
                              lr=args.lr, fp16=fp16)
        status["checkpoint"] = str(path)
    status["seconds"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        status["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_status.json").write_text(json.dumps(status, indent=2))
    print("MAGPIE_TRAIN " + json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
