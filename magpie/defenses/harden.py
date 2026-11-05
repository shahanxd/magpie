"""Train-time hardening of Tiny Aya with SafeTune (PLAN §4: M2 RR, M3 RepNoise).

Both defenses chain SafeTune and keep its **English** hardening data on
purpose. That is the natural experiment behind H1: does hardening trained on
English protect Hinglish? Hardening data (BeaverTails, Alpaca and SafeTune's
refusal set) never overlaps the evaluation seeds (HarmBench, StrongREJECT).

* **M3 RepNoise** (Rosati et al. 2024) uses SafeTune's ``RepNoiseTrainer``
  unchanged. The train set is clean benign SFT data (``harmful_frac=0``, we
  harden rather than attack); BeaverTails harmful QA feeds the noise and
  gradient-ascent terms.
* **M2 Circuit Breakers / RR** (Zou et al. 2024). SafeTune provides the
  faithful RR loss (``CircuitBreakerModel.compute_rr_loss``) but not a
  training loop, so :func:`train_circuit_breakers` is the LoRA loop of the
  reference implementation (GraySwanAI ``lorra_circuit_breaker.py``):
  * the frozen original model is the same network with the adapter disabled,
    so only one copy of the 3.35B weights sits on the T4;
  * rerouting acts on harmful-*response* tokens, while retain keeps benign and
    refusal representations close to the original;
  * the retain/reroute coefficients follow the paper's linear schedule.

T4 note: ``torch.cuda.is_bf16_supported()`` is True on Turing but bf16 is
emulated there (slow), so everything runs fp16 with fp32 LoRA weights and a
gradient scaler.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Tiny Aya's chat template adds a ~370-token preamble; leave room for content.
DEFAULT_MAX_LEN = 768
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def configure_safetune(dtype: str = "float16", max_len: int = DEFAULT_MAX_LEN, n: int = 512,
                       seed: int = 0, **extra: Any) -> None:
    """Pin SafeTune's runtime: fp16 (never emulated bf16 on a T4), length, data size, seed."""
    import safetune

    safetune.configure(dtype=dtype, max_len=max_len, harden_data_n=n, seed=seed, **extra)


def scaled_layers(n_layers: int, reference: Sequence[int] = (10, 20), ref_depth: int = 32
                  ) -> list[int]:
    """Map the paper's RR layers (10, 20 of a 32-layer model) to ``n_layers``."""
    return sorted({max(1, min(n_layers - 1, round(layer * n_layers / ref_depth)))
                   for layer in reference})


# ---------------------------------------------------------------------------
# M3 — RepNoise (SafeTune trainer, unchanged)
# ---------------------------------------------------------------------------


def train_repnoise(model: Any, tokenizer: Any, out_dir: str | Path, *,
                   train_dataset: Any = None, harmful_dataset: Any = None,
                   safety_dataset: Any = None, n: int = 512, epochs: int = 1,
                   batch_size: int = 4, lr: float = 1e-4, fp16: bool = True) -> str:
    """Harden ``model`` with SafeTune RepNoise; returns the merged checkpoint dir.

    Datasets default to SafeTune's hardening data: benign SFT (no contamination)
    for training, BeaverTails harmful QA for the noise and ascent terms. Pass
    datasets explicitly for offline tests.
    """
    from safetune.runner import harden

    if train_dataset is None:
        from safetune.runner.utils.data_utils import build_sft_dataset

        train_dataset = build_sft_dataset(tokenizer, n_total=n, harmful_frac=0.0)
    trainer = harden.RepNoiseTrainer(model, tokenizer, epochs=epochs, batch_size=batch_size,
                                     lr=lr, fp16=fp16, bf16=False)
    return trainer.train(train_dataset, out_dir=str(out_dir), harmful_dataset=harmful_dataset,
                         safety_dataset=safety_dataset)


# ---------------------------------------------------------------------------
# M2 — Circuit Breakers / Representation Rerouting (LoRA loop around SafeTune's loss)
# ---------------------------------------------------------------------------


@dataclass
class RRConfig:
    steps: int = 150            # paper: 150 steps
    batch_size: int = 2
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    rr_alpha: float = 5.0       # paper: 5 (Llama-3) / 10 (Mistral)
    target_layers: list[int] | None = None  # None -> paper's (10, 20) scaled to depth
    grad_checkpointing: bool = True
    fp16: bool = True
    max_grad_norm: float = 1.0
    log_every: int = 10
    seed: int = 0


@dataclass
class RRResult:
    out_dir: str
    steps: int
    target_layers: list[int]
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rows(dataset: Any) -> list[dict[str, Any]]:
    """Accept an HF Dataset or a list of {input_ids, attention_mask, labels} rows."""
    return list(dataset) if not isinstance(dataset, list) else dataset


def _collate(rows: Sequence[dict[str, Any]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    def as_list(x: Any) -> list[int]:
        return x.tolist() if hasattr(x, "tolist") else list(x)

    ids = [as_list(r["input_ids"]) for r in rows]
    att = [as_list(r.get("attention_mask", [1] * len(i))) for r, i in zip(rows, ids, strict=True)]
    lab = [as_list(r.get("labels", i)) for r, i in zip(rows, ids, strict=True)]
    width = max(len(i) for i in ids)

    def pad(seq: list[int], value: int) -> list[int]:
        return seq + [value] * (width - len(seq))  # right-pad: positions stay aligned

    return {
        "input_ids": torch.tensor([pad(i, pad_id) for i in ids], device=device),
        "attention_mask": torch.tensor([pad(a, 0) for a in att], device=device),
        "labels": torch.tensor([pad(lb, -100) for lb in lab], device=device),
    }


def _hidden(model: Any, batch: dict[str, Any], layers: Iterable[int]) -> dict[int, Any]:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple(n_layers + 1): index 0 is the embeddings
    return {layer: hs[layer] for layer in layers}


def train_circuit_breakers(model: Any, tokenizer: Any, retain_dataset: Any, cb_dataset: Any,
                           out_dir: str | Path, config: RRConfig | None = None,
                           log=print) -> RRResult:
    """Train RR LoRA adapters with SafeTune's faithful loss, merge, save.

    ``retain_dataset`` is benign + refusal rows; ``cb_dataset`` is harmful
    request + harmful response rows whose ``labels`` mark the response tokens
    (-100 on the prompt), so rerouting acts on harmful generation.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from safetune.interventions.steer.circuit_breaker import CircuitBreakerModel

    cfg = config or RRConfig()
    torch.manual_seed(cfg.seed)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    layers = cfg.target_layers or scaled_layers(n_layers)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(LORA_TARGETS),
                      layers_to_transform=list(range(max(layers) + 1)), task_type="CAUSAL_LM")
    pm = get_peft_model(model, lora)
    for p in pm.parameters():  # fp32 trainable weights so the grad scaler can unscale them
        if p.requires_grad:
            p.data = p.data.float()
    if cfg.grad_checkpointing:
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm.enable_input_require_grads()
    pm.train()
    cb = CircuitBreakerModel(pm, target_layers=layers, rr_alpha=cfg.rr_alpha,
                             rr_schedule_steps=cfg.steps)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=cfg.lr)
    use_scaler = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    retain, harmful = _rows(retain_dataset), _rows(cb_dataset)
    r_iter = itertools.cycle(range(0, len(retain), cfg.batch_size))
    c_iter = itertools.cycle(range(0, len(harmful), cfg.batch_size))
    result = RRResult(out_dir=str(out_dir), steps=cfg.steps, target_layers=layers)
    t0 = time.time()
    for step in range(cfg.steps):
        i, j = next(r_iter), next(c_iter)
        rb = _collate(retain[i:i + cfg.batch_size], pad_id, device)
        cbb = _collate(harmful[j:j + cfg.batch_size], pad_id, device)
        with torch.no_grad(), pm.disable_adapter():
            orig_r = _hidden(pm, rb, layers)
            orig_c = _hidden(pm, cbb, layers)
        lora_r = _hidden(pm, rb, layers)
        lora_c = _hidden(pm, cbb, layers)
        cb_mask = (cbb["labels"] != -100) & cbb["attention_mask"].bool()
        loss = cb.compute_rr_loss(orig_r, lora_r, orig_c, lora_c, step=step,
                                  retain_mask=rb["attention_mask"], cb_mask=cb_mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad],
                                       cfg.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"RR loss became non-finite at step {step}")
        result.losses.append(round(value, 5))
        if cfg.log_every and step % cfg.log_every == 0:
            log(f"rr step {step}/{cfg.steps} loss {value:.4f}")

    pm.eval()
    merged = pm.merge_and_unload()
    _save(merged, tokenizer, out_dir, method="magpie.defenses.circuit_breakers_rr",
          params={**asdict(cfg), "target_layers": layers})
    result.seconds = round(time.time() - t0, 1)
    (Path(out_dir) / "rr_training.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


def _save(model: Any, tokenizer: Any, out_dir: str | Path, method: str,
          params: dict[str, Any]) -> None:
    """Save weights + tokenizer + a lexsi.provenance/1 stamp (SafeTune's format)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from safetune.runner.utils.model_utils import save_checkpoint

        save_checkpoint(model, tokenizer, out.name, out_dir=str(out.parent), method=method,
                        inputs=[], params=params)
    except Exception:  # noqa: BLE001 - fall back to a plain save if the helper changes
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        (out / "lexsi_provenance.json").write_text(json.dumps(
            {"schema": "lexsi.provenance/1", "method": method, "params": params}, default=str))


def default_rr_data(tokenizer: Any, n: int = 256) -> tuple[list, list]:
    """SafeTune's English hardening data for RR: (retain, circuit-breaker) rows.

    Retain mixes benign SFT with refusal examples, so refusing harmful requests
    stays a "kept" behaviour; the circuit-breaker set is BeaverTails harmful QA.
    """
    from safetune.runner.utils.data_utils import build_sft_dataset, harden_contamination_sets

    contamination, _tar, refusal = harden_contamination_sets(tokenizer, n=n)
    benign = build_sft_dataset(tokenizer, n_total=n, harmful_frac=0.0)
    retain = list(benign) + list(refusal)
    # Interleave benign and refusal rows deterministically (str hash() is salted per process).
    retain.sort(key=lambda r: hashlib.sha1(str(list(r["input_ids"])[:16]).encode()).hexdigest())
    return retain, list(contamination)
