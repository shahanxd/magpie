"""Per-prompt refusal margins -- the white-box predictor for C3 / H4.

The refusal direction (Arditi et al.) is the single direction whose ablation
removes refusal. A prompt's **refusal margin** is the projection of its
last-token hidden state (at the chosen layer) onto that unit direction: how far
"into refusal space" the model is before it answers. H4 asks whether this cheap,
attack-free measurement predicts how expensive the prompt is to break.

:func:`extract_margins` fits the direction once on a disjoint contrast set (via
SafeTune, same primitive as the abliteration attack), then projects every
evaluation prompt. It needs the HF model (hidden states), so it runs with the
HF backend / in a GPU kernel; the projection itself is cheap. Standardised
margins are what the Cox model consumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from magpie.serve.base import render_chat
from magpie.types import PromptRecord


def fit_refusal_direction(model: Any, tokenizer: Any, harmful: Sequence[str],
                          harmless: Sequence[str], pick_layer: int | None = 24):
    """Fit the refusal direction on a contrast set; return (direction, layer_idx)."""
    from safetune.interventions.steer.refusal_direction import (
        RefusalDirectionConfig,
        extract_refusal_direction,
    )

    cfg = RefusalDirectionConfig(
        pick_layer=pick_layer,
        target_layers=[pick_layer] if pick_layer is not None else None)
    direction, layer_idx, _ = extract_refusal_direction(
        model, tokenizer, harmful_prompts=list(harmful), harmless_prompts=list(harmless),
        config=cfg)
    return direction, layer_idx


def prompt_margins(model: Any, tokenizer: Any, records: Sequence[PromptRecord],
                   direction: Any, layer_idx: int, batch_size: int = 8) -> dict[str, float]:
    """Projection of each prompt's last-token hidden state at ``layer_idx`` onto ``direction``."""
    import torch

    from magpie.serve.base import Request

    unit = direction / (direction.norm() + 1e-8)
    unit = unit.to(model.device).float()
    out: dict[str, float] = {}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    recs = list(records)
    for i in range(0, len(recs), batch_size):
        chunk = recs[i:i + batch_size]
        texts = [render_chat(tokenizer, Request(prompt=r.text)) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer_idx]
        last = hs[:, -1, :].float()  # left-padded, so -1 is the real last token
        proj = (last @ unit).tolist()
        for r, p in zip(chunk, proj, strict=True):
            out[r.prompt_id] = float(p)
    return out


def standardize(margins: dict[str, float]) -> dict[str, float]:
    """Z-score the margins (mean 0, sd 1) for use as a Cox covariate."""
    vals = list(margins.values())
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    sd = var ** 0.5 or 1.0
    return {k: (v - mean) / sd for k, v in margins.items()}


def extract_margins(model: Any, tokenizer: Any, records: Sequence[PromptRecord],
                    harmful_contrast: Sequence[str], harmless_contrast: Sequence[str],
                    pick_layer: int | None = 24) -> dict[str, Any]:
    """Fit the direction and return standardized per-prompt margins + metadata."""
    direction, layer_idx = fit_refusal_direction(model, tokenizer, harmful_contrast,
                                                 harmless_contrast, pick_layer)
    raw = prompt_margins(model, tokenizer, records, direction, layer_idx)
    return {"layer": int(layer_idx), "n_prompts": len(raw),
            "margins": raw, "margins_z": standardize(raw)}
