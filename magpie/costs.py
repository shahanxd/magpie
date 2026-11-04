"""The attacker-cost model — the x-axis of every Magpie survival curve.

Magpie's thesis is that refusal rate is the wrong safety number for an
open-weight on-device model: what matters is *how much effort* it takes to
break it. This module turns every attack into one comparable number so that
input-space attacks (prompting, best-of-N) and weight-space attacks
(abliteration, fine-tuning) sit on a single cost axis.

Three cost currencies, logged for every attempt:

* **FLOPs** — the primary axis. Using the standard transformer approximation
  (Kaplan et al. 2020; Hoffmann et al. 2022):

      forward pass      ≈ 2 · P · tokens
      training step     ≈ 6 · P · tokens      (forward + backward)

  with ``P`` the parameter count. ``P = 3.35e9`` for Tiny Aya Fire.
* **queries** — black-box API calls, the currency an L1 attacker actually pays.
* **wall-clock seconds** — on named reference hardware (a Kaggle T4, the team's
  RTX 3050 laptop), derived from measured throughput so the "minutes on a
  gaming laptop" story is concrete.

Nothing here imports torch; it is pure arithmetic and is exercised by the CPU
test-suite. The constants are derived from the published Tiny Aya Fire config
(``cohere2``: 36 layers, hidden 2048, d_mlp 11008, 16 heads / 4 KV heads,
vocab 262144, embeddings tied) and cross-checked against the 3.35B figure in
the Tiny Aya tech report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ModelShape:
    """Shape of a decoder-only transformer, enough to count parameters/FLOPs."""

    name: str
    n_layers: int
    d_model: int
    d_mlp: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool = True
    gated_mlp: bool = True

    @property
    def attn_params_per_layer(self) -> int:
        q = self.d_model * (self.n_heads * self.head_dim)
        k = self.d_model * (self.n_kv_heads * self.head_dim)
        v = self.d_model * (self.n_kv_heads * self.head_dim)
        o = (self.n_heads * self.head_dim) * self.d_model
        return q + k + v + o

    @property
    def mlp_params_per_layer(self) -> int:
        # Gated MLP (SwiGLU-style): gate + up + down. Non-gated: up + down.
        if self.gated_mlp:
            return 2 * (self.d_model * self.d_mlp) + (self.d_mlp * self.d_model)
        return 2 * (self.d_model * self.d_mlp)

    @property
    def params_non_embedding(self) -> int:
        return self.n_layers * (self.attn_params_per_layer + self.mlp_params_per_layer)

    @property
    def params_embedding(self) -> int:
        # Tied embeddings are counted once (input == output projection).
        factor = 1 if self.tied_embeddings else 2
        return factor * self.vocab_size * self.d_model

    @property
    def params_total(self) -> int:
        return self.params_non_embedding + self.params_embedding


# The main target. P = params_total ≈ 3.35B, matching the Tiny Aya tech report.
TINY_AYA_FIRE = ModelShape(
    name="CohereLabs/tiny-aya-fire",
    n_layers=36,
    d_model=2048,
    d_mlp=11008,
    n_heads=16,
    n_kv_heads=4,
    head_dim=128,
    vocab_size=262144,
    tied_embeddings=True,
    gated_mlp=True,
)


@dataclass(frozen=True)
class AttackCost:
    """The price of one attack *attempt* against one prompt.

    Costs are additive across the stages of a composite attack, so an attack
    that fits a direction once and then generates can report the amortised cost
    per prompt via :meth:`amortise`. ``flops`` is the single comparable axis;
    the rest are kept for the ROI story and sanity checks.
    """

    flops: float = 0.0
    queries: int = 0
    wall_clock_s: float = 0.0
    # FLOPs spent on *training* (weight updates), separated so L3 attacks can be
    # reported both ways (training is a one-time cost amortised over all prompts).
    train_flops: float = 0.0
    gpu: str = "T4"
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def total_flops(self) -> float:
        return self.flops + self.train_flops

    @property
    def log10_flops(self) -> float:
        import math

        return math.log10(self.total_flops) if self.total_flops > 0 else 0.0

    def __add__(self, other: AttackCost) -> AttackCost:
        if not isinstance(other, AttackCost):
            return NotImplemented
        merged = {**self.notes, **other.notes}
        return AttackCost(
            flops=self.flops + other.flops,
            queries=self.queries + other.queries,
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
            train_flops=self.train_flops + other.train_flops,
            gpu=self.gpu or other.gpu,
            notes=merged,
        )

    def amortise(self, n_prompts: int) -> AttackCost:
        """Spread a one-time training cost over ``n_prompts`` prompts."""
        if n_prompts <= 0:
            return self
        return replace(self, train_flops=self.train_flops / n_prompts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flops": self.flops,
            "train_flops": self.train_flops,
            "total_flops": self.total_flops,
            "queries": self.queries,
            "wall_clock_s": self.wall_clock_s,
            "gpu": self.gpu,
            "notes": self.notes,
        }


# Decode throughput (tokens/s) per reference device, for a-priori wall-clock
# estimates only; a real run records measured latency and overrides these.
# T4 is the measured HF-transformers single-stream rate from the day-1 smoke
# test (git ae39951): Tiny Aya Fire fp16 on one T4, ~19.5 tok/s single stream,
# ~135 tok/s at batch 10. vLLM (batched) is higher and is updated once measured.
REFERENCE_THROUGHPUT: dict[str, float] = {
    "T4": 19.5,         # HF transformers, fp16, single stream (measured, smoke v1)
    "T4_batch": 135.0,  # HF transformers, fp16, batch 10 (measured, smoke v1)
    "T4_vllm": 208.5,   # vLLM 0.33, fp16, batch (measured, smoke v2)
    "T4x2": 208.5,      # vLLM on the 2xT4 box (single-GPU figure; TP updated later)
    "RTX3050": 18.0,    # Q4_K_M GGUF on the 4 GB laptop (placeholder, pending)
}


class CostModel:
    """Turn (tokens, samples, training work) into an :class:`AttackCost`.

    Example::

        cm = CostModel(TINY_AYA_FIRE)
        cost = cm.generation(prompt_tokens=400, output_tokens=128, n_samples=1)
        cost.flops            # 2 · P · (400 + 128)
    """

    def __init__(self, shape: ModelShape = TINY_AYA_FIRE, gpu: str = "T4") -> None:
        self.shape = shape
        self.gpu = gpu
        self.P = float(shape.params_total)

    # -- forward / generation -------------------------------------------------

    def forward_flops(self, tokens: float) -> float:
        """2 · P · tokens — one forward pass over ``tokens`` tokens."""
        return 2.0 * self.P * float(tokens)

    def generation(
        self,
        prompt_tokens: float,
        output_tokens: float,
        n_samples: int = 1,
        queries: int | None = None,
    ) -> AttackCost:
        """Cost of generating ``output_tokens`` from ``prompt_tokens``, ``n_samples`` times.

        Prefill processes the prompt once; decode processes each generated
        token once (2·P per token, the context-length term is second order for
        Tiny Aya's sequence lengths and is folded into the constant).
        """
        per_sample = self.forward_flops(prompt_tokens + output_tokens)
        total = per_sample * n_samples
        wall = self._wall_clock(output_tokens * n_samples)
        return AttackCost(
            flops=total,
            queries=queries if queries is not None else n_samples,
            wall_clock_s=wall,
            gpu=self.gpu,
            notes={"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                   "n_samples": n_samples},
        )

    # -- training (L3) --------------------------------------------------------

    def train_step_flops(self, tokens: float) -> float:
        """6 · P · tokens — one forward+backward step over ``tokens`` tokens."""
        return 6.0 * self.P * float(tokens)

    def finetune(
        self,
        n_examples: int,
        avg_seq_len: float,
        epochs: float = 1.0,
        trainable_fraction: float = 1.0,
    ) -> AttackCost:
        """Cost of a (LoRA or full) fine-tune.

        ``trainable_fraction`` scales only the backward term's dominant cost for
        parameter-efficient tuning; the forward term is unchanged. We keep the
        simple 6·P·tokens upper bound by default (``trainable_fraction=1``) and
        expose the knob so a LoRA run can be reported more faithfully.
        """
        tokens = float(n_examples) * float(avg_seq_len) * float(epochs)
        # forward (2·P) always; backward (4·P) scaled by trainable fraction.
        flops = (2.0 + 4.0 * trainable_fraction) * self.P * tokens
        return AttackCost(
            train_flops=flops,
            gpu=self.gpu,
            notes={"n_examples": n_examples, "avg_seq_len": avg_seq_len,
                   "epochs": epochs, "trainable_fraction": trainable_fraction},
        )

    def direction_fit(self, n_contrast_prompts: int, avg_seq_len: float) -> AttackCost:
        """Cost of fitting a refusal direction (forward passes over contrast prompts).

        Abliteration needs no backward pass: it reads activations from forward
        passes over harmful/harmless contrast prompts, so the fit is pure
        inference cost. The weight edit itself is negligible (a few matmuls).
        """
        tokens = float(n_contrast_prompts) * float(avg_seq_len)
        return AttackCost(
            flops=self.forward_flops(tokens),
            queries=n_contrast_prompts,
            gpu=self.gpu,
            notes={"n_contrast_prompts": n_contrast_prompts, "avg_seq_len": avg_seq_len},
        )

    # -- helpers --------------------------------------------------------------

    def _wall_clock(self, output_tokens: float) -> float:
        tps = REFERENCE_THROUGHPUT.get(self.gpu, 40.0)
        return float(output_tokens) / tps if tps > 0 else 0.0

    def empty(self) -> AttackCost:
        return AttackCost(gpu=self.gpu)
