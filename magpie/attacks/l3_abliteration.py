"""L3a -- abliteration: ablate the refusal direction from the weights (PLAN §6).

The canonical weight-space attack (Arditi et al. 2024), via SafeTune's
``AbliterationAttack``. It is a one-time edit, not a per-prompt search: the
attacker fits a refusal direction from a *separate* contrast set once, then the
model no longer refuses anything. So :meth:`fit` does the work (and carries the
cost -- forward passes over the contrast prompts), and each prompt is then a
single generation on the ablated model.

Because it installs forward hooks, abliteration needs the HF model (not vLLM):
run it with ``backend="hf"``. The direction can be fit on English or on
Hinglish contrast prompts to test transfer (H4 / the en-vs-hing direction
question); ``contrast_form`` selects which.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from magpie.attacks.base import Attack, Variant
from magpie.config import GenerationConfig
from magpie.costs import CostModel
from magpie.serve.base import Generator, Request
from magpie.types import Level, PromptRecord


class L3Abliteration(Attack):
    """Refusal-direction ablation. ``fit`` ablates the weights; one gen per prompt."""

    name = "l3_abliteration"
    level = Level.L3

    def __init__(self, pick_layer: int | None = 24, n_contrast: int = 128,
                 mode: str = "runtime_ablate", strength: float = 1.0,
                 contrast_form: str = "en", avg_contrast_len: float = 32.0) -> None:
        super().__init__(pick_layer=pick_layer, n_contrast=n_contrast, mode=mode,
                         strength=strength, contrast_form=contrast_form)
        self.pick_layer = pick_layer
        self.n_contrast = n_contrast
        self.mode = mode
        self.strength = strength
        self.contrast_form = contrast_form
        self.avg_contrast_len = avg_contrast_len
        self._attack = None

    def _contrast(self, tokenizer) -> tuple[list[str], list[str]]:
        """Separate harmful/harmless contrast prompts (disjoint from the eval set).

        Uses SafeTune's BeaverTails/Alpaca pairs -- the same disjoint set its own
        refusal-direction code uses -- so we never fit the direction on the
        prompts we then score.
        """
        from safetune.runner.utils.data_utils import refusal_prompt_pairs_large

        harmful, harmless = refusal_prompt_pairs_large(self.n_contrast)
        return list(harmful)[: self.n_contrast], list(harmless)[: self.n_contrast]

    def fit(self, generator: Generator, records: Sequence[PromptRecord]):
        backend = generator.backend
        model = getattr(backend, "model", None)
        tokenizer = getattr(backend, "tokenizer", None)
        if model is None or tokenizer is None:
            raise RuntimeError("L3Abliteration needs an HF backend exposing .model/.tokenizer; "
                               "run with backend='hf'.")
        from safetune.instrumentation.evaluate.redteam.abliteration import AbliterationAttack
        from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig

        # Pin extraction to the chosen layer: with pick_layer set, the validation
        # sweep is bypassed, and target_layers=[pick_layer] caches activations for
        # that one layer only (all 36 layers OOMs the direction fit on a T4).
        cfg = RefusalDirectionConfig(
            pick_layer=self.pick_layer,
            target_layers=[self.pick_layer] if self.pick_layer is not None else None)
        harmful, harmless = self._contrast(tokenizer)
        self._attack = AbliterationAttack(model, tokenizer, config=cfg)
        self._attack.fit(harmful_prompts=harmful, harmless_prompts=harmless)
        self._attack.run(mode=self.mode, strength=self.strength)
        cm: CostModel = generator.cost_model
        # Cost is the forward passes over the contrast prompts (both sides).
        return cm.direction_fit(n_contrast_prompts=2 * len(harmful),
                                avg_seq_len=self.avg_contrast_len)

    def revert(self, generator: Generator) -> None:
        if self._attack is not None:
            self._attack.revert()
            self._attack = None

    def variants(self, record: PromptRecord, gen: GenerationConfig) -> Iterator[Variant]:
        # The model is already ablated; one direct generation shows the result.
        yield Variant(Request(prompt=record.text),
                      params={"attack": self.name, "mode": self.mode,
                              "pick_layer": self.pick_layer,
                              "contrast_form": self.contrast_form})

    @property
    def picked_layer(self) -> Any:
        return getattr(self._attack, "picked_layer", None) if self._attack else None
