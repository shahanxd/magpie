"""The one attack interface, with automatic cost logging and survival bookkeeping.

Every attack — from typing a prompt (L0) to editing weights (L3) — subclasses
:class:`Attack` and only has to say *how it varies a prompt*. The shared
:class:`AttackRunner` does the rest: it generates (batched), judges, accumulates
:class:`~magpie.costs.AttackCost`, stops at the first *valid* success, and emits
:class:`~magpie.types.Attempt` / :class:`~magpie.types.Outcome` rows. That is
what makes the survival analysis automatic and identical across attacks.

A prompt is the survival unit: it "dies" at the first attempt a judge marks a
valid success, and is censored if the attacker's budget runs out first. For
batched attacks (best-of-N) the runner generates in chunks and stops issuing
new chunks once a prompt is broken, so logged cost reflects an early-stopping
attacker even though generation is batched for throughput.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from magpie.config import GenerationConfig
from magpie.costs import AttackCost, CostModel
from magpie.serve.base import Generator, Request
from magpie.types import Attempt, Level, Outcome, PromptRecord, Verdict


@dataclass
class Variant:
    """One concrete thing to send the model, plus its bookkeeping.

    ``extra_cost`` covers work the generation call does not capture (e.g. a
    one-time direction fit amortised per prompt, or encoding/search overhead).
    ``n_samples`` > 1 means "draw this many independent samples" (decoding BoN);
    the runner prices all samples and a success on any one breaks the prompt.
    """

    request: Request
    params: dict = field(default_factory=dict)
    n_samples: int = 1
    extra_cost: AttackCost = field(default_factory=AttackCost)
    # Per-variant decoding override (L0 greedy-vs-sampled, L1d decoding BoN).
    # ``None`` uses the runner's default GenerationConfig.
    gen_override: GenerationConfig | None = None


@runtime_checkable
class Judge(Protocol):
    """Scores attempts (mutating ``attempt.judgements``) and rules on success."""

    def score(self, attempts: Sequence[Attempt], records: Sequence[PromptRecord]) -> None:
        ...

    def verdict(self, attempt: Attempt) -> Verdict:
        ...

    def is_success(self, attempt: Attempt, record: PromptRecord) -> bool:
        ...


@dataclass
class AttackRun:
    """Everything one attack produced over a set of prompts."""

    attack: str
    level: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)
    fit_cost: AttackCost = field(default_factory=AttackCost)


class Attack(abc.ABC):
    """Base class. Subclasses implement :meth:`variants`; L3 attacks also :meth:`fit`."""

    name: str = "attack"
    level: Level = Level.L1

    def __init__(self, **params: object) -> None:
        self.params = params

    @abc.abstractmethod
    def variants(self, record: PromptRecord, gen: GenerationConfig) -> Iterator[Variant]:
        """Yield the variants to try for ``record``, cheapest first.

        The runner consumes these lazily and stops pulling once the prompt is
        broken or the budget is spent, so an attack may yield an unbounded /
        large sequence (best-of-N) without wasting generation.
        """

    def fit(self, generator: Generator, records: Sequence[PromptRecord]) -> AttackCost:
        """One-time setup that mutates the model (L3). Default: no-op, no cost."""
        return AttackCost(gpu=getattr(generator.cost_model, "gpu", "T4"))

    def revert(self, generator: Generator) -> None:  # noqa: B027 - optional hook
        """Undo :meth:`fit`. Default: nothing to undo."""
        return None


class AttackRunner:
    """Drives an attack over prompts: generate → judge → account → survive."""

    def __init__(
        self,
        generator: Generator,
        judge: Judge,
        gen: GenerationConfig | None = None,
        *,
        budget_flops: float | None = None,
        max_attempts: int | None = None,
        chunk_size: int = 16,
        early_stop: bool = True,
        run_id: str = "",
        git_sha: str = "",
    ) -> None:
        self.generator = generator
        self.judge = judge
        self.gen = gen or GenerationConfig()
        self.budget_flops = budget_flops
        self.max_attempts = max_attempts
        self.chunk_size = max(1, chunk_size)
        self.early_stop = early_stop
        self.run_id = run_id
        self.git_sha = git_sha

    def run(self, attack: Attack, records: Sequence[PromptRecord]) -> AttackRun:
        fit_cost = attack.fit(self.generator, records)
        amortised = fit_cost.amortise(len(records)) if len(records) else fit_cost
        out = AttackRun(attack=attack.name, level=str(attack.level),
                        model=self.generator.name, fit_cost=fit_cost)
        try:
            for record in records:
                attempts, outcome = self._run_one(attack, record, amortised)
                out.attempts.extend(attempts)
                out.outcomes.append(outcome)
        finally:
            attack.revert(self.generator)
        return out

    def _run_one(
        self, attack: Attack, record: PromptRecord, amortised_fit: AttackCost
    ) -> tuple[list[Attempt], Outcome]:
        cum = amortised_fit
        attempts: list[Attempt] = []
        first_success: Attempt | None = None
        idx = 0
        exhausted = False  # budget or attempt cap reached
        for chunk in _chunked(attack.variants(record, self.gen), self.chunk_size):
            # Expand n_samples into one request each so a batch can mix variants.
            flat: list[tuple[Variant, Request]] = []
            for v in chunk:
                flat.extend((v, v.request) for _ in range(max(1, v.n_samples)))
            priced = self._generate_grouped_by_gen(flat)
            batch: list[Attempt] = []
            for (variant, _req), (comp, cost) in zip(flat, priced, strict=True):
                step_cost = cost + variant.extra_cost
                cum = cum + step_cost
                batch.append(Attempt(
                    prompt_id=record.prompt_id, model=self.generator.name,
                    attack=attack.name, level=str(attack.level), form=record.form,
                    prompt=variant.request.prompt, response=comp.full_text,
                    params=variant.params, cost=step_cost, cum_cost=cum,
                    attempt_index=idx, seed=self.gen.seed,
                    is_harmful_prompt=record.is_harmful,
                    run_id=self.run_id, git_sha=self.git_sha,
                ))
                idx += 1
            # Judge the whole chunk together (judges may batch GPU work).
            self.judge.score(batch, [record] * len(batch))
            for a in batch:
                attempts.append(a)
                if (first_success is None and record.is_harmful
                        and self.judge.is_success(a, record)):
                    first_success = a
                    if self.early_stop:
                        break  # an early-stopping attacker never pays for the rest
                if self.max_attempts and a.attempt_index + 1 >= self.max_attempts:
                    exhausted = True
                    break
                if self.budget_flops and a.cum_cost.total_flops >= self.budget_flops:
                    exhausted = True
                    break
            if exhausted or (first_success is not None and self.early_stop):
                break
        # The survival time is the cost at the *first* valid success; censored
        # prompts report everything the attacker spent.
        anchor = first_success or (attempts[-1] if attempts else None)
        outcome = Outcome(
            prompt_id=record.prompt_id, model=self.generator.name, attack=attack.name,
            form=record.form, level=str(attack.level), broken=first_success is not None,
            cost_to_break=anchor.cum_cost.total_flops if anchor else 0.0,
            queries_to_break=anchor.cum_cost.queries if anchor else 0,
            n_attempts=len(attempts),
            covariates={"category": record.category, "parallel_id": record.parallel_id,
                        **record.covariates},
        )
        return attempts, outcome

    def _generate_grouped_by_gen(self, flat: list[tuple[Variant, Request]]):
        """Generate, batching requests that share a decoding config.

        Most attacks use one GenerationConfig, so this is a single batched call;
        L0 (greedy vs sampled) and L1d (decoding sweep) mix configs, so we group
        by config to keep batching within each setting and preserve input order.
        """
        results: list = [None] * len(flat)
        groups: dict[tuple, list[int]] = {}
        for i, (variant, _req) in enumerate(flat):
            gen = variant.gen_override or self.gen
            key = (gen.max_new_tokens, gen.temperature, gen.top_p, gen.seed, gen.max_model_len)
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            gen = flat[idxs[0]][0].gen_override or self.gen
            priced = self.generator.generate_priced([flat[i][1] for i in idxs], gen)
            for i, pr in zip(idxs, priced, strict=True):
                results[i] = pr
        return [r for r in results if r is not None]


def _chunked(it: Iterable[Variant], n: int) -> Iterator[list[Variant]]:
    batch: list[Variant] = []
    for item in it:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def cost_model_for(generator: Generator) -> CostModel:
    return generator.cost_model
