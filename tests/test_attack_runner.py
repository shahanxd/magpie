"""The runner's survival bookkeeping, on the dependency-free echo backend."""

from collections.abc import Iterator, Sequence

from magpie.attacks.base import Attack, AttackRunner, Variant
from magpie.config import GenerationConfig
from magpie.serve import EchoBackend, Generator, Request
from magpie.types import Attempt, Level, PromptRecord, Verdict


class PrefixJudge:
    """Toy judge: anything not starting with an apology is compliance."""

    def score(self, attempts: Sequence[Attempt], records: Sequence[PromptRecord]) -> None:
        for a in attempts:
            refused = a.response.strip().lower().startswith("i'm sorry")
            a.judgements["toy"] = {"verdict": "refusal" if refused else "compliance"}

    def verdict(self, attempt: Attempt) -> Verdict:
        return Verdict(attempt.judgements["toy"]["verdict"])

    def is_success(self, attempt: Attempt, record: PromptRecord) -> bool:
        return self.verdict(attempt) is Verdict.COMPLIANCE


class Rephrase(Attack):
    """Toy attack: the k-th variant drops the trigger once k reaches ``breaks_at``."""

    name = "toy_rephrase"
    level = Level.L1

    def __init__(self, n: int = 10, breaks_at: int | None = 3) -> None:
        super().__init__(n=n, breaks_at=breaks_at)
        self.n, self.breaks_at = n, breaks_at

    def variants(self, record: PromptRecord, gen: GenerationConfig) -> Iterator[Variant]:
        for k in range(self.n):
            text = record.text
            if self.breaks_at is not None and k >= self.breaks_at:
                text = text.replace("TRIGGER", "")
            yield Variant(Request(prompt=text), params={"k": k})


def _runner(**kw):
    return AttackRunner(Generator(EchoBackend()), PrefixJudge(),
                        GenerationConfig(max_new_tokens=32), **kw)


def test_prompt_dies_at_first_success_and_cost_is_cumulative():
    rec = PromptRecord(prompt_id="p", text="please TRIGGER now")
    run = _runner(chunk_size=2).run(Rephrase(n=10, breaks_at=3), [rec])
    (out,) = run.outcomes
    assert out.broken and out.n_attempts == 4  # k = 0,1,2 refused; k = 3 breaks
    assert out.queries_to_break == 4
    costs = [a.cum_cost.total_flops for a in run.attempts]
    assert costs == sorted(costs) and out.cost_to_break == costs[-1]


def test_unbroken_prompt_is_censored_at_budget():
    rec = PromptRecord(prompt_id="p", text="TRIGGER")
    run = _runner().run(Rephrase(n=5, breaks_at=None), [rec])
    (out,) = run.outcomes
    assert not out.broken and out.n_attempts == 5
    assert out.cost_to_break == run.attempts[-1].cum_cost.total_flops


def test_max_attempts_caps_work():
    rec = PromptRecord(prompt_id="p", text="TRIGGER")
    run = _runner(max_attempts=2, chunk_size=1).run(Rephrase(n=50, breaks_at=None), [rec])
    assert run.outcomes[0].n_attempts == 2


def test_benign_prompts_are_never_counted_as_broken():
    rec = PromptRecord(prompt_id="b", text="hello", is_harmful=False)
    run = _runner().run(Rephrase(n=3), [rec])
    assert not run.outcomes[0].broken and run.outcomes[0].n_attempts == 3


def test_full_scan_keeps_every_attempt_but_dates_death_at_first_success():
    # early_stop=False is the full-ASR-scaling mode: run all N, but the prompt's
    # survival time is still the cost at the *first* success.
    rec = PromptRecord(prompt_id="p", text="TRIGGER")
    run = _runner(early_stop=False, chunk_size=4).run(Rephrase(n=8, breaks_at=2), [rec])
    (out,) = run.outcomes
    assert len(run.attempts) == 8 and out.n_attempts == 8
    assert out.broken and out.queries_to_break == 3
    assert out.cost_to_break == run.attempts[2].cum_cost.total_flops
