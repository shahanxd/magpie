"""Turn attack runs into result tables and persist them.

Bridges the attack layer and the survival math: given the outcomes of one or
more :class:`~magpie.attacks.base.AttackRun`, produce per-(model, form, attack)
rows carrying ASR, median cost-to-break with a bootstrap CI, and a paired
over-refusal rate from the benign twins — the "every number next to
over-refusal" rule (PLAN §7). Writes tidy JSON/JSONL that the figures and the
article read, and a Markdown table for quick inspection.

Raw attempt text (which may contain harmful completions) is written only under
``raw/`` and is gitignored; the aggregates committed to the repo carry rates
and costs, never completions (PLAN §16).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from magpie.analysis.survival import asr_wilson, bootstrap_median_ci
from magpie.types import Attempt, Outcome


@dataclass
class ResultRow:
    """One (model, form, attack) cell of a result table."""

    model: str
    form: str
    attack: str
    level: str
    n_harmful: int
    n_broken: int
    asr: float
    asr_lo: float
    asr_hi: float
    median_log10_flops: float | None
    median_ci_lo: float | None
    median_ci_hi: float | None
    # Over-refusal measured on the benign prompts under the same attack/form.
    n_benign: int = 0
    over_refusal: float | None = None
    collapse_rate: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _refused_fraction(attempts: Iterable[Attempt], judge_name: str = "refusal_lexicon") -> float:
    rows = list(attempts)
    if not rows:
        return 0.0
    refused = sum(1 for a in rows if a.judgements.get(judge_name, {}).get("is_refusal"))
    return refused / len(rows)


def _collapse_fraction(attempts: Iterable[Attempt], judge_name: str = "refusal_lexicon") -> float:
    rows = list(attempts)
    if not rows:
        return 0.0
    collapsed = sum(1 for a in rows if a.judgements.get(judge_name, {}).get("collapsed"))
    return collapsed / len(rows)


def build_rows(
    outcomes: Sequence[Outcome],
    benign_attempts: Sequence[Attempt] = (),
    n_boot: int = 1000,
    seed: int = 0,
) -> list[ResultRow]:
    """Aggregate outcomes into result rows, keyed by (model, form, attack).

    ``benign_attempts`` are the over-refusal attempts (is_harmful=False); their
    refusal fraction per (model, form, attack) becomes the row's over-refusal.
    """
    groups: dict[tuple[str, str, str], list[Outcome]] = {}
    for o in outcomes:
        groups.setdefault((o.model, o.form, o.attack), []).append(o)

    benign: dict[tuple[str, str, str], list[Attempt]] = {}
    for a in benign_attempts:
        benign.setdefault((a.model, a.form, a.attack), []).append(a)

    rows: list[ResultRow] = []
    for (model, form, attack), outs in sorted(groups.items()):
        asr = asr_wilson(outs)
        point, lo, hi = bootstrap_median_ci(outs, n_boot=n_boot, seed=seed)
        b = benign.get((model, form, attack), [])
        level = outs[0].level if outs else ""
        rows.append(ResultRow(
            model=model, form=form, attack=attack, level=level,
            n_harmful=asr.n, n_broken=asr.n_broken, asr=asr.asr, asr_lo=asr.lo, asr_hi=asr.hi,
            median_log10_flops=round(point, 3) if point is not None else None,
            median_ci_lo=round(lo, 3) if lo is not None else None,
            median_ci_hi=round(hi, 3) if hi is not None else None,
            n_benign=len(b),
            over_refusal=round(_refused_fraction(b), 4) if b else None,
            collapse_rate=round(_collapse_fraction(b), 4) if b else None,
        ))
    return rows


def rows_to_markdown(rows: Sequence[ResultRow], title: str = "L0 results") -> str:
    """A compact Markdown table: ASR (with CI), median cost, over-refusal per cell."""
    lines = [f"### {title}", "",
             "| model | form | attack | n | ASR (95% CI) | median log10 FLOPs | over-refusal |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        asr = f"{r.asr:.2f} [{r.asr_lo:.2f}, {r.asr_hi:.2f}]"
        med = "—" if r.median_log10_flops is None else (
            f"{r.median_log10_flops}"
            + ("" if r.median_ci_lo is None else f" [{r.median_ci_lo}, {r.median_ci_hi}]"))
        ov = "—" if r.over_refusal is None else f"{r.over_refusal:.2f}"
        lines.append(f"| {r.model} | {r.form} | {r.attack} | {r.n_harmful} | {asr} | {med} | {ov} |")
    return "\n".join(lines) + "\n"


@dataclass
class ResultBundle:
    """A table plus its provenance, ready to write to ``results/<name>/``."""

    name: str
    rows: list[ResultRow]
    provenance: dict = field(default_factory=dict)

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "table.json").write_text(
            json.dumps({"name": self.name, "provenance": self.provenance,
                        "rows": [r.to_dict() for r in self.rows]}, indent=2), encoding="utf-8")
        (out_dir / "table.md").write_text(
            rows_to_markdown(self.rows, title=self.name), encoding="utf-8")


def write_attempts(attempts: Sequence[Attempt], path: Path, include_response: bool = True) -> None:
    """Write attempts as JSONL. ``include_response=False`` drops completion text
    (for a committable, no-harmful-content copy); the full copy goes under raw/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in attempts:
            d = a.to_dict()
            if not include_response:
                d.pop("response", None)
                d.pop("prompt", None)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def write_outcomes(outcomes: Sequence[Outcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")
