"""Kernel job: rescore a pulled run's attempts with the HarmBench model judge.

Reads the raw attempts of a previous run (the ones carrying prompt + response
text), scores them with SafeTune's HarmBench classifier, applies the composite
validity gate (HarmBench harmful AND the lexicon guard did not flag collapse),
rebuilds the per-prompt outcomes and the result table, and writes a
``table_harmbench.*`` beside a short comparison to the lexicon numbers.

Input attempts are uploaded as a Kaggle dataset (the raw JSONL from a prior
run); ``config["attempts_dir"]`` points at the mounted copy. This runs under
the shared vLLM launcher (HarmBench loads via vLLM, TP=2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magpie.analysis.aggregate import build_rows, rows_to_markdown, write_outcomes
from magpie.config import provenance
from magpie.judges import LexiconJudge
from magpie.runners.jobs.common import write_json
from magpie.types import Attempt, Outcome, PromptRecord


def _load_attempts(attempts_dir: Path) -> list[Attempt]:
    rows: list[Attempt] = []
    for p in sorted(Path(attempts_dir).glob("attempts_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(Attempt.from_dict(json.loads(line)))
    return rows


def _outcome_for(group: list[Attempt], judge, record: PromptRecord) -> Outcome:
    first = next((a for a in group if judge.is_success(a, record)), None)
    anchor = first or group[-1]
    return Outcome(
        prompt_id=group[0].prompt_id, model=group[0].model, attack=group[0].attack,
        form=group[0].form, level=group[0].level, broken=first is not None,
        cost_to_break=anchor.cum_cost.total_flops, queries_to_break=anchor.cum_cost.queries,
        n_attempts=len(group), covariates=group[0].params,
    )


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    judge_name = config.get("judge", "harmbench")
    attempts_dir = Path(config.get("attempts_dir") or _find_attempts_mount())
    attempts = _load_attempts(attempts_dir)
    harmful = [a for a in attempts if a.is_harmful_prompt]
    benign = [a for a in attempts if not a.is_harmful_prompt]
    print(f"loaded {len(attempts)} attempts ({len(harmful)} harmful, {len(benign)} benign) "
          f"from {attempts_dir}", flush=True)

    # Lexicon first (collapse flags feed the validity gate), then the model judge.
    lex = LexiconJudge()
    lex.score(attempts, [PromptRecord(prompt_id=a.prompt_id, text="", form=a.form,
                                      is_harmful=a.is_harmful_prompt) for a in attempts])
    from magpie.judges.safetune_judge import SafeTuneJudge

    model_judge = SafeTuneJudge(judge_name, tensor_parallel_size=config.get("tp", 2))
    model_judge.score(attempts, [None] * len(attempts))

    # Per-(model, form, prompt) grouping, ordered by attempt index for survival.
    groups: dict[tuple, list[Attempt]] = {}
    for a in harmful:
        groups.setdefault((a.model, a.form, a.attack, a.prompt_id), []).append(a)
    outcomes = []
    for key, g in groups.items():
        g.sort(key=lambda a: a.attempt_index)
        rec = PromptRecord(prompt_id=key[3], text="", form=key[1], is_harmful=True)
        outcomes.append(_outcome_for(g, model_judge, rec))

    rows = build_rows(outcomes, benign_attempts=benign, n_boot=2000, seed=config.get("seed", 0))
    prov = provenance(run_id="rescore", judge=judge_name, source=str(attempts_dir))
    (out / f"table_{judge_name}.md").write_text(
        rows_to_markdown(rows, f"L0 ({judge_name} judge)"), encoding="utf-8")
    write_json(out / f"table_{judge_name}.json", {
        "name": f"L0 ({judge_name} judge)", "provenance": {**prov, "judge": judge_name},
        "rows": [r.to_dict() for r in rows]})
    write_outcomes(outcomes, out / f"outcomes_{judge_name}.jsonl")
    write_json(out / "rescore_summary.json", {
        "judge": judge_name, "n_attempts": len(attempts), "n_harmful_prompts": len(outcomes),
        "n_broken": sum(o.broken for o in outcomes), "provenance": prov})
    print(rows_to_markdown(rows, f"L0 {judge_name}"), flush=True)


def _find_attempts_mount() -> str:
    """Find a mounted dataset dir that holds attempts_*.jsonl."""
    root = Path("/kaggle/input")
    if root.is_dir():
        for p in sorted(root.rglob("attempts_*.jsonl")):
            return str(p.parent)
    raise FileNotFoundError("no attempts_*.jsonl under /kaggle/input; set config['attempts_dir']")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = Path(args.config).read_text() if Path(args.config).is_file() else args.config
    main(json.loads(cfg), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
