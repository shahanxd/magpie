"""Native-speaker review sheets for the translated core (PLAN §5.3).

The plan's real translation-quality gate is native verification (≥15% of each
set, 100% of the gold set). This builds a stratified review sheet pairing each
Hinglish / Devanagari translation with its English source, so a native speaker
marks each row ok / fix and (for Hinglish) confirms it reads as natural
code-mixing. Rows are prioritised by lowest back-translation agreement first
(the likeliest to be wrong), then stratified across form × harm category.

The sheet contains prompt text, so it is written to the gated dataset, never to
git. Pure-Python; CSV output.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from magpie.types import PromptRecord


def _en_by_parallel(records: Sequence[PromptRecord]) -> dict[str, str]:
    return {r.parallel_id: r.text for r in records if r.form == "en"}


def select_for_review(records: Sequence[PromptRecord], frac: float = 0.15, seed: int = 0,
                      forms: Sequence[str] = ("hing", "hi-Deva")) -> list[PromptRecord]:
    """Pick ~``frac`` of each form's translations, stratified by harm category.

    Deterministic: within a (form, category) stratum, rows are ordered by
    ascending back-translation agreement (lowest first, most likely wrong) then
    by a seed-salted hash, and the first ceil(frac·|stratum|) are taken.
    """
    import math

    strata: dict[tuple[str, str], list[PromptRecord]] = defaultdict(list)
    for r in records:
        if r.form in forms:
            strata[(r.form, r.category)].append(r)
    picked: list[PromptRecord] = []
    for rows in strata.values():
        rows.sort(key=lambda r: (r.covariates.get("agreement", 1.0),
                                 hashlib.sha1(f"{seed}|{r.prompt_id}".encode()).hexdigest()))
        k = max(1, math.ceil(frac * len(rows)))
        picked.extend(rows[:k])
    return picked


def write_review_csv(records: Sequence[PromptRecord], path: Path | str, frac: float = 0.15,
                     seed: int = 0) -> int:
    """Write a native-review CSV (translation vs English source); return row count."""
    path = Path(path)
    en = _en_by_parallel(records)
    chosen = select_for_review(records, frac=frac, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["prompt_id", "form", "category", "cmi", "agreement",
            "english_source", "translation", "ok (y/n)", "fixed_translation", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in chosen:
            w.writerow([r.prompt_id, r.form, r.category,
                        r.covariates.get("cmi", ""), r.covariates.get("agreement", ""),
                        en.get(r.parallel_id, ""), r.text, "", "", ""])
    return len(chosen)
