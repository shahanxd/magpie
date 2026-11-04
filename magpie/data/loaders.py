"""PromptRecord IO and dataset assembly.

A Magpie dataset split is a list of :class:`~magpie.types.PromptRecord` in
JSONL, plus a small manifest (counts, provenance, category histogram). Harmful
splits are written under a directory the repo gitignores; only manifests and
rate tables are committed. Building the parallel core (en + hing + hi-Deva) is
done by :func:`assemble_core`, which pairs every form under one ``parallel_id``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from magpie.data.seeds import category_counts
from magpie.types import PromptRecord


def write_records(records: Iterable[PromptRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: Path) -> list[PromptRecord]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(PromptRecord.from_dict(json.loads(line)))
    return out


@dataclass
class DatasetManifest:
    """Counts + provenance for a built split (committable; no prompt text)."""

    name: str
    n_records: int
    by_form: dict[str, int]
    by_category: dict[str, int]
    by_source: dict[str, int]
    n_harmful: int
    n_benign: int
    provenance: dict = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")


def manifest_for(name: str, records: Sequence[PromptRecord], provenance: dict | None = None,
                 ) -> DatasetManifest:
    by_form: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in records:
        by_form[r.form] = by_form.get(r.form, 0) + 1
        by_source[r.source] = by_source.get(r.source, 0) + 1
    return DatasetManifest(
        name=name, n_records=len(records), by_form=dict(sorted(by_form.items())),
        by_category=category_counts(list(records)), by_source=dict(sorted(by_source.items())),
        n_harmful=sum(1 for r in records if r.is_harmful),
        n_benign=sum(1 for r in records if not r.is_harmful),
        provenance=provenance or {},
    )


def assemble_core(
    en_records: Sequence[PromptRecord],
    translations: dict[str, dict[str, str]] | None = None,
) -> list[PromptRecord]:
    """Build the parallel core: for each en record, emit its hing / hi-Deva twins.

    ``translations[parallel_id][form] -> text``. Missing translations are simply
    skipped (the en record is always kept), so a partial translation run still
    yields a valid, if smaller, parallel set. Every emitted record shares the
    en record's ``parallel_id``, ``category`` and ``is_harmful``.
    """
    out: list[PromptRecord] = []
    translations = translations or {}
    for r in en_records:
        out.append(r)
        for form, text in translations.get(r.parallel_id, {}).items():
            if not text or form == r.form:
                continue
            out.append(PromptRecord(
                prompt_id=f"{r.prompt_id}#{form}", text=text, form=form,
                is_harmful=r.is_harmful, category=r.category, parallel_id=r.parallel_id,
                source=f"{r.source}+translated", covariates={**r.covariates, "translated_from": r.form},
            ))
    return out


def benign_twin(record: PromptRecord, twin_text: str) -> PromptRecord:
    """A benign twin of a harmful seed: same topic framing, harmless intent (H2b/§7)."""
    return PromptRecord(
        prompt_id=f"{record.prompt_id}#benign", text=twin_text, form=record.form,
        is_harmful=False, category="benign_twin", parallel_id=record.parallel_id,
        source=f"{record.source}+benign_twin",
        covariates={"twin_of": record.prompt_id, "twin_category": record.category},
    )
