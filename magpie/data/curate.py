"""Curate the dataset with CuratorKIT: dedup + provenance (chains CuratorKIT).

The hackathon rules require chaining the Track 2 repos; CuratorKIT is Magpie's
data-curation link. We convert :class:`~magpie.types.PromptRecord` to
CuratorKIT ``DataSample`` objects, run MinHash de-duplication (so near-identical
seeds or translations do not inflate counts), and attach the shared
``lexsi.provenance/1`` object so every sample is auditable back to its source.

Only CuratorKIT core is required (no heavy extras). The diversity and
hallucination gates (which need embeddings / an LLM judge) are left for the
full data build; dedup + provenance is what data-v1 needs and is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from magpie.types import PromptRecord


def to_datasample(record: PromptRecord):
    from curatorkit.schema import DataSample

    return DataSample(
        source_uri=record.source or "magpie",
        instruction=record.text,
        output="",
        task_type="prompt_only",
        metadata={
            "prompt_id": record.prompt_id, "form": record.form, "category": record.category,
            "parallel_id": record.parallel_id, "is_harmful": record.is_harmful,
            **record.covariates,
        },
    )


def from_datasample(sample) -> PromptRecord:
    m = sample.metadata or {}
    known = {"prompt_id", "form", "category", "parallel_id", "is_harmful"}
    return PromptRecord(
        prompt_id=m.get("prompt_id", sample.id), text=sample.instruction,
        form=m.get("form", "en"), is_harmful=bool(m.get("is_harmful", True)),
        category=m.get("category", ""), parallel_id=m.get("parallel_id", ""),
        source=sample.source_uri,
        covariates={k: v for k, v in m.items() if k not in known},
    )


@dataclass
class CurationResult:
    kept: list[PromptRecord]
    n_in: int
    n_out: int
    n_removed: int
    provenance: dict
    minhash_threshold: float


def curate(records: Sequence[PromptRecord], minhash_threshold: float = 0.9,
           base_model: str | None = None) -> CurationResult:
    """De-duplicate (MinHash) within each form and attach provenance.

    Dedup runs per-form so an en seed and its hing translation (different text)
    are never collapsed, while a duplicated seed within a form is removed.
    """
    from curatorkit.manifest import build_provenance
    from curatorkit.normalizers.dedup import MinHashDeduplicator

    by_form: dict[str, list] = {}
    for r in records:
        by_form.setdefault(r.form, []).append(to_datasample(r))

    kept: list[PromptRecord] = []
    for _form, samples in by_form.items():
        dedup = MinHashDeduplicator(threshold=minhash_threshold)
        kept.extend(from_datasample(s) for s in dedup.run(samples))

    prov = build_provenance(
        method="magpie.curate",
        params={"minhash_threshold": minhash_threshold, "per_form": True},
        base_model=base_model,
    )
    return CurationResult(
        kept=kept, n_in=len(records), n_out=len(kept), n_removed=len(records) - len(kept),
        provenance=prov, minhash_threshold=minhash_threshold,
    )
