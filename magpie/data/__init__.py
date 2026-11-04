"""Data (PLAN §5): seed loaders, the parallel core, Hinglish tools, CuratorKIT gates.

Harmful seeds come from their canonical public sources (HarmBench, StrongREJECT);
over-refusal from XSTest safe and OR-Bench-Hard. The Hinglish tools
(:mod:`magpie.data.hinglish`) measure code-mixing and generate the natural
spelling variation used by C2. Harmful prompt text lives only in gitignored
outputs and the gated HF dataset, never in the repo.
"""

from magpie.data.hinglish import HinglishAugmenter, cmi, natural_bon_variants
from magpie.data.loaders import (
    DatasetManifest,
    assemble_core,
    benign_twin,
    manifest_for,
    read_records,
    write_records,
)
from magpie.data.taxonomy import CATEGORIES, normalize_category

__all__ = [
    "cmi", "HinglishAugmenter", "natural_bon_variants",
    "CATEGORIES", "normalize_category",
    "DatasetManifest", "manifest_for", "assemble_core", "benign_twin",
    "read_records", "write_records",
]
