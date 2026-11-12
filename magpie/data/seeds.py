"""Load harmful seeds and over-refusal prompts from their canonical public sources.

Sources (PLAN §5.2), all public and human-written:

* **HarmBench** standard behaviours -- GitHub CSV (MIT).
* **StrongREJECT** -- GitHub CSV (MIT), carries the scoring rubric categories.
* **XSTest** safe prompts -- ``Paul/XSTest`` on HF (CC-BY-4.0), over-refusal.
* **OR-Bench-Hard-1k** -- ``bench-llm/or-bench`` on HF, over-refusal.

The loaders return :class:`~magpie.types.PromptRecord` objects and never print
prompt text; harmful text stays in records and in gitignored outputs. Each
record gets a stable ``prompt_id`` (``<source>:<id>``) and ``parallel_id`` (the
same id across forms and the harmful→benign twin). The CSVs are fetched to a
cache dir and parsed as data (``csv``/parquet), never executed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

from magpie.data.taxonomy import normalize_category
from magpie.types import Language, PromptRecord

HARMBENCH_CSV = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
                 "main/data/behavior_datasets/harmbench_behaviors_text_all.csv")
STRONGREJECT_CSV = ("https://raw.githubusercontent.com/alexandrasouly/strongreject/"
                    "main/strongreject_dataset/strongreject_dataset.csv")

CACHE_DIR = Path(os.environ.get("MAGPIE_SEED_CACHE", "/tmp/magpie-seeds"))


def _short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _fetch(url: str, name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if not dest.is_file() or dest.stat().st_size == 0:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URLs
            dest.write_bytes(resp.read())
    return dest


def _read_csv(path_or_text: Path | str) -> list[dict[str, str]]:
    text = path_or_text.read_text(encoding="utf-8") if isinstance(path_or_text, Path) \
        else path_or_text
    return list(csv.DictReader(io.StringIO(text)))


# ---------------------------------------------------------------------------
# Parsers (take already-loaded rows, so they are unit-testable without network)
# ---------------------------------------------------------------------------


def parse_harmbench(rows: list[dict[str, str]], functional: str = "standard") -> list[PromptRecord]:
    out = []
    for r in rows:
        if functional and (r.get("FunctionalCategory") or "").strip() != functional:
            continue
        text = (r.get("Behavior") or "").strip()
        if not text:
            continue
        bid = (r.get("BehaviorID") or _short(text)).strip()
        out.append(PromptRecord(
            prompt_id=f"harmbench:{bid}", text=text, form=Language.EN, is_harmful=True,
            category=normalize_category(r.get("SemanticCategory")),
            parallel_id=f"harmbench:{bid}", source="harmbench_standard",
            covariates={"semantic_category": (r.get("SemanticCategory") or "").strip()},
        ))
    return out


def parse_strongreject(rows: list[dict[str, str]]) -> list[PromptRecord]:
    out = []
    for r in rows:
        text = (r.get("forbidden_prompt") or "").strip()
        if not text:
            continue
        pid = _short(text)
        out.append(PromptRecord(
            prompt_id=f"strongreject:{pid}", text=text, form=Language.EN, is_harmful=True,
            category=normalize_category(r.get("category")),
            parallel_id=f"strongreject:{pid}", source="strongreject",
            covariates={"source_category": (r.get("category") or "").strip(),
                        "source_ref": (r.get("source") or "").strip()},
        ))
    return out


def parse_xstest_safe(rows: list[dict[str, str]]) -> list[PromptRecord]:
    """XSTest *safe* prompts (label == 'safe') as over-refusal probes."""
    out = []
    for r in rows:
        label = (r.get("label") or r.get("type_label") or "").strip().lower()
        text = (r.get("prompt") or "").strip()
        # XSTest 'type' encodes safe vs contrast; safe prompts are the ones whose
        # type does not start with 'contrast'. Prefer an explicit label when present.
        typ = (r.get("type") or "").strip()
        is_safe = (label == "safe") or (not label and not typ.startswith("contrast"))
        if not text or not is_safe:
            continue
        pid = (str(r.get("id") or r.get("id_v2") or _short(text))).strip()
        out.append(PromptRecord(
            prompt_id=f"xstest:{pid}", text=text, form=Language.EN, is_harmful=False,
            category="over_refusal", parallel_id=f"xstest:{pid}", source="xstest_safe",
            covariates={"xstest_type": typ},
        ))
    return out


# ---------------------------------------------------------------------------
# Network loaders
# ---------------------------------------------------------------------------


def load_harmbench(functional: str = "standard") -> list[PromptRecord]:
    return parse_harmbench(_read_csv(_fetch(HARMBENCH_CSV, "harmbench.csv")), functional)


def load_strongreject() -> list[PromptRecord]:
    return parse_strongreject(_read_csv(_fetch(STRONGREJECT_CSV, "strongreject.csv")))


def load_xstest_safe() -> list[PromptRecord]:
    from datasets import load_dataset

    ds = load_dataset("Paul/XSTest", split="train")
    return parse_xstest_safe([dict(r) for r in ds])


def load_orbench_hard(max_items: int | None = 400) -> list[PromptRecord]:
    """OR-Bench-Hard-1k over-refusal prompts (benign prompts that look harmful)."""
    from datasets import load_dataset

    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    out = []
    for i, r in enumerate(ds):
        if max_items and i >= max_items:
            break
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        out.append(PromptRecord(
            prompt_id=f"orbench:{i}", text=text, form=Language.EN, is_harmful=False,
            category="over_refusal", parallel_id=f"orbench:{i}", source="orbench_hard",
            covariates={"orbench_category": (r.get("category") or "").strip()},
        ))
    return out


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(records: list[PromptRecord], n: int, seed: int = 0,
                      key: str = "category") -> list[PromptRecord]:
    """Pick ~``n`` records balanced across ``key`` (round-robin by stable hash).

    Deterministic: within each stratum, records are ordered by a seed-salted
    hash, then drawn round-robin across strata until ``n`` is reached. No RNG
    state, so it is reproducible and independent of input order.
    """
    strata: dict[str, list[PromptRecord]] = defaultdict(list)
    for r in records:
        strata[getattr(r, key, "") or r.covariates.get(key, "other")].append(r)
    for k in strata:
        strata[k].sort(key=lambda r: hashlib.sha1(f"{seed}|{r.prompt_id}".encode()).hexdigest())
    order = sorted(strata)
    picked: list[PromptRecord] = []
    idx = dict.fromkeys(order, 0)
    while len(picked) < n and any(idx[k] < len(strata[k]) for k in order):
        for k in order:
            if idx[k] < len(strata[k]):
                picked.append(strata[k][idx[k]])
                idx[k] += 1
                if len(picked) >= n:
                    break
    return picked


def category_counts(records: list[PromptRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.category] += 1
    return dict(sorted(counts.items()))
