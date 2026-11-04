"""Build the Magpie data-v1 parallel core in a kernel (PLAN §5).

Loads the stratified harmful core and the over-refusal set (English), then with
Aya Expanse 8B on the T4: translates the core into Hinglish and Devanagari with
a back-translation agreement gate, assembles the parallel core, de-duplicates
and stamps provenance with CuratorKIT, annotates CMI, and writes:

* ``records.jsonl`` (full, with prompt text) under ``raw/`` — gitignored; this
  is the private/gated dataset payload pushed to HF from the dev environment,
* ``manifest.json`` and ``dataset_card.md`` (committable; counts + provenance,
  no prompt text),
* ``curation_report.json`` (dedup + back-translation agreement stats).

Over-refusal prompts are translated too (benign twins of the forms), so H2b's
orthographic over-refusal has a Hinglish benign set. Runs under the generic
vLLM launcher; on a GPU-less fallback it still runs with the HF backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magpie.config import provenance
from magpie.data.curate import curate
from magpie.data.hinglish import cmi
from magpie.data.loaders import assemble_core, manifest_for
from magpie.data.translate import AyaTranslator, token_overlap_agreement, translate_core
from magpie.runners.jobs.common import write_json
from magpie.types import PromptRecord


def _load_en(config: dict[str, Any]) -> tuple[list[PromptRecord], list[PromptRecord]]:
    from magpie.data.seeds import (
        load_harmbench,
        load_orbench_hard,
        load_strongreject,
        load_xstest_safe,
        stratified_sample,
    )

    seed = int(config.get("seed", 0))
    pool = load_harmbench("standard") + load_strongreject()
    harmful = stratified_sample(pool, n=int(config.get("n_harmful", 200)), seed=seed)
    benign: list[PromptRecord] = []
    try:
        benign += load_xstest_safe()
    except Exception as exc:  # noqa: BLE001
        print(f"xstest failed: {exc}", flush=True)
    try:
        benign += load_orbench_hard(max_items=int(config.get("n_benign", 200)))
    except Exception as exc:  # noqa: BLE001
        print(f"orbench failed: {exc}", flush=True)
    benign = stratified_sample(benign, n=int(config.get("n_benign", 200)), seed=seed, key="source")
    return harmful, benign


def _resolve_translate_model(config: dict[str, Any]) -> str:
    """Resolve the translation model dir on Kaggle.

    Default is tiny-aya Fire: it mounts freely (Aya Expanse is gated on Kaggle
    and needs manual licence acceptance), fits one T4, and is the most
    Hindi-capable Cohere model available. A MODELS key resolves via
    find_model_dir; an explicit path or any dir with a config.json is used as
    is; otherwise we search /kaggle/input for a config whose path matches a hint.
    """
    from magpie.config import MODELS
    from magpie.runners.jobs.common import find_model_dir

    key = config.get("translate_model", "fire")
    if key in MODELS:
        return find_model_dir(key)
    if Path(key).joinpath("config.json").is_file():
        return key
    hint = config.get("translate_model_hint", key).lower()
    root = Path("/kaggle/input")
    if root.is_dir():
        for cfg in sorted(root.rglob("config.json")):
            if hint in str(cfg.parent).lower():
                return str(cfg.parent)
    return key


def _annotate_cmi(records: list[PromptRecord]) -> None:
    for r in records:
        if r.form in ("hing", "hi-Deva"):
            r.covariates["cmi"] = cmi(r.text).cmi


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    raw = out / "raw"
    forms = config.get("forms", ["hing", "hi-Deva"])
    backend = config.get("backend", "vllm")
    prov = provenance(run_id="build_data", forms=forms,
                      translate_model=config.get("translate_model", "fire"))

    harmful, benign = _load_en(config)
    en_all = harmful + benign
    print(f"loaded {len(harmful)} harmful + {len(benign)} benign (en)", flush=True)

    resolved = _resolve_translate_model(config)
    print(f"translation model dir: {resolved}", flush=True)
    translator = AyaTranslator.from_pretrained(
        resolved, backend=backend, max_new_tokens=int(config.get("max_new_tokens", 512)),
        tensor_parallel_size=int(config.get("tensor_parallel_size", 1)))

    translations, report = translate_core(
        en_all, translator, forms=forms,
        back_translate=bool(config.get("back_translate", True)),
        agreement_fn=token_overlap_agreement,
        agreement_threshold=float(config.get("agreement_threshold", 0.25)),
    )
    core = assemble_core(en_all, translations)
    _annotate_cmi(core)
    print(f"assembled {len(core)} records across forms", flush=True)

    curation = curate(core, minhash_threshold=float(config.get("minhash_threshold", 0.92)),
                      base_model="CohereLabs/tiny-aya-fire")
    kept = curation.kept

    # Full payload (with text) -> raw/ (gitignored, the gated HF dataset payload).
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    manifest = manifest_for("magpie-core-v1", kept,
                            provenance={**prov, "lexsi_provenance": curation.provenance})
    manifest.write(out / "manifest.json")
    write_json(out / "curation_report.json", {
        "n_in": curation.n_in, "n_out": curation.n_out, "n_removed": curation.n_removed,
        "minhash_threshold": curation.minhash_threshold,
        "translation_report_sample": dict(list(report.items())[:5]),
        "n_translated": sum(len(v) for v in translations.values()),
    })
    (out / "dataset_card.md").write_text(_dataset_card(manifest, prov), encoding="utf-8")
    print(f"curated {curation.n_out}/{curation.n_in} (removed {curation.n_removed})", flush=True)


def _dataset_card(manifest, prov: dict) -> str:
    return (
        f"# Magpie core v1\n\n"
        f"Parallel red-team + over-refusal core for Magpie (Tiny Aya Fire robustness).\n\n"
        f"- records: **{manifest.n_records}** "
        f"({manifest.n_harmful} harmful, {manifest.n_benign} benign/over-refusal)\n"
        f"- forms: {manifest.by_form}\n- categories: {manifest.by_category}\n"
        f"- sources: {manifest.by_source}\n\n"
        f"Harmful seeds: HarmBench (standard) and StrongREJECT. Over-refusal: XSTest-safe "
        f"and OR-Bench-Hard. Hinglish and Devanagari by tiny-aya (v1 translator) with a "
        f"back-translation agreement gate, pending native-speaker verification.\n\n"
        f"**Harmful prompts are gated.** Non-commercial safety research only "
        f"(Tiny Aya CC-BY-NC-4.0 + Cohere Labs AUP; sources under their own licences).\n\n"
        f"Provenance: git `{prov.get('git_sha')}`, translator `{prov.get('translate_model')}`.\n"
    )


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
