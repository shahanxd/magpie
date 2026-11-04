"""L0 + over-refusal on Fire and Global — the first table (PLAN §6 L0, D1 gate).

Runs inside a Kaggle T4 kernel. Loads the harmful parallel core (HarmBench
standard + StrongREJECT, stratified) and the over-refusal set (XSTest-safe +
OR-Bench-Hard), then for each target model generates L0 (greedy + one sampled)
on every prompt, judges with the multilingual refusal matcher, and writes the
L0 table: ASR with a Wilson CI and median cost-to-break, paired with the
benign over-refusal rate and a collapse rate — the "every number next to
over-refusal" rule.

The judge here is the fast lexical pre-filter (`refusal_lexicon`); the model
judges (Llama-Guard / WildGuard / StrongREJECT scorer) rescoring comes next and
only tightens the validity gate. Harmful completion text is written under
``raw/`` (gitignored on pull); the committable copies carry no prompt/response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magpie.analysis.aggregate import ResultBundle, build_rows, rows_to_markdown, write_outcomes
from magpie.attacks import L0Direct
from magpie.attacks.base import AttackRunner
from magpie.config import GenerationConfig, provenance
from magpie.costs import TINY_AYA_FIRE, CostModel
from magpie.judges import LexiconJudge
from magpie.runners.jobs.common import find_model_dir, write_json
from magpie.serve.base import Generator
from magpie.types import PromptRecord


def _load_prompts(config: dict[str, Any]) -> tuple[list[PromptRecord], list[PromptRecord]]:
    from magpie.data.seeds import (
        load_harmbench,
        load_orbench_hard,
        load_strongreject,
        load_xstest_safe,
        stratified_sample,
    )

    n_harm = int(config.get("n_harmful", 150))
    n_benign = int(config.get("n_benign", 150))
    pool = load_harmbench("standard") + load_strongreject()
    harmful = stratified_sample(pool, n=n_harm, seed=config.get("seed", 0))
    benign: list[PromptRecord] = []
    try:
        benign += load_xstest_safe()
    except Exception as exc:  # noqa: BLE001
        print(f"xstest load failed: {exc}", flush=True)
    try:
        benign += load_orbench_hard(max_items=n_benign)
    except Exception as exc:  # noqa: BLE001
        print(f"orbench load failed: {exc}", flush=True)
    benign = stratified_sample(benign, n=n_benign, seed=config.get("seed", 0),
                               key="source") if benign else []
    return harmful, benign


def _build_generator(model_key: str, backend_kind: str = "vllm") -> Generator:
    from magpie.serve import load_backend

    model_dir = find_model_dir(model_key)
    # vLLM when available (in the venv), else HF transformers — load_backend falls back.
    backend = load_backend(backend_kind, model_dir, dtype="float16", max_model_len=2048)
    # All Tiny Aya variants share the cohere2 3.35B shape; label by key.
    gen = Generator(backend, CostModel(TINY_AYA_FIRE, gpu="T4"))
    gen.backend.name = model_key  # table rows are keyed by the friendly model name
    return gen


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    raw = out / "raw"
    models = config.get("models", ["fire", "global"])
    gen_cfg = GenerationConfig(max_new_tokens=int(config.get("max_new_tokens", 192)),
                               temperature=0.0)
    seed = int(config.get("seed", 0))
    prov = provenance(run_id="l0", models=models)

    harmful, benign = _load_prompts(config)
    from magpie.data.loaders import manifest_for

    manifest_for("l0_core", harmful + benign, provenance=prov).write(out / "data_manifest.json")
    print(f"loaded {len(harmful)} harmful, {len(benign)} benign", flush=True)

    judge = LexiconJudge()
    all_outcomes = []
    all_benign_attempts = []
    per_model_summary: dict[str, Any] = {}

    backend_kind = config.get("backend", "vllm")
    for model_key in models:
        print(f"=== model {model_key} ===", flush=True)
        gen = None
        try:
            gen = _build_generator(model_key, backend_kind)
            # L0 has 1-2 variants per prompt, so batch generation across records
            # (a per-record loop would send single prompts and waste the GPU).
            runner = AttackRunner(gen, judge, gen_cfg, chunk_size=256,
                                  batch_across_records=True, run_id="l0",
                                  git_sha=prov.get("git_sha", ""))
            attack = L0Direct(sampled=True, temperature=1.0, seed=seed)
            harm_run = runner.run(attack, harmful)
            benign_run = runner.run(attack, benign)
            all_outcomes += harm_run.outcomes
            all_benign_attempts += benign_run.attempts

            write_outcomes(harm_run.outcomes, out / f"outcomes_{model_key}.jsonl")
            # Full attempts (with text) under raw/ only; a redacted copy is committable.
            _write_attempts(harm_run.attempts + benign_run.attempts,
                            raw / f"attempts_{model_key}.jsonl", include_text=True)
            _write_attempts(harm_run.attempts + benign_run.attempts,
                            out / f"attempts_notext_{model_key}.jsonl", include_text=False)
            per_model_summary[model_key] = {
                "ok": True, "n_harmful": len(harm_run.outcomes),
                "n_harmful_broken": sum(o.broken for o in harm_run.outcomes),
                "n_benign_attempts": len(benign_run.attempts),
            }
        except Exception as exc:  # one model's failure must not lose the others
            import traceback

            per_model_summary[model_key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            (out / "logs").mkdir(parents=True, exist_ok=True)
            (out / "logs" / f"error_{model_key}.log").write_text(traceback.format_exc())
            print(f"model {model_key} FAILED: {exc}", flush=True)
        finally:
            if gen is not None and hasattr(gen.backend, "close"):
                gen.backend.close()  # release vLLM GPU memory before the next model
            gen = None
            _free_gpu()

    # Always write a table from whatever models completed.
    rows = build_rows(all_outcomes, benign_attempts=all_benign_attempts, n_boot=2000, seed=seed)
    bundle = ResultBundle("L0 (refusal_lexicon judge)", rows,
                          provenance={**prov, "judge": "refusal_lexicon", "level": "L0",
                                      "gen": {"max_new_tokens": gen_cfg.max_new_tokens}})
    bundle.write(out)
    write_json(out / "summary.json", {"provenance": prov, "per_model": per_model_summary,
                                      "n_rows": len(rows)})
    print(rows_to_markdown(rows, "L0"), flush=True)


def _write_attempts(attempts, path: Path, include_text: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in attempts:
            d = a.to_dict()
            if not include_text:
                d.pop("response", None)
                d.pop("prompt", None)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _cli(argv: list[str] | None = None) -> int:
    """Run the L0 worker from argv (the venv subprocess entry): --config JSON --out DIR."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON string or path to a JSON file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg_text = Path(args.config).read_text() if Path(args.config).is_file() else args.config
    main(json.loads(cfg_text), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
