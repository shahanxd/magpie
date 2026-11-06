"""Extract per-prompt refusal margins for the harmful core (HF; runs in a kernel).

Loads a model, fits the refusal direction on a disjoint contrast set, and
projects every harmful eval prompt onto it, writing ``margins_<model>.json``
(standardized margins + raw + layer). The Cox join against cost-to-break
happens locally (CPU) from the pulled margins and the attack outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    config = json.loads(Path(args.config).read_text() if Path(args.config).is_file()
                        else args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import safetune
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from magpie.mech.margins import extract_margins
    from magpie.runners.jobs.common import find_model_dir

    model_key = config.get("model", "fire")
    pick_layer = config.get("pick_layer", 24)
    safetune.configure(dtype="float16", calib_n=config.get("n_contrast", 64))
    model_dir = config.get("model_dir") or find_model_dir(model_key)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16,
                                                 device_map="cuda").eval()

    from magpie.data.seeds import load_harmbench, load_strongreject, stratified_sample

    pool = load_harmbench("standard") + load_strongreject()
    harmful = stratified_sample(pool, n=int(config.get("n_harmful", 100)),
                                seed=config.get("seed", 0))

    from safetune.runner.utils.data_utils import refusal_prompt_pairs_large

    hc, hl = refusal_prompt_pairs_large(int(config.get("n_contrast", 64)))
    res = extract_margins(model, tok, harmful, hc[:config.get("n_contrast", 64)],
                          hl[:config.get("n_contrast", 64)], pick_layer=pick_layer)
    res["model"] = model_key
    (out / f"margins_{model_key}.json").write_text(json.dumps(res, indent=2))
    print(f"MAGPIE_MARGINS model={model_key} layer={res['layer']} n={res['n_prompts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
