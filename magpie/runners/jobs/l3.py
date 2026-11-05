"""Kernel job: L3a abliteration on the vendor models, in Kaggle's system Python.

Abliteration installs forward hooks, so it needs the HF model, not vLLM. This
job installs SafeTune (for AbliterationAttack and the contrast data), removes
Kaggle's torchao (PEFT/refusal-direction quirk), then runs the eval worker in
system Python with ``backend=hf`` and ``attack=l3_abliteration``. No vLLM venv.

The worker fits the refusal direction once per model from a disjoint contrast
set, ablates the weights, and generates one response per harmful prompt — so
the L3a table shows the weight-space attack's effect next to L0/L1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from magpie.runners.jobs.common import env_report, python_env_versions, run_cmd, write_json


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    status: dict[str, Any] = {"env": env_report()}
    write_json(out / "l3_env.json", status["env"])

    status["safetune_install"] = run_cmd(
        [sys.executable, "-m", "pip", "install", "-q",
         config.get("safetune_spec", "safetune==0.1.12")],
        out / "logs" / "safetune_install.log", timeout=1800)
    status["torchao_removed"] = run_cmd(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        out / "logs" / "torchao_uninstall.log", timeout=300)
    status["system_versions"] = python_env_versions(sys.executable, out, "system")

    l0_cfg = {
        "attack": "l3_abliteration", "backend": "hf",
        "models": config.get("models", ["fire", "global"]),
        "model_dirs": config.get("model_dirs", {}),
        "n_harmful": config.get("n_harmful", 120), "n_benign": config.get("n_benign", 0),
        "max_new_tokens": config.get("max_new_tokens", 192), "seed": config.get("seed", 0),
        "attack_params": config.get("attack_params", {"pick_layer": 24, "n_contrast": 128}),
    }
    cfg_path = out / "_l3_config.json"
    cfg_path.write_text(json.dumps(l0_cfg))
    status["worker"] = run_cmd(
        [sys.executable, "-m", "magpie.runners.jobs.l0", "--config", str(cfg_path),
         "--out", str(out)],
        out / "logs" / "worker_l3.log", timeout=int(config.get("timeout_s", 4 * 3600)) - 300,
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    write_json(out / "l3_status.json", status)
    print(json.dumps({"attack": "l3_abliteration", "worker_rc": status["worker"]["returncode"]}),
          flush=True)
