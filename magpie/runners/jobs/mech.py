"""Kernel job: extract refusal margins (installs SafeTune, runs the mech worker).

Mirrors the L3a job: install SafeTune, remove Kaggle's torchao, then run the
margin extraction in a subprocess (system Python, HF backend). Writes
``margins_<model>.json`` for the local Cox join against cost-to-break (H4).
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
    write_json(out / "mech_env.json", status["env"])
    status["safetune_install"] = run_cmd(
        [sys.executable, "-m", "pip", "install", "-q",
         config.get("safetune_spec", "safetune==0.1.12")],
        out / "logs" / "safetune_install.log", timeout=1800)
    status["torchao_removed"] = run_cmd(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        out / "logs" / "torchao_uninstall.log", timeout=300)
    status["system_versions"] = python_env_versions(sys.executable, out, "system")

    cfg_path = out / "_mech_config.json"
    cfg_path.write_text(json.dumps({
        "model": config.get("model", "fire"), "pick_layer": config.get("pick_layer", 24),
        "n_harmful": config.get("n_harmful", 100), "n_contrast": config.get("n_contrast", 64),
        "seed": config.get("seed", 0)}))
    status["worker"] = run_cmd(
        [sys.executable, "-m", "magpie.runners.jobs.mech_worker", "--config", str(cfg_path),
         "--out", str(out)],
        out / "logs" / "worker_mech.log", timeout=int(config.get("timeout_s", 3 * 3600)) - 300,
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    write_json(out / "mech_status.json", status)
    print(json.dumps({"worker_rc": status["worker"]["returncode"]}), flush=True)
