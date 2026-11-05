"""Kernel job: harden Tiny Aya Fire (M2 RR or M3 RepNoise), then L0-evaluate it.

Runs in Kaggle's system Python (the kernel bootstrap calls ``main`` directly):

1. install SafeTune into system Python (its torch / transformers floors are
   already met there, so nothing CUDA-related is replaced);
2. train in a subprocess (``magpie.defenses.train``), so the GPU is fully freed;
3. install vLLM in the isolated venv and run the L0 worker on the hardened
   checkpoint by path, producing the same table as for the vendor models.

The merged checkpoint stays in this kernel's output (``ckpt/``). Later attack
kernels attach it as a kernel source; the runner pulls results but not weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from magpie.runners.jobs.common import (
    VENV_ENV,
    VENV_PY,
    env_report,
    find_model_dir,
    install_vllm_venv,
    python_env_versions,
    run_cmd,
    write_json,
)


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    method = config["method"]
    name = f"fire_{method}"
    ckpt = out.parent / "ckpt" / name  # /kaggle/working/ckpt/<name>, beside out/
    status: dict[str, Any] = {"method": method, "name": name, "env": env_report()}
    write_json(out / "harden_env.json", status["env"])

    status["safetune_install"] = run_cmd(
        [sys.executable, "-m", "pip", "install", "-q", config.get("safetune_spec",
                                                                  "safetune==0.1.12")],
        out / "logs" / "safetune_install.log", timeout=1800)
    status["system_versions"] = python_env_versions(sys.executable, out, "system")

    base = find_model_dir(config.get("base", "fire"))
    train_cmd = [sys.executable, "-m", "magpie.defenses.train", "--method", method,
                 "--model", base, "--out", str(ckpt), "--n", str(config.get("n", 256)),
                 "--steps", str(config.get("steps", 150)),
                 "--batch-size", str(config.get("batch_size", 2)),
                 "--lr", str(config.get("lr", 1e-4)), "--max-len", str(config.get("max_len", 768))]
    status["train"] = run_cmd(train_cmd, out / "logs" / "train.log",
                              timeout=int(config.get("train_timeout_s", 4 * 3600)))
    train_status = ckpt / "train_status.json"
    if train_status.is_file():
        status["train_status"] = json.loads(train_status.read_text())
        write_json(out / "train_status.json", status["train_status"])
    trained = status["train"]["returncode"] == 0 and (ckpt / "config.json").is_file()
    status["trained"] = trained

    if trained and config.get("eval", True):
        install = install_vllm_venv(config.get("vllm_spec", "vllm==0.33.0"), out)
        status["vllm_install"] = {k: install.get(k) for k in ("ok", "seconds", "returncode")}
        l0_cfg = {"models": [name], "model_dirs": {name: str(ckpt)},
                  "n_harmful": config.get("n_harmful", 150),
                  "n_benign": config.get("n_benign", 150),
                  "max_new_tokens": config.get("max_new_tokens", 192),
                  "seed": config.get("seed", 0)}
        cfg_path = out / "_l0_config.json"
        python, env = (VENV_PY, VENV_ENV) if install["ok"] else (sys.executable, None)
        if not install["ok"]:
            l0_cfg["backend"] = "hf"
        cfg_path.write_text(json.dumps(l0_cfg))
        status["eval"] = run_cmd([python, "-m", "magpie.runners.jobs.l0", "--config",
                                  str(cfg_path), "--out", str(out / "l0")],
                                 out / "logs" / "eval_l0.log", timeout=3 * 3600, env=env)
    write_json(out / "harden_status.json", status)
    print(json.dumps({"method": method, "trained": trained,
                      "eval_rc": status.get("eval", {}).get("returncode")}), flush=True)
