"""Generic in-kernel launcher: install vLLM in a venv, then run a worker there.

The kernel bootstrap runs in Kaggle's system Python (no vLLM 0.33). This
launcher installs vLLM into an isolated venv and runs ``config["worker"]``
(``module`` with a ``main(config, out_dir)`` and a ``--config/--out`` CLI)
inside it. If the venv has no GPU, it falls back to system Python where
``load_backend('vllm', ...)`` degrades to HF transformers. Used by every
GPU job that needs vLLM (L0, data build, attacks), so the venv plumbing
lives in exactly one place.
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
    install_vllm_venv,
    python_env_versions,
    run_cmd,
    write_json,
)


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    worker_module = config.get("worker")
    if not worker_module:
        raise ValueError("launch job needs config['worker'] = '<module with main(config,out)>'")
    status: dict[str, Any] = {"worker_module": worker_module, "env": env_report()}
    write_json(out / "launch_env.json", status["env"])

    install = install_vllm_venv(config.get("vllm_spec", "vllm==0.33.0"), out)
    status["vllm_install"] = {k: install.get(k) for k in ("ok", "seconds", "returncode", "stage")}
    venv_ok = False
    if install["ok"]:
        ver = python_env_versions(VENV_PY, out, "venv", env=VENV_ENV)
        status["venv_versions"] = ver
        venv_ok = ver.get("gpu_ok") is True

    cfg_path = out / "_job_config.json"
    budget = int(config.get("timeout_s", 3 * 3600)) - 300
    worker = ["-m", worker_module, "--config", str(cfg_path), "--out", str(out)]
    if venv_ok:
        status["backend"] = "vllm (venv)"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        status["worker"] = run_cmd([VENV_PY, *worker], out / "logs" / "worker.log",
                                   timeout=budget, env=VENV_ENV)
    else:
        status["backend"] = "hf (system fallback)"
        cfg_path.write_text(json.dumps({**config, "backend": "hf"}), encoding="utf-8")
        status["worker"] = run_cmd([sys.executable, *worker], out / "logs" / "worker.log",
                                   timeout=budget)
    write_json(out / "launch_status.json", status)
    print(json.dumps({"worker": worker_module, "backend": status["backend"],
                      "rc": status["worker"].get("returncode")}), flush=True)
