"""Kernel entrypoint for the L0 job: install vLLM in a venv, then run the worker there.

The kernel bootstrap runs in Kaggle's system Python, which has no vLLM 0.33, so
this launcher mirrors the smoke test: install vLLM into an isolated venv, then
run ``magpie.runners.jobs.l0`` inside it. If the install or the venv's GPU
check fails, it falls back to running the worker in system Python, where
``load_backend('vllm', ...)`` degrades to HF transformers — slower, but the
table still comes out. Either way the worker writes the L0 table to ``out``.
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
    status: dict[str, Any] = {"env": env_report()}
    write_json(out / "launch_env.json", status["env"])

    install = install_vllm_venv(config.get("vllm_spec", "vllm==0.33.0"), out)
    status["vllm_install"] = {k: install.get(k) for k in ("ok", "seconds", "returncode", "stage")}
    venv_ok = False
    if install["ok"]:
        ver = python_env_versions(VENV_PY, out, "venv", env=VENV_ENV)
        status["venv_versions"] = ver
        venv_ok = ver.get("gpu_ok") is True

    cfg_path = out / "_l0_config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    worker = ["-m", "magpie.runners.jobs.l0", "--config", str(cfg_path), "--out", str(out)]

    if venv_ok:
        status["backend"] = "vllm (venv)"
        status["worker"] = run_cmd([VENV_PY, *worker], out / "logs" / "worker_l0.log",
                                   timeout=config.get("timeout_s", 3 * 3600) - 300, env=VENV_ENV)
    else:
        status["backend"] = "hf (system fallback)"
        cfg2 = {**config, "backend": "hf"}
        cfg_path.write_text(json.dumps(cfg2), encoding="utf-8")
        status["worker"] = run_cmd([sys.executable, *worker], out / "logs" / "worker_l0.log",
                                   timeout=config.get("timeout_s", 3 * 3600) - 300)
    write_json(out / "launch_status.json", status)
    print(json.dumps({"backend": status["backend"],
                      "worker_rc": status["worker"].get("returncode")}), flush=True)
