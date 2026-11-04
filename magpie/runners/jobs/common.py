"""Helpers shared by in-kernel jobs: model discovery, env reports, subprocess runs."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from magpie.config import MODELS

KAGGLE_INPUT = Path("/kaggle/input")

# The kernel bootstrap unpacks the magpie source here and puts it on PYTHONPATH.
SRC = "/tmp/magpie_src"
VENV = Path("/tmp/venv-vllm")
# Subprocesses in the fresh vLLM venv get PYTHONPATH=SRC only (plus no user-site)
# so magpie imports but Kaggle's sitecustomize (needs `wrapt`) never pollutes stdout.
VENV_ENV = {"PYTHONPATH": SRC, "PYTHONNOUSERSITE": "1"}
VENV_PY = str(VENV / "bin" / "python")


def install_vllm_venv(spec: str, out: Path) -> dict[str, Any]:
    """Create an isolated venv and install vLLM into it (its pinned torch/CUDA).

    Kept out of Kaggle's system stack so a mismatched wheel cannot break the HF
    fallback. Returns a status dict; never raises.
    """
    uv = run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"],
                 out / "logs" / "uv_install.log", timeout=300)
    if uv["returncode"] != 0:
        return {"ok": False, "stage": "uv", **uv}
    venv = run_cmd([sys.executable, "-m", "uv", "venv", "--python", sys.executable, str(VENV)],
                   out / "logs" / "venv.log", timeout=300)
    if venv["returncode"] != 0:
        return {"ok": False, "stage": "venv", **venv}
    # Install the worker's deps into the venv too: accelerate (HF device_map
    # fallback), datasets (XSTest / OR-Bench loaders), curatorkit (data-build
    # curation). The harmful-seed CSV loaders use urllib and need none of these.
    inst = run_cmd([sys.executable, "-m", "uv", "pip", "install", "--python", VENV_PY,
                    spec, "accelerate>=1.0", "datasets>=3.0", "curatorkit>=0.1.22",
                    "sentencepiece>=0.1.99", "tiktoken>=0.5"],
                   out / "logs" / "vllm_install.log", timeout=1800, env=VENV_ENV)
    return {"ok": inst["returncode"] == 0, "stage": "install", "spec": spec,
            "seconds": inst["seconds"], "returncode": inst["returncode"],
            "tail": inst["tail"][-1500:]}


def python_env_versions(python: str, out: Path, tag: str,
                        env: dict[str, str] | None = None) -> dict[str, Any]:
    """torch/transformers/CUDA versions and GPU visibility for a Python, robustly parsed."""
    code = ("import json,torch,transformers;"
            "print('MAGPIE_VERSIONS ' + json.dumps({'torch':torch.__version__,"
            "'cuda':torch.version.cuda,'transformers':transformers.__version__,"
            "'gpu_ok':torch.cuda.is_available(),'n_gpu':torch.cuda.device_count()}))")
    r = run_cmd([python, "-c", code], out / "logs" / f"versions_{tag}.log", timeout=300, env=env)
    for line in reversed(r["tail"].splitlines()):
        if line.startswith("MAGPIE_VERSIONS "):
            try:
                return json.loads(line[len("MAGPIE_VERSIONS "):])
            except json.JSONDecodeError:
                break
    return {"error": r["tail"][-800:]}


def find_model_dir(key: str, search_root: Path = KAGGLE_INPUT) -> str:
    """Locate an attached Kaggle Models mirror for ``key``; fall back to the HF id.

    Kaggle mounts models at ``/kaggle/input/<model>/<framework>/<variation>/<version>``
    (sometimes under an extra ``models/`` level), so search instead of hard-coding.
    """
    spec = MODELS[key]
    expected = Path(spec["kaggle_path"])
    if (expected / "config.json").is_file():
        return str(expected)
    variation = spec["kaggle_source"].split("/")[3]
    if search_root.is_dir():
        for cfg in sorted(search_root.rglob("config.json")):
            parts = [p.lower() for p in cfg.parent.parts]
            if "tiny-aya" in parts and variation in parts:
                return str(cfg.parent)
    return spec["hf_id"]


def run_cmd(cmd: list[str], log_path: Path | None = None, timeout: int = 1800,
            env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a command, capture its output to ``log_path``, never raise."""
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, **(env or {})})
        out, rc = (proc.stdout or "") + (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = f"TIMEOUT after {timeout}s\n{exc.stdout or ''}{exc.stderr or ''}"
        rc = -9
    except OSError as exc:
        out, rc = f"{type(exc).__name__}: {exc}", -1
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out[-200_000:], encoding="utf-8")  # keep the tail only
    return {"cmd": cmd[:3] + (["..."] if len(cmd) > 3 else []), "returncode": rc,
            "seconds": round(time.time() - t0, 1), "tail": out[-2000:]}


def _cmd_out(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def env_report() -> dict[str, Any]:
    """Hardware + software facts about this kernel (no secrets, no env dump)."""
    rep: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi": _cmd_out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,"
                                "compute_cap", "--format=csv,noheader"]),
        "cuda_version_smi": next((ln.split("CUDA Version:")[1].strip(" |")
                                  for ln in _cmd_out(["nvidia-smi"]).splitlines()
                                  if "CUDA Version:" in ln), ""),
    }
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        rep["ram_gb"] = round(int(meminfo[0].split()[1]) / 1e6, 1)
    except (OSError, IndexError, ValueError):
        pass
    total, _, free = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists()
                                       else "/")
    rep["disk_free_gb"] = round(free / 1e9, 1)
    rep["disk_total_gb"] = round(total / 1e9, 1)
    for mod in ("torch", "transformers", "vllm", "accelerate", "tokenizers"):
        try:
            m = __import__(mod)
            rep[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - report, never fail
            rep[f"{mod}_version"] = f"unavailable ({type(exc).__name__})"
    try:
        import torch

        rep["torch_cuda"] = torch.version.cuda
        rep["cuda_available"] = torch.cuda.is_available()
        rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if rep["gpus"]:
            rep["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            rep["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:  # noqa: BLE001
        rep["torch_error"] = f"{type(exc).__name__}: {exc}"
    if KAGGLE_INPUT.is_dir():
        rep["kaggle_input"] = sorted(str(p.relative_to(KAGGLE_INPUT))
                                     for p in KAGGLE_INPUT.glob("*/*/*/*"))[:50]
    return rep


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
