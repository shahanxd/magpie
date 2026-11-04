"""Day-1 smoke test (CONTEXT §6.2), run inside a Kaggle T4 kernel.

1. Report the kernel's hardware and software (GPU, driver, CUDA, torch, RAM).
2. Install vLLM into an isolated venv (so a mismatched torch/CUDA wheel cannot
   break the system stack the HF fallback needs) and time the install.
3. vLLM worker: load Tiny Aya Fire in fp16, generate 5 en + 5 Hinglish benign
   prompts, report tokens/s.
4. HF worker (always, as the fallback and the white-box path): same prompts,
   same metrics, plus the fp16 numerics check against an fp32 CPU reference.
5. Write ``summary.json`` + ``summary.md`` with the go/no-go facts for the plan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from magpie.runners.jobs.common import env_report, find_model_dir, run_cmd, write_json

VENV = Path("/tmp/venv-vllm")
# The kernel bootstrap unpacks the magpie source here and puts it on PYTHONPATH.
# Subprocesses in the fresh vLLM venv get PYTHONPATH=SRC only (plus no user-site),
# so magpie imports but Kaggle's sitecustomize (which needs `wrapt`) does not load
# and pollute stdout.
SRC = "/tmp/magpie_src"
VENV_ENV = {"PYTHONPATH": SRC, "PYTHONNOUSERSITE": "1"}


def _install_vllm(spec: str, out: Path) -> dict[str, Any]:
    pip = run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"],
                  out / "logs" / "uv_install.log", timeout=300)
    if pip["returncode"] != 0:
        return {"ok": False, "stage": "uv", **pip}
    venv = run_cmd([sys.executable, "-m", "uv", "venv", "--python", sys.executable, str(VENV)],
                   out / "logs" / "venv.log", timeout=300)
    if venv["returncode"] != 0:
        return {"ok": False, "stage": "venv", **venv}
    inst = run_cmd([sys.executable, "-m", "uv", "pip", "install", "--python",
                    str(VENV / "bin" / "python"), spec],
                   out / "logs" / "vllm_install.log", timeout=1800, env=VENV_ENV)
    return {"ok": inst["returncode"] == 0, "stage": "install", "spec": spec,
            "seconds": inst["seconds"], "returncode": inst["returncode"],
            "tail": inst["tail"][-1500:]}


def _python_env_versions(python: str, out: Path, tag: str,
                         env: dict[str, str] | None = None) -> dict[str, Any]:
    code = ("import json,torch,transformers;"
            "print('MAGPIE_VERSIONS ' + json.dumps({'torch':torch.__version__,"
            "'cuda':torch.version.cuda,'transformers':transformers.__version__,"
            "'gpu_ok':torch.cuda.is_available(),"
            "'n_gpu':torch.cuda.device_count()}))")
    r = run_cmd([python, "-c", code], out / "logs" / f"versions_{tag}.log", timeout=300, env=env)
    # Scan for our tagged line; other libraries (and Kaggle's sitecustomize) may
    # print warnings around it, so never trust position alone.
    for line in reversed(r["tail"].splitlines()):
        if line.startswith("MAGPIE_VERSIONS "):
            try:
                return json.loads(line[len("MAGPIE_VERSIONS "):])
            except json.JSONDecodeError:
                break
    return {"error": r["tail"][-800:]}


def main(config: dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    model_dir = find_model_dir(config.get("model", "fire"))
    max_new = int(config.get("max_new_tokens", 128))
    summary: dict[str, Any] = {"model_dir": model_dir, "max_new_tokens": max_new}

    summary["env"] = env_report()
    write_json(out / "env.json", summary["env"])

    install = _install_vllm(config.get("vllm_spec", "vllm==0.33.0"), out)
    summary["vllm_install"] = install
    venv_py = str(VENV / "bin" / "python")
    if install["ok"]:
        summary["venv_versions"] = _python_env_versions(venv_py, out, "venv", env=VENV_ENV)

    worker = ["-m", "magpie.runners.jobs.smoke_worker", "--model", model_dir,
              "--out", str(out), "--max-new-tokens", str(max_new)]
    venv_gpu_ok = install["ok"] and summary.get("venv_versions", {}).get("gpu_ok") is True
    if venv_gpu_ok:
        summary["vllm_run"] = run_cmd([venv_py, *worker, "--backend", "vllm"],
                                      out / "logs" / "worker_vllm.log", timeout=2400,
                                      env=VENV_ENV)

    # HF path: in the venv when its torch sees the GPU (same stack as vLLM),
    # otherwise in Kaggle's system Python (ensure a transformers that knows cohere2).
    if venv_gpu_ok:
        hf_python = venv_py
    else:
        hf_python = sys.executable
        summary["system_transformers_upgrade"] = run_cmd(
            [sys.executable, "-m", "pip", "install", "-q", "transformers>=5.16,<5.21"],
            out / "logs" / "transformers_upgrade.log", timeout=900)
    summary["hf_python"] = "venv" if hf_python == venv_py else "system"
    summary["hf_run"] = run_cmd([hf_python, *worker, "--backend", "hf", "--fp16-check"],
                                out / "logs" / "worker_hf.log", timeout=3600,
                                env=VENV_ENV if venv_gpu_ok else None)

    for backend in ("vllm", "hf"):
        p = out / f"{backend}.json"
        if p.is_file():
            summary[backend] = json.loads(p.read_text())
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(s: dict[str, Any]) -> str:
    env = s.get("env", {})
    lines = ["# Magpie smoke test", "",
             f"- GPU: {env.get('nvidia_smi', '?')} (driver CUDA {env.get('cuda_version_smi', '?')})",
             f"- system torch {env.get('torch_version')} (CUDA {env.get('torch_cuda')}), "
             f"python {env.get('python')}, RAM {env.get('ram_gb')} GB",
             f"- vLLM install: {'ok' if s.get('vllm_install', {}).get('ok') else 'FAILED'} "
             f"({s.get('vllm_install', {}).get('seconds', '?')} s)",
             f"- model dir: `{s.get('model_dir')}`", ""]
    lines += ["| backend | ok | load s | batch tok/s | single tok/s | degenerate |",
              "|---|---|---|---|---|---|"]
    for b in ("vllm", "hf"):
        r = s.get(b)
        if not r:
            lines.append(f"| {b} | not run | | | | |")
            continue
        lines.append(f"| {b} | {r.get('ok')} | {r.get('load_seconds')} | "
                     f"{r.get('batch_tokens_per_s')} | {r.get('single_stream_tokens_per_s')} | "
                     f"{r.get('n_degenerate')} |")
    fp = (s.get("hf") or {}).get("fp16_check")
    if fp:
        lines += ["", f"fp16 check: **{fp['verdict']}**; logits finite {fp['logits_all_finite']}, "
                  f"max layer activation {fp['max_layer_peak']} (headroom ×{fp['fp16_headroom']}), "
                  "fp32 reference: " + ", ".join(
                      f"{c['form']} KL={c['kl_fp32_fp16']:.4f} top1={c['top1_match']}"
                      for c in fp.get("fp32_reference", []))]
    return "\n".join(lines) + "\n"
