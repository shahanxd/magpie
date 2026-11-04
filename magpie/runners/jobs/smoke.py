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
    model_dir = find_model_dir(config.get("model", "fire"))
    max_new = int(config.get("max_new_tokens", 128))
    summary: dict[str, Any] = {"model_dir": model_dir, "max_new_tokens": max_new}

    summary["env"] = env_report()
    write_json(out / "env.json", summary["env"])

    install = install_vllm_venv(config.get("vllm_spec", "vllm==0.33.0"), out)
    summary["vllm_install"] = install
    venv_py = VENV_PY
    if install["ok"]:
        summary["venv_versions"] = python_env_versions(venv_py, out, "venv", env=VENV_ENV)

    worker = ["-m", "magpie.runners.jobs.smoke_worker", "--model", model_dir,
              "--out", str(out), "--max-new-tokens", str(max_new)]
    venv_gpu_ok = install["ok"] and summary.get("venv_versions", {}).get("gpu_ok") is True
    if venv_gpu_ok:
        summary["vllm_run"] = run_cmd([venv_py, *worker, "--backend", "vllm"],
                                      out / "logs" / "worker_vllm.log", timeout=2400,
                                      env=VENV_ENV)

    # HF path always runs in Kaggle's system Python: it ships accelerate (which
    # transformers needs for device_map) and a cohere2-aware transformers. The
    # fresh vLLM venv deliberately has neither, so HF stays decoupled from it.
    hf_python = sys.executable
    summary["hf_python"] = "system"
    summary["system_transformers"] = python_env_versions(sys.executable, out, "system")
    summary["hf_run"] = run_cmd([hf_python, *worker, "--backend", "hf", "--fp16-check"],
                                out / "logs" / "worker_hf.log", timeout=3600)

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
