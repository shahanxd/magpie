"""Run Magpie jobs on Kaggle GPUs end to end: build → push → poll → pull.

Usage::

    python -m magpie.runners.kaggle quota
    python -m magpie.runners.kaggle run smoke            # push, wait, pull into results/
    python -m magpie.runners.kaggle status smoke
    python -m magpie.runners.kaggle pull smoke

Credentials come from the environment only, never from code or logs. Kaggle
issues two token styles and both are supported: the new single API token
(``KAGGLE_API_TOKEN``; we also accept it under ``KAGGLE_KEY``, where this
team stores a copy) and the legacy ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` pair.
More accounts (more weekly GPU quota) use suffixed variables:
``KAGGLE_USERNAME_2``, ``KAGGLE_API_TOKEN_2`` / ``KAGGLE_KEY_2``, ...

Each job runs as a private *script* kernel with GPU (T4 ×2) and internet on,
with Tiny Aya attached from the Kaggle Models mirror, so no HF token ever
leaves this machine. The kernel carries the ``magpie`` source as an embedded,
hash-checked bundle (see :mod:`magpie.runners.bundle`), so every result is
pinned to a git SHA.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from magpie.config import MODELS, REPO_ROOT
from magpie.runners.bundle import build_bundle
from magpie.runners.kernel_template import render_kernel

BUILD_DIR = REPO_ROOT / ".kaggle-build"
RESULTS_DIR = REPO_ROOT / "results"
T4 = "NvidiaTeslaT4"  # Kaggle's "GPU T4 x2" machine shape (P100 is retired)
_LEGACY_KEY = re.compile(r"^[0-9a-f]{32}$")
_TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KaggleCredentials:
    """One Kaggle account's credentials, resolved from the environment."""

    username: str | None
    token: str | None  # new-style API token
    legacy_key: str | None  # old 32-hex key (needs username)
    account: int = 1

    @property
    def style(self) -> str:
        return "token" if self.token else ("legacy" if self.legacy_key else "missing")

    @classmethod
    def from_env(cls, account: int = 1, env: dict[str, str] | None = None) -> KaggleCredentials:
        """Resolve account ``n``: ``KAGGLE_API_TOKEN[_n]`` first, then ``KAGGLE_KEY[_n]``."""
        env = dict(os.environ if env is None else env)
        sfx = "" if account == 1 else f"_{account}"
        username = env.get(f"KAGGLE_USERNAME{sfx}") or None
        token = env.get(f"KAGGLE_API_TOKEN{sfx}") or None
        key = env.get(f"KAGGLE_KEY{sfx}") or None
        legacy = None
        if not token and key:
            # A 32-hex key is the legacy style; anything else is a new-style token.
            if _LEGACY_KEY.match(key):
                legacy = key
            else:
                token = key
        return cls(username=username, token=token, legacy_key=legacy, account=account)

    def env_overrides(self) -> dict[str, str]:
        """The variables the Kaggle SDK reads, for this account."""
        out: dict[str, str] = {}
        if self.username:
            out["KAGGLE_USERNAME"] = self.username
        if self.token:
            out["KAGGLE_API_TOKEN"] = self.token
        elif self.legacy_key:
            out["KAGGLE_KEY"] = self.legacy_key
        return out


@contextlib.contextmanager
def _patched_env(overrides: dict[str, str], clear: tuple[str, ...]) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in (*overrides, *clear)}
    try:
        for k in clear:
            os.environ.pop(k, None)
        os.environ.update(overrides)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class KernelJob:
    """A Kaggle GPU job: what to run, on what hardware, with which models."""

    name: str  # kernel slug suffix, e.g. "smoke" -> <user>/magpie-smoke
    entrypoint: str  # "module:function", called as function(config, out_dir)
    config: dict[str, Any] = field(default_factory=dict)
    models: list[str] = field(default_factory=lambda: ["fire"])  # keys of config.MODELS
    dataset_sources: list[str] = field(default_factory=list)
    extra_model_sources: list[str] = field(default_factory=list)  # raw Kaggle model refs
    machine_shape: str = T4
    gpu: bool = True  # False = CPU-only kernel (costs no GPU quota)
    # Regex a pulled output file must match (Kaggle SDK uses re.search). Used to
    # leave model weights in Kaggle while pulling results.
    pull_pattern: str | None = None
    enable_internet: bool = True
    timeout_s: int = 3 * 3600

    @property
    def slug(self) -> str:
        return f"magpie-{self.name}".lower().replace("_", "-")

    def metadata(self, username: str) -> dict[str, Any]:
        return {
            "id": f"{username}/{self.slug}",
            "title": self.slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": self.gpu,
            "enable_tpu": False,
            "enable_internet": self.enable_internet,
            "machine_shape": self.machine_shape if self.gpu else "",
            "dataset_sources": list(self.dataset_sources),
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": ([MODELS[m]["kaggle_source"] for m in self.models]
                              + list(self.extra_model_sources)),
        }


# Registry of named jobs. Jobs live in magpie.runners.jobs.* and run *inside* the kernel.
JOBS: dict[str, Callable[..., KernelJob]] = {}


def job(name: str) -> Callable[[Callable[..., KernelJob]], Callable[..., KernelJob]]:
    def register(factory: Callable[..., KernelJob]) -> Callable[..., KernelJob]:
        JOBS[name] = factory
        return factory

    return register


@job("smoke")
def smoke_job(**overrides: Any) -> KernelJob:
    """Day-1 smoke test: Tiny Aya Fire fp16 on a T4, vLLM vs HF, fp16 stability."""
    cfg = {"model": "fire", "vllm_spec": "vllm==0.33.0", "max_new_tokens": 128}
    cfg.update(overrides)
    return KernelJob(name="smoke", entrypoint="magpie.runners.jobs.smoke:main", config=cfg,
                     models=["fire"], timeout_s=2 * 3600)


@job("l0")
def l0_job(**overrides: Any) -> KernelJob:
    """L0 + over-refusal on Fire and Global: the first robustness table (en)."""
    cfg = {"models": ["fire", "global"], "n_harmful": 150, "n_benign": 150,
           "max_new_tokens": 192, "seed": 0, "vllm_spec": "vllm==0.33.0",
           "worker": "magpie.runners.jobs.l0", "timeout_s": 3 * 3600}
    cfg.update(overrides)
    # vLLM lives in an isolated venv; the shared launcher installs it.
    return KernelJob(name="l0", entrypoint="magpie.runners.jobs.launch:main", config=cfg,
                     models=cfg["models"], timeout_s=cfg["timeout_s"])


@job("attack")
def attack_job(**overrides: Any) -> KernelJob:
    """Run any registered attack (default l1_templates) on the vendor models + L0.

    Pass ``core_dataset=<owner/slug>`` to attach the prebuilt parallel core and
    evaluate its forms (en/hing/hi-Deva) instead of the English benchmarks; the
    worker then loads core.jsonl from the mount and the table groups by form.
    """
    attack = overrides.pop("attack", "l1_templates")
    core_dataset = overrides.pop("core_dataset", None)
    cfg = {"attack": attack, "models": ["fire", "global"], "n_harmful": 150, "n_benign": 150,
           "max_new_tokens": 192, "seed": 0, "vllm_spec": "vllm==0.33.0",
           "worker": "magpie.runners.jobs.l0", "timeout_s": 3 * 3600}
    cfg.update(overrides)
    if core_dataset:
        cfg["use_core"] = True
    models = cfg.get("models", ["fire", "global"])
    model_keys = [m for m in models if m in MODELS]
    return KernelJob(name=f"attack-{attack.replace('_', '-')}",
                     entrypoint="magpie.runners.jobs.launch:main", config=cfg,
                     models=model_keys, dataset_sources=[core_dataset] if core_dataset else [],
                     timeout_s=cfg["timeout_s"])


# Aya Expanse 8B on the Kaggle Models mirror (needs a one-time licence accept);
# Kaggle requires the version number in a model source ref.
AYA_KAGGLE_SOURCE = "cohereforai/aya-expanse/transformers/8b/5"


@job("build_data")
def build_data_job(**overrides: Any) -> KernelJob:
    """Build the parallel core: translate the seeds to Hinglish/Devanagari.

    translate_model="aya" uses Aya Expanse 8B (tensor-parallel over both T4s;
    the quality translator); "fire"/"global"/"base" use a tiny-aya variant.
    """
    cfg = {"n_harmful": 200, "n_benign": 200, "forms": ["hing", "hi-Deva"],
           "translate_model": "aya", "translate_model_hint": "aya-expanse",
           "tensor_parallel_size": 2, "back_translate": True, "max_new_tokens": 400,
           "seed": 0, "vllm_spec": "vllm==0.33.0",
           "worker": "magpie.runners.jobs.build_data", "timeout_s": 4 * 3600}
    cfg.update(overrides)
    tm = cfg["translate_model"]
    if tm in MODELS:
        return KernelJob(name="build-data", entrypoint="magpie.runners.jobs.launch:main",
                         config=cfg, models=[tm], timeout_s=cfg["timeout_s"])
    # Aya (or any non-tiny-aya translator): attach it as an extra model source.
    return KernelJob(name="build-data", entrypoint="magpie.runners.jobs.launch:main", config=cfg,
                     models=[], extra_model_sources=[AYA_KAGGLE_SOURCE],
                     timeout_s=cfg["timeout_s"])


# Pull everything except model weights (they stay in Kaggle for later kernels).
NO_WEIGHTS = r"^(?!.*\.(?:safetensors|bin|pt|pth|gguf)$)"


@job("harden")
def harden_job(**overrides: Any) -> KernelJob:
    """Harden Fire with SafeTune (method=rr | repnoise), then run L0 on it."""
    cfg = {"method": "rr", "base": "fire", "n": 256, "steps": 150, "batch_size": 2,
           "lr": 1e-4, "max_len": 768, "n_harmful": 150, "n_benign": 150,
           "max_new_tokens": 192, "seed": 0, "vllm_spec": "vllm==0.33.0",
           "safetune_spec": "safetune==0.1.12"}
    cfg.update(overrides)
    if cfg["method"] not in ("rr", "repnoise"):
        raise ValueError(f"unknown hardening method {cfg['method']!r}")
    return KernelJob(name=f"harden-{cfg['method']}", entrypoint="magpie.runners.jobs.harden:main",
                     config=cfg, models=["fire"], timeout_s=6 * 3600, pull_pattern=NO_WEIGHTS)


@job("ablation")
def ablation_job(**overrides: Any) -> KernelJob:
    """L3a abliteration on the vendor models (HF backend; needs SafeTune)."""
    cfg = {"models": ["fire", "global"], "n_harmful": 120, "n_benign": 0,
           "max_new_tokens": 192, "seed": 0, "safetune_spec": "safetune==0.1.12",
           "attack_params": {"pick_layer": 24, "n_contrast": 64}, "timeout_s": 4 * 3600}
    cfg.update(overrides)
    model_keys = [m for m in cfg["models"] if m in MODELS]
    return KernelJob(name="ablation", entrypoint="magpie.runners.jobs.l3:main", config=cfg,
                     models=model_keys, timeout_s=cfg["timeout_s"], pull_pattern=NO_WEIGHTS)


@job("rescore")
def rescore_job(**overrides: Any) -> KernelJob:
    """HarmBench-rescore pulled attempts attached as a Kaggle dataset."""
    ds = overrides.pop("attempts_dataset", None)
    if not ds:
        raise ValueError("rescore job needs attempts_dataset=<owner/slug>")
    cfg = {"worker": "magpie.runners.jobs.rescore", "judge": "harmbench", "tp": 2,
           "vllm_spec": "vllm==0.33.0", "timeout_s": 3 * 3600}
    cfg.update(overrides)
    return KernelJob(name="rescore", entrypoint="magpie.runners.jobs.launch:main", config=cfg,
                     models=[], dataset_sources=[ds], timeout_s=cfg["timeout_s"])


@job("mech")
def mech_job(**overrides: Any) -> KernelJob:
    """Extract per-prompt refusal margins on a model (HF; needs SafeTune) for H4."""
    cfg = {"model": "fire", "pick_layer": 24, "n_harmful": 100, "n_contrast": 64, "seed": 0,
           "safetune_spec": "safetune==0.1.12", "timeout_s": 3 * 3600}
    cfg.update(overrides)
    key = cfg["model"] if cfg["model"] in MODELS else "fire"
    return KernelJob(name=f"mech-{cfg['model']}", entrypoint="magpie.runners.jobs.mech:main",
                     config=cfg, models=[key], timeout_s=cfg["timeout_s"], pull_pattern=NO_WEIGHTS)


@job("probe")
def probe_job(**overrides: Any) -> KernelJob:
    """CPU-only: list what model sources actually mount (no GPU quota)."""
    cfg = {"extra_sources": []}
    cfg.update(overrides)
    return KernelJob(name="probe", entrypoint="magpie.runners.jobs.probe:main", config=cfg,
                     models=list(cfg.get("models", [])),
                     extra_model_sources=list(cfg.get("extra_sources", [])),
                     gpu=False, enable_internet=True, timeout_s=1800)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class KernelRun:
    """Local record of one pushed kernel version (written to results/<job>/<ts>/)."""

    job: str
    ref: str
    version: int | None
    url: str | None
    git_sha: str
    git_dirty: bool
    bundle_sha256: str
    pushed_at: str
    status: str = "pushed"
    failure_message: str | None = None
    finished_at: str | None = None
    wall_clock_s: float | None = None
    output_dir: str | None = None
    account: int = 1


class KaggleRunner:
    """Authenticated wrapper over the Kaggle SDK for Magpie's kernel lifecycle."""

    def __init__(self, account: int = 1, credentials: KaggleCredentials | None = None) -> None:
        self.creds = credentials or KaggleCredentials.from_env(account)
        if self.creds.style == "missing":
            raise RuntimeError(
                f"No Kaggle credentials for account {self.creds.account}: set KAGGLE_API_TOKEN "
                "(or KAGGLE_USERNAME + KAGGLE_KEY) in the environment."
            )
        self._api: Any = None

    # -- auth -----------------------------------------------------------------

    @property
    def api(self) -> Any:
        if self._api is None:
            # The SDK reads credentials from os.environ at authenticate(); scope this
            # account's variables to that call so several accounts can coexist.
            clear = ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME")
            with _patched_env(self.creds.env_overrides(), clear=clear):
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
            self._api = api
        return self._api

    @property
    def username(self) -> str:
        user = self.api.get_config_value(self.api.CONFIG_NAME_USER)
        if not user:
            raise RuntimeError("authenticated, but Kaggle did not report a username")
        return str(user)

    # -- quota ----------------------------------------------------------------

    def quota(self) -> dict[str, Any]:
        q = self.api.quota_view()
        out: dict[str, Any] = {"account": self.creds.account, "username": self.username,
                               "refresh": str(q.quota_refresh_time)}
        for name, quota in (("gpu", q.gpu_quota), ("tpu", q.tpu_quota)):
            if quota is None:
                continue
            used = quota.time_used.total_seconds() / 3600
            total = quota.total_time_allowed.total_seconds() / 3600
            out[name] = {"used_h": round(used, 2), "total_h": round(total, 2),
                         "left_h": round(total - used, 2)}
        return out

    # -- build / push ---------------------------------------------------------

    def build(self, kjob: KernelJob) -> tuple[Path, dict[str, Any]]:
        """Write ``kernel.py`` + ``kernel-metadata.json`` into ``.kaggle-build/<slug>/``."""
        bundle = build_bundle()
        folder = BUILD_DIR / kjob.slug
        folder.mkdir(parents=True, exist_ok=True)
        script = render_kernel(
            name=kjob.name, entrypoint=kjob.entrypoint, config=kjob.config,
            bundle_b64=bundle.b64, bundle_sha256=bundle.sha256,
            git_sha=bundle.git_sha, git_dirty=bundle.git_dirty,
        )
        (folder / "kernel.py").write_text(script, encoding="utf-8")
        meta = kjob.metadata(self.username)
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        info = {"git_sha": bundle.git_sha, "git_dirty": bundle.git_dirty,
                "bundle_sha256": bundle.sha256, "bundle_bytes": bundle.size_bytes,
                "n_files": bundle.n_files}
        return folder, info

    def push(self, kjob: KernelJob) -> KernelRun:
        folder, info = self.build(kjob)
        resp = self.api.kernels_push(str(folder), timeout=str(kjob.timeout_s),
                                     acc=kjob.machine_shape if kjob.gpu else None)
        err = getattr(resp, "error", None)
        if err:
            raise RuntimeError(f"Kaggle rejected the push of {kjob.slug}: {err}")
        ref = getattr(resp, "ref", None) or f"{self.username}/{kjob.slug}"
        ref = ref.strip("/")
        if ref.startswith("code/"):
            ref = ref[len("code/"):]
        return KernelRun(
            job=kjob.name, ref=ref, version=getattr(resp, "version_number", None),
            url=getattr(resp, "url", None), git_sha=info["git_sha"],
            git_dirty=info["git_dirty"], bundle_sha256=info["bundle_sha256"],
            pushed_at=_now(), account=self.creds.account,
        )

    # -- poll / pull ----------------------------------------------------------

    def status(self, ref: str) -> tuple[str, str | None]:
        resp = self.api.kernels_status(ref)
        raw = getattr(resp, "status", resp)
        name = getattr(raw, "name", str(raw))
        # Enum names look like KERNEL_WORKER_STATUS_COMPLETE; keep the last word.
        norm = name.lower().replace("kernel_worker_status_", "").replace("_", "")
        return norm, getattr(resp, "failure_message", None) or None

    def wait(self, run: KernelRun, poll_s: int = 30, timeout_s: int = 4 * 3600,
             log: Callable[[str], None] = print) -> KernelRun:
        t0 = time.time()
        last = None
        while True:
            state, msg = self.status(run.ref)
            if state != last:
                log(f"[{_now()}] {run.ref}: {state}" + (f" ({msg})" if msg else ""))
                last = state
            if state in _TERMINAL:
                run.status, run.failure_message = state, msg
                run.finished_at = _now()
                run.wall_clock_s = round(time.time() - t0, 1)
                return run
            if time.time() - t0 > timeout_s:
                run.status = "timeout"
                return run
            time.sleep(poll_s)

    def upload_dataset(self, local_dir: Path, slug: str, title: str | None = None) -> str:
        """Create or version a private Kaggle dataset from ``local_dir``; return its ref.

        Used to ship pulled raw attempts into a HarmBench rescoring kernel. Writes
        the required ``dataset-metadata.json``, then creates the dataset or, if it
        already exists, a new version.
        """
        local = Path(local_dir)
        ref = f"{self.username}/{slug}"
        meta = {"title": (title or slug)[:50], "id": ref, "licenses": [{"name": "CC0-1.0"}]}
        (local / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
        # dir_mode="skip" stores the loose JSONL files directly (not a zip the
        # kernel would have to unpack). Create the dataset, or version it if it
        # already exists.
        exists = False
        try:
            self.api.dataset_status(ref)
            exists = True
        except Exception:  # noqa: BLE001 - not found -> create new
            exists = False
        if exists:
            self.api.dataset_create_version(str(local), version_notes="update",
                                            convert_to_csv=False, dir_mode="skip")
        else:
            self.api.dataset_create_new(str(local), public=False, convert_to_csv=False,
                                        dir_mode="skip")
        return ref

    def pull(self, ref: str, dest: Path, file_pattern: str | None = None) -> list[str]:
        """Download the output files of the latest version of ``ref`` into ``dest``.

        ``file_pattern`` keeps only matching files (e.g. :data:`NO_WEIGHTS`).
        """
        dest.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        token: str | None = None
        while True:
            got, token = self.api.kernels_output(ref, str(dest), file_pattern=file_pattern,
                                                 force=True, quiet=True, page_token=token)
            files.extend(got or [])
            if not token:
                break
        return files

    def run(self, kjob: KernelJob, poll_s: int = 30, log: Callable[[str], None] = print,
            wait: bool = True) -> KernelRun:
        """Push → (wait → pull). Writes ``results/<job>/<ts>/_run.json``."""
        run = self.push(kjob)
        log(f"pushed {run.ref} v{run.version} (git {run.git_sha}"
            f"{', dirty' if run.git_dirty else ''}) {run.url or ''}")
        dest = RESULTS_DIR / kjob.name / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run.output_dir = str(dest.relative_to(REPO_ROOT))
        if wait:
            run = self.wait(run, poll_s=poll_s, timeout_s=kjob.timeout_s + 3600, log=log)
            if run.status in ("complete", "error"):
                files = self.pull(run.ref, dest, kjob.pull_pattern)
                log(f"pulled {len(files)} file(s) into {run.output_dir}")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "_run.json").write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        return run


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m magpie.runners.kaggle")
    p.add_argument("--account", type=int, default=1, help="1 = KAGGLE_*, 2 = KAGGLE_*_2, ...")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quota")
    sub.add_parser("jobs")
    for name in ("run", "build"):
        sp = sub.add_parser(name)
        sp.add_argument("job", choices=sorted(JOBS))
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a job config value (JSON-parsed when possible)")
        if name == "run":
            sp.add_argument("--no-wait", action="store_true")
            sp.add_argument("--poll", type=int, default=30)
    for name in ("status", "pull"):
        sp = sub.add_parser(name)
        sp.add_argument("job")
        sp.add_argument("--dest", default=None)
    rs = sub.add_parser("rescore")
    rs.add_argument("run_dirs", nargs="+", help="result dirs with raw/attempts_*.jsonl")
    rs.add_argument("--slug", default="magpie-attempts", help="Kaggle dataset slug to upload to")
    rs.add_argument("--poll", type=int, default=30)
    rs.add_argument("--no-wait", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "jobs":
        for name, factory in sorted(JOBS.items()):
            print(f"{name:12} {factory.__doc__.strip().splitlines()[0] if factory.__doc__ else ''}")
        return 0
    runner = KaggleRunner(account=args.account)
    if args.cmd == "quota":
        print(json.dumps(runner.quota(), indent=2))
        return 0
    if args.cmd == "rescore":
        import shutil

        stage = BUILD_DIR / "attempts-upload"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        n = 0
        for rd in args.run_dirs:
            raw = Path(rd) / "out" / "raw"
            raw = raw if raw.is_dir() else Path(rd) / "raw"
            tag = Path(rd).parent.name  # e.g. attack-l1-bon-input
            for p_ in sorted(raw.glob("attempts_*.jsonl")):
                shutil.copy(p_, stage / f"attempts_{tag}_{p_.stem[len('attempts_'):]}.jsonl")
                n += 1
        print(f"staged {n} attempt files from {len(args.run_dirs)} run(s)")
        ref = runner.upload_dataset(stage, args.slug, title="Magpie attempts for rescoring")
        print(f"uploaded dataset {ref}")
        kjob = JOBS["rescore"](attempts_dataset=ref)
        run = runner.run(kjob, poll_s=args.poll, wait=not args.no_wait)
        print(json.dumps(asdict(run), indent=2))
        return 0 if run.status in ("complete", "pushed") else 1
    if args.cmd in ("run", "build"):
        overrides: dict[str, Any] = {}
        for kv in args.set:
            k, _, v = kv.partition("=")
            try:
                overrides[k] = json.loads(v)
            except json.JSONDecodeError:
                overrides[k] = v
        kjob = JOBS[args.job](**overrides)
        if args.cmd == "build":
            folder, info = runner.build(kjob)
            print(json.dumps({"folder": str(folder), **info}, indent=2))
            return 0
        run = runner.run(kjob, poll_s=args.poll, wait=not args.no_wait)
        print(json.dumps(asdict(run), indent=2))
        return 0 if run.status in ("complete", "pushed") else 1
    ref = args.job if "/" in args.job else f"{runner.username}/magpie-{args.job}"
    if args.cmd == "status":
        state, msg = runner.status(ref)
        print(json.dumps({"ref": ref, "status": state, "message": msg}))
        return 0
    dest = Path(args.dest) if args.dest else (
        RESULTS_DIR / ref.split("/")[-1].removeprefix("magpie-")
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    files = runner.pull(ref, dest)
    print(json.dumps({"ref": ref, "dest": str(dest), "files": len(files)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
