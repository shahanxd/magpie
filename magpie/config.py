"""Run configuration, seeding and provenance.

Every result Magpie produces must be traceable to a config and a git SHA
(PLAN §11 quality bar). :class:`RunConfig` is loaded from YAML in
``configs/``; :func:`provenance` stamps a run with the config hash, git SHA,
dirty flag and library versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Model registry (PLAN §4). Kaggle paths come from the Kaggle Models mirror
# (``cohere-labs/tiny-aya``), so GPU notebooks need no HF token.
MODELS: dict[str, dict[str, str]] = {
    "fire": {
        "hf_id": "CohereLabs/tiny-aya-fire",
        "kaggle_source": "cohere-labs/tiny-aya/transformers/fire/1",
        "kaggle_path": "/kaggle/input/tiny-aya/transformers/fire/1",
        "role": "M0 main target (vendor-hardened, South-Asia variant)",
    },
    "global": {
        "hf_id": "CohereLabs/tiny-aya-global",
        "kaggle_source": "cohere-labs/tiny-aya/transformers/global/1",
        "kaggle_path": "/kaggle/input/tiny-aya/transformers/global/1",
        "role": "M1 comparison (vendor's safest Tiny Aya)",
    },
    "base": {
        "hf_id": "CohereLabs/tiny-aya-base",
        "kaggle_source": "cohere-labs/tiny-aya/transformers/base/1",
        "kaggle_path": "/kaggle/input/tiny-aya/transformers/base/1",
        "role": "pretrained base, reference for KL / safety depth",
    },
}


@dataclass
class GenerationConfig:
    """Decoding settings. Greedy by default; L1d sweeps temperature / top-p."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    # Tiny Aya's chat template adds a ~370-token system preamble; never let a
    # context cap truncate the prompt. 2048 leaves room for long prompts.
    max_model_len: int = 2048

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class RunConfig:
    """One experiment run: which models, which data, which attack, which judges."""

    name: str
    models: list[str] = field(default_factory=lambda: ["fire"])
    forms: list[str] = field(default_factory=lambda: ["en", "hing"])
    attack: str = "l0_direct"
    attack_params: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judges: list[str] = field(default_factory=lambda: ["refusal_lexicon"])
    backend: str = "vllm"  # vllm | hf | echo
    dtype: str = "float16"  # T4 has no bf16
    seed: int = 0
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        gen = GenerationConfig(**(raw.pop("generation", None) or {}))
        return cls(generation=gen, **raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if present) torch. Fixed seeds everywhere."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_sha(short: bool = True) -> tuple[str, bool]:
    """Return ``(sha, dirty)`` for the repo, or ``("unknown", False)`` outside git.

    Inside a Kaggle kernel there is no ``.git``; the runner injects
    ``MAGPIE_GIT_SHA`` / ``MAGPIE_GIT_DIRTY`` so provenance survives the trip.
    """
    env_sha = os.environ.get("MAGPIE_GIT_SHA")
    if env_sha:
        return env_sha, os.environ.get("MAGPIE_GIT_DIRTY", "0") == "1"
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        sha = subprocess.check_output(args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
                                      text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def provenance(config: RunConfig | None = None, **extra: Any) -> dict[str, Any]:
    """A provenance stamp for a run: git SHA, config hash, library versions."""
    from importlib.metadata import PackageNotFoundError, version

    sha, dirty = git_sha()
    libs: dict[str, str] = {}
    for lib in ("magpie", "torch", "transformers", "vllm", "safetune", "auditkit",
                "curatorkit", "circuitkit"):
        try:
            libs[lib] = version(lib)
        except PackageNotFoundError:
            continue
    out: dict[str, Any] = {
        "git_sha": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "libraries": libs,
    }
    if config is not None:
        out["config_hash"] = config.config_hash
        out["config_name"] = config.name
    out.update(extra)
    return out
