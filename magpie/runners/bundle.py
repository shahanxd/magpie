"""Ship the magpie source to a remote kernel as one deterministic zip.

Kaggle kernels can't see this repo (it may be private, and a run must pin an
exact revision anyway). Instead the runner zips the ``magpie`` package as it
is on disk, stamps it with the git SHA and a content hash, and embeds it in the
kernel script. The kernel unpacks it onto ``sys.path``; nothing is
``pip install``-ed, so Kaggle's preinstalled CUDA stack is left alone.
"""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from magpie.config import REPO_ROOT, git_sha

# Fixed timestamp so identical sources give byte-identical zips (stable hashes).
_EPOCH = (2026, 1, 1, 0, 0, 0)
_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class Bundle:
    """A packed source tree: base64 zip + provenance."""

    b64: str
    sha256: str
    git_sha: str
    git_dirty: bool
    n_files: int

    @property
    def size_bytes(self) -> int:
        return len(self.b64) * 3 // 4


def _iter_files(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() or any(part in _EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix in _EXCLUDE_SUFFIXES:
            continue
        files.append(p)
    return files


def build_bundle(package_dir: Path | None = None) -> Bundle:
    """Zip ``magpie/`` (relative paths ``magpie/...``) deterministically."""
    package_dir = package_dir or (REPO_ROOT / "magpie")
    base = package_dir.parent
    buf = io.BytesIO()
    files = _iter_files(package_dir)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(base)), date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    raw = buf.getvalue()
    sha, dirty = git_sha()
    return Bundle(
        b64=base64.b64encode(raw).decode("ascii"),
        sha256=hashlib.sha256(raw).hexdigest(),
        git_sha=sha,
        git_dirty=dirty,
        n_files=len(files),
    )


def unpack_bundle(b64: str, dest: Path) -> Path:
    """Inverse of :func:`build_bundle`; returns the directory to put on ``sys.path``."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        zf.extractall(dest)
    return dest
