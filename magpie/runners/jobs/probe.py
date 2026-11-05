"""CPU-only probe: report which attached model sources actually mount, and where."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from magpie.runners.jobs.common import write_json

ROOT = Path("/kaggle/input")


def main(config: dict[str, Any], out_dir: Path) -> None:
    found: list[dict[str, Any]] = []
    tree: list[str] = []
    if ROOT.is_dir():
        for dirpath, dirnames, filenames in os.walk(ROOT):
            depth = len(Path(dirpath).relative_to(ROOT).parts)
            if depth <= 6:
                tree.append(f"{dirpath} ({len(filenames)} files)")
            if "config.json" in filenames:
                size = sum((Path(dirpath) / f).stat().st_size for f in filenames)
                found.append({"dir": dirpath, "n_files": len(filenames),
                              "size_gb": round(size / 1e9, 2)})
            if depth >= 7:
                dirnames[:] = []
    write_json(Path(out_dir) / "probe.json", {"root_exists": ROOT.is_dir(), "model_dirs": found,
                                              "tree": tree[:200]})
    print(f"model dirs: {found}", flush=True)
