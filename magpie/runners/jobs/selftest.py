"""A no-GPU job that proves the kernel bootstrap works (bundle, imports, outputs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from magpie.config import git_sha
from magpie.runners.jobs.common import write_json


def main(config: dict[str, Any], out_dir: Path) -> None:
    import magpie

    sha, dirty = git_sha()
    write_json(Path(out_dir) / "selftest.json",
               {"magpie_version": magpie.__version__, "git_sha": sha, "git_dirty": dirty,
                "echo": config.get("echo")})
