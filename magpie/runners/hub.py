"""Publish datasets, models and the Space to the Hugging Face Hub.

Runs in the dev environment (which holds ``HF_TOKEN``), never in a Kaggle
kernel — so the write token never leaves here. The data-build job writes the
dataset payload under ``results/.../raw/``; this uploads it to a **private**
dataset repo (gating is configured on the Hub afterwards). Attacked or
abliterated weights are never published (PLAN §16); only hardened variants and
the dataset/scores/logs go up.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_OWNER_ENV = "MAGPIE_HF_OWNER"


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in the environment")
    return HfApi(token=token), token


def whoami() -> dict[str, Any]:
    api, _ = _api()
    return api.whoami()


def push_folder(local_dir: str | Path, repo_id: str, repo_type: str = "dataset",
                private: bool = True, path_in_repo: str = "",
                commit_message: str = "magpie upload", allow_patterns: list[str] | None = None,
                dry_run: bool = False) -> dict[str, Any]:
    """Create ``repo_id`` (if needed) and upload ``local_dir`` into it.

    ``private=True`` by default; harmful datasets go up private and are gated on
    the Hub before any sharing. ``dry_run`` lists what would upload without
    touching the Hub.
    """
    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(local)
    files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
    if dry_run:
        return {"repo_id": repo_id, "repo_type": repo_type, "private": private,
                "would_upload": files, "n_files": len(files)}
    api, _ = _api()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(local),
                      path_in_repo=path_in_repo, commit_message=commit_message,
                      allow_patterns=allow_patterns)
    return {"repo_id": repo_id, "repo_type": repo_type, "private": private,
            "uploaded": len(files), "url": f"https://huggingface.co/{repo_type}s/{repo_id}"}


def push_dataset(local_dir: str | Path, repo_id: str, private: bool = True,
                 dry_run: bool = False) -> dict[str, Any]:
    """Upload a built dataset directory to a (private) HF dataset repo."""
    return push_folder(local_dir, repo_id, repo_type="dataset", private=private,
                       commit_message="magpie dataset v1", dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(prog="python -m magpie.runners.hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    pd = sub.add_parser("push-dataset")
    pd.add_argument("local_dir")
    pd.add_argument("repo_id")
    pd.add_argument("--public", action="store_true")
    pd.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "whoami":
        print(json.dumps({"name": whoami().get("name")}))
        return 0
    res = push_dataset(args.local_dir, args.repo_id, private=not args.public, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
