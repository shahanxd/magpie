"""Kaggle runner pieces that need no network: bundle, kernel render, credentials."""

import json
import subprocess
import sys

import pytest

from magpie.runners.bundle import build_bundle, unpack_bundle
from magpie.runners.kaggle import JOBS, KaggleCredentials, KernelJob
from magpie.runners.kernel_template import render_kernel


def test_bundle_is_deterministic_and_roundtrips(tmp_path):
    a, b = build_bundle(), build_bundle()
    assert a.sha256 == b.sha256 and a.n_files > 10
    root = unpack_bundle(a.b64, tmp_path / "src")
    assert (root / "magpie" / "costs.py").is_file()
    assert not list(root.rglob("__pycache__"))


def test_rendered_kernel_runs_the_job_end_to_end(tmp_path):
    bundle = build_bundle()
    out = tmp_path / "out"
    script = render_kernel(
        name="selftest", entrypoint="magpie.runners.jobs.selftest:main", config={"echo": 42},
        bundle_b64=bundle.b64, bundle_sha256=bundle.sha256, git_sha="deadbee",
        git_dirty=False, out_dir=str(out),
    )
    compile(script, "kernel.py", "exec")  # valid Python
    path = tmp_path / "kernel.py"
    path.write_text(script)
    # Run in a clean interpreter *outside* the repo so only the bundle can satisfy imports.
    proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    status = json.loads((out / "_status.json").read_text())
    assert status["ok"], status.get("error", proc.stderr)
    got = json.loads((out / "selftest.json").read_text())
    assert got == {"magpie_version": "0.1.0", "git_sha": "deadbee", "git_dirty": False,
                   "echo": 42}
    assert "MAGPIE_STATUS" in proc.stdout


def test_kernel_reports_job_errors_without_crashing(tmp_path):
    bundle = build_bundle()
    out = tmp_path / "out"
    script = render_kernel(
        name="bad", entrypoint="magpie.runners.jobs.selftest:missing", config={},
        bundle_b64=bundle.b64, bundle_sha256=bundle.sha256, git_sha="x", git_dirty=True,
        out_dir=str(out),
    )
    path = tmp_path / "kernel.py"
    path.write_text(script)
    proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0
    status = json.loads((out / "_status.json").read_text())
    assert status["ok"] is False and "AttributeError" in status["error"]


@pytest.mark.parametrize(
    ("env", "style", "token"),
    [
        ({"KAGGLE_API_TOKEN": "KGAT_new", "KAGGLE_KEY": "KGAT_copy"}, "token", "KGAT_new"),
        ({"KAGGLE_KEY": "KGAT_only_here", "KAGGLE_USERNAME": "u"}, "token", "KGAT_only_here"),
        ({"KAGGLE_KEY": "0123456789abcdef0123456789abcdef", "KAGGLE_USERNAME": "u"},
         "legacy", None),
        ({}, "missing", None),
    ],
)
def test_credentials_prefer_api_token_then_key(env, style, token):
    c = KaggleCredentials.from_env(env=env)
    assert c.style == style and c.token == token
    if style == "legacy":
        assert c.env_overrides() == {"KAGGLE_USERNAME": "u",
                                     "KAGGLE_KEY": "0123456789abcdef0123456789abcdef"}


def test_extra_accounts_use_suffixed_variables():
    env = {"KAGGLE_API_TOKEN": "one", "KAGGLE_USERNAME_2": "two", "KAGGLE_API_TOKEN_2": "tok2"}
    c = KaggleCredentials.from_env(account=2, env=env)
    assert (c.username, c.token, c.account) == ("two", "tok2", 2)


def test_build_data_job_attaches_translation_model():
    job = JOBS["build_data"]()
    meta = job.metadata("u")
    # v1 uses tiny-aya Fire (ungated); Aya Expanse is gated on Kaggle.
    assert meta["model_sources"] == ["cohere-labs/tiny-aya/transformers/fire/1"]
    assert job.config["translate_model"] == "fire"
    assert job.config["worker"] == "magpie.runners.jobs.build_data"
    assert job.entrypoint == "magpie.runners.jobs.launch:main"


def test_l0_job_uses_shared_launcher():
    job = JOBS["l0"]()
    assert job.entrypoint == "magpie.runners.jobs.launch:main"
    assert job.config["worker"] == "magpie.runners.jobs.l0"
    assert set(job.metadata("u")["model_sources"]) == {
        "cohere-labs/tiny-aya/transformers/fire/1",
        "cohere-labs/tiny-aya/transformers/global/1"}


def test_smoke_job_metadata():
    job = JOBS["smoke"]()
    assert isinstance(job, KernelJob)
    meta = job.metadata("someone")
    assert meta["id"] == "someone/magpie-smoke" and meta["title"] == "magpie-smoke"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["model_sources"] == ["cohere-labs/tiny-aya/transformers/fire/1"]
    assert JOBS["smoke"](max_new_tokens=16).config["max_new_tokens"] == 16
