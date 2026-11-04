
from magpie.runners.hub import push_dataset


def test_push_dataset_dry_run_lists_files(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "records.jsonl").write_text("{}\n")
    res = push_dataset(tmp_path, "me/magpie-core-v1", dry_run=True)
    assert res["repo_id"] == "me/magpie-core-v1" and res["private"] is True
    assert set(res["would_upload"]) == {"manifest.json", "raw/records.jsonl"}


def test_push_dataset_missing_dir(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        push_dataset(tmp_path / "nope", "me/x", dry_run=True)
