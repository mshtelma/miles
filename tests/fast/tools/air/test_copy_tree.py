"""copy_tree copies a directory with parallel shutil copies; the files it is told to copy last go last."""

import shutil

import pytest
from tools.air.copy_tree import copy_tree


def _tree(root):
    (root / "a/b").mkdir(parents=True)
    (root / "a/b/x.bin").write_bytes(b"x" * 100)
    (root / "y.txt").write_text("y")
    (root / "latest_checkpointed_iteration.txt").write_text("release")
    return root


def test_every_file_keeps_its_relative_path(tmp_path):
    copy_tree(_tree(tmp_path / "src"), tmp_path / "dst")

    assert (tmp_path / "dst/a/b/x.bin").read_bytes() == b"x" * 100
    assert (tmp_path / "dst/y.txt").read_text() == "y"


def test_the_files_named_last_are_copied_after_the_rest(tmp_path, monkeypatch):
    order, copyfile = [], shutil.copyfile
    monkeypatch.setattr(
        shutil, "copyfile", lambda source, target: order.append(target.name) or copyfile(source, target)
    )

    copy_tree(_tree(tmp_path / "src"), tmp_path / "dst", last=("latest_checkpointed_iteration.txt",))

    assert order[-1] == "latest_checkpointed_iteration.txt"
    assert sorted(order) == ["latest_checkpointed_iteration.txt", "x.bin", "y.txt"]


def test_a_missing_source_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        copy_tree(tmp_path / "missing", tmp_path / "dst")
