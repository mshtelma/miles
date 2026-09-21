import json
import os
from pathlib import Path

import pytest
import torch

from miles.backends.training_utils.artifact_io import ArtifactStore


def test_artifact_store_publishes_staged_directory(tmp_path):
    store = ArtifactStore()
    destination = tmp_path / "checkpoint"

    with store.staging_dir(destination) as staging:
        store.atomic_write_bytes(staging / "payload.bin", b"payload")
    store.publish(destination, metadata={"version": 3}, marker="READY")

    assert (destination / "payload.bin").read_bytes() == b"payload"
    assert json.loads((destination / "META.json").read_text()) == {"version": 3}
    assert (destination / "READY").exists()


def test_artifact_store_writes_and_verifies_manifests_and_trackers(tmp_path):
    store = ArtifactStore()
    manifest = {"version": 4, "files": {"model.safetensors": "sha256:abc"}}

    store.write_manifest(tmp_path / "manifest.json", manifest)
    store.verify_manifest(tmp_path / "manifest.json", manifest)
    store.write_tracker(tmp_path / "latest", 4)

    assert (tmp_path / "latest").read_text() == "4"


def test_artifact_store_writes_hf_index_and_copies_metadata(tmp_path):
    store = ArtifactStore()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text("old index")

    store.copy_hf_metadata(source, target)
    store.write_hf_index(
        target / "model.safetensors.index.json", {"layer.weight": "model-00001.safetensors"}, total_size=7
    )

    assert (target / "config.json").exists()
    assert (target / "tokenizer.json").exists()
    assert not (target / "model.safetensors").exists()
    assert json.loads((target / "model.safetensors.index.json").read_text()) == {
        "metadata": {"total_size": 7},
        "weight_map": {"layer.weight": "model-00001.safetensors"},
    }


def test_artifact_store_writes_torch_files_atomically(tmp_path):
    path = tmp_path / "state.pt"

    ArtifactStore().atomic_torch_save(path, {"step": 4})

    assert torch.load(path, weights_only=True) == {"step": 4}


def test_publish_failure_removes_orphaned_version_and_staging(tmp_path, monkeypatch):
    store = ArtifactStore()
    destination = tmp_path / "checkpoint"
    with store.staging_dir(destination) as staging:
        (staging / "payload").write_text("payload")

    real_replace = os.replace

    def fail_after_version_move(source, target):
        real_replace(source, target)
        if Path(target).name.startswith("_version_checkpoint_"):
            raise OSError("publish link failed")

    monkeypatch.setattr(os, "replace", fail_after_version_move)

    with pytest.raises(OSError, match="publish link failed"):
        store.publish(destination)

    assert not (tmp_path / "_tmp_checkpoint").exists()
    assert not list(tmp_path.glob("_version_checkpoint_*"))
