import io
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from tests.fast.tools.air.conftest import copy_save
from tools.air import dcp_stage
from tools.air import fuse_transfer as fuse
from tools.air import publisher, resume, stage_models
from torch.distributed.checkpoint import FileSystemReader, load


def _stager(remote, local, document, state, limiter):
    return dcp_stage.ReadPlanStager(
        [{"remote_root": str(remote), "local_root": str(local), "document": document}], state, limiter
    )


def test_real_read_plan_stages_only_requested_model_shards(reference_root, tmp_path, monkeypatch):
    remote, document = reference_root
    local = tmp_path / "local/reference"
    limiter = fuse.NodeTransferLimiter(tmp_path / "slots", 2)
    stage_models.stage_checkpoint(remote, local, document, limiter)
    assert not list(local.rglob("*.distcp"))
    monkeypatch.setattr(FileSystemReader, "read_data", FileSystemReader.read_data)
    dcp_stage.install_reader_hook(_stager(remote, local, document, tmp_path / "state", limiter))
    target = {"model.weight": torch.empty(4, dtype=torch.int64)}
    load(target, storage_reader=FileSystemReader(local / "release"), no_dist=True)
    assert torch.equal(target["model.weight"], torch.arange(4))
    metadata = dcp_stage.read_metadata(remote / "release")
    requested = {info.relative_path for index, info in metadata.storage_data.items() if index.fqn == "model.weight"}
    assert {p.name for p in local.rglob("*.distcp")} == requested


def test_two_rank_caches_deduplicate_on_the_node(reference_root, tmp_path, monkeypatch):
    remote, document = reference_root
    local, state = tmp_path / "local/reference", tmp_path / "state"
    limiter = fuse.NodeTransferLimiter(tmp_path / "slots", 2)
    names = [f["path"].removeprefix("release/") for f in document["files"] if f["path"].endswith(".distcp")]
    copies = []
    original = fuse._stream_copy

    def copy(source, destination, **kwargs):
        copies.append(source)
        return original(source, destination, **kwargs)

    monkeypatch.setattr(fuse, "_stream_copy", copy)
    stagers = [_stager(remote, local, document, state, limiter) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda stager: stager.stage(local / "release", names + names), stagers))
    assert len(copies) == len(names)
    for name in names:
        assert (local / "release" / name).read_bytes() == (remote / "release" / name).read_bytes()


def test_published_save_restores_full_state_through_the_resume_read_plan(distributed_save, tmp_path, monkeypatch):
    checkpoint = distributed_save()
    publisher.finalize_save(
        checkpoint.target, copy_save(checkpoint), run_id="run", attempt_id="attempt", rollout_id=3, configuration={}
    )
    remote, document = resume.latest_complete(checkpoint.target.parents[4], "run")
    local = tmp_path / "local/resume"
    limiter = fuse.NodeTransferLimiter(tmp_path / "slots", 2)
    stage_models.stage_checkpoint(remote, local, document, limiter)
    monkeypatch.setattr(FileSystemReader, "read_data", FileSystemReader.read_data)
    dcp_stage.install_reader_hook(_stager(remote, local, document, tmp_path / "state", limiter))
    restored = {
        "model.weight": torch.full((4,), -1, dtype=torch.int64),
        "optimizer.exp_avg": torch.full((4,), -1.0),
        "common_state/shard_0_1": io.BytesIO(),
    }
    load(restored, storage_reader=FileSystemReader(local / "iter_0000003"), no_dist=True)
    assert torch.equal(restored["model.weight"], torch.arange(4))
    assert torch.equal(restored["optimizer.exp_avg"], torch.zeros(4))
    restored["common_state/shard_0_1"].seek(0)
    common = torch.load(restored["common_state/shard_0_1"], weights_only=False)[0]
    assert common["iteration"] == 3
    assert common["opt_param_scheduler"] == checkpoint.common["opt_param_scheduler"]
    assert torch.equal(common["rng_state"]["torch_rng_state"], checkpoint.common["rng_state"]["torch_rng_state"])
    cursor = torch.load(local / "rollout/global_dataset_state_dict_3.pt", weights_only=False)
    assert cursor["sample_offset"] == 2
    assert cursor["metadata"]["curriculum"]["phase"] == "hard"


def test_plan_cannot_request_an_uninventoried_shard(reference_root, tmp_path):
    remote, document = reference_root
    local = tmp_path / "local/reference"
    stager = _stager(remote, local, document, tmp_path / "state", fuse.NodeTransferLimiter(tmp_path / "slots", 1))
    with pytest.raises(ValueError, match="absent from the manifest"):
        stager.stage(local / "release", ["absent.distcp"])
    with pytest.raises(ValueError, match="invalid artifact path"):
        stager.stage(local / "release", ["../escape"])
    stager.stage(tmp_path / "unmanaged", ["unrelated.distcp"])
    assert not local.exists()


def test_initialization_marks_actual_rank_and_installs_the_reader(reference_root, tmp_path, monkeypatch):
    from types import SimpleNamespace

    remote, document = reference_root
    state = tmp_path / "state"
    local = state / "reference"
    path = tmp_path / "runtime.json"
    path.write_bytes(
        stage_models.json_bytes(
            {
                "state_root": str(state),
                "slot_root": str(tmp_path / "slots"),
                "workers": 2,
                "sources": [{"remote_root": str(remote), "local_root": str(local), "document": document}],
            }
        )
    )
    monkeypatch.setenv("MILES_CHECKPOINT_CONFIG", str(path))
    monkeypatch.setattr(FileSystemReader, "read_data", FileSystemReader.read_data)
    dcp_stage.initialize(SimpleNamespace(rank=17, world_size=32))
    assert stage_models.read_json(state / "trainer-ranks/17.json") == {"rank": 17, "world_size": 32}
