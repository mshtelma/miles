import json
from types import SimpleNamespace

import pytest
import torch
from tests.fast.tools.air.conftest import copy_save, save_inventory

from tools.air import fuse_transfer as fuse
from tools.air import publisher, stage_models


def _finalize(checkpoint, files, **kwargs):
    return publisher.finalize_save(
        checkpoint.target,
        files,
        run_id="run",
        attempt_id=checkpoint.attempt,
        rollout_id=checkpoint.step,
        configuration={"model": {"revision": "a"}},
        **kwargs,
    )


def test_save_publishes_the_union_and_complete_is_last(distributed_save, monkeypatch):
    checkpoint = distributed_save()
    writes = []
    original = fuse.write_bytes

    def write(path, data, **kwargs):
        writes.append(path.name)
        if path.name == "COMPLETE.json":
            document = stage_models.read_json(checkpoint.target / "manifest.json")
            for entry in document["files"]:
                fuse.verify(checkpoint.target / entry["path"], size_bytes=entry["size_bytes"], sha256=entry["sha256"])
        return original(path, data, **kwargs)

    monkeypatch.setattr(fuse, "write_bytes", write)
    files = copy_save(checkpoint)
    document = _finalize(checkpoint, files)

    assert writes == ["manifest.json", "COMPLETE.json"]
    assert stage_models.read_complete(checkpoint.target) == document
    assert {"iteration", "opt_param_scheduler", "rng_state"} <= set(document["common_keys"])
    cursor = torch.load(
        checkpoint.target / f"rollout/global_dataset_state_dict_{checkpoint.step}.pt", weights_only=False
    )
    assert cursor["sample_offset"] == 2
    assert cursor["metadata"]["curriculum"] == {"phase": "hard", "steps": 3}
    assert publisher.inventory_node(checkpoint.roots["serving"], checkpoint.step) == []


def test_missing_node_inventory_fails(distributed_save):
    checkpoint = distributed_save()
    with pytest.raises(ValueError, match="missing a node"):
        publisher.merge_inventories(
            {"trainer": publisher.inventory_node(checkpoint.roots["trainer"], 3)}, list(checkpoint.roots)
        )


def test_missing_referenced_node_file_never_completes(distributed_save):
    checkpoint = distributed_save()
    next(checkpoint.roots["rollout"].glob("iter_*/*.distcp")).unlink()
    files = copy_save(checkpoint)
    with pytest.raises(ValueError, match="storage reference is missing"):
        _finalize(checkpoint, files)
    assert not (checkpoint.target / "COMPLETE.json").exists()


@pytest.mark.parametrize("missing", ["cursor", "tracker", "metadata"])
def test_required_controls_cannot_be_omitted(distributed_save, missing):
    checkpoint = distributed_save(cursor=missing != "cursor")
    if missing == "tracker":
        for root in checkpoint.roots.values():
            (root / "latest_checkpointed_iteration.txt").unlink(missing_ok=True)
    if missing == "metadata":
        for root in checkpoint.roots.values():
            (root / f"iter_{checkpoint.step:07d}/.metadata").unlink(missing_ok=True)
    with pytest.raises(ValueError, match="missing required files"):
        publisher.require_save_files(save_inventory(checkpoint), checkpoint.step)


def test_conflicting_duplicate_metadata_fails_before_copy(distributed_save):
    checkpoint = distributed_save()
    (checkpoint.roots["rollout"] / "iter_0000003/metadata.json").write_text('{"different":true}')
    with pytest.raises(ValueError, match="conflicting duplicate"):
        save_inventory(checkpoint)
    assert not checkpoint.target.exists()


@pytest.mark.parametrize("field,value", [("size_bytes", 1), ("sha256", "0" * 64)])
def test_copy_rejects_wrong_size_or_hash(distributed_save, field, value):
    checkpoint = distributed_save()
    files = [f for f in save_inventory(checkpoint) if f["owner"] == "trainer"]
    files[0] = {**files[0], field: value}
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        publisher.copy_node(
            checkpoint.roots["trainer"], checkpoint.target, checkpoint.roots["trainer"] / "slots", 2, files=files
        )
    assert not (checkpoint.target / "COMPLETE.json").exists()


@pytest.mark.parametrize("part", ["optimizer", "scheduler", "rng"])
def test_resume_state_is_required(distributed_save, part):
    checkpoint = distributed_save(**{part: False})
    with pytest.raises(ValueError, match=part.upper() if part == "rng" else part):
        _finalize(checkpoint, copy_save(checkpoint))
    assert not (checkpoint.target / "COMPLETE.json").exists()


def test_hf_export_is_opaque_and_uses_only_receipt_status(distributed_save):
    checkpoint = distributed_save()
    iteration = checkpoint.roots["trainer"] / "iter_0000003"
    (iteration / "hf").mkdir()
    (iteration / "hf/opaque.bin").write_bytes(b"export")
    (iteration / "hf_export.json").write_text('{"status":"ok","arbitrary":{"future":"fields"}}')
    document = _finalize(checkpoint, copy_save(checkpoint), save_hf=True)
    assert document["hf_export_status"] == "ok"
    assert (checkpoint.target / "iter_0000003/hf/opaque.bin").read_bytes() == b"export"


def test_requested_hf_export_cannot_be_missing(distributed_save):
    checkpoint = distributed_save()
    with pytest.raises(ValueError, match="HF export is missing"):
        _finalize(checkpoint, copy_save(checkpoint), save_hf=True)
    assert not (checkpoint.target / "COMPLETE.json").exists()


def test_pruning_keeps_two_completed_generations_and_partial_saves(tmp_path):
    for step in (1, 2, 3, 4, 99):
        (tmp_path / f"iter_{step:07d}").mkdir()
        path = tmp_path / f"rollout/global_dataset_state_dict_{step}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"cursor")
    assert publisher.prune_node(tmp_path, [1, 2, 3, 4]) == [1, 2]
    assert sorted(p.name for p in tmp_path.glob("iter_*")) == ["iter_0000003", "iter_0000004", "iter_0000099"]
    assert not (tmp_path / "rollout/global_dataset_state_dict_1.pt").exists()


def test_capacity_uses_actual_trainer_ranks_and_hf_writer(tmp_path):
    config = {
        "state_root": str(tmp_path),
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "reference_bytes": 1000,
        "hf_bytes": 500,
        "save_hf": True,
    }
    serving = publisher.capacity_node(config, [])
    ranks = tmp_path / "trainer-ranks"
    ranks.mkdir()
    for rank in (2, 3):
        (ranks / f"{rank}.json").write_text(json.dumps({"rank": rank, "world_size": 8}))
    trainer = publisher.capacity_node(config, [])
    assert trainer["trainer_ranks"] == 2
    assert trainer["required_bytes"] - serving["required_bytes"] == 1750
    (ranks / "2.json").rename(ranks / "0.json")
    (ranks / "0.json").write_text('{"rank":0,"world_size":8}')
    writer = publisher.capacity_node(config, [])
    assert writer["required_bytes"] - trainer["required_bytes"] == 500
    iteration = tmp_path / "checkpoints/iter_0000001"
    iteration.mkdir(parents=True)
    (iteration / "weights").write_bytes(b"x" * 1000)
    measured = publisher.capacity_node(config, [1])
    assert measured["required_bytes"] - serving["required_bytes"] == 1100


async def test_driver_rpc_publication_prunes_only_after_complete(distributed_save, tmp_path, monkeypatch):
    checkpoint = distributed_save()
    config = {
        "node_ids": list(checkpoint.roots),
        "checkpoint_root": str(checkpoint.roots["trainer"]),
        "state_root": str(tmp_path),
        "slot_root": str(tmp_path / "slots"),
        "volume_root": str(tmp_path / "volume"),
        "run_id": "run",
        "attempt_id": "attempt",
        "workers": 2,
        "save_hf": False,
        "configuration": {
            "lr_schedule": {
                "num_rollout": 100,
                "global_batch_size": 8,
                "rollout_batch_size": 4,
                "n_samples_per_prompt": 2,
            }
        },
    }
    path = tmp_path / "runtime.json"
    path.write_bytes(stage_models.json_bytes(config))
    monkeypatch.setenv("MILES_CHECKPOINT_CONFIG", str(path))
    calls = []

    async def local_rpc(function, node_ids, *args, per_node=None, **kwargs):
        calls.append((function.__name__, node_ids))
        results = []
        for node in node_ids:
            local_args = (checkpoint.roots[node], *args[1:])
            if function == publisher.prune_node:
                assert (checkpoint.target / "COMPLETE.json").exists()
            results.append(function(*local_args, **(per_node or {}).get(node, {})))
        return results

    monkeypatch.setattr(publisher, "on_nodes", local_rpc)

    async def local_io(function, *args, **kwargs):
        return function(*args, **kwargs)

    # The simulated node RPCs and driver I/O run inline over real files.
    monkeypatch.setattr(publisher.asyncio, "to_thread", local_io)
    args = SimpleNamespace(
        save=str(checkpoint.roots["trainer"]),
        ckpt_format="torch_dist",
        no_save_optim=False,
        no_save_rng=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        num_rollout=100,
        global_batch_size=8,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
    )
    manager = publisher.CheckpointPublisher(args)
    await manager.after_save(3)
    assert [name for name, _ in calls] == ["inventory_node", "copy_node", "prune_node"]
    assert manager._completed == [3]
