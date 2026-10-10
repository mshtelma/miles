import hashlib
import io
import shutil
from types import SimpleNamespace

import pytest
import torch
from tools.air import publisher, stage_models
from tools.air.fuse_transfer import NodeTransferLimiter
from torch.distributed.checkpoint import FileSystemWriter, save


@pytest.fixture
def distributed_save(tmp_path):
    def build(step=3, *, optimizer=True, scheduler=True, rng=True, cursor=True, attempt="attempt"):
        roots = {name: tmp_path / attempt / name for name in ("trainer", "rollout", "serving")}
        iteration = roots["trainer"] / f"iter_{step:07d}"
        iteration.mkdir(parents=True)
        common = {"iteration": step}
        if scheduler:
            common["opt_param_scheduler"] = {"num_steps": step * 8, "lr_decay_steps": 800, "wd_incr_steps": 800}
        if rng:
            common["rng_state"] = {"torch_rng_state": torch.get_rng_state()}
        serialized = io.BytesIO()
        torch.save([common], serialized)
        state = {"common_state/shard_0_1": serialized, "model.weight": torch.arange(4)}
        if optimizer:
            state["optimizer.exp_avg"] = torch.zeros(4)
        save(state, storage_writer=FileSystemWriter(iteration, single_file_per_rank=False), no_dist=True)
        (iteration / "metadata.json").write_text('{"backend":"torch_dist"}')
        (roots["trainer"] / "latest_checkpointed_iteration.txt").write_text(str(step))
        other = roots["rollout"] / iteration.name
        other.mkdir(parents=True)
        for path in sorted(iteration.glob("*.distcp"))[::2]:
            shutil.move(path, other / path.name)
        for name in (".metadata", "metadata.json"):
            shutil.copyfile(iteration / name, other / name)
        shutil.copyfile(
            roots["trainer"] / "latest_checkpointed_iteration.txt",
            roots["rollout"] / "latest_checkpointed_iteration.txt",
        )
        if cursor:
            path = roots["rollout"] / f"rollout/global_dataset_state_dict_{step}.pt"
            path.parent.mkdir()
            torch.save(
                {
                    "sample_offset": 2,
                    "epoch_id": 1,
                    "sample_index": 20,
                    "sample_group_index": 10,
                    "metadata": {"curriculum": {"phase": "hard", "steps": 3}},
                },
                path,
            )
        target = tmp_path / "volume/runs/run/checkpoints" / attempt / iteration.name
        return SimpleNamespace(roots=roots, target=target, step=step, attempt=attempt, common=common)

    return build


@pytest.fixture
def input_manifest(tmp_path):
    model = tmp_path / "volume/models/model"
    dataset = tmp_path / "volume/datasets/officeqa"
    model.mkdir(parents=True)
    dataset.mkdir(parents=True)
    for name, raw in {"config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"HF weights"}.items():
        (model / name).write_bytes(raw)
    (dataset / "train.jsonl").write_text('{"prompt":"q"}\n')
    (dataset / "curriculum.json").write_text('{"phase":"easy"}')
    manifest = {
        "schema": 1,
        "volume_root": str(tmp_path / "volume"),
        "artifacts": [
            {
                "kind": "model",
                "name": "GLM-5.3-Flash",
                "repo": "example/model",
                "revision": "a" * 40,
                "source": "models/model",
                "target": "models/GLM-5.3-Flash",
                "files": stage_models.inventory(model),
            },
            {
                "kind": "dataset",
                "name": "officeqa",
                "repo": "local/officeqa",
                "revision": "b" * 40,
                "source": "datasets/officeqa",
                "target": "data/officeqa",
                "files": stage_models.inventory(dataset),
            },
        ],
    }
    raw = stage_models.json_bytes(manifest)
    path = tmp_path / "volume/inputs" / f"{hashlib.sha256(raw).hexdigest()}.json"
    path.parent.mkdir()
    path.write_bytes(raw)
    (path.parent / "latest").write_text(str(path))
    return path, manifest


def save_inventory(checkpoint):
    return publisher.merge_inventories(
        {node: publisher.inventory_node(root, checkpoint.step) for node, root in checkpoint.roots.items()},
        list(checkpoint.roots),
    )


def copy_save(checkpoint):
    files = save_inventory(checkpoint)
    for node, root in checkpoint.roots.items():
        publisher.copy_node(root, checkpoint.target, root / "slots", 2, files=[f for f in files if f["owner"] == node])
    return files


@pytest.fixture
def reference_root(tmp_path, input_manifest):
    source = tmp_path / "converted"
    iteration = source / "release"
    save(
        {"model.weight": torch.arange(4), "model.unused": torch.zeros(2)},
        storage_writer=FileSystemWriter(iteration, single_file_per_rank=False),
        no_dist=True,
    )
    (iteration / "metadata.json").write_text('{"backend":"torch_dist"}')
    (source / "latest_checkpointed_iteration.txt").write_text("release")
    target = tmp_path / "volume/ckpt/reference"
    document = stage_models.publish_reference(
        source, target, input_manifest[0].stem, NodeTransferLimiter(tmp_path / "slots", 2)
    )
    return target, document
