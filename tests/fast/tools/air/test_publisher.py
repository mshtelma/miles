"""Each node copies its part of a save to the Volume and frees its disk; the tracker goes last."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.air import publisher


def _save(root: Path, step: int, *, ranks: list[int], writer: bool) -> None:
    """One node's part of a Miles save: its ranks' shards, plus on the writer's node the rest."""
    iteration = root / f"iter_{step:07d}"
    iteration.mkdir(parents=True)
    for rank in ranks:
        (iteration / f"__{rank}_0.distcp").write_bytes(b"shard")
    if writer:
        (iteration / ".metadata").write_bytes(b"metadata")
        (iteration / "hf").mkdir()
        (iteration / "hf/model-00001.safetensors").write_bytes(b"hf")
        (root / "rollout").mkdir()
        (root / f"rollout/global_dataset_state_dict_{step}.pt").write_bytes(b"cursor")
        (root / "latest_checkpointed_iteration.txt").write_text(str(step))


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_a_node_copies_its_part_of_the_save_and_frees_its_disk(tmp_path):
    local, remote = tmp_path / "node", tmp_path / "volume"
    _save(local, 3, ranks=[0, 1], writer=True)

    publisher.copy_node_save(local, remote, 3)

    assert _files(remote) == [
        "iter_0000003/.metadata",
        "iter_0000003/__0_0.distcp",
        "iter_0000003/__1_0.distcp",
        "iter_0000003/hf/model-00001.safetensors",
        "rollout/global_dataset_state_dict_3.pt",
    ]
    assert _files(local) == ["latest_checkpointed_iteration.txt"]


def test_a_node_without_the_save_copies_nothing(tmp_path):
    publisher.copy_node_save(tmp_path / "rollout-node", tmp_path / "volume", 3)

    assert not (tmp_path / "volume").exists()


class _Copies:
    """Stands in for ray.remote(copy_node_save): runs each live node's copy in place."""

    def __init__(self, nodes: dict[str, Path], fail: str | None = None):
        self.nodes, self.fail, self.ran = nodes, fail, []

    def options(self, *, scheduling_strategy):
        self.node = scheduling_strategy
        return self

    def remote(self, local, remote, rollout_id):
        future = asyncio.get_running_loop().create_future()
        self.ran.append(self.node)
        if self.node == self.fail:
            future.set_exception(OSError("copy failed"))
        else:
            publisher.copy_node_save(self.nodes[self.node], remote, rollout_id)
            future.set_result(None)
        return future


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    nodes = {"head": tmp_path / "head", "trainer": tmp_path / "trainer"}
    _save(nodes["head"], 3, ranks=[0], writer=True)
    _save(nodes["trainer"], 3, ranks=[1], writer=False)
    monkeypatch.setattr(
        publisher.ray,
        "nodes",
        lambda: [{"NodeID": node, "Alive": True} for node in nodes] + [{"NodeID": "gone", "Alive": False}],
    )
    monkeypatch.setattr(publisher, "NodeAffinitySchedulingStrategy", lambda node, soft: node)
    monkeypatch.setenv("MILES_PERSIST_DIR", str(tmp_path / "volume"))
    return nodes


def _older_save(remote: Path) -> Path:
    """A save published before: its DCP state and its HF export."""
    iteration = remote / "iter_0000001"
    (iteration / "hf").mkdir(parents=True)
    (iteration / "hf/model-00001.safetensors").write_bytes(b"hf")
    (iteration / "__0_0.distcp").write_bytes(b"shard")
    (iteration / ".metadata").write_bytes(b"metadata")
    return iteration


def test_the_tracker_is_written_once_every_live_node_copied(tmp_path, cluster, monkeypatch):
    copies = _Copies(cluster)
    monkeypatch.setattr(publisher.ray, "remote", lambda **options: lambda function: copies)

    asyncio.run(publisher.VolumePublisher(SimpleNamespace(save=str(tmp_path / "head"))).after_save(3))

    assert sorted(copies.ran) == ["head", "trainer"]
    remote = tmp_path / "volume"
    assert (remote / "latest_checkpointed_iteration.txt").read_text() == "3"
    assert {"iter_0000003/__0_0.distcp", "iter_0000003/__1_0.distcp"} <= set(_files(remote))


def test_a_published_save_leaves_the_older_saves_only_their_hf_export(tmp_path, cluster, monkeypatch):
    older = _older_save(tmp_path / "volume")
    copies = _Copies(cluster)
    monkeypatch.setattr(publisher.ray, "remote", lambda **options: lambda function: copies)

    asyncio.run(publisher.VolumePublisher(SimpleNamespace(save=str(tmp_path / "head"))).after_save(3))

    assert _files(older) == ["hf/model-00001.safetensors"]
    assert _files(tmp_path / "volume/iter_0000003") == [
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
        "hf/model-00001.safetensors",
    ]


def test_a_failed_node_copy_leaves_the_tracker_and_the_older_saves_untouched(tmp_path, cluster, monkeypatch):
    older = _older_save(tmp_path / "volume")
    copies = _Copies(cluster, fail="trainer")
    monkeypatch.setattr(publisher.ray, "remote", lambda **options: lambda function: copies)

    with pytest.raises(OSError, match="copy failed"):
        asyncio.run(publisher.VolumePublisher(SimpleNamespace(save=str(tmp_path / "head"))).after_save(3))

    assert not (tmp_path / "volume/latest_checkpointed_iteration.txt").exists()
    assert _files(older) == [".metadata", "__0_0.distcp", "hf/model-00001.safetensors"]
