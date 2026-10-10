import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def iter_hf_weights(self, weights):
        yield list(weights.items())


class _Prefixed:
    """Renames into the base, casts to BF16, and adds one base tensor the trainer does not hold."""

    def place(self, name, tensor):
        return f"base.{name}", tensor.to(torch.bfloat16)

    def untrained_shards(self, written):
        assert written == {"base.w"}
        yield {"base.visual": torch.ones(2)}


def test_a_layout_places_each_tensor_and_adds_the_untrained_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)

    SnapshotPublisher(_Iterator()).write_model(
        tmp_path, weights={"w": torch.zeros(3)}, hf_checkpoint=str(tmp_path / "base"), layout=_Prefixed()
    )

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"base.w": "model-00001.safetensors", "base.visual": "model-00002.safetensors"}
    assert index["metadata"]["total_size"] == 3 * 2 + 2 * 4
    assert load_file(tmp_path / "model-00001.safetensors")["base.w"].dtype == torch.bfloat16
