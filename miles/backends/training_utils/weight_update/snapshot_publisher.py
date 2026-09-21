from collections.abc import Callable
from pathlib import Path

import torch

from miles.backends.training_utils.artifact_io import ArtifactStore
from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.multi_lora import AdapterSpec

SnapshotWriter = Callable[[Path, ArtifactStore], None]


class SnapshotPublisher:
    """Coordinate one distributed snapshot transaction.

    Writers own snapshot-format content; this class owns the transaction
    boundary and gives them the store used by that transaction.
    """

    def __init__(self, *, shared_storage: bool = True) -> None:
        self._shared_storage = shared_storage

    def publish(
        self,
        path: str | Path,
        writer: SnapshotWriter,
        *,
        metadata: dict | None = None,
        overwrite: bool = True,
    ) -> None:
        store = ArtifactStore(shared_storage=self._shared_storage)
        write_checkpoint_dir(
            path,
            lambda staging: writer(staging, store),
            metadata=metadata,
            overwrite=overwrite,
            shared_storage=self._shared_storage,
            artifact_store=store,
        )


class WeightPublisher:
    """Write a gathered adapter through the shared snapshot transaction."""

    def __init__(self, iterator: HfWeightIteratorBase, adapter_config: dict) -> None:
        assert iterator.placement.is_full_gather, "publishing requires the full adapter on rank 0"
        self._iterator = iterator
        self._adapter_config = adapter_config
        self._snapshot_publisher = SnapshotPublisher()

    @torch.no_grad()
    def publish_adapter(self, adapter: AdapterSpec, path: str, metadata: dict | None = None) -> None:
        def write_adapter(tmp_dir: Path, store: ArtifactStore) -> None:
            is_writer = store.is_publisher
            tensors = {
                name: tensor.detach().contiguous().cpu()
                for name, tensor in self._iterator.materialize_adapter(adapter, materialize=is_writer).items()
            }

            if is_writer:
                config = self._adapter_config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
                store.atomic_write_json(tmp_dir / "adapter_config.json", config)
                store.write_safetensors_shard(tmp_dir / "adapter_model.safetensors", tensors)

        self._snapshot_publisher.publish(path, write_adapter, metadata=metadata, overwrite=False)
