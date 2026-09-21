"""Compatibility wrapper for the shared artifact directory writer."""

from collections.abc import Callable
from pathlib import Path

from miles.backends.training_utils.artifact_io import ArtifactStore


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
    shared_storage: bool = True,
    artifact_store: ArtifactStore | None = None,
) -> None:
    """Write collectively, then atomically point ``path`` at the completed version.

    All ranks must call. Readers may still hold an older version, so retain it.
    With shared storage, only the publisher rank mutates the public path. With
    node-local storage, every rank publishes on its own filesystem.
    """
    store = artifact_store or ArtifactStore(shared_storage=shared_storage)
    with store.staging_dir(path, overwrite=overwrite) as staging:
        store.run_local_phase("checkpoint.write_shards", lambda: write_shards(staging))
        store.wait_for_all()
    store.publish(path, metadata=metadata)
