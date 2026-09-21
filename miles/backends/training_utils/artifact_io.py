"""Shared atomic and collective primitives for durable training artifacts."""

import json
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_phase import DistributedPhaseGate

HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


class ArtifactStore:
    """Write files and collectively publish complete artifact directories.

    Shared storage lets the publisher rank perform the filesystem mutation while
    every rank participates in the phase gates. Node-local storage makes each
    rank perform the same mutation in its own local filesystem.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        control_group: dist.ProcessGroup | None = None,
        publisher_rank: int = 0,
        shared_storage: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.publisher_rank = publisher_rank
        self.shared_storage = shared_storage
        self._phase_gate = DistributedPhaseGate(control_group)

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def is_publisher(self) -> bool:
        return self.rank == self.publisher_rank

    def path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if self.root is None or candidate.is_absolute() else self.root / candidate

    def ensure_dir(self, path: str | Path) -> Path:
        directory = self.path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def atomic_write_bytes(self, path: str | Path, data: bytes, *, fsync: bool = True) -> Path:
        destination = self.path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                if fsync:
                    os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def atomic_write_json(
        self,
        path: str | Path,
        value: Any,
        *,
        indent: int = 2,
        sort_keys: bool = False,
        fsync: bool = True,
    ) -> Path:
        return self.atomic_write_bytes(
            path, json.dumps(value, indent=indent, sort_keys=sort_keys).encode(), fsync=fsync
        )

    def atomic_torch_save(self, path: str | Path, value: Any, *, fsync: bool = True) -> Path:
        import io

        import torch

        buffer = io.BytesIO()
        torch.save(value, buffer)
        return self.atomic_write_bytes(path, buffer.getvalue(), fsync=fsync)

    @contextmanager
    def staging_dir(self, final_dir: str | Path, *, overwrite: bool = True) -> Iterator[Path]:
        """Prepare staging; callers publish it after all local phases pass."""
        destination = self.path(final_dir)
        staging = destination.parent / f"_tmp_{destination.name}"

        def prepare() -> None:
            if self.shared_storage and not self.is_publisher:
                return
            if not overwrite and destination.exists():
                raise FileExistsError(f"artifact {destination} already exists")
            if destination.exists() and not destination.is_symlink():
                raise NotImplementedError(
                    f"cannot overwrite a legacy artifact directory {destination}; save under a new name"
                )
            if staging.is_symlink():
                staging.unlink()
            elif staging.exists():
                shutil.rmtree(staging)

        self._phase_gate.run_local_phase("artifact.stage", prepare)
        self._phase_gate.wait_for_all()

        self._phase_gate.run_local_phase("artifact.stage.mkdir", lambda: staging.mkdir(parents=True, exist_ok=True))
        self._phase_gate.wait_for_all()
        try:
            yield staging
        except Exception:
            self._remove_path(staging)
            raise

    def publish(
        self,
        final_dir: str | Path,
        *,
        marker: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically publish staging at ``final_dir`` according to storage scope."""
        destination = self.path(final_dir)
        staging = destination.parent / f"_tmp_{destination.name}"
        version_dir: Path | None = None

        def publish_local() -> None:
            nonlocal version_dir
            if self.shared_storage and not self.is_publisher:
                return
            if metadata is not None:
                self.atomic_write_json(staging / "META.json", dict(metadata))
            if marker is not None:
                self.atomic_write_bytes(staging / marker, b"")
            version_dir = destination.parent / f"_version_{destination.name}_{uuid4().hex}"
            os.replace(staging, version_dir)
            staging.symlink_to(version_dir.name, target_is_directory=True)
            os.replace(staging, destination)

        try:
            self._phase_gate.run_local_phase("artifact.publish", publish_local)
        except Exception:
            self._remove_path(staging)
            if (
                version_dir is not None
                and version_dir.exists()
                and (not destination.is_symlink() or destination.resolve() != version_dir)
            ):
                self._remove_path(version_dir)
            raise
        self._phase_gate.wait_for_all()

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                # Another rank may have removed the shared staging path first.
                pass

    def write_tracker(self, path: str | Path, version: int | str) -> Path:
        return self.atomic_write_bytes(path, str(version).encode())

    def version_dir(self, version: int | str, *, prefix: str = "iter_") -> Path:
        return self.path(f"{prefix}{version:07d}" if isinstance(version, int) else f"{prefix}{version}")

    def write_manifest(self, path: str | Path, manifest: Mapping[str, Any]) -> Path:
        return self.atomic_write_json(path, dict(manifest))

    def verify_manifest(self, path: str | Path, expected: Mapping[str, Any]) -> None:
        actual = json.loads(self.path(path).read_text())
        if actual != dict(expected):
            raise ValueError(f"manifest {self.path(path)} does not match the expected contents")

    def copy_tree_exact(self, source: str | Path, destination: str | Path, *, prune: bool = True) -> Path:
        source_path = self.path(source)
        destination_path = self.path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        if prune:
            for child in destination_path.iterdir():
                if not (source_path / child.name).exists():
                    shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        for source_child in source_path.iterdir():
            target = destination_path / source_child.name
            if source_child.is_dir():
                shutil.copytree(source_child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source_child, target)
        return destination_path

    def write_safetensors_shard(
        self,
        path: str | Path,
        tensors: Mapping[str, Any],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> Path:
        import safetensors.torch

        data = safetensors.torch.save(dict(tensors), metadata=dict(metadata) if metadata is not None else None)
        return self.atomic_write_bytes(path, data)

    def write_hf_index(
        self,
        path: str | Path,
        weight_map: Mapping[str, str],
        *,
        total_size: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        index_metadata = dict(metadata or {})
        if total_size is not None:
            index_metadata["total_size"] = total_size
        return self.atomic_write_json(path, {"metadata": index_metadata, "weight_map": dict(weight_map)})

    def copy_hf_metadata(
        self, source_dir: str | Path, target_dir: str | Path, *, exclude_weights: bool = True
    ) -> None:
        source = self.path(source_dir)
        target = self.ensure_dir(target_dir)
        for source_file in source.iterdir():
            if not source_file.is_file():
                continue
            if exclude_weights and (
                source_file.suffix in HF_WEIGHT_SUFFIXES or source_file.name.endswith(".index.json")
            ):
                continue
            shutil.copy2(source_file, target / source_file.name)

    def run_local_phase(self, phase: str, operation: Callable[[], None]) -> None:
        self._phase_gate.run_local_phase(phase, operation)

    def wait_for_all(self) -> None:
        self._phase_gate.wait_for_all()


class HfShardWriter:
    """Track and atomically write Hugging Face safetensor shards and their index."""

    def __init__(self, output_dir: str | Path, *, store: ArtifactStore | None = None) -> None:
        self.store = store or ArtifactStore()
        self.output_dir = self.store.ensure_dir(output_dir)
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self._shard_index = 0

    def add_shard(
        self,
        tensors: Mapping[str, Any],
        *,
        shard_name: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Path:
        self._shard_index += 1
        filename = shard_name or f"model-{self._shard_index:05d}.safetensors"
        for name, tensor in tensors.items():
            if name in self.weight_map:
                raise ValueError(f"HF shard contains duplicate tensor {name!r}")
            self.weight_map[name] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        return self.store.write_safetensors_shard(self.output_dir / filename, tensors, metadata=metadata)

    def finalize(
        self,
        *,
        index_name: str = "model.safetensors.index.json",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if not self.weight_map:
            raise ValueError(f"HF export to {self.output_dir} produced no weights")
        return self.store.write_hf_index(
            self.output_dir / index_name,
            self.weight_map,
            total_size=self.total_size,
            metadata=metadata,
        )
