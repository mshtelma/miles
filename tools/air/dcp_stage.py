"""Megatron DCP validation and rank-local read-plan staging, without a coordinator."""

import hashlib
import io
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.air import fuse_transfer as fuse
from tools.air.stage_models import json_bytes, read_json, safe_path, validate_files

MAX_COMMON_BYTES = 256 * 1024 * 1024
_MODEL_ROOTS = {"model", "embedding", "decoder", "output_layer", "language_model", "encoder"}


def read_metadata(iteration: Path):
    return pickle.loads(fuse.read_bounded(iteration / ".metadata"))


def common_files(metadata) -> set[str]:
    return {
        info.relative_path for index, info in metadata.storage_data.items() if index.fqn.startswith("common_state/")
    }


def read_common(iteration: Path, metadata=None) -> dict:
    # torch is optional for the input preparation tooling.
    import torch

    if fuse.exists(iteration / "common.pt"):
        raw = fuse.read_bounded(iteration / "common.pt", max_bytes=MAX_COMMON_BYTES)
    else:
        metadata = read_metadata(iteration) if metadata is None else metadata
        spans = [info for index, info in metadata.storage_data.items() if index.fqn.startswith("common_state/")]
        if len(spans) != 1 or spans[0].length > MAX_COMMON_BYTES:
            raise ValueError("checkpoint must contain one bounded Megatron common state")
        info = spans[0]
        raw = fuse.read_span(safe_path(iteration, info.relative_path), info.offset, info.length)
    common = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if isinstance(common, io.BytesIO):
        common.seek(0)
        common = torch.load(common, map_location="cpu", weights_only=False)
    if isinstance(common, list) and len(common) == 1:
        common = common[0]
    if not isinstance(common, dict):
        raise ValueError("checkpoint has invalid Megatron common state")
    return common


def validate_checkpoint(iteration: Path, files: list[dict], rollout_id: int | None = None) -> list[str]:
    entries = validate_files(files)
    prefix = iteration.name + "/"
    for name in (".metadata", "metadata.json"):
        if not entries.get(prefix + name, {}).get("size_bytes"):
            raise ValueError(f"checkpoint lacks {prefix + name}")
    metadata_entry = entries[prefix + ".metadata"]
    fuse.verify(iteration / ".metadata", size_bytes=metadata_entry["size_bytes"], sha256=metadata_entry["sha256"])
    metadata = read_metadata(iteration)
    if not metadata.storage_data or not any(
        key.split(".", 1)[0] in _MODEL_ROOTS for key in metadata.state_dict_metadata
    ):
        raise ValueError("DCP metadata lacks model storage")
    for info in metadata.storage_data.values():
        safe_path(iteration, info.relative_path)
        entry = entries.get(prefix + info.relative_path)
        if info.offset < 0 or info.length <= 0 or entry is None or entry["size_bytes"] < info.offset + info.length:
            raise ValueError(f"DCP storage reference is missing or short: {prefix + info.relative_path}")
    if rollout_id is None:
        return []
    if not any("optim" in key.lower() for key in metadata.state_dict_metadata):
        raise ValueError("checkpoint lacks optimizer state")
    common = read_common(iteration, metadata)
    if common.get("iteration") != rollout_id:
        raise ValueError("checkpoint common state has a different rollout ID")
    if "opt_param_scheduler" not in common:
        raise ValueError("checkpoint lacks scheduler state")
    if "rng_state" not in common and not any(key.startswith("rng_state") for key in metadata.state_dict_metadata):
        raise ValueError("checkpoint lacks RNG state")
    return sorted(common)


class ReadPlanStager:
    def __init__(self, sources: list[dict], state_root: Path, limiter):
        self.sources = [
            (Path(s["remote_root"]), Path(s["local_root"]), validate_files(s["document"]["files"])) for s in sources
        ]
        self.state_root, self.limiter = state_root, limiter

    def _stage_file(self, remote: Path, local: Path, name: str, entry: dict) -> None:
        target = safe_path(local, name)
        lock = self.state_root / "read-locks" / hashlib.sha256(str(target).encode()).hexdigest()
        with fuse.file_lock(lock):
            # Only verified atomic downloads create final files in this attempt's cache.
            if target.is_file() and target.stat().st_size == entry["size_bytes"]:
                return
            with self.limiter.slot():
                fuse.download_verified(
                    safe_path(remote, name), target, size_bytes=entry["size_bytes"], sha256=entry["sha256"]
                )

    def stage(self, root: Path, names: list[str]) -> None:
        source = next((s for s in self.sources if root.is_relative_to(s[1])), None)
        if source is None:
            return
        remote, local, entries = source
        relative = sorted({safe_path(root, n).relative_to(local).as_posix() for n in names})
        absent = set(relative) - entries.keys()
        if absent:
            raise ValueError(f"DCP read plan requests files absent from the manifest: {sorted(absent)}")
        with ThreadPoolExecutor(max_workers=self.limiter.workers) as pool:
            list(pool.map(lambda n: self._stage_file(remote, local, n, entries[n]), relative))


def install_reader_hook(stager: ReadPlanStager, reader_class=None) -> None:
    # Megatron has already imported this class; patch its method rather than its module binding.
    if reader_class is None:
        from torch.distributed.checkpoint import FileSystemReader

        reader_class = FileSystemReader
    original = reader_class.read_data

    def read_data(reader, plan, planner):
        names = [reader.storage_data[item.storage_index].relative_path for item in plan.items]
        stager.stage(Path(reader.path), names)
        return original(reader, plan, planner)

    reader_class.read_data = read_data


def initialize(args) -> None:
    config = read_json(Path(os.environ["MILES_CHECKPOINT_CONFIG"]))
    state = Path(config["state_root"])
    ranks = state / "trainer-ranks"
    ranks.mkdir(parents=True, exist_ok=True)
    (ranks / f"{args.rank}.json").write_bytes(json_bytes({"rank": args.rank, "world_size": args.world_size}))
    limiter = fuse.NodeTransferLimiter(Path(config["slot_root"]), config["workers"])
    install_reader_hook(ReadPlanStager(config["sources"], state, limiter))
