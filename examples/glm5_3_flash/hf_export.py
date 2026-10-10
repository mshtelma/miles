"""Drop-in HF checkpoints of GLM-5.3-Flash inside each training save.

With ``--save-hf <save>/iter_{rollout_id:07d}/hf``, Miles writes an HF export right after each DCP save.
The trainer exports through the weight-sync converter, which names tensors as SGLang updates them
(``model.layers.*``), in the dtypes the trainer holds, and only for the text model. The base is the
multimodal ``Glm5NextForConditionalGeneration`` checkpoint (``model.language_model.layers.*``).

``base_layout`` (``--save-hf-layout-path``) closes that gap as the shards are written: base names, the
base's float dtype for tensors the trainer upcasts (GLM-5.3 trains its KDA convolutions in FP32 over a
BF16 checkpoint), the base's shape where only size-1 dimensions differ, and the vision tower and the
multi-token-prediction layer, which Megatron neither holds nor trains, copied from the base. It reads
the base from ``--hf-checkpoint``, which must hold the weights.

``finalize_hf_export`` (``--custom-megatron-post-save-hook-path``, rank 0) then checks the export
against the base's tensor set (hf_checks.py) and writes ``iter_{id:07d}/hf_export.json`` beside the DCP
files. A failed export is removed and recorded, never kept; training continues, because the DCP save
remains the source of truth and tools/convert_torch_dist_to_hf.py can convert it later.
"""

import json
import logging
import os
import re
import resource
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from examples.glm5_3_flash import hf_checks
from safetensors import safe_open

from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

logger = logging.getLogger(__name__)

RECEIPT_NAME = "hf_export.json"
HF_DIR_NAME = "hf"
RECEIPT_SCHEMA = 1
RECEIPT_KIND = "hf_export_receipt"
_LANGUAGE_LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")
_FLOAT_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
_UNTRAINED_SHARD_BYTES = 4 * 1024**3


def base_layout(args) -> "BaseLayout":
    return BaseLayout(Path(args.hf_checkpoint))


class BaseLayout:
    """The GLM-5.3-Flash base checkpoint's layout, for ``SnapshotPublisher.write_model``."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._table: dict[str, hf_checks.TensorInfo] | None = None
        self._placed: set[str] = set()

    @property
    def table(self) -> dict[str, hf_checks.TensorInfo]:
        # Read on the first tensor: every rank builds the layout, only the writer uses it.
        if self._table is None:
            self._table = hf_checks.tensor_table(self._base_dir)
        return self._table

    def place(self, name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
        if name not in self.table and base_name(name) in self.table:
            name = base_name(name)
        if name in self._placed:
            raise ValueError(f"the export writes {name} twice")
        self._placed.add(name)
        reference = self.table.get(name)
        if reference is None or _squeezed(reference.shape) != _squeezed(tuple(tensor.shape)):
            return name, tensor  # left for the structural check to report
        dtype = _FLOAT_DTYPES.get(reference.dtype)
        if dtype is not None and tensor.dtype != dtype and tensor.dtype in _FLOAT_DTYPES.values():
            tensor = tensor.to(dtype)
        return name, tensor.reshape(reference.shape)

    def untrained_shards(self, written: set[str]) -> Iterator[dict[str, torch.Tensor]]:
        untrained = untrained_in(self._base_dir)
        names = sorted(name for name in self.table if name not in written and untrained(name))
        shard, size = [], 0
        for name in names:
            shard.append(name)
            size += self.table[name].end - self.table[name].begin
            if size >= _UNTRAINED_SHARD_BYTES:
                yield self._read(shard)
                shard, size = [], 0
        if shard:
            yield self._read(shard)

    def _read(self, names: list[str]) -> dict[str, torch.Tensor]:
        tensors = {}
        by_file: dict[str, list[str]] = {}
        for name in names:
            by_file.setdefault(self.table[name].file, []).append(name)
        for file, file_names in by_file.items():
            with safe_open(str(self._base_dir / file), framework="pt") as shard:
                tensors.update({name: shard.get_tensor(name) for name in file_names})
        return tensors


def finalize_hf_export(args, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None) -> None:
    """Post-save hook (rank 0): check the export, remove it if it fails, always write a receipt."""
    receipt_path = Path(checkpoint_dir) / RECEIPT_NAME
    try:
        hf_dir = Path(hf_checkpoint_dir) if hf_checkpoint_dir else None
        receipt = build_receipt(Path(args.hf_checkpoint), hf_dir, rollout_id=rollout_id)
        if not receipt_ok(receipt) and hf_dir is not None and hf_dir.exists():
            shutil.rmtree(hf_dir)  # a failed export is never kept; the receipt keeps the details
    except Exception as error:  # the hook must never fail the save
        logger.exception("HF export check failed")
        receipt = _receipt(rollout_id, status="failed", reason=f"hook error: {type(error).__name__}: {error}")
    try:
        _write_atomic(receipt_path, receipt)
    except OSError:
        logger.exception("could not write the HF export receipt %s", receipt_path)
    _announce(receipt)


def receipt_ok(receipt: object) -> bool:
    return isinstance(receipt, dict) and receipt.get("status") == "ok"


def build_receipt(base_dir: Path, hf_dir: Path | None, *, rollout_id: int | None) -> dict:
    """The check's verdict on one export directory, with its size, timing and rank-0 memory peaks."""
    if hf_dir is None or not hf_dir.is_dir():
        return _receipt(rollout_id, status="failed", reason="no export directory; see the rank-0 trainer log")
    if not (hf_dir / HF_EXPORT_COMPLETE_MARKER).is_file():
        return _receipt(rollout_id, status="failed", reason="export has no .complete marker")
    # Only the weight shards: the metadata files keep the base checkpoint's mtimes (copied with copy2).
    export_span = _write_span(list(hf_dir.glob("*.safetensors")))
    started = time.monotonic()
    structure = hf_checks.structural_report(base_dir, hf_dir)
    files = [path for path in hf_dir.rglob("*") if path.is_file()]
    receipt = _receipt(
        rollout_id,
        status="ok" if structure["ok"] else "failed",
        reason=None if structure["ok"] else structure["error"] or _structure_reason(structure),
    )
    receipt.update(
        files=len(files),
        total_bytes=sum(path.stat().st_size for path in files),
        tensors=structure["tensors"],
        base={
            "path": str(base_dir),
            "index_sha256": hf_checks.file_sha256(base_dir / hf_checks.INDEX_NAME),
            "config_sha256": hf_checks.file_sha256(base_dir / hf_checks.CONFIG_NAME),
        },
        structural={
            key: structure[key] for key in ("checked", "counts", "missing", "allowed_missing", "extra", "mismatched")
        },
        seconds={"export_span": export_span, "check": time.monotonic() - started},
    )
    return receipt


def base_name(name: str) -> str:
    """The base checkpoint's name for a tensor the weight-sync converter calls ``model.*``."""
    return "model.language_model." + name.removeprefix("model.") if name.startswith("model.") else name


def untrained_in(base_dir: Path):
    """Base tensors Megatron never holds: the vision tower and the multi-token-prediction layers.

    The trainer is the 45-layer text model (no MTP training), so these keep the base's values in every
    checkpoint; serving uses neither (text-only prompts, no speculative decoding).
    """
    config = json.loads((base_dir / hf_checks.CONFIG_NAME).read_text())
    num_layers = config.get("text_config", config)["num_hidden_layers"]

    def untrained(name: str) -> bool:
        layer = _LANGUAGE_LAYER.match(name)
        return name.startswith("model.visual.") or (layer is not None and int(layer.group(1)) >= num_layers)

    return untrained


def _squeezed(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dim for dim in shape if dim != 1)


def _structure_reason(structure: dict) -> str:
    counts = structure["counts"]
    if not structure["config_identical"]:
        return "config.json differs from the base checkpoint"
    return (
        f"tensor set differs: {counts['missing']} missing, {counts['extra']} extra, {counts['mismatched']} mismatched"
    )


def _receipt(rollout_id: int | None, *, status: str, reason: str | None) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": RECEIPT_KIND,
        "status": status,
        "reason": reason,
        "rollout_id": rollout_id,
        "hf_dir": HF_DIR_NAME,
        "files": 0,
        "total_bytes": 0,
        "tensors": 0,
        "base": None,
        "structural": None,
        "seconds": None,
        "memory": _memory_peaks(),
    }


def _write_span(files: list[Path]) -> float:
    """Seconds between the first and the last write of ``files``: the export's visible duration on rank 0."""
    mtimes = [path.stat().st_mtime for path in files]
    return max(mtimes) - min(mtimes) if mtimes else 0.0


def _memory_peaks() -> dict:
    """Rank-0 peaks so far in this process: the export runs here, so they bound its footprint."""
    gpu = None
    try:
        import torch  # deferred: the CPU tests have no CUDA context

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            gpu = int(torch.cuda.max_memory_allocated())
    except Exception:
        gpu = None
    # Linux reports ru_maxrss in KiB.
    return {"gpu_peak_bytes": gpu, "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def _write_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _announce(receipt: dict) -> None:
    keys = ("status", "reason", "rollout_id", "files", "total_bytes", "seconds")
    print("MILES_HF_EXPORT=" + json.dumps({key: receipt[key] for key in keys}, sort_keys=True), flush=True)
    if receipt["status"] != "ok" and receipt["structural"] is not None:
        details = {key: receipt["structural"][key] for key in ("counts", "missing", "extra", "mismatched")}
        print("MILES_HF_EXPORT_STRUCTURAL=" + json.dumps(details, sort_keys=True), flush=True)
