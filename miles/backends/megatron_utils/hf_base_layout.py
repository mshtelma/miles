"""Drop-in HF checkpoints of GLM-5.3-Flash or Qwen3.5-35B-A3B inside each training save.

With ``--save-hf <save>/iter_{rollout_id:07d}/hf``, Miles writes an HF export right after each DCP save.
The trainer exports through the weight-sync converter, which names tensors as SGLang updates them
(``model.layers.*``), in the dtypes the trainer holds, and only for the text model. Both bases are
multimodal checkpoints (``model.language_model.layers.*``, ``Glm5NextForConditionalGeneration`` and
``Qwen3_5MoeForConditionalGeneration``).

``base_layout`` (``--save-hf-layout-path``) closes that gap as the shards are written: base names, the
base's float dtype for tensors the trainer upcasts (GLM-5.3 trains its KDA convolutions in FP32 over a
BF16 checkpoint), the base's shape where only size-1 dimensions differ, and the vision tower and the
multi-token-prediction layer, which Megatron neither holds nor trains, copied from the base. It reads
the base from ``--hf-checkpoint``, which must hold the weights. Miles writes ``hf/.complete`` once every
rank has finished; a failed export leaves no directory, and the DCP save remains the source of truth.
"""

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

_LANGUAGE_LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")
_FLOAT_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
_ELEMENT_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2}
_UNTRAINED_SHARD_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    file: str
    nbytes: int


def tensor_table(model_dir: Path) -> dict[str, TensorInfo]:
    """Every tensor of an HF checkpoint, read from its safetensors headers."""
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    table = {}
    for file in sorted(set(weight_map.values())):
        with safe_open(str(model_dir / file), framework="pt") as shard:
            for name in shard.keys():
                piece = shard.get_slice(name)
                shape, dtype = tuple(piece.get_shape()), piece.get_dtype()
                table[name] = TensorInfo(dtype, shape, file, math.prod(shape) * _ELEMENT_BYTES.get(dtype, 1))
    return table


def base_layout(args) -> "BaseLayout":
    return BaseLayout(Path(args.hf_checkpoint))


class BaseLayout:
    """The base checkpoint's layout, for ``SnapshotPublisher.write_model``."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._table: dict[str, TensorInfo] | None = None
        self._placed: set[str] = set()

    @property
    def table(self) -> dict[str, TensorInfo]:
        # Read on the first tensor: every rank builds the layout, only the writer uses it.
        if self._table is None:
            self._table = tensor_table(self._base_dir)
        return self._table

    def place(self, name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
        if name not in self.table and base_name(name) in self.table:
            name = base_name(name)
        if name in self._placed:
            raise ValueError(f"the export writes {name} twice")
        self._placed.add(name)
        reference = self.table.get(name)
        if reference is None or _squeezed(reference.shape) != _squeezed(tuple(tensor.shape)):
            # Raising here would strand the other ranks in the weight gathers; serving such an
            # export fails on load instead.
            return name, tensor
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
            size += self.table[name].nbytes
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


def base_name(name: str) -> str:
    """The base checkpoint's name for a tensor the weight-sync converter calls ``model.*``."""
    return "model.language_model." + name.removeprefix("model.") if name.startswith("model.") else name


def untrained_in(base_dir: Path):
    """Base tensors Megatron never holds: the vision tower and the multi-token-prediction layers, which
    GLM-5.3-Flash stores as language layers past the text model's and Qwen3.5 under ``mtp.``.

    The trainer is the text model (no MTP training), so these keep the base's values in every checkpoint;
    serving uses neither (text-only prompts, no speculative decoding).
    """
    config = json.loads((base_dir / "config.json").read_text())
    num_layers = config.get("text_config", config)["num_hidden_layers"]

    def untrained(name: str) -> bool:
        layer = _LANGUAGE_LAYER.match(name)
        if name.startswith(("model.visual.", "mtp.")):
            return True
        return layer is not None and int(layer.group(1)) >= num_layers

    return untrained


def _squeezed(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dim for dim in shape if dim != 1)
