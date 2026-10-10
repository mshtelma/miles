"""Check an exported HF checkpoint against the base checkpoint it was converted from.

The check reads only safetensors headers, so it is cheap enough to run on the trainer's rank 0 after
every save: the export must carry exactly the base's tensors with the same dtypes and shapes, its
index must cover exactly the tensors on disk, and its config must be the base's bytes.
"""

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
_ELEMENT_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_LISTED = 50
_SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z")


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    file: str
    # Absolute byte range of the tensor's data inside its shard file.
    begin: int
    end: int


def read_header(path: Path) -> dict[str, TensorInfo]:
    """Parse one safetensors header, checking every tensor's byte range against its shape."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"safetensors file is shorter than its header length: {path}")
        (length,) = struct.unpack("<Q", raw_length)
        if not 0 < length <= _MAX_HEADER_BYTES or 8 + length > size:
            raise ValueError(f"safetensors header length is invalid: {path}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    data_start = 8 + length
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = _tensor_info(path, name, entry, data_start=data_start, file_size=size)
    return tensors


def _tensor_info(path: Path, name: str, entry: object, *, data_start: int, file_size: int) -> TensorInfo:
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise ValueError(f"safetensors entry {name} has an invalid schema: {path}")
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if dtype not in _ELEMENT_BYTES:
        raise ValueError(f"safetensors entry {name} has an unknown dtype {dtype!r}: {path}")
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError(f"safetensors entry {name} has an invalid shape: {path}")
    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(value) is not int for value in offsets):
        raise ValueError(f"safetensors entry {name} has invalid data offsets: {path}")
    begin, end = data_start + offsets[0], data_start + offsets[1]
    if not data_start <= begin <= end <= file_size or end - begin != math.prod(shape) * _ELEMENT_BYTES[dtype]:
        raise ValueError(f"safetensors entry {name} has a byte range that does not fit its shape: {path}")
    return TensorInfo(dtype=dtype, shape=tuple(shape), file=path.name, begin=begin, end=end)


def tensor_table(model_dir: Path) -> dict[str, TensorInfo]:
    """Every tensor of an HF checkpoint directory, cross-checked against its index."""
    weight_map = _weight_map(model_dir)
    files = sorted(set(weight_map.values())) if weight_map is not None else [SINGLE_FILE_NAME]
    table: dict[str, TensorInfo] = {}
    for file_name in files:
        if not _SHARD_NAME.fullmatch(file_name):
            raise ValueError(f"checkpoint index names an unsafe shard: {file_name!r}")
        for name, info in read_header(model_dir / file_name).items():
            if name in table:
                raise ValueError(f"tensor {name} appears in two shards of {model_dir}")
            table[name] = info
    if weight_map is not None:
        if set(weight_map) != set(table):
            unindexed = len(set(table) - set(weight_map))
            dangling = len(set(weight_map) - set(table))
            raise ValueError(f"index and shards disagree in {model_dir}: {unindexed} unindexed, {dangling} dangling")
        misplaced = [name for name, file_name in weight_map.items() if table[name].file != file_name]
        if misplaced:
            raise ValueError(f"index places {len(misplaced)} tensors in the wrong shard of {model_dir}")
    return table


def _weight_map(model_dir: Path) -> dict[str, str] | None:
    index_path = model_dir / INDEX_NAME
    if index_path.is_file():
        if index_path.stat().st_size > _MAX_INDEX_BYTES:
            raise ValueError(f"checkpoint index is oversized: {index_path}")
        weight_map = json.loads(index_path.read_bytes()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint index has no weight map: {index_path}")
        if any(type(name) is not str or type(file_name) is not str for name, file_name in weight_map.items()):
            raise ValueError(f"checkpoint index has a non-text entry: {index_path}")
        return weight_map
    if (model_dir / SINGLE_FILE_NAME).is_file():
        return None
    raise ValueError(f"no safetensors index or single-file checkpoint in {model_dir}")


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def structural_report(base_dir: Path, exported_dir: Path, *, allow_missing: Iterable[str] = ()) -> dict:
    """Whether an export carries exactly the base's tensors, dtypes, shapes and config.

    ``allow_missing`` holds full-match regexes for base tensors the exporter is known not to produce;
    they are reported separately and never silently dropped. Lists are capped; counts are exact.
    """
    patterns = [re.compile(pattern) for pattern in allow_missing]
    try:
        base = tensor_table(base_dir)
        exported = tensor_table(exported_dir)
    except (OSError, ValueError, KeyError) as error:
        return _failed_structure(f"{type(error).__name__}: {error}")
    absent = sorted(set(base) - set(exported))
    allowed = [name for name in absent if any(pattern.fullmatch(name) for pattern in patterns)]
    allowed_names = set(allowed)
    missing = [name for name in absent if name not in allowed_names]
    extra = sorted(set(exported) - set(base))
    shared = sorted(set(base) & set(exported))
    mismatched = [
        {
            "name": name,
            "base": [base[name].dtype, list(base[name].shape)],
            "exported": [exported[name].dtype, list(exported[name].shape)],
        }
        for name in shared
        if (base[name].dtype, base[name].shape) != (exported[name].dtype, exported[name].shape)
    ]
    base_config = file_sha256(base_dir / CONFIG_NAME)
    config_identical = base_config is not None and base_config == file_sha256(exported_dir / CONFIG_NAME)
    return {
        "ok": not missing and not extra and not mismatched and config_identical,
        "error": None,
        "checked": len(shared),
        "tensors": len(exported),
        "counts": {
            "missing": len(missing),
            "allowed_missing": len(allowed),
            "extra": len(extra),
            "mismatched": len(mismatched),
        },
        "missing": missing[:_MAX_LISTED],
        "allowed_missing": allowed[:_MAX_LISTED],
        "extra": extra[:_MAX_LISTED],
        "mismatched": mismatched[:_MAX_LISTED],
        "config_identical": config_identical,
    }


def _failed_structure(error: str) -> dict:
    return {
        "ok": False,
        "error": error,
        "checked": 0,
        "tensors": 0,
        "counts": {"missing": 0, "allowed_missing": 0, "extra": 0, "mismatched": 0},
        "missing": [],
        "allowed_missing": [],
        "extra": [],
        "mismatched": [],
        "config_identical": False,
    }
