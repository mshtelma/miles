"""The HF export check catches tensor-set drift from safetensors headers alone."""

import json
import struct
from pathlib import Path

import pytest

from examples.glm5_3_flash import hf_checks


def _write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset, payload = {}, 0, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload += data
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _write_checkpoint(root: Path, shards: dict[str, dict], *, config: bytes = b'{"model_type": "glm5_next"}') -> Path:
    root.mkdir(parents=True)
    weight_map = {}
    for file_name, tensors in shards.items():
        _write_shard(root / file_name, tensors)
        weight_map.update(dict.fromkeys(tensors, file_name))
    (root / hf_checks.INDEX_NAME).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (root / hf_checks.CONFIG_NAME).write_bytes(config)
    return root


def _base_shards() -> dict[str, dict]:
    return {
        "model-00001.safetensors": {
            "model.embed_tokens.weight": ("BF16", [2, 2], bytes(range(8))),
            "model.layers.3.mlp.experts.0.down_proj.weight": ("F8_E4M3", [2, 2], bytes([1, 2, 0x81, 0x7E])),
        },
        "model-00002.safetensors": {
            "model.layers.3.mlp.experts.0.down_proj.weight_scale_inv": ("F32", [1], struct.pack("<f", 0.5)),
        },
    }


def test_identical_export_passes_the_structural_gate(tmp_path: Path) -> None:
    base = _write_checkpoint(tmp_path / "base", _base_shards())
    exported = _write_checkpoint(tmp_path / "export", {"model-00001.safetensors": _merged(_base_shards())})

    report = hf_checks.structural_report(base, exported)

    assert report["ok"] and report["checked"] == 3 and report["counts"]["missing"] == 0
    assert report["config_identical"]


def _merged(shards: dict[str, dict]) -> dict:
    # A different shard layout is fine: only names, dtypes and shapes are compared.
    return {name: value for tensors in shards.values() for name, value in tensors.items()}


def test_gate_reports_missing_extra_mismatched_and_config_drift(tmp_path: Path) -> None:
    base = _write_checkpoint(tmp_path / "base", _base_shards())
    tensors = _merged(_base_shards())
    del tensors["model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"]
    tensors["model.layers.3.extra.weight"] = ("BF16", [1], b"\x00\x00")
    tensors["model.embed_tokens.weight"] = ("F32", [2, 1], bytes(8))
    exported = _write_checkpoint(tmp_path / "export", {"model-00001.safetensors": tensors}, config=b"{}")

    report = hf_checks.structural_report(base, exported)

    assert not report["ok"]
    assert report["missing"] == ["model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"]
    assert report["extra"] == ["model.layers.3.extra.weight"]
    assert report["mismatched"] == [
        {"name": "model.embed_tokens.weight", "base": ["BF16", [2, 2]], "exported": ["F32", [2, 1]]}
    ]
    assert report["config_identical"] is False


def test_allowed_missing_tensors_are_reported_not_dropped(tmp_path: Path) -> None:
    base = _write_checkpoint(tmp_path / "base", _base_shards())
    tensors = _merged(_base_shards())
    del tensors["model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"]
    exported = _write_checkpoint(tmp_path / "export", {"model-00001.safetensors": tensors})

    report = hf_checks.structural_report(base, exported, allow_missing=[r".+\.weight_scale_inv"])

    assert report["ok"] and report["missing"] == []
    assert report["allowed_missing"] == ["model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"]


def test_index_that_disagrees_with_its_shards_fails_the_gate(tmp_path: Path) -> None:
    base = _write_checkpoint(tmp_path / "base", _base_shards())
    exported = _write_checkpoint(tmp_path / "export", _base_shards())
    index = json.loads((exported / hf_checks.INDEX_NAME).read_text())
    index["weight_map"]["model.layers.9.ghost.weight"] = "model-00001.safetensors"
    (exported / hf_checks.INDEX_NAME).write_text(json.dumps(index))

    report = hf_checks.structural_report(base, exported)

    assert not report["ok"] and "dangling" in report["error"]


def test_header_whose_byte_range_does_not_fit_its_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    _write_shard(path, {"w": ("BF16", [3], bytes(4))})  # 3 BF16 elements need 6 bytes

    with pytest.raises(ValueError, match="does not fit its shape"):
        hf_checks.read_header(path)


def test_the_gate_imports_without_numpy_or_torch(tmp_path: Path) -> None:
    import subprocess
    import sys

    # A None entry makes any import of that module raise, so this fails if the gate pulls either in.
    script = "import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None\nfrom examples.glm5_3_flash import hf_checks\n"
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[4])
