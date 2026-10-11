"""An HF export takes the base checkpoint's names, dtypes and shapes, plus its untrained tensors."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from examples.air_officeqa import hf_export
from safetensors.torch import load_file, save_file

_TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}
# The real base layout: the multimodal checkpoint, with a vision tower and an MTP layer (index 4 of 4).
_BASE_LAYOUT = {
    "lm_head.weight": ("BF16", [2], b"\x0a\x0b\x0c\x0d"),
    "model.language_model.embed_tokens.weight": ("BF16", [2], b"\x01\x02\x03\x04"),
    "model.language_model.layers.3.mlp.experts.0.down_proj.weight": ("F8_E4M3", [2], b"\x01\x02"),
    "model.language_model.layers.4.eh_proj.weight": ("BF16", [2], b"\x05\x06\x07\x08"),
    "model.visual.blocks.0.attn.qkv.weight": ("BF16", [1], b"\x09\x0a"),
}
# What the weight-sync converter writes: the trained text model under SGLang's update names.
_SYNC_NAMED = {
    "lm_head.weight": ("BF16", [2], b"\x0a\x0b\x0c\x0d"),
    "model.embed_tokens.weight": ("BF16", [2], b"\x11\x12\x13\x14"),
    "model.layers.3.mlp.experts.0.down_proj.weight": ("F8_E4M3", [2], b"\x21\x22"),
}


def _tensor(dtype: str, shape: list[int], data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(_TORCH_DTYPES[dtype]).reshape(shape).clone()


def _write_base(root: Path, tensors: dict) -> Path:
    root.mkdir(parents=True)
    save_file({name: _tensor(*value) for name, value in tensors.items()}, root / "model-00001.safetensors")
    weight_map = dict.fromkeys(tensors, "model-00001.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text('{"model_type": "glm5_next", "text_config": {"num_hidden_layers": 4}}')
    return root


def _export(tmp_path: Path, exported: dict, *, base_tensors: dict) -> tuple[Path, Path]:
    """Write ``exported`` the way SnapshotPublisher.write_model does with the base layout."""
    base = _write_base(tmp_path / "base", base_tensors)
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    layout = hf_export.base_layout(SimpleNamespace(hf_checkpoint=str(base)))
    shard = dict(layout.place(name, _tensor(*value)) for name, value in exported.items())
    save_file(shard, hf_dir / "model-00001.safetensors")
    for index, untrained in enumerate(layout.untrained_shards(set(shard)), start=2):
        save_file(untrained, hf_dir / f"model-{index:05d}.safetensors")
    return base, hf_dir


def _tensor_bytes(model_dir: Path) -> dict[str, bytes]:
    tensors = {}
    for shard in model_dir.glob("*.safetensors"):
        tensors.update(load_file(shard))
    return {name: bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()) for name, tensor in tensors.items()}


def test_the_table_reads_each_tensor_from_the_safetensors_headers(tmp_path) -> None:
    table = hf_export.tensor_table(_write_base(tmp_path / "base", _BASE_LAYOUT))

    info = table["model.language_model.embed_tokens.weight"]
    assert (info.dtype, info.shape, info.file, info.nbytes) == ("BF16", (2,), "model-00001.safetensors", 4)
    assert set(table) == set(_BASE_LAYOUT)


def test_a_sync_named_export_gets_the_base_layout_and_its_untrained_tensors(tmp_path) -> None:
    base, hf_dir = _export(tmp_path, _SYNC_NAMED, base_tensors=_BASE_LAYOUT)

    final, original = _tensor_bytes(hf_dir), _tensor_bytes(base)
    assert set(final) == set(_BASE_LAYOUT)
    # Trained values are the export's own bytes under the base's names; untrained ones are the base's.
    assert final["model.language_model.embed_tokens.weight"] == b"\x11\x12\x13\x14"
    assert final["model.language_model.layers.3.mlp.experts.0.down_proj.weight"] == b"\x21\x22"
    assert final["model.visual.blocks.0.attn.qkv.weight"] == original["model.visual.blocks.0.attn.qkv.weight"]
    eh_proj = "model.language_model.layers.4.eh_proj.weight"
    assert final[eh_proj] == original[eh_proj]


def test_qwen35_multi_token_prediction_layer_is_copied_from_the_base(tmp_path) -> None:
    # Qwen3.5 keeps its MTP layer under mtp., outside the language model's layers.
    base_tensors = {
        **_BASE_LAYOUT,
        "mtp.fc.weight": ("BF16", [2], b"\x31\x32\x33\x34"),
        "mtp.layers.0.mlp.gate.weight": ("BF16", [1], b"\x35\x36"),
    }
    base, hf_dir = _export(tmp_path, _SYNC_NAMED, base_tensors=base_tensors)

    final, original = _tensor_bytes(hf_dir), _tensor_bytes(base)
    assert final["mtp.fc.weight"] == original["mtp.fc.weight"]
    assert final["mtp.layers.0.mlp.gate.weight"] == original["mtp.layers.0.mlp.gate.weight"]


def test_a_trained_tensor_the_export_dropped_is_never_filled_from_the_base(tmp_path) -> None:
    dropped = {name: value for name, value in _SYNC_NAMED.items() if "experts" not in name}
    _, hf_dir = _export(tmp_path, dropped, base_tensors=_BASE_LAYOUT)

    assert "model.language_model.layers.3.mlp.experts.0.down_proj.weight" not in _tensor_bytes(hf_dir)


def test_two_exported_tensors_placed_onto_one_base_name_fail_the_export(tmp_path) -> None:
    base = _write_base(tmp_path / "base", _BASE_LAYOUT)
    layout = hf_export.base_layout(SimpleNamespace(hf_checkpoint=str(base)))
    layout.place("model.embed_tokens.weight", _tensor(*_SYNC_NAMED["model.embed_tokens.weight"]))

    with pytest.raises(ValueError, match="twice"):
        layout.place("model.language_model.embed_tokens.weight", _tensor("BF16", [2], b"\x01\x02\x03\x04"))


def _f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def test_float_tensors_the_trainer_upcast_are_cast_back_to_the_base_dtype(tmp_path) -> None:
    # The KDA convolutions are BF16 in the checkpoint and FP32 in training, so they export as FP32.
    conv = "model.language_model.layers.0.self_attn.q_conv1d.weight"
    base_tensors = {
        **_BASE_LAYOUT,
        conv: ("BF16", [4, 1], b"\x80\x3f" * 4),
        "model.language_model.layers.0.hc_attn_base": ("F32", [2], _f32(0.0, 0.0)),
        "model.language_model.layers.0.hc_attn_scale": ("F32", [3], _f32(0.0, 0.0, 0.0)),
    }
    # 1.0, the BF16 halfway points 1 + 2^-8 (rounds to even: down) and 1 + 3 * 2^-8 (up), and just past one.
    exported = {
        **_SYNC_NAMED,
        "model.layers.0.self_attn.q_conv1d.weight": (
            "F32",
            [4, 1],
            struct.pack("<4I", 0x3F800000, 0x3F808000, 0x3F818000, 0x3F808010),
        ),
        "model.layers.0.hc_attn_base": ("BF16", [2], b"\x80\x3f\x00\xc0"),  # 1.0, -2.0
        "model.layers.0.hc_attn_scale": ("F32", [1, 3], _f32(0.5, 1.5, 2.5)),
    }
    _, hf_dir = _export(tmp_path, exported, base_tensors=base_tensors)

    final = _tensor_bytes(hf_dir)
    assert struct.unpack("<4H", final[conv]) == (0x3F80, 0x3F80, 0x3F82, 0x3F81)
    assert struct.unpack("<2f", final["model.language_model.layers.0.hc_attn_base"]) == (1.0, -2.0)
    assert struct.unpack("<3f", final["model.language_model.layers.0.hc_attn_scale"]) == (0.5, 1.5, 2.5)
    assert load_file(hf_dir / "model-00001.safetensors")["model.language_model.layers.0.hc_attn_scale"].shape == (3,)
