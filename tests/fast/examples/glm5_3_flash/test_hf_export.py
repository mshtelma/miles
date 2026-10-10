"""A GLM-5.3 HF export is kept only if its tensor set matches the base, and always leaves a receipt."""

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from examples.glm5_3_flash import hf_checks, hf_export
from safetensors.torch import save_file


def _write_checkpoint(root: Path, tensors: dict[str, tuple[str, list[int], bytes]], *, complete: bool) -> Path:
    root.mkdir(parents=True)
    header, offset, payload = {}, 0, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload += data
    raw = json.dumps(header).encode()
    (root / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    weight_map = dict.fromkeys(tensors, "model-00001.safetensors")
    (root / hf_checks.INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    (root / hf_checks.CONFIG_NAME).write_text('{"model_type": "glm5_next", "text_config": {"num_hidden_layers": 4}}')
    if complete:
        (root / hf_export.HF_EXPORT_COMPLETE_MARKER).touch()
    return root


_TENSORS = {
    "model.embed_tokens.weight": ("BF16", [2], b"\x01\x02\x03\x04"),
    "model.layers.3.mlp.experts.0.down_proj.weight": ("F8_E4M3", [2], b"\x01\x02"),
}
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


def _save(tmp_path: Path, tensors=_TENSORS, *, complete: bool = True, base_tensors=_TENSORS):
    base = _write_checkpoint(tmp_path / "base", base_tensors, complete=False)
    iteration = tmp_path / "checkpoints/iter_0000099"
    iteration.mkdir(parents=True)
    hf_dir = _write_checkpoint(iteration / "hf", tensors, complete=complete)
    return SimpleNamespace(hf_checkpoint=str(base)), iteration, hf_dir


def _receipt(iteration: Path) -> dict:
    return json.loads((iteration / "hf_export.json").read_text())


def test_a_matching_export_ships_with_an_ok_receipt(tmp_path, capsys) -> None:
    args, iteration, hf_dir = _save(tmp_path)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    receipt = _receipt(iteration)
    assert receipt["status"] == "ok" and receipt["reason"] is None and hf_dir.is_dir()
    assert receipt["rollout_id"] == 99 and receipt["tensors"] == 2 and receipt["files"] == 4
    assert receipt["total_bytes"] == sum(path.stat().st_size for path in hf_dir.rglob("*") if path.is_file())
    assert receipt["structural"]["counts"] == {"missing": 0, "allowed_missing": 0, "extra": 0, "mismatched": 0}
    assert receipt["base"]["config_sha256"] == hf_checks.file_sha256(Path(args.hf_checkpoint) / "config.json")
    assert receipt["memory"]["host_peak_rss_bytes"] > 0
    assert hf_export.receipt_ok(receipt)
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("MILES_HF_EXPORT="))
    assert json.loads(line.split("=", 1)[1])["status"] == "ok"


def test_the_export_span_leaves_out_the_metadata_copied_from_the_base(tmp_path) -> None:
    args, iteration, hf_dir = _save(tmp_path)
    written = (hf_dir / "model-00001.safetensors").stat().st_mtime
    os.utime(hf_dir / hf_checks.CONFIG_NAME, (written - 7200, written - 7200))  # copy2 keeps the base's mtime

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    assert _receipt(iteration)["seconds"]["export_span"] == 0.0  # the one shard is the whole export


def test_an_export_missing_a_tensor_is_removed_and_recorded(tmp_path) -> None:
    tensors = {name: value for name, value in _TENSORS.items() if "experts" not in name}
    args, iteration, hf_dir = _save(tmp_path, tensors)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    receipt = _receipt(iteration)
    assert receipt["status"] == "failed" and "1 missing" in receipt["reason"]
    assert receipt["structural"]["missing"] == ["model.layers.3.mlp.experts.0.down_proj.weight"]
    assert not hf_dir.exists()  # a failed export never ships


def test_an_export_that_raised_leaves_a_failed_receipt(tmp_path) -> None:
    args, iteration, hf_dir = _save(tmp_path, complete=False)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    assert _receipt(iteration)["reason"] == "export has no .complete marker" and not hf_dir.exists()


def test_the_hook_never_raises_even_when_the_gate_does(tmp_path, monkeypatch) -> None:
    args, iteration, hf_dir = _save(tmp_path)

    def broken(*_args, **_kwargs):
        raise RuntimeError("disk vanished")

    monkeypatch.setattr(hf_checks, "structural_report", broken)
    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    receipt = _receipt(iteration)
    assert receipt["status"] == "failed" and "disk vanished" in receipt["reason"]


_TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def _tensor(dtype: str, shape: list[int], data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(_TORCH_DTYPES[dtype]).reshape(shape).clone()


def _export(tmp_path: Path, exported: dict, *, base_tensors: dict):
    """Write ``exported`` the way SnapshotPublisher.write_model does with the base layout."""
    base = _write_checkpoint(tmp_path / "base", base_tensors, complete=False)
    iteration = tmp_path / "checkpoints/iter_0000099"
    hf_dir = iteration / "hf"
    hf_dir.mkdir(parents=True)
    layout = hf_export.base_layout(SimpleNamespace(hf_checkpoint=str(base)))
    shard = dict(layout.place(name, _tensor(*value)) for name, value in exported.items())
    save_file(shard, hf_dir / "model-00001.safetensors")
    weight_map = dict.fromkeys(shard, "model-00001.safetensors")
    for index, untrained in enumerate(layout.untrained_shards(set(weight_map)), start=2):
        save_file(untrained, hf_dir / f"model-{index:05d}.safetensors")
        weight_map.update(dict.fromkeys(untrained, f"model-{index:05d}.safetensors"))
    (hf_dir / hf_checks.INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    (hf_dir / hf_checks.CONFIG_NAME).write_bytes((base / hf_checks.CONFIG_NAME).read_bytes())
    (hf_dir / hf_export.HF_EXPORT_COMPLETE_MARKER).touch()
    return SimpleNamespace(hf_checkpoint=str(base)), iteration, hf_dir


def _tensor_bytes(model_dir: Path) -> dict[str, bytes]:
    table = hf_checks.tensor_table(model_dir)
    return {name: (model_dir / info.file).read_bytes()[info.begin : info.end] for name, info in table.items()}


def test_a_sync_named_export_gets_the_base_layout_and_its_untrained_tensors(tmp_path) -> None:
    args, iteration, hf_dir = _export(tmp_path, _SYNC_NAMED, base_tensors=_BASE_LAYOUT)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    receipt = _receipt(iteration)
    assert receipt["status"] == "ok", receipt["reason"]
    final, base = _tensor_bytes(hf_dir), _tensor_bytes(Path(args.hf_checkpoint))
    assert set(final) == set(_BASE_LAYOUT)
    # Trained values are the export's own bytes under the base's names; untrained ones are the base's.
    assert final["model.language_model.embed_tokens.weight"] == b"\x11\x12\x13\x14"
    assert final["model.language_model.layers.3.mlp.experts.0.down_proj.weight"] == b"\x21\x22"
    assert final["model.visual.blocks.0.attn.qkv.weight"] == base["model.visual.blocks.0.attn.qkv.weight"]
    assert (
        final["model.language_model.layers.4.eh_proj.weight"] == base["model.language_model.layers.4.eh_proj.weight"]
    )


def test_a_trained_tensor_the_export_dropped_is_never_filled_from_the_base(tmp_path) -> None:
    dropped = {name: value for name, value in _SYNC_NAMED.items() if "experts" not in name}
    args, iteration, hf_dir = _export(tmp_path, dropped, base_tensors=_BASE_LAYOUT)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    receipt = _receipt(iteration)
    assert receipt["status"] == "failed" and "1 missing" in receipt["reason"]
    assert receipt["structural"]["missing"] == ["model.language_model.layers.3.mlp.experts.0.down_proj.weight"]


def test_two_exported_tensors_placed_onto_one_base_name_fail_the_export(tmp_path) -> None:
    base = _write_checkpoint(tmp_path / "base", _BASE_LAYOUT, complete=False)
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
    args, iteration, hf_dir = _export(tmp_path, exported, base_tensors=base_tensors)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    assert _receipt(iteration)["status"] == "ok", _receipt(iteration)["reason"]
    final = _tensor_bytes(hf_dir)
    assert struct.unpack("<4H", final[conv]) == (0x3F80, 0x3F80, 0x3F82, 0x3F81)
    assert struct.unpack("<2f", final["model.language_model.layers.0.hc_attn_base"]) == (1.0, -2.0)
    assert struct.unpack("<3f", final["model.language_model.layers.0.hc_attn_scale"]) == (0.5, 1.5, 2.5)
    assert hf_checks.tensor_table(hf_dir)["model.language_model.layers.0.hc_attn_scale"].shape == (3,)


def test_an_fp8_tensor_exported_in_another_dtype_still_fails_the_gate(tmp_path) -> None:
    widened = {**_SYNC_NAMED, "model.layers.3.mlp.experts.0.down_proj.weight": ("BF16", [2], b"\x21\x22\x23\x24")}
    args, iteration, hf_dir = _export(tmp_path, widened, base_tensors=_BASE_LAYOUT)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    receipt = _receipt(iteration)
    assert receipt["status"] == "failed" and "1 mismatched" in receipt["reason"]
    assert receipt["structural"]["mismatched"] == [
        {
            "name": "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
            "base": ["F8_E4M3", [2]],
            "exported": ["BF16", [2]],
        }
    ]


def test_a_failed_gate_prints_the_tensors_it_rejected(tmp_path, capsys) -> None:
    dropped = {name: value for name, value in _SYNC_NAMED.items() if "experts" not in name}
    args, iteration, hf_dir = _export(tmp_path, dropped, base_tensors=_BASE_LAYOUT)

    hf_export.finalize_hf_export(args, 99, str(iteration), str(hf_dir))

    line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("MILES_HF_EXPORT_STRUCTURAL=")
    )
    assert json.loads(line.split("=", 1)[1])["missing"] == [
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight"
    ]
