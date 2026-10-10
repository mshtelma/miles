"""FP8 weight sync leaves checkpoint BF16 modules in the receiver's original dtype."""

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8  # noqa: E402


@pytest.mark.parametrize(
    ("megatron_suffix", "hf_name", "kept"),
    [
        ("self_attention.linear_kv_up_proj.weight", "model.layers.3.self_attn.kv_b_proj.weight", True),
        ("self_attention.wq_b.weight", "model.layers.3.self_attn.indexer.wq_b.weight", True),
        ("self_attention.linear_attn.out_proj.weight", "model.layers.0.self_attn.o_proj.weight", True),
        ("self_attention.linear_q_up_proj.weight", "model.layers.3.self_attn.q_b_proj.weight", False),
        ("mlp.experts.linear_fc1.weight0", "model.layers.3.mlp.experts.0.gate_proj.weight", False),
    ],
)
def test_fp8_exclusions_control_export(monkeypatch, megatron_suffix, hf_name, kept):
    original = torch.ones(2, 2, dtype=torch.bfloat16)
    quantized = torch.zeros(2, 2)
    monkeypatch.setattr(quantizer_fp8, "_quantize_param", lambda *args: [(hf_name, quantized)])
    result = quantizer_fp8.quantize_params_fp8(
        Namespace(indexer_rope_interleave=False),
        f"module.module.decoder.layers.3.{megatron_suffix}",
        [(hf_name, original)],
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "model.layers.3.self_attn.kv_b_proj",
                "model.layers.3.self_attn.indexer.wq_b",
                "model.layers.0.self_attn.o_proj",
            ],
        },
    )
    assert len(result) == 1 and result[0][0] == hf_name
    assert result[0][1] is (original if kept else quantized)
    assert result[0][1].dtype == (torch.bfloat16 if kept else torch.float32)


_FP8_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


def _record_quantizers(monkeypatch) -> list[str]:
    # The engine asks for DeepGEMM's UE8M0 layout, as SGLang does on Blackwell.
    used = []
    monkeypatch.setattr(quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", lambda **_: True)
    monkeypatch.setattr(
        quantizer_fp8, "quant_weight_ue8m0", lambda weight, **_: used.append("ue8m0") or (weight, weight)
    )
    monkeypatch.setattr(quantizer_fp8, "transform_scale_ue8m0", lambda scale, **_: scale)
    monkeypatch.setattr(
        quantizer_fp8, "blockwise_cast_to_fp8_triton", lambda weight, _: used.append("fp32") or (weight, weight)
    )
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    return used


def test_an_hf_checkpoint_keeps_fp32_block_scales_where_weight_sync_uses_ue8m0(monkeypatch):
    # An earlier export carried UE8M0 scales packed as I32 [4096, 4] for the
    # checkpoint's F32 [32, 16] block scales, on all 179 non-expert FP8 linears.
    used = _record_quantizers(monkeypatch)
    name = "model.layers.3.self_attn.q_b_proj.weight"
    megatron_name = "module.module.decoder.layers.3.self_attention.linear_q_up_proj.weight"
    weight = torch.ones(2, 2, dtype=torch.bfloat16)

    quantizer_fp8.quantize_params_fp8(Namespace(), megatron_name, [(name, weight)], _FP8_CONFIG)
    quantizer_fp8.quantize_params_fp8(
        Namespace(), megatron_name, [(name, weight)], quantizer_fp8.checkpoint_quantization_config(_FP8_CONFIG)
    )

    assert used == ["ue8m0", "fp32"]


def test_only_fp8_configs_gain_the_checkpoint_scales_flag():
    assert quantizer_fp8.checkpoint_quantization_config(None) is None
    mxfp8 = {"quant_method": "mxfp8"}
    assert quantizer_fp8.checkpoint_quantization_config(mxfp8) is mxfp8
    marked = quantizer_fp8.checkpoint_quantization_config(_FP8_CONFIG)
    assert (
        marked[quantizer_fp8.CHECKPOINT_SCALES_KEY] is True and quantizer_fp8.CHECKPOINT_SCALES_KEY not in _FP8_CONFIG
    )
