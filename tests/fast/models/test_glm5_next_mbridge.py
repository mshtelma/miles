"""Keep the KDA checkpoint marker in Megatron without treating it as an HF weight."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


def _bridge_module():
    class BridgeBase:
        _ATTENTION_MAPPING = {}

        def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep=True):
            return {name: name for name in model.state_dict()}

        def _weight_name_mapping_mcore_to_hf(self, name):
            return [name.replace("decoder.", "model.").replace("self_attention.linear_attn", "self_attn")]

    bridge_core = ModuleType("mbridge.core")
    bridge_core.register_model = lambda _name: lambda cls: cls
    bridge_models = ModuleType("mbridge.models")
    bridge_models.DeepseekV3Bridge = BridgeBase
    ext = ModuleType("mbridge.models.ext")
    deepseek_ext = ModuleType("mbridge.models.ext.deepseek_v3")
    dequant_module = ModuleType("mbridge.models.ext.deepseek_v3.dequant_fp8_safetensor_io")
    dequant_module.DequantFP8SafeTensorIO = type("DequantFP8SafeTensorIO", (), {})
    kernel_module = ModuleType("mbridge.models.ext.deepseek_v3.kernel")
    kernel_module.weight_dequant = lambda weight, scale: weight.float() * scale
    deepseek_bridge = ModuleType("miles_plugins.mbridge.deepseek_v32")
    deepseek_bridge.GlmMoeDsaBridge = BridgeBase
    modules = {
        "mbridge": ModuleType("mbridge"),
        "mbridge.core": bridge_core,
        "mbridge.models": bridge_models,
        "mbridge.models.ext": ext,
        "mbridge.models.ext.deepseek_v3": deepseek_ext,
        "mbridge.models.ext.deepseek_v3.dequant_fp8_safetensor_io": dequant_module,
        "mbridge.models.ext.deepseek_v3.kernel": kernel_module,
        "miles_plugins.mbridge": ModuleType("miles_plugins.mbridge"),
        "miles_plugins.mbridge.deepseek_v32": deepseek_bridge,
    }
    path = Path(__file__).resolve().parents[3] / "miles_plugins/mbridge/glm5_next.py"
    spec = importlib.util.spec_from_file_location("test_glm5_next_mbridge_target", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def test_bridge_excludes_only_the_kda_layout_marker_from_hf_mapping():
    layer = torch.nn.Module()
    layer.layer_number = 1
    layer.self_attention = torch.nn.Module()
    layer.self_attention.linear_attn = torch.nn.Module()
    layer.self_attention.linear_attn.q_proj = torch.nn.Linear(2, 2, bias=False)
    layer.self_attention.linear_attn.register_buffer("weight_layout_version", torch.tensor([1], dtype=torch.int32))
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([layer])
    bridge = object.__new__(_bridge_module().Glm5NextBridge)

    state = model.state_dict()
    mapping = bridge._weight_name_mapping_mcore_local_to_global(model)
    export_mapping = bridge._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)

    marker = "decoder.layers.0.self_attention.linear_attn.weight_layout_version"
    weight = "decoder.layers.0.self_attention.linear_attn.q_proj.weight"
    assert marker in state
    assert marker not in mapping and marker not in export_mapping
    assert mapping[weight] == export_mapping[weight] == weight
    assert bridge._weight_name_mapping_mcore_to_hf(weight) == ["model.language_model.layers.0.self_attn.q_proj.weight"]


def test_fp8_loader_uses_the_selected_cuda_device_for_weights_and_scales(tmp_path):
    module = _bridge_module()
    io = object.__new__(module.Glm5NextFP8SafeTensorIO)
    io.hf_dir = str(tmp_path)
    io.index = {"weight": "weight.safetensors", "weight_scale_inv": "scale.safetensors"}
    opened = []

    class _Tensors:
        def __init__(self, path, device):
            opened.append((Path(path).name, device))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_tensor(self, name):
            if name == "weight":
                return torch.ones((2, 2), dtype=torch.uint8)
            return torch.ones((2, 2), dtype=torch.float32)

    initial_dtype = torch.get_default_dtype()
    with (
        patch.object(module.torch.cuda, "is_initialized", return_value=True),
        patch.object(module.torch.cuda, "current_device", return_value=3),
        patch.object(module, "safe_open", side_effect=lambda path, *, framework, device: _Tensors(path, device)),
    ):
        result = io.load_some_hf_weight(["weight"])["weight"]

    assert result.dtype == torch.float32
    assert opened == [("weight.safetensors", "cuda:3"), ("scale.safetensors", "cuda:3")]
    assert torch.get_default_dtype() == initial_dtype
