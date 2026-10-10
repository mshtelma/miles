import os
from collections import defaultdict

import torch
from safetensors import safe_open

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge
from mbridge.models.ext.deepseek_v3.dequant_fp8_safetensor_io import DequantFP8SafeTensorIO
from mbridge.models.ext.deepseek_v3.kernel import weight_dequant
from miles_plugins.mbridge.deepseek_v32 import GlmMoeDsaBridge

_GLM5_NEXT_VOCAB_SIZE = 154880
_KDA_LAYOUT_VERSION_SUFFIX = ".self_attention.linear_attn.weight_layout_version"

_HC_ALPHA_SLICES = {
    "alpha_pre": slice(0, 1),
    "alpha_post": slice(1, 2),
    "alpha_res": slice(2, 3),
}


class Glm5NextFP8SafeTensorIO(DequantFP8SafeTensorIO):
    """Read FP8 weights on the CUDA device selected by this distributed rank."""

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict[str, torch.Tensor]:
        if not torch.cuda.is_initialized():
            raise RuntimeError("select this rank's CUDA device before loading GLM FP8 weights")
        device = f"cuda:{torch.cuda.current_device()}"
        files = defaultdict(list)
        for name in hf_weight_names:
            files[self.index[name]].append(name)

        loaded = {}
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            for filename, names in files.items():
                with safe_open(os.path.join(self.hf_dir, filename), framework="pt", device=device) as tensors:
                    for name in names:
                        weight = tensors.get_tensor(name)
                        scale_name = f"{name}_scale_inv"
                        if weight.element_size() != 1 or scale_name not in self.index:
                            loaded[name] = weight
                            continue
                        scale_filename = self.index[scale_name]
                        if scale_filename == filename:
                            scale = tensors.get_tensor(scale_name)
                        else:
                            with safe_open(
                                os.path.join(self.hf_dir, scale_filename), framework="pt", device=device
                            ) as scales:
                                scale = scales.get_tensor(scale_name)
                        loaded[name] = weight_dequant(weight, scale)
        finally:
            torch.set_default_dtype(previous_dtype)
        return loaded


@register_model("glm5_next")
class Glm5NextBridge(GlmMoeDsaBridge):
    _KDA_MAPPING = {
        f"self_attention.linear_attn.{weight_name}": ["model.layers.{layer_number}.self_attn." + weight_name]
        for weight_name in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "b_proj.weight",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "g_a_proj.weight",
            "g_b_proj.weight",
            "A_log",
            "dt_bias",
        ]
    }
    _KDA_MAPPING["self_attention.linear_attn.norm.weight"] = ["model.layers.{layer_number}.self_attn.o_norm.weight"]
    _KDA_MAPPING["self_attention.linear_attn.out_proj.weight"] = [
        "model.layers.{layer_number}.self_attn.o_proj.weight"
    ]

    _KPOOL_MAPPING = {
        "self_attention.index_kpool_compress_gate": [
            "model.layers.{layer_number}.self_attn.indexer.index_kpool_compress_gate"
        ],
        "self_attention.index_kpool_compress_ape": [
            "model.layers.{layer_number}.self_attn.indexer.index_kpool_compress_ape"
        ],
    }

    _ATTENTION_MAPPING = {**GlmMoeDsaBridge._ATTENTION_MAPPING, **_KDA_MAPPING, **_KPOOL_MAPPING}

    _OTHER_MAPPING = {
        "self_attention_hyper_connection.mapping_proj.weight": ["model.layers.{layer_number}.hc_attn_fn"],
        "self_attention_hyper_connection.bias": ["model.layers.{layer_number}.hc_attn_base"],
        "self_attention_hyper_connection.alpha_pre": ["model.layers.{layer_number}.hc_attn_scale"],
        "self_attention_hyper_connection.alpha_post": ["model.layers.{layer_number}.hc_attn_scale"],
        "self_attention_hyper_connection.alpha_res": ["model.layers.{layer_number}.hc_attn_scale"],
        "mlp_hyper_connection.mapping_proj.weight": ["model.layers.{layer_number}.hc_ffn_fn"],
        "mlp_hyper_connection.bias": ["model.layers.{layer_number}.hc_ffn_base"],
        "mlp_hyper_connection.alpha_pre": ["model.layers.{layer_number}.hc_ffn_scale"],
        "mlp_hyper_connection.alpha_post": ["model.layers.{layer_number}.hc_ffn_scale"],
        "mlp_hyper_connection.alpha_res": ["model.layers.{layer_number}.hc_ffn_scale"],
    }

    def __init__(self, hf_config, *args, **kwargs):
        text_config = getattr(hf_config, "text_config", None) or hf_config
        super().__init__(text_config, *args, **kwargs)

    def _get_safetensor_io(self, weights_path: str):
        if self.dtype != torch.bfloat16:
            raise ValueError("GLM FP8 checkpoint loading requires bfloat16")
        return Glm5NextFP8SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _get_rope_theta(self):
        rope_theta = getattr(self.hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(self.hf_config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            if int(getattr(self.hf_config, "qk_rope_head_dim", 0) or 0) == 0:
                return 10000.0
            raise ValueError("GLM-5.3 config must provide rope_theta (directly or via rope_parameters)")
        return float(rope_theta)

    def _get_rope_scaling(self):
        rope_scaling = getattr(self.hf_config, "rope_parameters", None)
        if not isinstance(rope_scaling, dict):
            rope_scaling = getattr(self.hf_config, "rope_scaling", None)
        return self._normalize_rope_scaling(rope_scaling)

    def _build_base_config(self, **kwargs):
        assert self.hf_config.vocab_size == _GLM5_NEXT_VOCAB_SIZE
        hc_eps = getattr(self.hf_config, "hc_eps", 1e-6)
        assert hc_eps == 1e-6, f"Megatron mHC eps constants are 1e-6, config says {hc_eps}"
        kwargs.pop("mtp_num_layers", None)
        kwargs.pop("mtp_loss_scaling_factor", None)
        kwargs.update(
            enable_hyper_connections=True,
            num_residual_streams=getattr(self.hf_config, "hc_mult", 4),
            mhc_sinkhorn_iterations=getattr(self.hf_config, "hc_sinkhorn_iters", 20),
            use_fused_mhc=False,
        )
        return super()._build_base_config(**kwargs)

    @staticmethod
    def _nest_language_model(name: str) -> str:
        if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
            return "model.language_model." + name[len("model.") :]
        return name

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep: bool = True) -> dict[str, str]:
        mapping = super()._weight_name_mapping_mcore_local_to_global(model, consider_ep=consider_ep)
        return {
            local_name: global_name
            for local_name, global_name in mapping.items()
            if not local_name.endswith(_KDA_LAYOUT_VERSION_SUFFIX)
        }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        try:
            names = super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            names = self._weight_name_mapping_other(mcore_weights_name)
        return [self._nest_language_model(n) for n in names]

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if mcore_weights_name.endswith(
            (
                "self_attention.linear_attn.q_conv1d.weight",
                "self_attention.linear_attn.k_conv1d.weight",
                "self_attention.linear_attn.v_conv1d.weight",
                "self_attention.linear_attn.A_log",
                "self_attention.linear_attn.dt_bias",
            )
        ):
            assert len(hf_weights) == 1
            return hf_weights[0].float().contiguous()
        for alpha_name, alpha_slice in _HC_ALPHA_SLICES.items():
            if mcore_weights_name.endswith(f"_hyper_connection.{alpha_name}"):
                assert len(hf_weights) == 1
                return hf_weights[0].reshape(-1)[alpha_slice].clone()
        if len(hf_weights) == 1 and hf_weights[0].dtype == torch.float32:
            saved_dtype = getattr(self, "dtype", None)
            self.dtype = None
            try:
                return DeepseekV3Bridge._weight_to_mcore_format(self, mcore_weights_name, hf_weights)
            finally:
                self.dtype = saved_dtype
        return DeepseekV3Bridge._weight_to_mcore_format(self, mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        for alpha_name in _HC_ALPHA_SLICES:
            if mcore_weights_name.endswith(f"_hyper_connection.{alpha_name}"):
                raise NotImplementedError(
                    "hc_*_scale export needs all three alphas at once; use the raw megatron_to_hf converter (miles/backends/megatron_utils/megatron_to_hf/glm5_next.py)"
                )
        return DeepseekV3Bridge._weight_to_hf_format(self, mcore_weights_name, mcore_weights)
