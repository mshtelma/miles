"""Megatron -> HF weight-update conversion for GLM-5.3-Flash, the inverse of `miles_plugins/mbridge/glm5_next.py`.

DSA indexer names are mapped here, not in `convert_deepseekv3_to_hf`, so its rope-interleave half-swap never runs
(`qk_rope_head_dim == 0`). The three `alpha_*` of one mHC site arrive on the same rank and are buffered until they
can be emitted as one `hc_*_scale` tensor.
"""

import re

import torch

from .deepseekv3 import convert_deepseekv3_to_hf

_KDA_SUFFIX_MAPPING = {
    f"self_attention.linear_attn.{weight_name}": f"self_attn.{weight_name}"
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
_KDA_SUFFIX_MAPPING["self_attention.linear_attn.norm.weight"] = "self_attn.o_norm.weight"
_KDA_SUFFIX_MAPPING["self_attention.linear_attn.out_proj.weight"] = "self_attn.o_proj.weight"

_INDEXER_SUFFIX_MAPPING = {
    "self_attention.wq_b.weight": "self_attn.indexer.wq_b.weight",
    "self_attention.wk.weight": "self_attn.indexer.wk.weight",
    "self_attention.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
    "self_attention.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "self_attention.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "self_attention.index_kpool_compress_gate": "self_attn.indexer.index_kpool_compress_gate",
    "self_attention.index_kpool_compress_ape": "self_attn.indexer.index_kpool_compress_ape",
}

_HC_SUFFIX_MAPPING = {
    "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
    "self_attention_hyper_connection.bias": "hc_attn_base",
    "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
    "mlp_hyper_connection.bias": "hc_ffn_base",
}

_HC_ALPHA_ORDER = ("alpha_pre", "alpha_post", "alpha_res")
_HC_SITE_TO_SCALE = {
    "self_attention_hyper_connection": "hc_attn_scale",
    "mlp_hyper_connection": "hc_ffn_scale",
}

_LAYER_PATTERN = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")
_HC_ALPHA_PATTERN = re.compile(
    r"(self_attention_hyper_connection|mlp_hyper_connection)\.(alpha_pre|alpha_post|alpha_res)$"
)


class Glm5NextExportState:
    """Hold incomplete mHC alpha triples for exactly one weight export attempt."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self._closed = False

    def add_hc_scale(self, layer_idx: str, site: str, alpha_name: str, param: torch.Tensor):
        if self._closed:
            raise RuntimeError("GLM-5.3 export state is closed")
        key = (layer_idx, site)
        buffer = self._pending.setdefault(key, {})
        if alpha_name in buffer:
            raise ValueError(f"Duplicate GLM-5.3 mHC alpha {alpha_name} for layer {layer_idx} {site}")
        buffer[alpha_name] = param
        if len(buffer) < len(_HC_ALPHA_ORDER):
            return []
        self._pending.pop(key)
        scale = torch.cat([buffer[name].reshape(1) for name in _HC_ALPHA_ORDER])
        return [(f"model.layers.{layer_idx}.{_HC_SITE_TO_SCALE[site]}", scale)]

    def finish(self) -> None:
        if self._pending:
            raise ValueError(f"Incomplete GLM-5.3 mHC alpha triples: {sorted(self._pending)}")
        self._closed = True

    def abort(self) -> None:
        self._pending.clear()
        self._closed = True


def convert_glm5_next_to_hf(args, name, param, *, export_state: Glm5NextExportState | None = None):
    match = _LAYER_PATTERN.match(name)
    if match:
        layer_idx, rest = match.groups()

        hf_suffix = _KDA_SUFFIX_MAPPING.get(rest) or _INDEXER_SUFFIX_MAPPING.get(rest) or _HC_SUFFIX_MAPPING.get(rest)
        if hf_suffix is not None:
            return [(f"model.layers.{layer_idx}.{hf_suffix}", param)]

        alpha_match = _HC_ALPHA_PATTERN.match(rest)
        if alpha_match:
            if export_state is None:
                raise ValueError("GLM-5.3 mHC export requires a scoped Glm5NextExportState")
            site, alpha_name = alpha_match.groups()
            return export_state.add_hc_scale(layer_idx, site, alpha_name, param)

    return convert_deepseekv3_to_hf(args, name, param)
