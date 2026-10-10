"""GLM-5.3-Flash KDA on the shared head-sharded linear-attention layer.

GLM factors the output gate as ``g_b_proj(g_a_proj(x))``. The shared KDA supplies
the three FP32 convolutions, recurrence, output norm, and context parallelism.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from torch import nn

from miles.kernels.attention.delta_rule.heads import DeltaRuleHeads
from miles.kernels.attention.delta_rule.rule import KimiDeltaRule
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer, Projections


def _get_text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_fields(text_config) -> dict:
    """Read the checkpoint's KDA geometry without silently selecting defaults."""
    linear_config = getattr(text_config, "linear_attn_config", None)
    if linear_config is None:
        linear_config = {}
    if not isinstance(linear_config, dict):
        raise ValueError("GLM-5.3 linear_attn_config must be a mapping")

    def field(key: str, legacy_attr: str):
        value = linear_config.get(key)
        if value is None:
            value = getattr(text_config, legacy_attr, None)
        if value is None:
            raise ValueError(f"GLM-5.3 KDA requires {key} in the HF config")
        return value

    integer_fields = {
        "num_heads": field("num_heads", "linear_num_heads"),
        "head_dim": field("head_dim", "linear_head_dim"),
        "conv_kernel_size": field("short_conv_kernel_size", "linear_conv_kernel_dim"),
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"GLM-5.3 KDA {name} must be a positive integer, got {value!r}")

    bound = linear_config.get("gate_lower_bound")
    if bound is None:
        bound = getattr(text_config, "gate_lower_bound", None)
    if bound is None:
        bound = getattr(text_config, "linear_lower_bound", None)
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound >= 0:
        raise ValueError("GLM-5.3 KDA requires a finite negative gate_lower_bound in the HF config")

    return {**integer_fields, "gate_lower_bound": float(bound)}


@dataclass(frozen=True)
class Glm5NextKDAShape:
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    gate_lower_bound: float
    allgather_cp: bool


class Glm5NextDeltaAttention(KimiDeltaAttention):
    """Kimi's head-sharded KDA with GLM's low-rank output gate."""

    def _build_projections(self) -> None:
        hidden, local = self.config.hidden_size, self.local
        self.q_proj = self.sharded_linear("q_proj", hidden, local.key_dim)
        self.k_proj = self.sharded_linear("k_proj", hidden, local.key_dim)
        self.v_proj = self.sharded_linear("v_proj", hidden, local.value_dim)
        self.b_proj = self.sharded_linear("b_proj", hidden, local.num_v_heads)
        self.f_a_proj = self._replicated_down_projection(hidden)
        self.f_b_proj = self.sharded_linear("f_b_proj", self.heads.head_v_dim, local.value_dim)
        self.g_a_proj = self._replicated_down_projection(hidden)
        self.g_b_proj = self.sharded_linear("g_b_proj", self.heads.head_v_dim, local.value_dim)

    def _replicated_down_projection(self, hidden: int) -> nn.Linear:
        linear = nn.Linear(
            hidden,
            self.heads.head_v_dim,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        self.config.init_method(linear.weight)
        return linear

    def project(self, x: torch.Tensor) -> Projections:
        f_a_weight = copy_to_tensor_model_parallel_region(self.f_a_proj.weight, group=self.tp_group)
        g_a_weight = copy_to_tensor_model_parallel_region(self.g_a_proj.weight, group=self.tp_group)
        # BF16 down GEMMs change their rounded outputs with the padded sequence length on B300.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            f_down = F.linear(x_fp32, f_a_weight.float()).to(dtype=f_a_weight.dtype)
            g_down = F.linear(x_fp32, g_a_weight.float()).to(dtype=g_a_weight.dtype)
        return Projections(
            qkv=(self.q_proj(x), self.k_proj(x), self.v_proj(x)),
            gate=self.g_b_proj(g_down),
            beta_logits=self.b_proj(x),
            decay=self.f_b_proj(f_down),
        )


class Glm5NextKDAAttention(LinearAttentionLayer):
    """Megatron ``self_attention`` drop-in for a GLM KDA layer."""

    is_kda = True

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        del layer_number, cp_comm_type, name
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        shape: Glm5NextKDAShape = config.glm5_next_kda
        if shape.num_heads % pg_collection.tp.size() != 0:
            raise ValueError(f"GLM-5.3 KDA heads {shape.num_heads} must divide TP={pg_collection.tp.size()}")
        if pg_collection.cp.size() > 1 and not shape.allgather_cp:
            raise ValueError("GLM-5.3 KDA CP requires contiguous allgather-CP shards")
        heads = DeltaRuleHeads(
            num_k_heads=shape.num_heads,
            num_v_heads=shape.num_heads,
            head_k_dim=shape.head_dim,
            head_v_dim=shape.head_dim,
        )
        core = Glm5NextDeltaAttention(
            config,
            heads,
            KimiDeltaRule(gate_lower_bound=shape.gate_lower_bound),
            conv_kernel_size=shape.conv_kernel_size,
            norm_eps=config.layernorm_epsilon,
            tp_group=pg_collection.tp,
        )
        super().__init__(
            config,
            linear_attn=core,
            input_layernorm=nn.Identity(),
            pg_collection=pg_collection,
            allgather_cp=shape.allgather_cp,
        )
