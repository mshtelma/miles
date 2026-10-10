"""Compare GLM's two-stage-gate KDA with an unsharded FLA reference.

Run with ``torchrun --nproc_per_node=8 tests/fast-gpu/test_glm53_kda_cp.py``.
The three layouts isolate TP sharding, contiguous CP, and their combination.
"""

import json
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.kernels.attention.delta_rule import DeltaRuleHeads
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention, Glm5NextKDAShape

sys.path.insert(0, os.path.dirname(__file__))
from delta_rule_reference import ReplicatedKDA, packed, rel_err, shard  # noqa: E402


HIDDEN = 256
HEAD_DIM = 64
HEADS = DeltaRuleHeads(num_k_heads=8, num_v_heads=8, head_k_dim=HEAD_DIM, head_v_dim=HEAD_DIM)
SEQ_LEN = 512
CU_SEQLENS = (0, 256, SEQ_LEN)
OUTPUT_TOLERANCE = 2e-2
PARAM_TOLERANCE = 5e-2


def _setup(tp_size: int, cp_size: int) -> None:
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(1234)


def _reference() -> ReplicatedKDA:
    torch.manual_seed(7)
    reference = ReplicatedKDA(HIDDEN, HEADS, torch.bfloat16).cuda()
    reference.g_proj = nn.Sequential(
        nn.Linear(HIDDEN, HEAD_DIM, bias=False),
        nn.Linear(HEAD_DIM, HEADS.value_dim, bias=False),
    ).to(device="cuda", dtype=torch.bfloat16)
    return reference


def _config(tp_size: int, cp_size: int) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS.num_v_heads,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        layernorm_epsilon=1e-6,
    )
    config.glm5_next_kda = Glm5NextKDAShape(
        num_heads=HEADS.num_v_heads,
        head_dim=HEAD_DIM,
        conv_kernel_size=4,
        gate_lower_bound=-5.0,
        allgather_cp=True,
    )
    return config


def _load_reference_weights(layer: Glm5NextKDAAttention, reference: ReplicatedKDA) -> None:
    core = layer.linear_attn
    tp_group = parallel_state.get_tensor_model_parallel_group()
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj"):
            getattr(core, name).weight.copy_(shard(getattr(reference, name).weight, 0, tp_group))
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            getattr(core, name).weight.copy_(shard(getattr(reference, name).weight, 0, tp_group))
        core.g_a_proj.weight.copy_(reference.g_proj[0].weight)
        core.g_b_proj.weight.copy_(shard(reference.g_proj[1].weight, 0, tp_group))
        core.f_a_proj.weight.copy_(reference.f_a_proj.weight)
        core.A_log.copy_(shard(reference.A_log, 0, tp_group))
        core.dt_bias.copy_(shard(reference.dt_bias, 0, tp_group))
        core.norm.weight.copy_(reference.o_norm.weight)
        core.out_proj.weight.copy_(shard(reference.o_proj.weight, 1, tp_group))


def _build_layer(tp_size: int, cp_size: int, reference: ReplicatedKDA) -> Glm5NextKDAAttention:
    layer = Glm5NextKDAAttention(_config(tp_size, cp_size), layer_number=1)
    layer.bfloat16()
    enforce_marked_param_dtypes([layer])
    core = layer.linear_attn
    assert core.A_log.dtype == core.dt_bias.dtype == torch.float32
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        assert getattr(core, name).weight.dtype == torch.float32
    _load_reference_weights(layer, reference)
    return layer


def _parameter_errors(layer: Glm5NextKDAAttention, reference: ReplicatedKDA) -> dict[str, float]:
    core = layer.linear_attn
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pairs = {}
    for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj"):
        pairs[name] = (shard(getattr(reference, name).weight.grad, 0, tp_group), getattr(core, name).weight.grad)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        pairs[name] = (shard(getattr(reference, name).weight.grad, 0, tp_group), getattr(core, name).weight.grad)
    pairs.update(
        {
            "g_a_proj": (reference.g_proj[0].weight.grad, core.g_a_proj.weight.grad),
            "g_b_proj": (shard(reference.g_proj[1].weight.grad, 0, tp_group), core.g_b_proj.weight.grad),
            "f_a_proj": (reference.f_a_proj.weight.grad, core.f_a_proj.weight.grad),
            "A_log": (shard(reference.A_log.grad, 0, tp_group), core.A_log.grad),
            "dt_bias": (shard(reference.dt_bias.grad, 0, tp_group), core.dt_bias.grad),
            "norm": (reference.o_norm.weight.grad, core.norm.weight.grad),
            "out_proj": (shard(reference.o_proj.weight.grad, 1, tp_group), core.out_proj.weight.grad),
        }
    )
    return {name: rel_err(expected, actual) for name, (expected, actual) in pairs.items()}


def _check_case(tp_size: int, cp_size: int) -> dict[str, float]:
    _setup(tp_size, cp_size)
    reference = _reference()
    layer = _build_layer(tp_size, cp_size, reference)
    torch.manual_seed(11)
    full = torch.randn(1, SEQ_LEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn_like(full)
    cu_seqlens = torch.tensor(CU_SEQLENS, device="cuda", dtype=torch.int32)

    full_input = full.clone().requires_grad_()
    expected = reference(full_input, cu_seqlens)
    expected.backward(gradient)

    cp_rank = parallel_state.get_context_parallel_rank()
    local = full[0].chunk(cp_size, dim=0)[cp_rank].unsqueeze(1).contiguous().requires_grad_()
    local_gradient = gradient[0].chunk(cp_size, dim=0)[cp_rank].unsqueeze(1).contiguous()
    actual, _ = layer(local, packed_seq_params=packed(cu_seqlens))
    actual.backward(local_gradient)

    errors = {
        "out": rel_err(expected[0].chunk(cp_size, dim=0)[cp_rank], actual[:, 0]),
        "dx": rel_err(full_input.grad[0].chunk(cp_size, dim=0)[cp_rank], local.grad[:, 0]),
    }
    if cp_size == 1:
        errors.update(_parameter_errors(layer, reference))
    dist.barrier()
    parallel_state.destroy_model_parallel()
    return errors


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError("GLM-5.3 KDA probe requires eight GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    all_errors = {}
    for tp_size, cp_size in ((8, 1), (1, 8), (4, 2)):
        all_errors[f"tp{tp_size}_cp{cp_size}"] = _check_case(tp_size, cp_size)
    bad = {
        case: {
            name: value
            for name, value in errors.items()
            if value > (PARAM_TOLERANCE if name not in {"out", "dx"} else OUTPUT_TOLERANCE)
        }
        for case, errors in all_errors.items()
    }
    bad = {case: errors for case, errors in bad.items() if errors}
    failed = torch.tensor(bool(bad), device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("GLM53_KDA_PROBE=" + json.dumps({"errors": all_errors, "bad": bad}, sort_keys=True), flush=True)
    dist.destroy_process_group()
    return int(failed.item())


if __name__ == "__main__":
    sys.exit(main())
