"""Compare packed GLM-5.3 DSA selection and sparse backward across CP shards.

Run with ``torchrun --standalone --nproc-per-node=8 tests/fast-gpu/test_glm53_dsa_cp.py``.
The CP1 reference, CP2 pair, and CP8 group use the same packed documents and
global KV rows. Only rank zero computes the reference to avoid repeated work.
"""

import json
import math
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from miles.kernels.attention.dsa import build_pooled_keys, kpool_select_topk, sparse_attention
from miles.kernels.attention.dsa.kpool import pool_boundaries

LENS = (7, 1500, 2093, 500, 3, 1017)
KPOOL = 4
HEADS = 8
HEAD_DIM = 512
INDEX_DIM = 128
TOPK = 64
TOLERANCE = {"out": 0.01, "dq": 0.025, "dkv": 0.05}


def _relative_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected = expected.float()
    actual = actual.float()
    value = float((expected - actual).norm() / expected.norm().clamp_min(1e-12))
    if not math.isfinite(value):
        raise ValueError("DSA CP comparison produced a non-finite relative error")
    return value


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    total = sum(LENS)
    cu = torch.tensor([0, *LENS], device=device, dtype=torch.int32).cumsum(0)
    pool_cu = pool_boundaries(cu, KPOOL)
    positions = torch.arange(total, device=device)
    sequence = torch.searchsorted(cu, positions, right=True) - 1
    local = positions - cu[sequence]
    complete = local < ((cu[sequence + 1] - cu[sequence]) // KPOOL) * KPOOL
    pool_ids = torch.where(complete, pool_cu[sequence] + local // KPOOL, 0)

    index_k = torch.zeros((total, INDEX_DIM), dtype=torch.bfloat16, device=device)
    index_k[:, 0] = (pool_ids // 128).to(index_k.dtype)
    index_k[:, 1] = (pool_ids % 128).to(index_k.dtype)
    pooled = build_pooled_keys(
        index_k,
        torch.zeros_like(index_k),
        torch.zeros((KPOOL, INDEX_DIM), dtype=torch.float32, device=device),
        cu,
        KPOOL,
    )
    index_q = torch.zeros((total, HEADS, INDEX_DIM), dtype=torch.bfloat16, device=device)
    index_q[:, 0, 0] = 128
    index_q[:, 0, 1] = 1
    head_weights = torch.zeros((total, HEADS), dtype=torch.float32, device=device)
    head_weights[:, 0] = 1

    generator = torch.Generator(device=device).manual_seed(53)
    query = torch.randn((1, total, HEADS, HEAD_DIM), generator=generator, device=device, dtype=torch.bfloat16)
    kv = torch.randn((1, total, 1, HEAD_DIM), generator=generator, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn((1, total, HEADS, HEAD_DIM), generator=generator, device=device, dtype=torch.float32)
    grad_out[:, -LENS[-1] :] = 0
    return {
        "cu": cu,
        "index_q": index_q,
        "pooled": pooled,
        "head_weights": head_weights,
        "query": query,
        "kv": kv,
        "grad_out": grad_out,
    }


def _selection(inputs: dict[str, torch.Tensor], start: int, end: int, topk: int) -> torch.Tensor:
    return kpool_select_topk(
        inputs["index_q"][start:end],
        inputs["pooled"],
        inputs["head_weights"][start:end],
        inputs["cu"],
        topk,
        KPOOL,
        query_offset=start,
    ).unsqueeze(0)


def _sparse_step(inputs: dict[str, torch.Tensor], start: int, end: int, indices: torch.Tensor) -> dict:
    query = inputs["query"][:, start:end].clone().requires_grad_()
    kv = inputs["kv"].clone().requires_grad_()
    output = sparse_attention(query, kv, indices, HEAD_DIM**-0.5)
    (output.float() * inputs["grad_out"][:, start:end]).sum().backward()
    if query.grad is None or kv.grad is None:
        raise ValueError("DSA sparse backward produced no query or KV gradient")
    return {"out": output.detach(), "dq": query.grad.detach(), "dkv": kv.grad.detach()}


def _gather(group, local: torch.Tensor, cp_size: int) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def _cp_case(inputs: dict[str, torch.Tensor], cp_size: int, reference: dict | None) -> dict | None:
    ranks = list(range(cp_size))
    group = dist.new_group(ranks=ranks, backend="nccl")
    rank = dist.get_rank()
    result = None
    if rank < cp_size:
        width = sum(LENS) // cp_size
        start, end = rank * width, (rank + 1) * width
        chosen = _selection(inputs, start, end, TOPK)
        step = _sparse_step(inputs, start, end, chosen)
        gathered = {key: _gather(group, step[key], cp_size) for key in ("out", "dq")}
        gathered["chosen"] = _gather(group, chosen, cp_size)
        kv_gradient = step["dkv"].float()
        dist.all_reduce(kv_gradient, group=group)
        chosen_full = _gather(group, _selection(inputs, start, end, 2048), cp_size)
        if rank == 0:
            result = {
                "selection_exact": bool(torch.equal(gathered["chosen"], reference["chosen"])),
                "selection_2048_exact": bool(torch.equal(chosen_full, reference["chosen_2048"])),
                "out": _relative_error(reference["out"], gathered["out"]),
                "dq": _relative_error(reference["dq"], gathered["dq"]),
                "dkv": _relative_error(reference["dkv"], kv_gradient),
                "out_max_abs": float((reference["out"].float() - gathered["out"].float()).abs().max()),
            }
    dist.barrier()
    if rank < cp_size:
        dist.destroy_process_group(group)
    return result


def main() -> int:
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise ValueError("GLM-5.3 DSA CP probe requires eight GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        inputs = _inputs(device)
        reference = None
        noise = None
        if dist.get_rank() == 0:
            chosen = _selection(inputs, 0, sum(LENS), TOPK)
            baseline = _sparse_step(inputs, 0, sum(LENS), chosen)
            repeated = _sparse_step(inputs, 0, sum(LENS), chosen)
            noise = {key: _relative_error(baseline[key], repeated[key]) for key in baseline}
            reference = {**baseline, "chosen": chosen, "chosen_2048": _selection(inputs, 0, sum(LENS), 2048)}
            if reference["chosen_2048"].shape[-1] != 2112:
                raise ValueError("production kpool top-k width is not 2112")
        dist.barrier()
        cases = {f"cp{size}": _cp_case(inputs, size, reference) for size in (2, 8)}
        failed = torch.zeros((), dtype=torch.int32, device=device)
        if dist.get_rank() == 0:
            bad = {
                case: {
                    key: value
                    for key, value in result.items()
                    if (key.startswith("selection") and not value)
                    or (key in TOLERANCE and value > TOLERANCE[key])
                    or (key == "out_max_abs" and (not math.isfinite(value) or value > 0.1))
                }
                for case, result in cases.items()
            }
            bad = {case: errors for case, errors in bad.items() if errors}
            if any(noise[key] > TOLERANCE[key] / 2 for key in TOLERANCE):
                bad["baseline_noise"] = noise
            failed.fill_(bool(bad))
            print(
                "GLM53_DSA_CP="
                + json.dumps(
                    {"lens": LENS, "tolerance": TOLERANCE, "baseline_noise": noise, "cases": cases, "bad": bad},
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.broadcast(failed, src=0)
        return int(failed.item())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
