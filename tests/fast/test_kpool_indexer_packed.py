"""Kpool pools and queries stay inside each packed sequence, including across CP shards."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from miles.kernels.attention.dsa import kpool as kpool_module  # noqa: E402
from miles.kernels.attention.dsa.kpool import kpool_select_topk, pool_boundaries  # noqa: E402

KPOOL = 4
SEQ_LENS = [37, 13, 22]  # none is a multiple of KPOOL


def _cu_seqlens(seq_lens):
    return torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)


def test_pools_are_counted_per_sequence():
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)

    counts = (pool_cu_seqlens[1:] - pool_cu_seqlens[:-1]).tolist()
    assert counts == [length // KPOOL for length in SEQ_LENS]

    # A grid laid over the whole buffer would produce more pools than the per-sequence
    # count, and the extra ones are exactly the pools that straddle a boundary.
    assert sum(counts) <= int(cu_seqlens[-1]) // KPOOL


def test_eligible_pool_window_never_leaves_the_query_sequence():
    """Mirrors the index arithmetic in `kpool_select_topk`."""
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)
    total_tokens = int(cu_seqlens[-1])

    token_ids = torch.arange(total_tokens)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices]
    pool_base = pool_cu_seqlens[seq_indices]
    local_positions = token_ids - seq_token_base
    eligible_pools = torch.div(local_positions + 1, KPOOL, rounding_mode="floor")

    window_start = pool_base
    window_end = pool_base + eligible_pools
    own_window_end = pool_cu_seqlens[seq_indices + 1]

    assert bool((window_start >= pool_base).all())
    assert bool((window_end <= own_window_end).all()), "a query can see pools belonging to a later sequence"

    # Every query outside the first sequence starts strictly after that sequence's pools,
    # i.e. it can never be handed pool 0 of the packed buffer.
    is_later_seq = seq_indices > 0
    assert bool((window_start[is_later_seq] >= int(pool_cu_seqlens[1])).all())
    assert int(pool_cu_seqlens[1]) > 0


def test_local_positions_restart_per_sequence():
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    total_tokens = int(cu_seqlens[-1])
    token_ids = torch.arange(total_tokens)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    local_positions = token_ids - cu_seqlens[seq_indices]

    expected = torch.cat([torch.arange(length) for length in SEQ_LENS])
    torch.testing.assert_close(local_positions, expected)


def _select_highest_eligible_pools(pool_ranges, seq_bases, pool_bases, local_positions, topk, kpool):
    """CPU stand-in for the GPU selector when each pool's score equals its global pool ID."""
    width = (topk + kpool - 1 + kpool_module.SPARSE_MLA_BLOCK - 1) // kpool_module.SPARSE_MLA_BLOCK
    width *= kpool_module.SPARSE_MLA_BLOCK
    out = torch.full((pool_ranges.shape[0], width), -1, dtype=torch.int32)
    for row, (pool_range, base, pool_base, position) in enumerate(
        zip(pool_ranges.tolist(), seq_bases.tolist(), pool_bases.tolist(), local_positions.tolist(), strict=True)
    ):
        if position + 1 <= topk:
            out[row, : position + 1] = torch.arange(base, base + position + 1, dtype=torch.int32)
            continue

        first_pool, last_pool = pool_range
        selected = min(topk // kpool, last_pool - first_pool)
        pool_ids = last_pool - 1 - torch.arange(selected, dtype=torch.int32)
        offsets = torch.arange(kpool, dtype=torch.int32)
        out[row, : selected * kpool] = (base + (pool_ids[:, None] - pool_base) * kpool + offsets).flatten()
        tail = (position + 1) % kpool
        out[row, topk : topk + tail] = base + (position + 1) // kpool * kpool + offsets[:tail]
    return out


def _reference_selection(seq_lens, topk):
    """Select by global query position, independently of the kpool offset arithmetic."""
    width = (topk + KPOOL - 1 + 63) // 64 * 64
    out = torch.full((sum(seq_lens), width), -1, dtype=torch.int32)
    seq_start = 0
    offsets = torch.arange(KPOOL, dtype=torch.int32)
    for length in seq_lens:
        for position in range(length):
            row = seq_start + position
            if position + 1 <= topk:
                out[row, : position + 1] = torch.arange(seq_start, row + 1, dtype=torch.int32)
                continue

            eligible = (position + 1) // KPOOL
            selected = min(topk // KPOOL, eligible)
            pool_ids = eligible - 1 - torch.arange(selected, dtype=torch.int32)
            out[row, : selected * KPOOL] = (seq_start + pool_ids[:, None] * KPOOL + offsets).flatten()
            tail = (position + 1) % KPOOL
            out[row, topk : topk + tail] = seq_start + eligible * KPOOL + offsets[:tail]
        seq_start += length
    return out


@pytest.fixture
def cpu_kpool_selector(monkeypatch):
    """Replace only the GPU scorer/expander; exercise kpool_select_topk's real query mapping."""
    module_name = "miles.kernels.attention.dsa.tilelang.indexer_fwd"
    fake_indexer = types.ModuleType(module_name)

    def score_by_pool_id(_q, _kv, _weights, pool_starts, pool_ends, *, clean_logits):
        assert clean_logits
        return torch.stack((pool_starts, pool_ends), dim=1)

    fake_indexer.indexer_fwd_interface = score_by_pool_id
    monkeypatch.setitem(sys.modules, module_name, fake_indexer)
    monkeypatch.setattr(kpool_module, "_select_expand_tail", _select_highest_eligible_pools)
    return kpool_select_topk


@pytest.mark.parametrize(("cp_size", "padding"), [(2, 249), (4, 505), (8, 1017)])
@pytest.mark.parametrize("topk", [16, 2048])
def test_query_offset_matches_unsharded_packed_selection(cpu_kpool_selector, cp_size, padding, topk):
    seq_lens = [7, 1500, 2093, 500, 3, padding]
    total = sum(seq_lens)
    local_tokens = total // cp_size
    cu_seqlens = _cu_seqlens(seq_lens)
    index_q = torch.zeros((total, 1, 1), dtype=torch.bfloat16)
    pooled_k = torch.zeros((total // KPOOL, 1), dtype=torch.bfloat16)
    head_weights = torch.ones((total, 1))

    full = cpu_kpool_selector(index_q, pooled_k, head_weights, cu_seqlens, topk, KPOOL)
    shards = [
        cpu_kpool_selector(
            index_q[offset : offset + local_tokens],
            pooled_k,
            head_weights[offset : offset + local_tokens],
            cu_seqlens,
            topk,
            KPOOL,
            query_offset=offset,
        )
        for offset in range(0, total, local_tokens)
    ]

    torch.testing.assert_close(full[:, 0], _reference_selection(seq_lens, topk), rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(shards), full, rtol=0, atol=0)
    if topk == 2048:
        assert full.shape[-1] == 2112
        assert sum(max(0, length - topk) for length in seq_lens) == 45

    wrong_offset = cpu_kpool_selector(
        index_q[local_tokens : 2 * local_tokens],
        pooled_k,
        head_weights[local_tokens : 2 * local_tokens],
        cu_seqlens,
        topk,
        KPOOL,
    )
    assert not torch.equal(wrong_offset, full[local_tokens : 2 * local_tokens])
