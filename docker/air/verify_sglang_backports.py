"""Run CPU contract probes for the pinned GLM SGLang backports.

Run this after applying ``patches/sglang-glm53-150k.patch`` to the pinned
SGLang tree. The GPU top-k and memory checks remain AIR release gates.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as kpool


def _fake_logits(q, kv, _weights, _starts, _ends, *, clean_logits):
    assert clean_logits
    return q[:, 0, 0].float().unsqueeze(1).expand(-1, kv[0].shape[0]).contiguous()


def _fake_topk(
    _self,
    logits,
    pool_lens,
    *,
    seq_lens=None,
    page_table=None,
    topk_offsets=None,
    row_starts=None,
    out_rows=None,
    page_table_row_index=None,
    return_raw_indices=False,
):
    row_ids = logits[:, 0].int()
    if page_table_row_index is not None:
        mapping = page_table[page_table_row_index, 0]
    elif page_table is not None:
        mapping = page_table[:, 0]
    else:
        mapping = torch.zeros_like(row_ids)
    if topk_offsets is not None:
        mapping = mapping + topk_offsets
    if row_starts is not None:
        mapping = mapping + row_starts
    selected = torch.stack((row_ids, mapping), dim=1)
    raw = torch.stack((row_ids, pool_lens), dim=1) if return_raw_indices else None
    if out_rows is not None and out_rows > len(row_ids):
        pad = torch.full((out_rows - len(row_ids), 2), -1, dtype=torch.int32)
        selected = torch.cat((selected, pad))
        if raw is not None:
            raw = torch.cat((raw, pad))
    return selected, raw


def _check_row_chunk_mapping() -> None:
    layer = kpool.IndexerKPool.__new__(kpool.IndexerKPool)
    torch.nn.Module.__init__(layer)
    layer.index_topk = 2
    layer.index_kpool = 4
    values = torch.arange(6, dtype=torch.float32)
    q = values.reshape(6, 1, 1)
    keys = torch.zeros((3, 1))
    pool_lens = torch.arange(1, 7, dtype=torch.int32)
    row_index = torch.tensor([0, 1, 2, 0, 1, 2])
    request_table = torch.tensor([[10], [20], [30]], dtype=torch.int32)
    row_table = torch.tensor([[11], [22], [33], [44], [55], [66]], dtype=torch.int32)
    chunks = (slice(0, 2), slice(2, 4), slice(4, 6))
    common = {
        "q_fp8": q,
        "weights": torch.ones((6, 1)),
        "kv_fp8": (keys, torch.ones(3)),
        "logits_starts": torch.zeros(6, dtype=torch.int32),
        "logits_ends": pool_lens,
        "pool_lens": pool_lens,
        "seq_lens": pool_lens,
        "topk_offsets": torch.arange(6, dtype=torch.int32),
        "topk_row_starts": None,
        "out_rows": 8,
        "return_raw_indices": True,
    }
    with (
        patch.object(kpool, "deep_gemm", SimpleNamespace(fp8_mqa_logits=_fake_logits), create=True),
        patch.object(kpool, "_mqa_logits_row_chunks", return_value=chunks),
        patch.object(kpool.IndexerKPool, "_topk_from_kpool_logits", _fake_topk),
    ):
        selected, raw = layer._kpool_topk_by_row_chunks(
            **common,
            page_table=request_table,
            page_table_row_index=row_index,
        )
        expected = torch.stack((torch.arange(6), torch.tensor([10, 21, 32, 13, 24, 35])), dim=1)
        assert torch.equal(selected[:6], expected)
        assert torch.equal(raw[:6, 0], torch.arange(6))
        assert torch.equal(raw[:6, 1], pool_lens)
        assert torch.all(selected[6:] == -1) and torch.all(raw[6:] == -1)

        per_row, _ = layer._kpool_topk_by_row_chunks(
            **common,
            page_table=row_table,
            page_table_row_index=None,
        )
        assert torch.equal(per_row[:6, 1], torch.tensor([11, 23, 35, 47, 59, 71]))


def _check_budget_and_capture() -> None:
    with (
        patch.object(kpool, "mqa_logits_budget_bytes", return_value=4_096_000),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
    ):
        chunks = kpool._mqa_logits_row_chunks(
            num_rows=9_000,
            num_cols=1_024,
            device=torch.device("cuda:0"),
        )
    assert len(chunks) == 9
    assert chunks[0] == slice(0, 1_000)
    assert chunks[-1] == slice(8_000, 9_000)

    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
        captured = kpool._mqa_logits_row_chunks(
            num_rows=9_000,
            num_cols=1_024,
            device=torch.device("cuda:0"),
        )
    assert captured == (slice(0, 9_000),)


def main() -> None:
    _check_row_chunk_mapping()
    _check_budget_and_capture()
    print(json.dumps({"sglang_backport_cpu_contracts": "passed"}))


if __name__ == "__main__":
    main()
