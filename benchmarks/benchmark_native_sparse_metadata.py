#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded SM121 gate for real DSpark metadata and padded indexer kernels."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    import torch
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_dspark_noncausal_swa_indices_kernel as kernel,
    )
    from vllm.v1.attention.backends.mla.indexer import _prepare_uniform_decode_kernel
    from vllm.utils.dsv4_sparse_binary import verify_native_sparse_binary

    assert torch.cuda.get_device_capability() == (12, 1)
    verify_native_sparse_binary(32)
    device = "cuda"
    query_starts = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32, device=device)
    requests = torch.arange(4, device=device, dtype=torch.int32).repeat_interleave(5)
    block_table = torch.tensor([[2, 7, 4, 3], [6, 3, 5, 1], [7, 1, 5, 4], [0, 0, 0, 0]],
                               dtype=torch.int32, device=device)
    valid = requests != 3
    rows = []
    for context in (5, 133, 255, 256, 257, 511, 768):
        seq_lens = torch.tensor([context, max(5, context - 3), context, 0],
                                dtype=torch.int32, device=device)
        results = []
        for width in (192, 256, 512):
            indices = torch.full((20, 1, width), -1, dtype=torch.int32, device=device)
            lengths = torch.full((20,), -1, dtype=torch.int32, device=device)
            kernel[(20,)](indices, indices.stride(0), lengths, 128, width,
                          query_starts, seq_lens, requests, valid, block_table,
                          block_table.stride(0), 256, 0, TRITON_BLOCK_SIZE=1024)
            torch.cuda.synchronize()
            for row in range(20):
                req = row // 5
                length = int(seq_lens[req])
                expected = list(range(max(length - 5 - 128, 0), length)) if req < 3 else []
                slots = [int(block_table[req, pos // 256]) * 256 + pos % 256 for pos in expected]
                assert int(lengths[row]) == len(slots), (context, width, row)
                assert indices[row, 0, :len(slots)].tolist() == slots
                if req < 3:
                    assert torch.all(indices[row, 0, len(slots):] == -1)
            results.append(lengths.cpu().tolist())
        assert results[0] == results[1] == results[2]
        rows.append({"context": context, "widths": [192, 256, 512], "status": "passed"})

    # Uniform MTP/DSpark padding must never emit negative context lengths.
    lengths = torch.tensor([0, 1, 6, 128], dtype=torch.int32, device=device)
    expanded = torch.empty((24, 4), dtype=torch.int32, device=device)
    decoded = torch.empty(24, dtype=torch.int32, device=device)
    counts = torch.empty_like(decoded)
    _prepare_uniform_decode_kernel[(24,)](
        lengths, decoded, block_table, block_table.stride(0), expanded,
        expanded.stride(0), counts, 6, BLOCK_SIZE=4)
    torch.cuda.synchronize()
    expected = [max(length - 6 + i + 1, 0) for length in (0, 1, 6, 128) for i in range(6)]
    assert decoded.tolist() == expected
    assert counts.tolist() == [1] * 24
    report = {"status": "passed", "elapsed_s": time.monotonic() - start,
              "metadata_cases": rows, "padding_lengths": decoded.tolist(),
              "gpu": torch.cuda.get_device_name(), "binary_verified": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
