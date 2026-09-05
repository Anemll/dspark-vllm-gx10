# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image-only prefill visibility, backported from vLLM #54566.

Separate buffers and kernels preserve the 128-wide text/decode path, even
for text-only requests to the vision checkpoint.
"""

import torch
from vllm.triton_utils import tl, triton


class ImageVisibilityBuffers:
    def __init__(self, max_tokens, max_reqs, window, max_image_tokens, device):
        self.window = window
        self.max_image_tokens = max_image_tokens
        self.width = window + max_image_tokens
        if self.width != 512:
            raise ValueError("Spark vision prefill requires window128 + image384")
        self.indices = torch.empty((max_tokens, 1, self.width), dtype=torch.int32, device=device)
        self.lens = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.left = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.right = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.indptr = torch.empty(max_reqs + 1, dtype=torch.int32, device=device)
        self.starts = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.ends = torch.empty(max_tokens, dtype=torch.int32, device=device)

    def build(self, common, num_decodes, num_decode_tokens, num_prefill_tokens,
              token_to_req, is_valid, block_size):
        if not num_prefill_tokens or not common.mm_req_doc_ranges:
            return None
        # CPU metadata only: never inspect GPU values to choose a graph path.
        seqs = common.seq_lens_cpu_upper_bound
        qsl = common.query_start_loc_cpu
        starts, ends = [], []
        indptr = [0]
        for req in range(common.num_reqs):
            if req >= num_decodes:
                end = int(seqs[req])
                begin = end - int(qsl[req + 1] - qsl[req])
                for left, right in common.mm_req_doc_ranges.get(req, ()):
                    if right < begin or left >= end:
                        continue
                    if right >= end:
                        raise ValueError("DSv4 image prefill was split: require disable_chunked_mm_input")
                    if right - left + 1 > self.max_image_tokens:
                        raise ValueError("Image span exceeds checkpoint visibility budget")
                    starts.append(left)
                    ends.append(right)
            indptr.append(len(starts))
        if not starts:
            return None
        if len(starts) > self.starts.numel():
            raise ValueError("Too many active image spans")
        # Blocking copies deliberately keep temporary host storage alive until
        # transfer completes; never launch asynchronous copies from temporaries.
        self.indptr[:len(indptr)].copy_(torch.tensor(indptr, dtype=torch.int32))
        self.starts[:len(starts)].copy_(torch.tensor(starts, dtype=torch.int32))
        self.ends[:len(ends)].copy_(torch.tensor(ends, dtype=torch.int32))
        _compute_image_visibility_kernel[(num_prefill_tokens,)](
            self.left, self.right, self.indptr, self.starts, self.ends,
            common.query_start_loc, common.seq_lens, token_to_req,
            self.max_image_tokens, token_offset=num_decode_tokens,
        )
        _compute_image_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
            self.indices, self.indices.stride(0), self.lens, self.window, self.width,
            self.left, self.right, common.query_start_loc, common.seq_lens,
            token_to_req, is_valid, common.block_table_tensor,
            common.block_table_tensor.stride(0), block_size,
            token_offset=num_decode_tokens, HAS_IMAGE=True, TRITON_BLOCK_SIZE=512,
        )
        return self.indices[:num_prefill_tokens], self.lens[:num_prefill_tokens]

@triton.jit(do_not_specialize=["token_offset"])
def _compute_image_visibility_kernel(
    left_visible_ptr,
    right_visible_ptr,
    span_indptr_ptr,
    span_starts_ptr,
    span_ends_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    max_image_tokens,
    token_offset,
):
    """Per-token in-image visible counts (port of `get_image_visible`).

    One program per prefill token. A token at position pos inside a span
    [span_start, span_end] sees min(pos - span_start, max_image_tokens - 1)
    extra tokens to its left and min(span_end - pos, max_image_tokens) to its
    right; tokens outside every span get 0/0 (plain causal window).
    """
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    pos = seq_len - (query_end - query_start) + token_idx - query_start

    span_lo = tl.load(span_indptr_ptr + req_idx)
    span_hi = tl.load(span_indptr_ptr + req_idx + 1)
    left = tl.zeros((), dtype=tl.int32)
    right = tl.zeros((), dtype=tl.int32)
    for i in range(span_lo, span_hi):
        span_start = tl.load(span_starts_ptr + i)
        span_end = tl.load(span_ends_ptr + i)
        in_span = (pos >= span_start) & (pos <= span_end)
        left = tl.where(
            in_span, tl.minimum(pos - span_start, max_image_tokens - 1), left
        )
        right = tl.where(in_span, tl.minimum(span_end - pos, max_image_tokens), right)
    tl.store(left_visible_ptr + token_idx, left)
    tl.store(right_visible_ptr + token_idx, right)


# TODO(ben): unify this kernel to reduce duplication


@triton.jit(
    do_not_specialize=[
        "swa_indices_stride",
        "block_table_stride",
        "token_offset",
    ]
)
def _compute_image_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    left_visible_ptr,
    right_visible_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    HAS_IMAGE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < index_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    if HAS_IMAGE:
        # In-image bidirectional visibility widens the window: the window
        # starts up to max(left - (window - 1), 0) positions earlier and
        # extends `right` positions past the query token.
        left = tl.load(left_visible_ptr + token_idx)
        right = tl.load(right_visible_ptr + token_idx)
    else:
        left = 0
        right = 0
    left_add = tl.maximum(left - (window_size - 1), 0)
    start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)
    end_pos = pos + right + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )
