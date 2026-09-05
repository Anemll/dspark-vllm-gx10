# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Bounded image-only FlashInfer numerical canary; helpers from pinned FlashInfer."""
import json
import hashlib
import os
from pathlib import Path
import torch
from vllm.models.deepseek_v4.nvidia.vision_sparse import image_sparse_prefill

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round inverse scale to the nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv4 FP8 FOOTER format."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)

    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(
                torch.uint8
            )
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dsv4`."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, 512)


def _ref_sparse_attn(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SDPA over sparse-gathered KV."""
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2

    if attn_sink is not None:
        sink = attn_sink.float()
        sink_log2 = sink / LN2
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))
        out_f = out_f * factor.unsqueeze(-1)
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )

    return out_f.to(torch.bfloat16), lse_log2

def main():
    torch.manual_seed(4104)
    free, total = torch.cuda.mem_get_info()
    if free < 512 * 1024**2:
        raise RuntimeError("Insufficient free GPU memory for bounded numerical canary")
    torch.cuda.set_per_process_memory_fraction(0.01)
    results = []
    # TP2 uses H32; H64 guards the other compiled specialization. C4's
    # compressed cache has page64; C128's has page2 and runtime topk128.
    for heads, extra_pbs, extra_topk in ((32, 0, 0), (32, 64, 512),
                                        (32, 2, 128), (64, 64, 512), (64, 2, 128)):
        tokens, main_pbs, count = 65, 64, 1024
        q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 3
        main = quantize_kv_dsv4(torch.randn(count//main_pbs, main_pbs, 1, 512,
                                   device="cuda", dtype=torch.bfloat16) / 3)
        values = dequantize_kv_dsv4(main).reshape(-1, 512)
        indices = torch.randint(count, (tokens, 1, 512), device="cuda", dtype=torch.int32)
        indices[:, :, 389:] = -1
        lengths = torch.randint(0, 513, (tokens,), device="cuda", dtype=torch.int32)
        lengths[0] = 0
        sink = torch.randn(heads, device="cuda")
        extra = extra_indices = extra_lengths = None
        virtual_indices = indices.squeeze(1).clone()
        virtual_indices.masked_fill_(torch.arange(512, device="cuda")[None] >= lengths[:, None], -1)
        if extra_pbs:
            extra = quantize_kv_dsv4(torch.randn(count//extra_pbs, extra_pbs, 1, 512,
                                      device="cuda", dtype=torch.bfloat16) / 3)
            values = torch.cat((values, dequantize_kv_dsv4(extra).reshape(-1, 512)))
            extra_indices = torch.randint(count, (tokens, 1, extra_topk), device="cuda", dtype=torch.int32)
            extra_indices[:, :, extra_topk*3//4:] = -1
            extra_lengths = torch.randint(0, extra_topk+1, (tokens,), device="cuda", dtype=torch.int32)
            extra_lengths[0] = 0
            extra_ref = extra_indices.squeeze(1).clone()
            extra_ref.masked_fill_(torch.arange(extra_topk, device="cuda")[None] >= extra_lengths[:, None], -1)
            extra_ref = torch.where(extra_ref < 0, extra_ref, extra_ref + count)
            virtual_indices = torch.cat((virtual_indices, extra_ref), dim=-1)
        out = torch.empty_like(q)
        image_sparse_prefill(q, main, indices, out, 512**-0.5, sink, lengths,
                             extra, extra_indices, extra_lengths)
        if not extra_pbs:
            # The baseline already supports single-cache topk512. Verify that
            # this separately linked module exactly retains that implementation.
            import tvm_ffi
            baseline_path = Path(os.environ["BASELINE_SPARSE_LIBRARY"])
            assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == os.environ["BASELINE_SPARSE_SHA256"]
            baseline_module = tvm_ffi.load_module(str(baseline_path))
            baseline = torch.empty_like(out)
            baseline_lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
            baseline_module.sparse_mla_sm120_paged_attention(
                q, main, indices, baseline, baseline_lse, 512**-0.5,
                1, lengths, sink, None, None, None,
            )
            torch.testing.assert_close(out, baseline, atol=0, rtol=0)
        # Chunk the dense reference to bound allocation, not the tested kernel.
        maximum = 0.
        for start in range(0, tokens, 8):
            end = min(start+8, tokens)
            expected, _ = _ref_sparse_attn(q[start:end], values, virtual_indices[start:end],
                                           512**-0.5, 512, sink)
            # Match the pinned FlashInfer correctness suite's published
            # tolerance for its quantized prefill kernels.
            torch.testing.assert_close(out[start:end], expected, atol=5e-2, rtol=5e-2)
            maximum = max(maximum, float((out[start:end]-expected).abs().max()))
        torch.cuda.synchronize()
        results.append(dict(heads=heads, extra_page=extra_pbs, extra_topk=extra_topk,
                            max_abs_error=maximum))
        print(json.dumps(results[-1]), flush=True)
    peak = torch.cuda.max_memory_allocated()
    if peak > 128*1024**2:
        raise RuntimeError(f"Numerical canary exceeded 128 MiB allocation cap: {peak}")
    print(json.dumps(dict(passed=len(results), peak_bytes=peak, free_before=free)), flush=True)


if __name__ == "__main__":
    main()
