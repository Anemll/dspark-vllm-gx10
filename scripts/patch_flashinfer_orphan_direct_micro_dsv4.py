#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Port FlashInfer's exact orphan direct-MoE source to the current DSv4 ABI.

This is deliberately a narrow, hash-pinned source transform.  It starts from
the otherwise-unwired ``MoEDirectMicroKernel`` at FlashInfer revision
``0472b9b3`` and changes only contracts required to execute that literal
kernel against the current image:

* typed CuTeDSL pointer arguments and launches;
* a complete deterministic compile-cache key;
* an explicit ModelOpt ``w13 == [up/w3, gate/w1]`` contract; and
* DeepSeek-V4's clamped SwiGLU activation (``limit=10``).

It does not import B12X's later dot-product, prefetch, retile, or scale-format
optimizations.  The output therefore settles the performance of the literal
FlashInfer implementation rather than a descendant.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "abfad363fae29d15c0c2af127a54b7bafe2ae667c08ff976a2caf6d0828436b2"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def port_source(text: str) -> str:
    text = replace_once(
        text,
        """from flashinfer.cute_dsl.utils import (
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
)""",
        """from flashinfer.cute_dsl.utils import (
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)""",
        "make_ptr import",
    )
    text = replace_once(
        text,
        """        dynamic_down_scale: bool = False,
        w4a16_mode: bool = False,
    ):
        if activation not in {\"silu\", \"relu2\"}:
            raise ValueError(f\"unsupported activation {activation!r}\")
        self.sf_vec_size = sf_vec_size""",
        """        dynamic_down_scale: bool = False,
        w4a16_mode: bool = False,
        swiglu_limit: float | None = None,
        w13_layout: str = \"w13\",
    ):
        if activation not in {\"silu\", \"relu2\"}:
            raise ValueError(f\"unsupported activation {activation!r}\")
        if swiglu_limit is not None and activation != \"silu\":
            raise ValueError(\"swiglu_limit requires the gated (silu) activation\")
        if w13_layout != \"w13\":
            raise ValueError(
                \"literal FlashInfer direct kernel supports only ModelOpt \"
                \"w13=[up/w3,gate/w1] storage\"
            )
        self.w13_layout = w13_layout
        self.w13_gate_first = False
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
        self.sf_vec_size = sf_vec_size""",
        "DSv4 constructor contract",
    )
    text = replace_once(
        text,
        """        self.m1_fc2_onepass = False
        self.grid_x = 0

    @cute.jit
    def _fp4_dot4_for_math(""",
        """        self.m1_fc2_onepass = False
        self.grid_x = 0

    @property
    def __cache_key__(self):
        return (
            self.sf_vec_size,
            self.fast_math,
            self.activation,
            self.share_input_across_experts,
            self.share_expert_scales,
            self.single_token,
            self.dynamic_down_scale,
            self.w4a16_mode,
            self.w13_layout,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self._cfg,
            self.m_const,
            self.m1_fc2_onepass,
        )

    @cute.jit
    def _fp4_dot4_for_math(""",
        "compile cache key",
    )
    text = replace_once(
        text,
        """                    if cutlass.const_expr(self.is_gated):
                        sigmoid = Float32(1.0) / (
                            Float32(1.0) + cute.math.exp(-gate_red, fastmath=False)
                        )
                        activated = sigmoid * gate_red * up_red""",
        """                    if cutlass.const_expr(self.is_gated):
                        if cutlass.const_expr(self.has_swiglu_limit):
                            limit = Float32(self.swiglu_limit)
                            neg_limit = Float32(-self.swiglu_limit)
                            if gate_red > limit:
                                gate_red = limit
                            if up_red > limit:
                                up_red = limit
                            if up_red < neg_limit:
                                up_red = neg_limit
                        sigmoid = Float32(1.0) / (
                            Float32(1.0) + cute.math.exp(-gate_red, fastmath=False)
                        )
                        activated = sigmoid * gate_red * up_red""",
        "DSv4 SwiGLU clamp",
    )
    text = replace_once(
        text,
        """        x: cute.Tensor,
        w1_ptr: cute.Pointer,""",
        """        x_ptr: cute.Pointer,
        w1_ptr: cute.Pointer,""",
        "typed input pointer ABI",
    )
    text = replace_once(
        text,
        """        barrier_count_ptr: cute.Pointer,
        barrier_epoch_ptr: cute.Pointer,""",
        """        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,""",
        "tensor barrier ABI",
    )
    text = replace_once(
        text,
        """        a_input = cute.make_tensor(
            x.iterator, cute.make_layout(Int32(m_val * cfg.k_dim))
        )""",
        """        a_input = cute.make_tensor(
            x_ptr, cute.make_layout(Int32(m_val * cfg.k_dim))
        )""",
        "typed input tensor construction",
    )
    text = replace_once(
        text,
        """        barrier_slots = m_val * Int32(cfg.num_topk + 16)
        barrier_count = cute.make_tensor(
            barrier_count_ptr, cute.make_layout(barrier_slots)
        )
        barrier_epoch = cute.make_tensor(
            barrier_epoch_ptr, cute.make_layout(barrier_slots)
        )

        self.kernel(""",
        """        self.kernel(""",
        "preconstructed barrier tensors",
    )
    text = replace_once(
        text,
        """    ):
        stream = current_cuda_stream()

        compiled_fn(
            x,
            w1_fp4.data_ptr(),
            w1_blockscale.view(torch.uint8).data_ptr(),
            w1_alphas.data_ptr(),
            a1_gscale.data_ptr(),
            a2_gscale.data_ptr(),
            inter_fp32.view(torch.uint32).data_ptr(),
            w2_fp4.data_ptr(),
            w2_blockscale.view(torch.uint8).data_ptr(),
            w2_alphas.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            barrier_count.data_ptr(),
            barrier_epoch.data_ptr(),""",
        """    ):
        def ptr(dtype, tensor):
            return make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )

        ids_dtype = cutlass.Int64 if topk_ids.dtype == torch.int64 else cutlass.Int32
        stream = current_cuda_stream()

        compiled_fn(
            ptr(cutlass.BFloat16, x),
            ptr(cutlass.Uint8, w1_fp4),
            ptr(cutlass.Uint8, w1_blockscale.view(torch.uint8)),
            ptr(cutlass.Float32, w1_alphas),
            ptr(cutlass.Float32, a1_gscale),
            ptr(cutlass.Float32, a2_gscale),
            ptr(cutlass.Uint32, inter_fp32.view(torch.uint32)),
            ptr(cutlass.Uint8, w2_fp4),
            ptr(cutlass.Uint8, w2_blockscale.view(torch.uint8)),
            ptr(cutlass.Float32, w2_alphas),
            ptr(ids_dtype, topk_ids),
            ptr(cutlass.Float32, topk_weights),
            ptr(cutlass.BFloat16, out),
            barrier_count,
            barrier_epoch,""",
        "typed runtime launch ABI",
    )

    required = (
        'self.w13_layout = w13_layout',
        'self.has_swiglu_limit = swiglu_limit is not None',
        'def __cache_key__(self):',
        'if gate_red > limit:',
        'x_ptr: cute.Pointer',
        'out_ptr: cute.Pointer,\n        barrier_count: cute.Tensor',
        'ptr(cutlass.BFloat16, x)',
    )
    for marker in required:
        if text.count(marker) != 1:
            raise RuntimeError(f"ported source marker drifted: {marker!r}")
    if "fp4_dot8_dual_sum" in text or "prefetch_global_l2" in text:
        raise RuntimeError("descendant-only optimization leaked into literal port")
    return text


def patch(source: Path, output: Path) -> dict[str, object]:
    payload = source.read_bytes()
    observed = sha256_bytes(payload)
    if observed != SOURCE_SHA256:
        raise RuntimeError(
            f"orphan source drifted: expected {SOURCE_SHA256}, got {observed}"
        )
    text = payload.decode("utf-8")
    ported = port_source(text).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(ported)
    return {
        "source": str(source.resolve()),
        "source_sha256": observed,
        "output": str(output.resolve()),
        "output_sha256": sha256_bytes(ported),
        "source_bytes": len(payload),
        "output_bytes": len(ported),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = patch(args.source, args.output)
    for key, value in evidence.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
