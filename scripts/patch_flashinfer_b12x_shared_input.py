#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Enable pinned SM121 B12X tiny-decode fast paths.

Prepared DeepSeek V4 B12X layers bake their expert-global scales into the
block scales, leaving the two activation global-scale vectors uniformly 1.0.
For M=1 the same hidden-state row is routed to all six experts, so quantizing
that row once is sufficient.  FlashInfer's microkernel already implements the
gated shared-input path, but the pinned dispatcher only enables it for ReLU2
and the wrapper does not expose the now-shared activation scales separately
from its per-expert weight-alpha views.

This image-time patch is deliberately narrow and source-pinned:

* an adapter-owned opt-in attribute slices only the activation-scale call
  arguments to one element after full per-expert weight views are cached; and
* the dispatcher permits its existing shared-input implementation for any
  single-token activation when those call arguments are scalar.

For one decode token, every routed pair is also treated as an independent
one-row work item.  Top-k guarantees that the token's expert ids are unique,
so the microkernel can read each physical expert id directly from
``topk_ids``.  This removes the Triton expert-compaction launch plus row-count
atomics and token-map traffic while preserving the same FP4 MMA and reduction
arithmetic.

Multi-token decode keeps expert compaction.  Real routes can select the same
expert for several tokens; bypassing compaction in that case repeats the
expert's FC1/FC2 work and regresses concurrent serving even though a synthetic
all-distinct M=4 route gets slightly faster.  Prefill is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)
PINNED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
PINNED_MICRO_KERNEL_SHA256 = (
    "9ef89f9f9d806e8e2904e3bd345b69c9c8a0e1d0643d21d8975e6e3ae8c8a6ed"
)
DEFAULT_WRAPPER_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/b12x_moe.py"
)
DEFAULT_DISPATCH_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
DEFAULT_MICRO_KERNEL_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_micro_kernel.py"
)

_WRAPPER_RETURN_ANCHOR = """\
        return launch_sm120_moe(
            a=x,
"""
_WRAPPER_RETURN_REPLACEMENT = """\
        shared_unit_scales = getattr(self, "_dspark_unit_input_scales", False)
        input_gs = w1_alpha[:1] if shared_unit_scales else w1_alpha
        down_input_scale = (
            fc2_input_scale[:1]
            if shared_unit_scales and fc2_input_scale is not None
            else fc2_input_scale
        )

        return launch_sm120_moe(
            a=x,
"""
_WRAPPER_ARGS_ANCHOR = """\
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
"""
_WRAPPER_ARGS_REPLACEMENT = """\
            w1_alpha=input_gs,
            fc2_input_scale=down_input_scale,
"""

_DISPATCH_SHARE_INPUT_ANCHOR = """\
    share_input_across_experts = (
        activation == "relu2"
        and num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
"""
_DISPATCH_SHARE_INPUT_REPLACEMENT = """\
    share_input_across_experts = (
        num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
"""
_DISPATCH_SHARE_SCALE_ANCHOR = """\
    share_expert_scales = (
        activation == "relu2" and input_gs_is_shared and down_input_scale_is_shared
    )
"""
_DISPATCH_SHARE_SCALE_REPLACEMENT = """\
    share_expert_scales = input_gs_is_shared and down_input_scale_is_shared
"""

_DISPATCH_PAIRWISE_DECL_ANCHOR = """\
    if use_micro:
        assert flat_ids.numel() <= workspace.compact_topk_ids.numel(), (
"""
_DISPATCH_PAIRWISE_DECL_REPLACEMENT = """\
    if use_micro:
        # A token's top-k expert ids are unique, so M=1 may treat every route
        # as a one-row work item and address weights directly.  Keep compact
        # grouping for M>1 because different tokens can route to the same
        # expert; expanding those duplicates repeats full expert work.
        pairwise_routes = num_tokens == 1
        assert flat_ids.numel() <= workspace.compact_topk_ids.numel(), (
"""
_DISPATCH_ROUTING_ANCHOR = """\
        # Single-token ReLU2 is non-gated, so the micro kernel can launch on
        # the routed expert ids directly. Gated SiLU still goes through the
        # compact id buffer so the kernel can map compact launch ids back to
        # the physical gate/up weight experts.
        if num_tokens == 1 and activation == "relu2":
            launch_ids = flat_ids
        elif num_tokens == 1:
            compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            compact_ids.copy_(
                torch.arange(
                    flat_ids.numel(),
                    device=flat_ids.device,
                    dtype=torch.int32,
                )
            )
            workspace.weight_expert_ids[: flat_ids.numel()].copy_(
                flat_ids.to(torch.int32)
            )
            workspace.active_expert_count.fill_(flat_ids.numel())
            launch_ids = compact_ids
        else:
"""
_DISPATCH_ROUTING_REPLACEMENT = """\
        if pairwise_routes:
            launch_ids = flat_ids
        else:
"""
_DISPATCH_SINGLE_TOKEN_ANCHOR = """\
            single_token=num_tokens == 1,
"""
_DISPATCH_SINGLE_TOKEN_REPLACEMENT = """\
            single_token=pairwise_routes,
"""

_MICRO_TOKEN_ANCHOR = """\
            token_idx = Int32(0)
            weight = cutlass.Float32(0.0)
            if cutlass.const_expr(not self.single_token):
                token_idx = pair_idx // num_topk
                weight = topk_weights[pair_idx].to(cutlass.Float32)
"""
_MICRO_TOKEN_REPLACEMENT = """\
            token_idx = Int32(0)
            weight = cutlass.Float32(0.0)
            if cutlass.const_expr(self.single_token):
                token_idx = pair_idx // num_topk
                weight = topk_weights[pair_idx].to(cutlass.Float32)
            else:
                token_idx = pair_idx // num_topk
                weight = topk_weights[pair_idx].to(cutlass.Float32)
"""
_MICRO_EXPERT_ANCHOR = """\
            if cutlass.const_expr(self.single_token):
                local_expert_id = pair_idx
                if cutlass.const_expr(self.is_gated):
                    expert_id = weight_expert_ids[local_expert_id].to(Int32)
                else:
                    expert_id = topk_ids[local_expert_id].to(Int32)
"""
_MICRO_EXPERT_REPLACEMENT = """\
            if cutlass.const_expr(self.single_token):
                local_expert_id = pair_idx
                expert_id = topk_ids[local_expert_id].to(Int32)
"""
_MICRO_QUANT_EXPERT_ANCHOR = """\
                    if cutlass.const_expr(not self.is_gated):
                        quant_expert_id = topk_ids[Int32(0)].to(Int32)
                    else:
                        quant_expert_id = weight_expert_ids[Int32(0)]
"""
_MICRO_QUANT_EXPERT_REPLACEMENT = """\
                    quant_expert_id = topk_ids[Int32(0)].to(Int32)
"""
_MICRO_WEIGHT_EXPERT_ANCHOR = """\
                if cutlass.const_expr(self.single_token):
                    if cutlass.const_expr(not self.is_gated):
                        weight_expert_idx = topk_ids[local_expert_idx].to(Int32)
                    else:
                        weight_expert_idx = weight_expert_ids[local_expert_idx]
                else:
"""
_MICRO_WEIGHT_EXPERT_REPLACEMENT = """\
                if cutlass.const_expr(self.single_token):
                    weight_expert_idx = topk_ids[local_expert_idx].to(Int32)
                else:
"""
_MICRO_MMA_WEIGHT_EXPERT_ANCHOR = """\
                if cutlass.const_expr(self.single_token):
                    if cutlass.const_expr(not self.is_gated):
                        weight_expert_idx = topk_ids[local_expert_idx].to(Int32)
                    else:
                        weight_expert_idx = weight_expert_ids[local_expert_idx]
                    valid_rows = Int32(1)
                else:
"""
_MICRO_MMA_WEIGHT_EXPERT_REPLACEMENT = """\
                if cutlass.const_expr(self.single_token):
                    weight_expert_idx = topk_ids[local_expert_idx].to(Int32)
                    valid_rows = Int32(1)
                else:
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def patch_wrapper_source(source: str) -> str:
    source = _replace_once(
        source,
        _WRAPPER_RETURN_ANCHOR,
        _WRAPPER_RETURN_REPLACEMENT,
        "wrapper launch",
    )
    return _replace_once(
        source,
        _WRAPPER_ARGS_ANCHOR,
        _WRAPPER_ARGS_REPLACEMENT,
        "wrapper activation scales",
    )


def patch_dispatch_source(source: str) -> str:
    source = _replace_once(
        source,
        _DISPATCH_SHARE_INPUT_ANCHOR,
        _DISPATCH_SHARE_INPUT_REPLACEMENT,
        "dispatcher shared input",
    )
    source = _replace_once(
        source,
        _DISPATCH_SHARE_SCALE_ANCHOR,
        _DISPATCH_SHARE_SCALE_REPLACEMENT,
        "dispatcher shared scales",
    )
    source = _replace_once(
        source,
        _DISPATCH_PAIRWISE_DECL_ANCHOR,
        _DISPATCH_PAIRWISE_DECL_REPLACEMENT,
        "dispatcher pairwise declaration",
    )
    source = _replace_once(
        source,
        _DISPATCH_ROUTING_ANCHOR,
        _DISPATCH_ROUTING_REPLACEMENT,
        "dispatcher pairwise routing",
    )
    return _replace_once(
        source,
        _DISPATCH_SINGLE_TOKEN_ANCHOR,
        _DISPATCH_SINGLE_TOKEN_REPLACEMENT,
        "dispatcher pairwise specialization",
    )


def patch_micro_kernel_source(source: str) -> str:
    source = _replace_once(
        source,
        _MICRO_TOKEN_ANCHOR,
        _MICRO_TOKEN_REPLACEMENT,
        "microkernel pair token metadata",
    )
    source = _replace_once(
        source,
        _MICRO_EXPERT_ANCHOR,
        _MICRO_EXPERT_REPLACEMENT,
        "microkernel direct expert id",
    )
    source = _replace_once(
        source,
        _MICRO_QUANT_EXPERT_ANCHOR,
        _MICRO_QUANT_EXPERT_REPLACEMENT,
        "microkernel direct quant expert id",
    )
    source = _replace_once(
        source,
        _MICRO_MMA_WEIGHT_EXPERT_ANCHOR,
        _MICRO_MMA_WEIGHT_EXPERT_REPLACEMENT,
        "microkernel MMA expert id",
    )
    return _replace_once(
        source,
        _MICRO_WEIGHT_EXPERT_ANCHOR,
        _MICRO_WEIGHT_EXPERT_REPLACEMENT,
        "microkernel DMA expert id",
    )


def _patch_file(path: Path, expected_sha: str, transform, label: str) -> str:
    original = path.read_bytes()
    actual_sha = _sha256(original)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pinned {label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    patched = transform(original.decode("utf-8")).encode("utf-8")
    path.write_bytes(patched)
    return _sha256(patched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-target", type=Path, default=DEFAULT_WRAPPER_TARGET)
    parser.add_argument("--dispatch-target", type=Path, default=DEFAULT_DISPATCH_TARGET)
    parser.add_argument(
        "--micro-kernel-target", type=Path, default=DEFAULT_MICRO_KERNEL_TARGET
    )
    args = parser.parse_args()

    wrapper_result = _patch_file(
        args.wrapper_target,
        PINNED_WRAPPER_SHA256,
        patch_wrapper_source,
        "FlashInfer B12X wrapper",
    )
    dispatch_result = _patch_file(
        args.dispatch_target,
        PINNED_DISPATCH_SHA256,
        patch_dispatch_source,
        "FlashInfer B12X dispatcher",
    )
    micro_kernel_result = _patch_file(
        args.micro_kernel_target,
        PINNED_MICRO_KERNEL_SHA256,
        patch_micro_kernel_source,
        "FlashInfer B12X microkernel",
    )
    print(
        "patched FlashInfer B12X unit-scale shared input: "
        f"wrapper_result={wrapper_result} dispatch_result={dispatch_result} "
        f"micro_kernel_result={micro_kernel_result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
