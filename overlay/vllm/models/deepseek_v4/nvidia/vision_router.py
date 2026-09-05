# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision-only DSv4 routing; leaves the text model's compiled ABI unchanged.

Selection follows vLLM #54566: image sentinels use bias_vl, ordinary hash
tokens use tid2eid, and weights always come from UNBIASED sqrt(softplus).
Adapted from upstream router/dsv4_topk.py at 8277c42e.
"""

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _select_six(weights, bias, offsets, NUM_EXPERTS: tl.constexpr):
    current = tl.where(offsets < NUM_EXPERTS, weights + bias, -float("inf"))
    current = tl.where(current == current, current, -1e30)
    slots = tl.arange(0, 8)
    ids = tl.zeros([8], tl.int32)
    selected = tl.zeros([8], tl.float32)
    for slot in tl.static_range(6):
        maximum = tl.max(current, 0)
        expert = tl.min(tl.where(current == maximum, offsets, NUM_EXPERTS), 0)
        value = tl.sum(tl.where(offsets == expert, weights, 0.0), 0)
        ids = tl.where(slots == slot, expert, ids)
        selected = tl.where(slots == slot, value, selected)
        current = tl.where(offsets == expert, -float("inf"), current)
    return selected, ids


@triton.jit
def _vision_topk_kernel(
    logits, correction, bias_vl, input_ids, tid2eid, output_weights, output_ids,
    routed_scale, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr,
    HAS_HASH: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    slots = tl.arange(0, 8)
    token_id = tl.load(input_ids + row).to(tl.int64)
    is_image = (token_id >= 129257) & (token_id < 129262)
    raw = tl.load(logits + row * NUM_EXPERTS + offsets,
                  mask=offsets < NUM_EXPERTS, other=0).to(tl.float32)
    weights = tl.sqrt(tl.where(raw > 20., raw, tl.log(1. + tl.exp(raw))))
    regular_bias = tl.load(correction + offsets, mask=offsets < NUM_EXPERTS, other=0)
    image_bias = tl.load(bias_vl + offsets, mask=offsets < NUM_EXPERTS, other=0)
    bias = tl.where(is_image, image_bias, regular_bias)
    if HAS_HASH:
        if is_image:
            selected, ids = _select_six(weights, bias, offsets, NUM_EXPERTS)
        else:
            ids = tl.load(tid2eid + token_id * 6 + slots, mask=slots < 6, other=0)
            selected = tl.gather(weights, ids.to(tl.int32), axis=0)
            selected = tl.where(slots < 6, selected, 0.)
    else:
        selected, ids = _select_six(weights, bias, offsets, NUM_EXPERTS)
    total = tl.sum(selected, 0)
    selected *= routed_scale / tl.maximum(total, 1e-20)
    tl.store(output_weights + row * 6 + slots, selected, mask=slots < 6)
    tl.store(output_ids + row * 6 + slots, ids, mask=slots < 6)


class DeepseekV4VisionRouter(FusedTopKBiasRouter):
    def __init__(self, original: FusedTopKBiasRouter, correction, bias_vl):
        if original.top_k != 6 or not original.renormalize:
            raise NotImplementedError("DSv4 vision routing requires normalized top-6")
        if original.num_fused_shared_experts:
            raise NotImplementedError("DSv4 vision requires separate shared experts")
        super().__init__(
            top_k=original.top_k,
            global_num_experts=original.global_num_experts,
            e_score_correction_bias=correction,
            renormalize=original.renormalize,
            routed_scaling_factor=original.routed_scaling_factor,
            eplb_state=original.eplb_state,
            scoring_func=original.scoring_func,
            hash_indices_table=original._hash_indices_table,
        )
        self.bias_vl = bias_vl

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        if input_ids is None:
            raise ValueError("DSv4 vision routing requires raw input_ids")
        if router_logits.dtype != torch.float32 or not router_logits.is_contiguous():
            raise ValueError("DSv4 vision requires contiguous float32 router logits")
        if not input_ids.is_contiguous() or input_ids.numel() != router_logits.shape[0]:
            raise ValueError("DSv4 vision routing token/logit shape mismatch")
        rows, experts = router_logits.shape
        weights = torch.empty((rows, 6), device=router_logits.device, dtype=torch.float32)
        ids = torch.empty((rows, 6), device=router_logits.device,
                          dtype=indices_type or torch.int32)
        if rows:
            _vision_topk_kernel[(rows,)](
                router_logits, self.e_score_correction_bias, self.bias_vl,
                input_ids, self._hash_indices_table, weights, ids,
                self.routed_scaling_factor, NUM_EXPERTS=experts,
                BLOCK=triton.next_power_of_2(experts),
                HAS_HASH=self._hash_indices_table is not None, num_warps=1,
            )
        return weights, ids


def install_vision_router(moe) -> None:
    """Called only while constructing vision target/draft layers, before loading."""
    if moe.use_mega_moe or not isinstance(moe.experts.router, FusedTopKBiasRouter):
        raise NotImplementedError("Spark vision requires the external fused-bias router")
    if moe.gate.e_score_correction_bias is None:
        moe.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(moe.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
    moe.gate.bias_vl = nn.Parameter(
        torch.zeros(moe.n_routed_experts, dtype=torch.float32), requires_grad=False
    )
    moe.experts.router = DeepseekV4VisionRouter(
        moe.experts.router, moe.gate.e_score_correction_bias, moe.gate.bias_vl
    )
