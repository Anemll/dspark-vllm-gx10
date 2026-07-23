#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Add an opt-in active-24 descriptor path to FlashInfer CUTLASS MoE.

DeepSeek-V4 C4 decode expands four rows through top-k=6, so at most 24
experts can be active.  The pinned FlashInfer runner nevertheless presents all
256 expert problem descriptors to both grouped GEMMs.  This source-pinned
transform adds a fixed-capacity 24-descriptor path behind
``FLASHINFER_CUTLASS_ACTIVE24=1``.

The specialization is deliberately narrow and fail closed:

* NVFP4 activations and weights only;
* M=4, expanded M=24, E=256, EP=1, and local expert base zero;
* both GEMMs must use the TMA warp-specialized implementation; and
* the fixed descriptor count remains 24, so CUDA graph shape is invariant.

The existing 257-entry expert offset map remains authoritative for activation
quantization, buffer addressing, and final routing.  Only the TMA descriptor
arrays are compacted.  Unset or zero leaves the original source path intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PINNED_FLASHINFER_REVISION = "0472b9b3f2fba11b463f8526f390297d52a8aad7"
KERNEL_RELATIVE_PATH = Path(
    "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"
)
HEADER_RELATIVE_PATH = Path(
    "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
)
PINNED_KERNEL_SHA256 = "fd24f5f8234b0736f205dd2540f47dcaf90783a53c2fbbab66d0490c9494dbac"
PINNED_HEADER_SHA256 = "d5562b100214697950149718929fc6dd0bf6570ac79cd452d6da6c9df2ea6161"
PATCHED_KERNEL_SHA256 = "9a8fc3abd0d8bd3589adcf855c6f92e4534a6b914a5ab0a31bfa21339f12061b"
PATCHED_HEADER_SHA256 = "7be6f6f272b373157c796120e891da0b3389dee24d6348c92d909d21238a2cd8"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(anchor, replacement, 1)


ENV_ANCHOR = """\
constexpr int CVT_ELTS_PER_THREAD = 8;
"""
ENV_REPLACEMENT = """\
constexpr int CVT_ELTS_PER_THREAD = 8;
constexpr int CUTLASS_ACTIVE24_GROUPS = 24;

inline bool getEnvCutlassActive24() {
  static bool const enabled = []() {
    auto const value = tensorrt_llm::common::getIntEnv("FLASHINFER_CUTLASS_ACTIVE24");
    if (!value.has_value()) {
      return false;
    }
    TLLM_CHECK_WITH_INFO(value.value() == 0 || value.value() == 1,
                         "FLASHINFER_CUTLASS_ACTIVE24 must be exactly 0 or 1");
    return value.value() == 1;
  }();
  return enabled;
}
"""

SCALING_HELPER_ANCHOR = """\
template <class BSConfig>
__device__ void setupFP4BlockScalingFactors(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int expert, int gemm_m, int gemm_n, int gemm_k,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale,
    int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_act);
  assert(layout_info.fpX_block_scaling_factors_stride_weight);

  auto stride_act_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_act);
  auto stride_weight_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_weight);
  if (layout_info.swap_ab) {
    // M & N swapped for transpose
    stride_act_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
    stride_weight_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
  } else {
    stride_act_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
    stride_weight_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
  }

  // This assert validates our current assumption that A&B can be safely transposed without needing
  // to modify
  assert(
      BSConfig::tile_atom_to_shape_SFB(
          cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, 1)) ==
      BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, 1)));

  auto scaling_type =
      std::is_same_v<BSConfig, TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
          ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
          : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  layout_info.fpX_block_scaling_factors_act[expert] =
      fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k, scaling_type);

  layout_info.fpX_block_scaling_factors_weight[expert] =
      weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);
}
"""
SCALING_HELPER_REPLACEMENT = """\
template <class BSConfig>
__device__ void setupFP4BlockScalingFactors(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int expert, int out_idx, int gemm_m,
    int gemm_n, int gemm_k,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale,
    int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_act);
  assert(layout_info.fpX_block_scaling_factors_stride_weight);

  auto stride_act_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_act);
  auto stride_weight_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_weight);
  if (layout_info.swap_ab) {
    // M & N swapped for transpose
    stride_act_ptr[out_idx] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
    stride_weight_ptr[out_idx] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
  } else {
    stride_act_ptr[out_idx] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
    stride_weight_ptr[out_idx] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
  }

  // This assert validates our current assumption that A&B can be safely transposed without needing
  // to modify
  assert(
      BSConfig::tile_atom_to_shape_SFB(
          cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, 1)) ==
      BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, 1)));

  auto scaling_type =
      std::is_same_v<BSConfig, TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
          ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
          : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  layout_info.fpX_block_scaling_factors_act[out_idx] =
      fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k, scaling_type);

  layout_info.fpX_block_scaling_factors_weight[out_idx] =
      weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);
}
"""

FINALIZE_POINTER_ANCHOR = """\
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
    layout_info.fused_finalize_epilogue.ptr_source_token_index[expert] =
        permuted_row_to_unpermuted_row + num_tokens_before_expert;
    layout_info.fused_finalize_epilogue.ptr_router_scales[expert] =
        router_scales + num_tokens_before_expert;
    if (layout_info.fused_finalize_epilogue.ptr_bias != nullptr) {
      layout_info.fused_finalize_epilogue.ptr_bias[expert] = bias + gemm_n * expert;
    }
  }
"""
FINALIZE_POINTER_REPLACEMENT = """\
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
    layout_info.fused_finalize_epilogue.ptr_source_token_index[out_idx] =
        permuted_row_to_unpermuted_row + num_tokens_before_expert;
    layout_info.fused_finalize_epilogue.ptr_router_scales[out_idx] =
        router_scales + num_tokens_before_expert;
    if (layout_info.fused_finalize_epilogue.ptr_bias != nullptr) {
      layout_info.fused_finalize_epilogue.ptr_bias[out_idx] = bias + gemm_n * expert;
    }
  }
"""

NORMAL_SCALING_CALL_ANCHOR = """\
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info1, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
          quant_type.fc1.weight_block_scale, num_tokens_before_expert);
    }
    if (quant_type.fc2.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info2, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
          quant_type.fc2.weight_block_scale, num_tokens_before_expert);
"""
NORMAL_SCALING_CALL_REPLACEMENT = """\
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info1, expert, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
          quant_type.fc1.weight_block_scale, num_tokens_before_expert);
    }
    if (quant_type.fc2.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info2, expert, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
          quant_type.fc2.weight_block_scale, num_tokens_before_expert);
"""

COMPACT_KERNEL_ANCHOR = """\
}

// ========================== Permutation things =======================================
"""
COMPACT_KERNEL_REPLACEMENT = """\
}

// C4/top-k=6 has only 24 routed rows, hence at most 24 active experts. Keep the
// original expert offsets authoritative, but compact the two grouped-GEMM
// descriptor arrays to a fixed 24 slots so graph shape remains invariant.
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedActive24Kernel(
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
    int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
    int64_t const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
    WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
    float const* alpha_scale_flat2,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* gemm1_output,
    OutputType* gemm2_output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row) {
  assert(gridDim.x == 1 && blockDim.x == 32);
  assert(num_tokens == 4);
  assert(expanded_num_tokens == CUTLASS_ACTIVE24_GROUPS);
  assert(num_experts_per_node == 256);
  assert(!layout_info1.int4_groupwise_params.enabled);
  assert(!layout_info2.int4_groupwise_params.enabled);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  constexpr unsigned kFullWarp = 0xffffffffu;
  constexpr int kExpertsPerLane = 256 / 32;
  int const lane = threadIdx.x;

  // Cached workspace is sized for E=256. Only the leading 24 descriptors are
  // exposed to CUTLASS on this path; clear them before active lanes overwrite
  // the compact prefix.
  if (lane < CUTLASS_ACTIVE24_GROUPS) {
    layout_info1.shape_info.problem_shapes[lane] =
        TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
            layout_info1.swap_ab ? gemm1_n : 0, layout_info1.swap_ab ? 0 : gemm1_n, gemm1_k);
    layout_info2.shape_info.problem_shapes[lane] =
        TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
            layout_info2.swap_ab ? gemm2_n : 0, layout_info2.swap_ab ? 0 : gemm2_n, gemm2_k);
  }
  __syncwarp(kFullWarp);

  int const first_expert = lane * kExpertsPerLane;
  int local_active = 0;
#pragma unroll
  for (int i = 0; i < kExpertsPerLane; ++i) {
    int const expert = first_expert + i;
    local_active += expert_first_token_offset[expert + 1] > expert_first_token_offset[expert];
  }

  int inclusive_active = local_active;
#pragma unroll
  for (int delta = 1; delta < 32; delta *= 2) {
    int const preceding = __shfl_up_sync(kFullWarp, inclusive_active, delta);
    if (lane >= delta) {
      inclusive_active += preceding;
    }
  }
  int const active_count = __shfl_sync(kFullWarp, inclusive_active, 31);
  assert(active_count <= CUTLASS_ACTIVE24_GROUPS);

  int out_idx = inclusive_active - local_active;
#pragma unroll
  for (int i = 0; i < kExpertsPerLane; ++i) {
    int const expert = first_expert + i;
    auto const num_tokens_before_expert = expert_first_token_offset[expert];
    auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
    auto const gemm_m = num_tokens_including_expert - num_tokens_before_expert;
    if (gemm_m == 0) {
      continue;
    }
    assert(out_idx < CUTLASS_ACTIVE24_GROUPS);

    layout_info1.shape_info.problem_shapes[out_idx] =
        TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
            layout_info1.swap_ab ? gemm1_n : gemm_m,
            layout_info1.swap_ab ? gemm_m : gemm1_n, gemm1_k);
    layout_info2.shape_info.problem_shapes[out_idx] =
        TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
            layout_info2.swap_ab ? gemm2_n : gemm_m,
            layout_info2.swap_ab ? gemm_m : gemm2_n, gemm2_k);

    if (alpha_scale_flat1 && alpha_scale_flat2) {
      layout_info1.alpha_scale_ptr_array[out_idx] = alpha_scale_flat1 + expert;
      layout_info2.alpha_scale_ptr_array[out_idx] = alpha_scale_flat2 + expert;
    }

    auto setupIfSelected = [&](auto bs_config, auto quant_type) {
      if (quant_type.fc1.weight_block_scale) {
        setupFP4BlockScalingFactors<decltype(bs_config)>(
            layout_info1, expert, out_idx, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
            quant_type.fc1.weight_block_scale, num_tokens_before_expert);
      }
      if (quant_type.fc2.weight_block_scale) {
        setupFP4BlockScalingFactors<decltype(bs_config)>(
            layout_info2, expert, out_idx, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
            quant_type.fc2.weight_block_scale, num_tokens_before_expert);
      }
    };
    setupIfSelected(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig{}, quant_params.fp4);

    computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, out_idx);
    computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, out_idx);
    computeTmaWarpSpecializedInputPointers(
        layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert, gemm1_in, weights1,
        nullptr, bias1, gemm1_output, nullptr, nullptr, out_idx);
    computeTmaWarpSpecializedInputPointers(
        layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert, gemm2_in, weights2,
        nullptr, bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, out_idx);
    ++out_idx;
  }

  __syncwarp(kFullWarp);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ========================== Permutation things =======================================
"""

COMPUTE_SIGNATURE_ANCHOR = """\
        float const* router_scales, int const* permuted_row_to_unpermuted_row, bool enable_pdl,
        cudaStream_t stream) {
"""
COMPUTE_SIGNATURE_REPLACEMENT = """\
        float const* router_scales, int const* permuted_row_to_unpermuted_row,
        bool compact_active24, bool enable_pdl, cudaStream_t stream) {
"""

COMPUTE_LAUNCH_ANCHOR = """\
  // Use a smaller block size to spread work across multiple SMs. Each thread handles one expert,
  // so we only need num_experts_per_node threads total. With 1 warp per block, 128 experts
  // yields 4 blocks across 4 SMs instead of 1 block on 1 SM.
  int const threads = std::min(32, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance =
      &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1,
                     layout_info2, num_tokens, expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n,
                     gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1, weights2,
                     alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2,
                     quant_params, bias1, bias2, gemm1_output, gemm2_output, router_scales,
                     permuted_row_to_unpermuted_row);
"""
COMPUTE_LAUNCH_REPLACEMENT = """\
  // The default launch remains unchanged. Active24 uses one fixed warp and
  // exposes only the first 24 cached descriptor slots to both grouped GEMMs.
  int const threads = compact_active24 ? 32 : std::min(32, num_experts_per_node);
  int const blocks = compact_active24 ? 1 : (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance =
      compact_active24
          ? &computeStridesTmaWarpSpecializedActive24Kernel<T, WeightType, OutputType, ScaleBiasType>
          : &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;
  if (compact_active24) {
    layout_info1.shape_info.num_groups = CUTLASS_ACTIVE24_GROUPS;
    layout_info2.shape_info.num_groups = CUTLASS_ACTIVE24_GROUPS;
  }

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1,
                     layout_info2, num_tokens, expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n,
                     gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1, weights2,
                     alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2,
                     quant_params, bias1, bias2, gemm1_output, gemm2_output, router_scales,
                     permuted_row_to_unpermuted_row);
"""

SETUP_GATE_ANCHOR = """\
  // Set enable_pdl for both GEMM inputs
  gemm1_tma_ws_input.enable_pdl = enable_pdl;
  gemm2_tma_ws_input.enable_pdl = enable_pdl;
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
      !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }
"""
SETUP_GATE_REPLACEMENT = """\
  // Set enable_pdl for both GEMM inputs
  gemm1_tma_ws_input.enable_pdl = enable_pdl;
  gemm2_tma_ws_input.enable_pdl = enable_pdl;

  bool const compact_active24 =
      getEnvCutlassActive24() && use_fp4 && !min_latency_mode && !use_lora && num_rows == 4 &&
      expanded_num_rows == CUTLASS_ACTIVE24_GROUPS && num_experts_per_node == 256 &&
      parallelism_config.ep_size == 1 && start_expert == 0;
  if (compact_active24) {
    TLLM_CHECK_WITH_INFO(moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
                             moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_),
                         "FLASHINFER_CUTLASS_ACTIVE24 requires TMA warp-specialized FC1 and FC2");
  }
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
      !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }
"""

SETUP_COMPUTE_CALL_ANCHOR = """\
        reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), permuted_token_final_scales_,
        permuted_row_to_unpermuted_row_, enable_pdl, stream);
"""
SETUP_COMPUTE_CALL_REPLACEMENT = """\
        reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), permuted_token_final_scales_,
        permuted_row_to_unpermuted_row_, compact_active24, enable_pdl, stream);
"""

HEADER_DISPATCH_CALL_ANCHOR = """\
        reinterpret_cast<UnfusedGemmOutputType*>(gemm2_output), router_scales,
        permuted_row_to_unpermuted_row, enable_pdl, stream);
"""
HEADER_DISPATCH_CALL_REPLACEMENT = """\
        reinterpret_cast<UnfusedGemmOutputType*>(gemm2_output), router_scales,
        permuted_row_to_unpermuted_row, false, enable_pdl, stream);
"""

HEADER_STATIC_DECL_ANCHOR = """\
      UnfusedGemmOutputType* gemm2_output, float const* router_scales,
      int const* permuted_row_to_unpermuted_row, bool enable_pdl, cudaStream_t stream);
"""
HEADER_STATIC_DECL_REPLACEMENT = """\
      UnfusedGemmOutputType* gemm2_output, float const* router_scales,
      int const* permuted_row_to_unpermuted_row, bool compact_active24, bool enable_pdl,
      cudaStream_t stream);
"""


def patch_kernel_source(source: str) -> str:
    transforms: tuple[tuple[str, str, str], ...] = (
        (ENV_ANCHOR, ENV_REPLACEMENT, "active24 environment contract"),
        (SCALING_HELPER_ANCHOR, SCALING_HELPER_REPLACEMENT, "scale descriptor split"),
        (FINALIZE_POINTER_ANCHOR, FINALIZE_POINTER_REPLACEMENT, "finalize descriptor split"),
        (NORMAL_SCALING_CALL_ANCHOR, NORMAL_SCALING_CALL_REPLACEMENT, "normal scale calls"),
        (COMPACT_KERNEL_ANCHOR, COMPACT_KERNEL_REPLACEMENT, "active24 kernel insertion"),
        (COMPUTE_SIGNATURE_ANCHOR, COMPUTE_SIGNATURE_REPLACEMENT, "compute signature"),
        (COMPUTE_LAUNCH_ANCHOR, COMPUTE_LAUNCH_REPLACEMENT, "compute launch selection"),
        (SETUP_GATE_ANCHOR, SETUP_GATE_REPLACEMENT, "exact active24 gate"),
        (SETUP_COMPUTE_CALL_ANCHOR, SETUP_COMPUTE_CALL_REPLACEMENT, "setup compute call"),
    )
    for anchor, replacement, label in transforms:
        source = _replace_once(source, anchor, replacement, label)

    required = (
        'getIntEnv("FLASHINFER_CUTLASS_ACTIVE24")',
        "num_rows == 4",
        "expanded_num_rows == CUTLASS_ACTIVE24_GROUPS",
        "num_experts_per_node == 256",
        "parallelism_config.ep_size == 1",
        "start_expert == 0",
        "shape_info.num_groups = CUTLASS_ACTIVE24_GROUPS",
        "computeStridesTmaWarpSpecializedActive24Kernel",
        "ptr_source_token_index[out_idx]",
        "getOffsetWeightSF(expert",
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"patched kernel marker missing: {marker!r}")
    if "compact_expert_first_token_offset" in source:
        raise RuntimeError("active24 must not replace the authoritative expert offsets")
    return source


def patch_header_source(source: str) -> str:
    source = _replace_once(
        source,
        HEADER_DISPATCH_CALL_ANCHOR,
        HEADER_DISPATCH_CALL_REPLACEMENT,
        "public dispatch control",
    )
    return _replace_once(
        source,
        HEADER_STATIC_DECL_ANCHOR,
        HEADER_STATIC_DECL_REPLACEMENT,
        "static compute declaration",
    )


def _read_pinned(path: Path, expected_sha256: str, label: str) -> bytes:
    payload = path.read_bytes()
    observed = _sha256(payload)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} drifted: expected {expected_sha256}, observed {observed}: {path}"
        )
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.active24-{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def patch_tree(source_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    kernel_path = source_root / KERNEL_RELATIVE_PATH
    header_path = source_root / HEADER_RELATIVE_PATH

    # Validate and transform both inputs before publishing either output.
    kernel_before = _read_pinned(kernel_path, PINNED_KERNEL_SHA256, "kernel source")
    header_before = _read_pinned(header_path, PINNED_HEADER_SHA256, "runner header")
    kernel_after = patch_kernel_source(kernel_before.decode("utf-8")).encode("utf-8")
    header_after = patch_header_source(header_before.decode("utf-8")).encode("utf-8")
    if _sha256(kernel_after) != PATCHED_KERNEL_SHA256:
        raise RuntimeError("deterministic active24 kernel result drifted")
    if _sha256(header_after) != PATCHED_HEADER_SHA256:
        raise RuntimeError("deterministic active24 header result drifted")

    _atomic_write(kernel_path, kernel_after)
    _atomic_write(header_path, header_after)
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "flashinfer_revision": PINNED_FLASHINFER_REVISION,
        "environment": "FLASHINFER_CUTLASS_ACTIVE24",
        "files": {
            str(KERNEL_RELATIVE_PATH): {
                "before_sha256": PINNED_KERNEL_SHA256,
                "after_sha256": _sha256(kernel_after),
            },
            str(HEADER_RELATIVE_PATH): {
                "before_sha256": PINNED_HEADER_SHA256,
                "after_sha256": _sha256(header_after),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="root of the exact pinned FlashInfer source tree to patch in place",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(patch_tree(args.source_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
