#!/usr/bin/env python3
"""Restore the removed FlashInfer FP4 low-latency descriptor builder.

FlashInfer commit 20435b4 imported a newer TensorRT-LLM CUTLASS MoE runner and
replaced the complete low-latency stride builder with an unconditional throw.
The routing, workspace, GEMM1, and GEMM2 code paths remain present.  This
source-pinned transform ports the deleted descriptor builder to the current
``swap_ab``/``fpX_block_scaling`` layout used by FlashInfer 0.6.15.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


PINNED_SHA256 = "fd24f5f8234b0736f205dd2540f47dcaf90783a53c2fbbab66d0490c9494dbac"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


KERNEL_ANCHOR = """\
// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(
"""

LOW_LATENCY_KERNEL = r"""
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedLowLatencyKernel(
    TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n,
    int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
    int64_t const num_experts_per_node, T const* in1, T const* in2,
    WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
    float const* alpha_scale_flat2,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2,
    QuantParams quant_params, ScaleBiasType const* bias1, ScaleBiasType const* bias2,
    OutputType* output1, OutputType* output2, int const* num_active_experts_per,
    int const* active_expert_global_ids, int start_expert) {
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int64_t const num_tokens_before_expert = expert * num_tokens;
  bool const is_active_expert = expert < *num_active_experts_per;
  int const local_expert =
      is_active_expert ? active_expert_global_ids[expert] - start_expert : -1;
  int64_t const gemm_m = is_active_expert ? num_tokens : 0;

  layout_info1.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info1.swap_ab ? gemm1_n : gemm_m,
          layout_info1.swap_ab ? gemm_m : gemm1_n, gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info2.swap_ab ? gemm2_n : gemm_m,
          layout_info2.swap_ab ? gemm_m : gemm2_n, gemm2_k);

  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] =
        is_active_expert ? alpha_scale_flat1 + local_expert : nullptr;
    layout_info2.alpha_scale_ptr_array[expert] =
        is_active_expert ? alpha_scale_flat2 + local_expert : nullptr;
  }

  auto setupIfSelected = [&](auto bs_config, auto quant_type) {
    using BSConfig = decltype(bs_config);
    auto const scaling_type =
        std::is_same_v<BSConfig,
                       TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
            ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
            : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
    if (quant_type.fc1.weight_block_scale && is_active_expert) {
      setupFP4BlockScalingFactors<BSConfig>(
          layout_info1, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
          quant_type.fc1.weight_block_scale, num_tokens_before_expert);
      // GEMM1 reuses the same quantized token matrix for every active expert.
      layout_info1.fpX_block_scaling_factors_act[expert] = fp4_act_flat1;
      layout_info1.fpX_block_scaling_factors_weight[expert] =
          quant_type.fc1.weight_block_scale +
          getOffsetWeightSF(local_expert, gemm1_n, gemm1_k, scaling_type);
    }
    if (quant_type.fc2.weight_block_scale && is_active_expert) {
      setupFP4BlockScalingFactors<BSConfig>(
          layout_info2, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
          quant_type.fc2.weight_block_scale, num_tokens_before_expert);
      layout_info2.fpX_block_scaling_factors_weight[expert] =
          quant_type.fc2.weight_block_scale +
          getOffsetWeightSF(local_expert, gemm2_n, gemm2_k, scaling_type);
    }
  };
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig{},
                  quant_params.fp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.fp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.mxfp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.mxfp8_mxfp8);

  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k,
                                        expert);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k,
                                        expert);

  if (is_active_expert) {
    layout_info1.ptr_act[expert] = in1;
    layout_info2.ptr_act[expert] =
        safe_inc_ptr(in2, num_tokens_before_expert * gemm2_k);
    // Preserve the original low-latency packed-weight stride.  Both prepared
    // expert matrices occupy the same packed span despite their logical GEMM
    // shapes.
    auto const packed_weight_stride = int64_t(gemm1_n) * gemm2_k;
    layout_info1.ptr_weight[expert] =
        safe_inc_ptr(weights1, int64_t(local_expert) * packed_weight_stride);
    layout_info2.ptr_weight[expert] =
        safe_inc_ptr(weights2, int64_t(local_expert) * packed_weight_stride);
    layout_info1.ptr_d[expert] =
        safe_inc_ptr(output1, num_tokens_before_expert * gemm1_n);
    layout_info2.ptr_d[expert] =
        safe_inc_ptr(output2, num_tokens_before_expert * gemm2_n);
  } else {
    layout_info1.ptr_act[expert] = nullptr;
    layout_info1.ptr_weight[expert] = nullptr;
    layout_info1.ptr_d[expert] = nullptr;
    layout_info2.ptr_act[expert] = nullptr;
    layout_info2.ptr_weight[expert] = nullptr;
    layout_info2.ptr_d[expert] = nullptr;
    layout_info1.fpX_block_scaling_factors_act[expert] = nullptr;
    layout_info1.fpX_block_scaling_factors_weight[expert] = nullptr;
    layout_info2.fpX_block_scaling_factors_act[expert] = nullptr;
    layout_info2.fpX_block_scaling_factors_weight[expert] = nullptr;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(
"""


THROW_BODY = """\
  TLLM_THROW("Min latency mode is no longer supported");
"""

RESTORED_BODY = r"""  TLLM_CHECK_WITH_INFO(!use_w4_groupwise,
                       "W4AFP8 and WFP4A16 are not supported in low latency mode");

  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;
  layout_info1.fused_finalize_epilogue.ptr_bias = nullptr;
  layout_info2.fused_finalize_epilogue.ptr_bias = nullptr;

  auto alpha_scale_flat1 = use_fp4        ? quant_params.fp4.fc1.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc1.global_scale
                           : use_fp8      ? fp8_dequant1
                                          : nullptr;
  auto alpha_scale_flat2 = use_fp4        ? quant_params.fp4.fc2.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc2.global_scale
                           : use_fp8      ? fp8_dequant2
                                          : nullptr;
  if (!alpha_scale_flat1 && !alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array = nullptr;
    layout_info2.alpha_scale_ptr_array = nullptr;
  }

  layout_info1.int4_groupwise_params.enabled = false;
  layout_info2.int4_groupwise_params.enabled = false;
  layout_info1.int4_groupwise_params.use_wfp4a16 = false;
  layout_info2.int4_groupwise_params.use_wfp4a16 = false;
  layout_info1.fpX_block_scaling_type = getScalingType();
  layout_info2.fpX_block_scaling_type = getScalingType();

  int const threads = std::min(32, num_experts);
  int const blocks = (num_experts + threads - 1) / threads;
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
  cudaLaunchKernelEx(
      &config,
      computeStridesTmaWarpSpecializedLowLatencyKernel<T, WeightType, UnfusedGemmOutputType,
                                                       ScaleBiasType>,
      layout_info1, layout_info2, num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k,
      num_experts, input1, input2, weights1, weights2, alpha_scale_flat1,
      alpha_scale_flat2, fc1_fp4_act_flat, fc2_fp4_act_flat, quant_params, bias1, bias2,
      output1, output2, num_active_experts_per, active_expert_global_ids, start_expert);
  return std::make_pair(layout_info1, layout_info2);
"""


def patch(payload: bytes) -> bytes:
    actual = _sha256(payload)
    if actual != PINNED_SHA256:
        raise RuntimeError(f"unpinned FlashInfer kernel: {actual}")
    text = payload.decode()
    if text.count(KERNEL_ANCHOR) != 1:
        raise RuntimeError("low-latency kernel insertion anchor drifted")
    if text.count(THROW_BODY) != 1:
        raise RuntimeError("low-latency throw anchor drifted")
    text = text.replace(KERNEL_ANCHOR, LOW_LATENCY_KERNEL, 1)
    text = text.replace(THROW_BODY, RESTORED_BODY, 1)
    if "Min latency mode is no longer supported" in text:
        raise RuntimeError("removed low-latency throw survived patch")
    return text.encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    source = args.target.read_bytes()
    result = patch(source)
    temporary = args.target.with_name(f".{args.target.name}.minlat-{os.getpid()}.tmp")
    temporary.write_bytes(result)
    os.replace(temporary, args.target)
    print(f"SOURCE_SHA256={_sha256(source)}")
    print(f"PATCHED_SHA256={_sha256(result)}")


if __name__ == "__main__":
    main()
