#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Patch the exact target-only V1 GPU runner with bounded route capture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, anchor: str, replacement: str) -> str:
    if source.count(anchor) != 1:
        raise RuntimeError(
            f"expected one V1 route-capture anchor, found {source.count(anchor)}"
        )
    return source.replace(anchor, replacement, 1)


def patch(path: Path) -> str:
    if sha256(path) != SOURCE_SHA256:
        raise RuntimeError(f"V1 GPU runner source SHA drift: {sha256(path)}")
    source = path.read_text()
    source = replace_once(
        source,
        """        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
""",
        """        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.v1.worker.gpu.target_route_capture import (
            TargetRouteCaptureConfig,
            validate_target_only_runtime,
        )

        self.target_route_capture = None
        self._target_route_capture_config = (
            TargetRouteCaptureConfig.from_environment()
        )
        if self._target_route_capture_config is not None:
            validate_target_only_runtime(
                speculative_config=self.speculative_config,
                speculator=None,
                num_speculative_steps=(
                    0
                    if self.speculative_config is None
                    else self.speculative_config.num_speculative_tokens
                ),
                enable_return_routed_experts=bool(
                    getattr(self.model_config, "enable_return_routed_experts", False)
                ),
                tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            )
""",
    )
    source = replace_once(
        source,
        """        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
""",
        """        if self._target_route_capture_config is not None:
            from vllm.distributed.parallel_state import get_tp_group
            from vllm.v1.worker.gpu.target_route_capture import (
                bind_target_route_capture,
                validate_loaded_target_model,
            )

            tp_group = get_tp_group()
            validate_loaded_target_model(self.model)
            self.target_route_capture = bind_target_route_capture(
                config=self._target_route_capture_config,
                static_forward_context=self.compilation_config.static_forward_context,
                device=self.device,
                rank=int(tp_group.rank_in_group),
                world_size=int(tp_group.world_size),
            )
            logger.info(
                "Enabled bounded V1 target-only route capture: "
                "rank=%d/%d steps=%d warmup=%d output=%s",
                int(tp_group.rank_in_group),
                int(tp_group.world_size),
                self._target_route_capture_config.steps,
                self._target_route_capture_config.warmup_steps,
                self._target_route_capture_config.output_dir,
            )

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
""",
    )
    source = replace_once(
        source,
        """        # Run the model.
        # Use persistent buffers for CUDA graphs.
""",
        """        capture_target_routes = False
        if self.target_route_capture is not None:
            capture_target_routes = self.target_route_capture.begin_v1_step(
                num_reqs=num_reqs,
                num_tokens=num_tokens_unpadded,
                num_scheduled_tokens=num_scheduled_tokens_np,
                use_spec_decode=use_spec_decode,
            )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
""",
    )
    source = replace_once(
        source,
        """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
""",
        """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if capture_target_routes:
            route_manifest = self.target_route_capture.end_step()
            if route_manifest is not None:
                logger.info("Completed V1 target route capture: %s", route_manifest)

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
""",
    )
    path.write_text(source)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(patch(args.target))


if __name__ == "__main__":
    main()
