#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DSpark confidence-head and physical variable-verifier probe."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open


WEIGHT_NAME = "mtp.2.confidence_head.proj.weight"
EXPECTED_SHAPE = (1, 4352)


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_checkpoint_weight(checkpoint: Path) -> tuple[torch.Tensor, Path]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"][WEIGHT_NAME]
    shard_path = checkpoint / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(WEIGHT_NAME)
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"unexpected confidence weight contract: {weight.dtype} "
            f"{tuple(weight.shape)}"
        )
    return weight, shard_path


def verify_async_scheduler_contract() -> dict[str, object]:
    """Exercise physical row compaction and truthful metric adjustment."""

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.dspark.confidence import (
        mask_draft_tokens_by_confidence,
        trim_invalid_draft_tail,
    )
    from vllm.v1.worker.gpu.spec_decode.utils import (
        DraftTokensHandler,
        compact_scheduler_output_for_variable_drafts,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.variable_verifier import (
        complete_async_copy_if_needed,
    )

    class Request:
        @staticmethod
        def is_finished() -> bool:
            return False

    class StructuredOutputManager:
        @staticmethod
        def should_advance(request: object) -> bool:
            return False

    scheduler = SimpleNamespace(
        requests={"probe": Request()},
        structured_output_manager=StructuredOutputManager(),
        log_stats=True,
        num_spec_tokens=5,
    )
    raw_tokens = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int64)
    probabilities = torch.tensor(
        [[0.9, 0.8, 0.2, 0.99, 0.99]], dtype=torch.float32
    )
    masked, prefix_lengths = mask_draft_tokens_by_confidence(
        raw_tokens,
        torch.logit(probabilities),
        threshold=0.5,
    )
    truncated = trim_invalid_draft_tail(masked.tolist()[0])
    if prefix_lengths.tolist() != [2] or truncated != [10, 11]:
        raise RuntimeError(
            "confidence policy did not produce the expected two-token prefix: "
            f"lengths={prefix_lengths.tolist()}, tokens={truncated}"
        )
    truncated_length = len(truncated)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"probe": 6}
    output.total_num_scheduled_tokens = 6
    output.scheduled_spec_decode_tokens = {"probe": [-1] * 5}
    invalid = compact_scheduler_output_for_variable_drafts(
        output,
        ["probe"],
        [masked.tolist()[0]],
    )
    physical = output.scheduled_spec_decode_tokens["probe"]
    if (
        physical != [10, 11]
        or invalid != {"probe": 3}
        or output.num_scheduled_tokens != {"probe": 3}
        or output.total_num_scheduled_tokens != 3
    ):
        raise RuntimeError(
            "physical verifier compaction failed: "
            f"tokens={physical}, invalid={invalid}, "
            f"scheduled={output.num_scheduled_tokens}, "
            f"total={output.total_num_scheduled_tokens}"
        )

    runner_output = ModelRunnerOutput(
        req_ids=["probe"],
        req_id_to_index={"probe": 0},
        confidence_invalid_spec_tokens=invalid,
        confidence_physical_target_rows=[3],
        confidence_d2h_copy_fallback=False,
    )
    if runner_output.confidence_invalid_spec_tokens != {"probe": 3}:
        raise RuntimeError("physical trim evidence was not preserved in runner output")

    runner_source = inspect.getsource(GPUModelRunner.execute_model)
    compact_pos = runner_source.index("compact_scheduler_output(")
    dispatch_pos = runner_source.index("dispatch_cg_and_sync_dp(")
    if compact_pos >= dispatch_pos:
        raise RuntimeError("physical compaction is not before CUDA-graph dispatch")
    scheduler_source = inspect.getsource(AsyncScheduler.update_from_output)
    if (
        "confidence_invalid_spec_tokens" not in scheduler_source
        or "max(merged.get(req_id, 0), count)" not in scheduler_source
        or "observe_engine_compaction_telemetry" not in scheduler_source
    ):
        raise RuntimeError("async scheduler lacks physical verifier metric correction")
    cudagraph_source = inspect.getsource(CudaGraphManager._init_candidates)
    if (
        "variable_dspark = (" not in cudagraph_source
        or "list(range(1, self.decode_query_len + 1))" not in cudagraph_source
        or "and rounded_num_reqs > 1" not in cudagraph_source
    ):
        raise RuntimeError(
            "CUDA graph manager lacks bounded exact C=1 DSpark verifier shapes"
        )

    get_draft_source = inspect.getsource(DraftTokensHandler.get_draft_tokens)
    compact_source = inspect.getsource(DraftTokensHandler.compact_scheduler_output)
    completion_source = inspect.getsource(complete_async_copy_if_needed)
    if (
        "self.scheduler_requires_draft_tokens" not in get_draft_source
        or "complete_async_copy_if_needed" not in compact_source
        or "last_physical_target_rows" not in compact_source
        or "last_d2h_copy_fallback" not in compact_source
        or completion_source.index("event.query()")
        >= completion_source.index("event.synchronize()")
    ):
        raise RuntimeError(
            "DSpark proposal D2H copy is not overlap-first with a measured "
            "fail-closed wait"
        )

    observed: dict[str, int] = {}

    class Stats:
        @staticmethod
        def observe_draft(
            *, num_draft_tokens: int, num_accepted_tokens: int
        ) -> None:
            observed["draft"] = num_draft_tokens
            observed["accepted"] = num_accepted_tokens

    stats = Stats()
    returned = Scheduler.make_spec_decoding_stats(
        scheduler,
        stats,
        num_draft_tokens=5,
        num_accepted_tokens=1,
        num_invalid_spec_tokens=runner_output.confidence_invalid_spec_tokens,
        request_id="probe",
    )
    if returned is not stats or observed != {"draft": 2, "accepted": 1}:
        raise RuntimeError(
            "installed speculative metrics did not subtract invalid draft slots: "
            f"observed={observed}"
        )
    scheduled_slots = len(physical)
    return {
        "raw_slots": 5,
        "truncated_proposal_length": truncated_length,
        "scheduled": physical,
        "scheduled_slots_seen_by_runner": scheduled_slots,
        "invalid_slots": invalid["probe"],
        "target_rows_including_bonus": output.num_scheduled_tokens["probe"],
        "metrics_draft_tokens": observed["draft"],
        "metrics_accepted_tokens": observed["accepted"],
        "metrics_proposed_equals_truncated": (
            observed["draft"] == truncated_length
        ),
        "physical_verifier_shortened": scheduled_slots == truncated_length,
        "variable_length_verify_ready": True,
        "diagnosis": "confidence prefix physically compacts target rows",
        "integration": {
            "compaction_before_cuda_graph_dispatch": True,
            "physical_trim_returned_to_scheduler": True,
            "grammar_overlap_uses_max_not_sum": True,
            "exact_c1_cuda_graph_shapes_1_to_6": True,
            "unstructured_scheduler_copy_wait_elided": True,
            "d2h_completion_ready_vs_fallback_telemetry": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expect-scheduler", choices=("off", "on"), required=True)
    parser.add_argument("--expect-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vllm.model_executor import parameter as parameter_module
    from vllm.model_executor.layers import linear as linear_module
    from vllm.models.deepseek_v4.nvidia.dspark import (
        DSparkConfidenceHead,
        DSparkDeepseekV4ForCausalLM,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.confidence import (
        mask_draft_tokens_by_confidence,
        parse_confidence_config,
        trim_invalid_draft_tail,
    )

    config = parse_confidence_config(os.environ)
    if config.scheduler != args.expect_scheduler or not math.isclose(
        config.threshold, args.expect_threshold, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError(
            f"confidence env mismatch: observed={config.as_dict()} "
            f"expected={args.expect_scheduler}/{args.expect_threshold}"
        )

    mapped = DSparkDeepseekV4ForCausalLM._remap_dspark_name(None, WEIGHT_NAME)
    if mapped != "model.confidence_head.proj.weight":
        raise RuntimeError(f"confidence weight remapped incorrectly: {mapped!r}")

    weight, shard_path = load_checkpoint_weight(args.checkpoint)
    async_scheduler = verify_async_scheduler_contract()

    # ReplicatedLinear normally reads the initialized TP group. This isolated
    # one-parameter probe is intentionally TP=1 and patches only the imported
    # rank/world-size accessors used while constructing the real vLLM head.
    # ModelWeightParameter keeps its own imported rank symbol, so patching the
    # linear module alone is insufficient.
    linear_module.get_tensor_model_parallel_rank = lambda: 0
    linear_module.get_tensor_model_parallel_world_size = lambda: 1
    parameter_module.get_tensor_model_parallel_rank = lambda: 0
    parameter_module.get_tensor_model_parallel_world_size = lambda: 1
    head = DSparkConfidenceHead(EXPECTED_SHAPE[1], prefix="probe.confidence_head")
    head.proj.weight_loader(head.proj.weight, weight)
    if head.proj.weight.dtype != torch.float32 or not torch.equal(
        head.proj.weight.detach().cpu(), weight.float()
    ):
        raise RuntimeError("real vLLM confidence weight loader lost FP32 parity")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA probe requires exactly one visible GPU")
        head = head.to(device)

    hidden = torch.linspace(-0.5, 0.5, 2 * 4096, dtype=torch.float32).reshape(
        2, 4096
    )
    markov = torch.linspace(0.25, -0.25, 2 * 256, dtype=torch.float32).reshape(
        2, 256
    )
    inputs = torch.cat((hidden, markov), dim=-1)
    if inputs.shape[-1] != EXPECTED_SHAPE[1] or head.proj.input_size != 4352:
        raise RuntimeError(
            "confidence head input width drifted: "
            f"concat={inputs.shape[-1]}, projection={head.proj.input_size}"
        )
    expected_logits = F.linear(inputs, weight.float()).squeeze(-1)
    actual_logits = head(hidden.to(device), markov.to(device)).float().cpu()
    if not torch.isfinite(actual_logits).all() or not torch.allclose(
        actual_logits, expected_logits, rtol=1e-5, atol=1e-4
    ):
        raise RuntimeError("confidence forward is non-finite or mismatched")

    draft_tokens = torch.tensor([[10, 11, 12, 13, 14]], device=device)
    policy_logits = torch.logit(
        torch.tensor([[0.9, 0.8, 0.2, 0.99, 0.99]], device=device)
    )
    half_tokens, half_lengths = mask_draft_tokens_by_confidence(
        draft_tokens, policy_logits, threshold=0.5
    )
    full_tokens, full_lengths = mask_draft_tokens_by_confidence(
        draft_tokens, policy_logits, threshold=0.0
    )
    empty_tokens, empty_lengths = mask_draft_tokens_by_confidence(
        draft_tokens, policy_logits, threshold=1.0
    )
    if half_lengths.cpu().tolist() != [2] or trim_invalid_draft_tail(
        half_tokens.cpu().tolist()[0]
    ) != [10, 11]:
        raise RuntimeError("threshold 0.5 did not preserve the expected prefix")
    if full_lengths.cpu().tolist() != [5] or not torch.equal(
        full_tokens.cpu(), draft_tokens.cpu()
    ):
        raise RuntimeError("threshold 0.0 did not preserve the full proposal")
    if empty_lengths.cpu().tolist() != [0] or empty_tokens.cpu().tolist() != [
        [-1] * 5
    ]:
        raise RuntimeError("threshold 1.0 did not suppress the proposal")

    result = {
        "ok": True,
        "config": config.as_dict(),
        "device": str(device),
        "cuda": (
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
            }
            if device.type == "cuda"
            else None
        ),
        "checkpoint": {
            "weight_name": WEIGHT_NAME,
            "mapped_name": mapped,
            "shard": shard_path.name,
            "shape": list(weight.shape),
            "checkpoint_dtype": str(weight.dtype),
            "runtime_dtype": str(head.proj.weight.dtype),
            "tensor_sha256": tensor_sha256(weight),
        },
        "forward": {
            "finite": True,
            "logits": actual_logits.tolist(),
            "input_width": inputs.shape[-1],
            "input_contract": "concat(backbone_hidden_4096,markov_embed_256)",
        },
        "policy": {
            "threshold_0_0_length": 5,
            "threshold_0_5_length": 2,
            "threshold_1_0_length": 0,
            "tail_sentinel": -1,
        },
        "async_scheduler": async_scheduler,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
