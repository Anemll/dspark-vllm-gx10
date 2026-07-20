#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the real V2 execute-state -> sample_tokens boundary without weights."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch


EXPECTED_TRIM = {"probe": 3}


class _CapturedAsyncOutput:
    def __init__(
        self,
        *,
        model_runner_output,
        sampler_output,
        num_sampled_tokens,
        main_stream,
        copy_stream,
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens


class _PromptLogprobs:
    @staticmethod
    def compute_prompt_logprobs(*_args, **_kwargs):
        return {}


class _KVConnector:
    @staticmethod
    def pre_forward(_scheduler_output):
        return None

    @staticmethod
    def post_forward(_finished_req_ids):
        return None


class _EPLB:
    steps = 0

    def step(self, **_kwargs):
        self.steps += 1

    @staticmethod
    def prepare_forward(*_args, **_kwargs):
        return None


class _DraftTokensHandler:
    def __init__(self, trim: dict[str, int]):
        self.trim = trim
        self.calls = 0

    def compact_scheduler_output(self, _scheduler_output):
        self.calls += 1
        return dict(self.trim)


class _ModelState:
    @staticmethod
    def preprocess_state(*_args, **_kwargs):
        return None

    @staticmethod
    def prepare_attn(*_args, **_kwargs):
        return None

    @staticmethod
    def prepare_inputs(*_args, **_kwargs):
        return {}


class _BlockTables:
    @staticmethod
    def apply_staged_writes():
        return None


def _input_batch(device: torch.device):
    return SimpleNamespace(
        input_ids=torch.tensor([1], dtype=torch.int64, device=device),
        positions=torch.tensor([0], dtype=torch.int64, device=device),
        num_tokens=1,
        num_tokens_after_padding=1,
        is_padding=False,
        req_ids=["probe"],
        idx_mapping=torch.tensor([0], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
    )


def _scheduler_output():
    return SimpleNamespace(
        num_scheduled_tokens={"probe": 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
    )


def _nested(value):
    return SimpleNamespace(gpu=value)


def _make_runner(model_runner_module, *, device: torch.device, trim: dict[str, int]):
    sampled = torch.tensor([[7]], dtype=torch.int64, device=device)
    input_batch = _input_batch(device)
    runner = SimpleNamespace(
        execute_model_state=None,
        is_last_pp_rank=True,
        is_first_pp_rank=True,
        pp_handler=None,
        lora_config=None,
        lora_state=None,
        is_encoder_decoder=False,
        supports_mm_inputs=False,
        use_aux_hidden_state_outputs=False,
        cudagraph_manager=None,
        dp_size=1,
        dp_rank=0,
        model_config=None,
        kv_cache_config=None,
        attn_groups=None,
        vllm_config=None,
        input_buffers=None,
        block_tables=_BlockTables(),
        model_state=_ModelState(),
        draft_tokens_handler=_DraftTokensHandler(trim),
        prompt_logprobs_worker=_PromptLogprobs(),
        model=SimpleNamespace(
            compute_logits=lambda *_args, **_kwargs: None,
            requires_raw_input_tokens=False,
        ),
        req_states=SimpleNamespace(
            all_token_ids=_nested(None),
            num_computed_tokens=_nested(None),
            prompt_len=SimpleNamespace(np=None),
        ),
        main_stream=None,
        output_copy_stream=None,
        speculator=None,
        num_speculative_steps=0,
        kv_connector=_KVConnector(),
        eplb=_EPLB(),
    )
    runner.model = SimpleNamespace(
        compute_logits=lambda *_args, **_kwargs: None,
        requires_raw_input_tokens=False,
        __call__=None,
    )
    runner.model = lambda **_kwargs: torch.zeros(
        (1, 1), dtype=torch.float32, device=device
    )
    runner.model.compute_logits = lambda *_args, **_kwargs: None
    runner.model.requires_raw_input_tokens = False
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = lambda *_args: None
    runner.free_states = lambda *_args: None
    runner.add_requests = lambda *_args: None
    runner.update_requests = lambda *_args: None
    runner.prepare_inputs = lambda *_args: input_batch
    runner.prepare_attn = lambda *_args: (object(), object())
    runner.sample = lambda *_args, **_kwargs: (
        SimpleNamespace(sampled_token_ids=sampled),
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
    )
    runner.postprocess_sampled = lambda *_args, **_kwargs: None
    return runner


def run_case(*, device: torch.device, trim: dict[str, int], dummy_run: bool):
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    runner = _make_runner(model_runner_module, device=device, trim=trim)
    scheduler_output = _scheduler_output()
    original_async_output = model_runner_module.AsyncOutput
    original_dispatch = model_runner_module.dispatch_cg_and_sync_dp
    original_build_slots = model_runner_module.build_slot_mappings_by_layer
    original_forward_context = model_runner_module.set_forward_context
    original_make_dummy = model_runner_module.InputBatch.make_dummy
    model_runner_module.AsyncOutput = _CapturedAsyncOutput
    model_runner_module.dispatch_cg_and_sync_dp = lambda *_args, **_kwargs: (
        SimpleNamespace(
            num_reqs=1,
            num_tokens=1,
            cg_mode=model_runner_module.CUDAGraphMode.NONE,
            num_active_loras=0,
        ),
        None,
    )
    model_runner_module.build_slot_mappings_by_layer = lambda *_args: None
    model_runner_module.set_forward_context = lambda *_args, **_kwargs: nullcontext()
    model_runner_module.InputBatch.make_dummy = staticmethod(
        lambda *_args, **_kwargs: _input_batch(device)
    )
    try:
        execute = inspect.unwrap(model_runner_module.GPUModelRunner.execute_model)
        sample = inspect.unwrap(model_runner_module.GPUModelRunner.sample_tokens)
        execute(
            runner,
            scheduler_output,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=dummy_run,
        )
        fields = tuple(runner.execute_model_state._fields)
        output = sample(runner, None)
    finally:
        model_runner_module.AsyncOutput = original_async_output
        model_runner_module.dispatch_cg_and_sync_dp = original_dispatch
        model_runner_module.build_slot_mappings_by_layer = original_build_slots
        model_runner_module.set_forward_context = original_forward_context
        model_runner_module.InputBatch.make_dummy = original_make_dummy
    return output, fields, runner.draft_tokens_handler.calls


def verify_source_contract() -> dict[str, object]:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    execute_source = inspect.getsource(GPUModelRunner.execute_model)
    sample_source = inspect.getsource(GPUModelRunner.sample_tokens)
    save = "confidence_invalid_spec_tokens=confidence_invalid_spec_tokens"
    read = "self.execute_model_state.confidence_invalid_spec_tokens"
    if save not in execute_source or read not in sample_source:
        raise RuntimeError("execute/sample physical-trim state handoff is absent")
    return {
        "execute_state_write": True,
        "sample_state_read": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--expect", choices=("old-nameerror", "pass"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["VLLM_DSPARK_CONFIDENCE_SCHEDULER"] = "on"
    os.environ["VLLM_DSPARK_CONFIDENCE_THRESHOLD"] = "0.5"
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one available CUDA device")

    try:
        trimmed_output, state_fields, compact_calls = run_case(
            device=device, trim=EXPECTED_TRIM, dummy_run=False
        )
        warmup_output, warmup_fields, warmup_compact_calls = run_case(
            device=device, trim={}, dummy_run=True
        )
    except NameError as error:
        if args.expect != "old-nameerror" or "confidence_invalid_spec_tokens" not in str(error):
            raise
        result = {
            "ok": True,
            "expectation": args.expect,
            "reproduced": type(error).__name__,
            "message": str(error),
            "device": args.device,
            "state_field_present": False,
        }
    else:
        if args.expect != "pass":
            raise RuntimeError("old code unexpectedly survived the split boundary")
        trimmed = trimmed_output.model_runner_output.confidence_invalid_spec_tokens
        warmup = warmup_output.model_runner_output.confidence_invalid_spec_tokens
        if trimmed != EXPECTED_TRIM or warmup is not None:
            raise RuntimeError(
                f"split evidence drift: trimmed={trimmed}, warmup={warmup}"
            )
        if compact_calls != 1 or warmup_compact_calls != 0:
            raise RuntimeError(
                "execute path drift: "
                f"compact_calls={compact_calls}, "
                f"warmup_compact_calls={warmup_compact_calls}"
            )
        source = verify_source_contract()
        result = {
            "ok": True,
            "expectation": args.expect,
            "device": args.device,
            "state_fields": list(state_fields),
            "warmup_state_fields": list(warmup_fields),
            "trimmed_output": trimmed,
            "warmup_output": warmup,
            "execute_compaction_calls": compact_calls,
            "warmup_compaction_calls": warmup_compact_calls,
            "cuda": (
                {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                }
                if device.type == "cuda"
                else None
            ),
            **source,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
