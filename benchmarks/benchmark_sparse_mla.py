#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded SM12x DSV4 sparse-width correctness and component timing.

All arms attend to the same active entries; only sentinel padding changes.
This uses FlashInfer's 584-byte FP8-footer DSV4 interface. It does not measure
model throughput or claim generic NVFP4 KV support. Requires torch/FlashInfer
inside a compatible GPU environment; no checkpoint is loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", default="512", help="Compare 512,256,192 after verifying installed support")
    parser.add_argument("--tokens", default="5,10,20,60,64,65")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--active-entries", type=int, default=133)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--isolate-aot-dir", type=Path,
                        help="Empty diagnostic AOT directory; prevents loading an old packaged kernel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = [int(value) for value in args.widths.split(",")]
    tokens = [int(value) for value in args.tokens.split(",")]
    if (not widths or not tokens or min(tokens) < 1 or max(tokens) > 256
            or not 1 <= args.active_entries <= min(widths) or max(widths) > 1024
            or args.heads not in (8, 16, 32, 64, 128)
            or args.trials < 1 or args.repetitions < 1 or args.timeout <= 0):
        parser.error("invalid or unbounded benchmark dimensions")
    import torch
    import flashinfer
    from flashinfer.mla import _sparse_mla_sm120 as sparse
    from flashinfer.jit import env as jit_env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.isolate_aot_dir:
        args.isolate_aot_dir.mkdir(parents=True, exist_ok=True)
        if any(args.isolate_aot_dir.iterdir()):
            raise RuntimeError("isolated AOT directory must be empty")
        jit_env.FLASHINFER_AOT_DIR = args.isolate_aot_dir.resolve()

    start = time.monotonic()
    report = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "seed": args.seed, "heads": args.heads,
        "active_entries": args.active_entries, "widths": widths, "tokens": tokens,
        "repetitions": args.repetitions, "trials": args.trials,
        "kv_format": "DSV4 584-byte FP8 footer, BF16 RoPE, page64",
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "source_sha256": hashlib.sha256(Path(sparse.__file__).read_bytes()).hexdigest(),
        "timing_mode": "eager public-call CUDA event envelope including enqueue gaps and internal overhead; no autotune context",
        "jit_workspace": str(jit_env.FLASHINFER_WORKSPACE_DIR),
        "aot_directory": str(jit_env.FLASHINFER_AOT_DIR),
        "runtime_source_hashes": {},
        "correctness_tolerance": {"atol": 0.05, "rtol": 0.05},
        "results": [],
    }

    def save():
        report["elapsed_s"] = time.monotonic() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def deadline():
        if time.monotonic() - start > args.timeout:
            raise TimeoutError("component benchmark exceeded its time budget")

    try:
        for path in [Path(sparse.__file__),
                     jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_decode_dsv4.cu",
                     jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_jit_binding.cu",
                     jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_prefill.cu",
                     jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh"]:
            report["runtime_source_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        capability = torch.cuda.get_device_capability()
        if capability[0] != 12:
            raise RuntimeError(f"requires SM12x, found {capability}")
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {"name": torch.cuda.get_device_name(), "capability": capability,
                         "free_bytes_before": free, "total_bytes": total}
        if free < args.min_free_gib * 1024**3:
            raise RuntimeError("insufficient free GPU memory for isolated component test")
        dispatch = sparse._DECODE_DSV4_DISPATCH
        unsupported = [width for width in widths if (args.heads, width) not in dispatch]
        if unsupported:
            raise RuntimeError(f"installed FlashInfer does not declare decode shapes: {unsupported}")
        torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        device = "cuda"
        blocks, page, dimension = 32, 64, 512
        count = blocks * page
        # Quantize each 64-value non-RoPE group using power-of-two FP8 scales.
        # Unit-scale inputs keep the reference well above the absolute tolerance;
        # tiny Q/K values could let an all-zero or stale output pass by accident.
        original = torch.randn(blocks, page, dimension, device=device)
        groups = original[..., :448].reshape(blocks, page, 7, 64)
        scales = torch.pow(2.0, torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)))
        quantized = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        rope = original[..., 448:].to(torch.bfloat16).contiguous()
        # Footer stores seven exponent bytes and one reserved byte per token.
        packed = torch.zeros(blocks, page * 584, device=device, dtype=torch.uint8)
        data = packed[:, :page * 576].view(blocks, page, 576)
        data[..., :448] = quantized.view(torch.uint8).reshape(blocks, page, 448)
        data[..., 448:] = rope.view(torch.uint8)
        footer = packed[:, page * 576:].view(blocks, page, 8)
        footer[..., :7] = (torch.log2(scales) + 127).to(torch.uint8)
        cache = packed.view(blocks, page, 1, 584)
        dequantized = torch.cat(((quantized.float() * scales[..., None]).reshape(blocks, page, 448),
                                 rope.float()), dim=-1).reshape(count, dimension)
        workspace = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
        sinks = torch.randn(args.heads, device=device, dtype=torch.float32)
        scale = dimension**-0.5
        for num_tokens in tokens:
            deadline()
            query = torch.randn(num_tokens, args.heads, dimension, device=device).to(torch.bfloat16)
            live_indices = torch.randint(count, (num_tokens, args.active_entries), device=device, dtype=torch.int32)
            lengths = torch.full((num_tokens,), args.active_entries, device=device, dtype=torch.int32)
            # Exercise both padding and partially populated context rows.
            if num_tokens > 1:
                lengths[0] = 0
                lengths[1] = max(1, args.active_entries // 2)
            def reference_for_query(sign=1):
                result = torch.empty_like(query)
                for token in range(num_tokens):
                    active = int(lengths[token].item())
                    keys = dequantized[live_indices[token, :active].long()]
                    logits = (query[token].float() * sign) @ keys.T * scale
                    weights = torch.softmax(torch.cat((logits, sinks[:, None]), dim=1), dim=1)[:, :active]
                    result[token] = (weights @ keys).to(torch.bfloat16)
                return result

            reference = reference_for_query()
            negative_reference = reference_for_query(-1)
            if reference.float().abs().max().item() <= 0.05:
                raise RuntimeError("reference has insufficient signal to reject a zero output")
            if (reference.float() - negative_reference.float()).abs().max().item() <= 0.1:
                raise RuntimeError("query replay has insufficient signal to reject a stale output")
            arms = {}
            for width in widths:
                indices = torch.full((num_tokens, width), -1, device=device, dtype=torch.int32)
                indices[:, :args.active_entries] = live_indices
                output = torch.empty_like(query).unsqueeze(1)

                def run(indices=indices, output=output):
                    return flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
                        query=query.unsqueeze(1), swa_kv_cache=cache,
                        workspace_buffer=workspace, sparse_indices=indices,
                        swa_topk_lens=lengths, bmm1_scale=scale, sinks=sinks,
                        out=output, kv_layout="NHD",
                    )

                run()
                torch.cuda.synchronize()
                torch.testing.assert_close(output.squeeze(1), reference, atol=0.05, rtol=0.05)
                max_error = (output.squeeze(1).float() - reference.float()).abs().max().item()
                # Replay with changed query values to catch stale cached outputs.
                query.neg_()
                run()
                torch.cuda.synchronize()
                torch.testing.assert_close(output.squeeze(1), negative_reference, atol=0.05, rtol=0.05)
                query.neg_()
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                torch.testing.assert_close(output.squeeze(1), reference, atol=0.05, rtol=0.05)
                arms[width] = {"run": run, "times_us": [], "max_abs_error": max_error}
                deadline()
            # Alternate arm order across trials to reduce warm/thermal bias.
            for trial in range(args.trials):
                order = widths if trial % 2 == 0 else list(reversed(widths))
                for width in order:
                    deadline()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(args.repetitions):
                        arms[width]["run"]()
                    end.record()
                    end.synchronize()
                    arms[width]["times_us"].append(begin.elapsed_time(end) * 1000 / args.repetitions)
            for width in widths:
                arm = arms[width]
                report["results"].append({"tokens": num_tokens, "width": width,
                                          "max_abs_error": arm["max_abs_error"], "correct": True,
                                          "reference_rms": reference.float().square().mean().sqrt().item(),
                                          "trials_us": arm["times_us"],
                                          "median_us": statistics.median(arm["times_us"])})
            save()
            print(json.dumps(report["results"][-len(widths):]), flush=True)
        report["status"] = "passed"
        report["gpu"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        libraries = list(jit_env.FLASHINFER_JIT_DIR.glob("*sparse_mla_sm120*/*.so"))
        libraries.extend(jit_env.FLASHINFER_AOT_DIR.glob("*sparse_mla_sm120*/*.so"))
        report["library_artifacts"] = [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in libraries
        ]
        save()


if __name__ == "__main__":
    main()
