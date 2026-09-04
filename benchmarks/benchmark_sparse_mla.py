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
import math
import os
from pathlib import Path
import statistics
import tempfile
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
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--compiled-provenance", type=Path,
                        help="Reuse a hash-verified library from a successful isolated diagnostic")
    parser.add_argument("--image-id", help="Coordinator-verified image ID; record alongside Docker inspect evidence")
    parser.add_argument("--isolate-aot-dir", type=Path,
                        help="Empty diagnostic AOT directory; prevents loading an old packaged kernel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compiled_provenance and args.isolate_aot_dir:
        parser.error("choose compiled provenance or an empty AOT directory")
    widths = [int(value) for value in args.widths.split(",")]
    tokens = [int(value) for value in args.tokens.split(",")]
    if (not widths or not tokens or min(tokens) < 1 or max(tokens) > 256
            or not 1 <= args.active_entries <= min(widths) or max(widths) > 1024
            or args.heads not in (8, 16, 32, 64, 128)
            or not 1 <= args.trials <= 100 or not 1 <= args.repetitions <= 1000
            or not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600
            or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0
            or len(set(widths)) != len(widths) or len(set(tokens)) != len(tokens)):
        parser.error("invalid or unbounded benchmark dimensions")
    if args.compiled_provenance and not args.image_id:
        parser.error("binary reuse requires --image-id and matching external Docker inspect evidence")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be new; refusing to overwrite existing evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    autotune_dir = args.output.with_name(args.output.stem + "-autotune")
    autotune_dir.mkdir(exist_ok=False)
    os.environ["FLASHINFER_AUTOTUNE_DIR"] = str(autotune_dir.resolve())
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
        "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(sparse.__file__).read_bytes()).hexdigest(),
        "coordinator_verified_image_id": args.image_id,
        "image_identity_note": "Not introspected inside container; join with preserved Docker inspect evidence. Required when reusing old five-file manifests.",
        "tvm_ffi_version": importlib.metadata.version("apache-tvm-ffi"),
        "autotune_cache": str(sparse._decode_dsv4_default_cache_path()),
        "autotune_policy": "Fresh empty disk cache; no autotune context. Actual hot-cache tactics recorded at exit.",
        "timing_mode": (
            "captured public-call batch, CUDA event envelope per call; no host gaps inside batch"
            if args.timing_mode == "graph" else
            "eager public-call CUDA event envelope including enqueue gaps and internal overhead; no autotune context"
        ),
        "jit_workspace": str(jit_env.FLASHINFER_WORKSPACE_DIR),
        "aot_directory": str(jit_env.FLASHINFER_AOT_DIR),
        "runtime_source_hashes": {},
        "correctness_tolerance": {"atol": 0.05, "rtol": 0.05},
        "results": [],
        "trial_orders": [],
    }

    def save():
        report["elapsed_s"] = time.monotonic() - start
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        try:
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)

    def deadline():
        if time.monotonic() - start > args.timeout:
            raise TimeoutError("component benchmark exceeded its time budget")

    save()
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
        from flashinfer.jit.mla import gen_sparse_mla_sm120_module
        spec = gen_sparse_mla_sm120_module()
        report["all_jit_sources_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in spec.sources
        }
        report["extra_cuda_cflags"] = spec.extra_cuda_cflags
        report["public_wrapper_sha256"] = hashlib.sha256(
            Path(sparse.__file__).with_name("_core.py").read_bytes()
        ).hexdigest()
        if args.compiled_provenance:
            provenance = json.loads(args.compiled_provenance.read_text())
            if provenance["status"] not in ("passed", "compile_only_passed"):
                raise RuntimeError("reuse requires a successful isolated build")
            if (provenance["torch_version"] != torch.__version__
                    or provenance["cuda_version"] != torch.version.cuda
                    or provenance["flashinfer_distribution_version"] != report["flashinfer_version"]
                    or tuple(provenance["gpu"]["capability"]) != capability):
                raise RuntimeError("compiled runtime/GPU provenance mismatch")
            old_hashes = {Path(row["path"]).name: row["sha256"] for row in provenance["runtime_files"]}
            if old_hashes != report["runtime_source_hashes"]:
                raise RuntimeError("compiled source hashes differ from this runtime")
            library = Path(provenance["library"]["path"])
            if hashlib.sha256(library.read_bytes()).hexdigest() != provenance["library"]["sha256"]:
                raise RuntimeError("compiled library hash mismatch")
            jit_env.FLASHINFER_AOT_DIR = library.parent.parent
            spec = gen_sparse_mla_sm120_module()
            if not spec.is_aot or spec.aot_path.resolve() != library.resolve():
                raise RuntimeError("reused library does not match the resolved AOT path")
            report["aot_directory"] = str(jit_env.FLASHINFER_AOT_DIR)
            report["reused_library"] = provenance["library"]
            report["compiled_provenance_sha256"] = hashlib.sha256(args.compiled_provenance.read_bytes()).hexdigest()
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
                record = {"tokens": num_tokens, "width": width, "correct": False,
                          "checks": {}, "trials_us": [], "median_us": None,
                          "reference_rms": reference.float().square().mean().sqrt().item()}
                report["results"].append(record)
                save()
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
                def check(expected, phase, output=output, record=record):
                    error = (output.squeeze(1).float() - expected.float()).abs().max().item()
                    record["checks"][phase] = {"max_abs_error": error, "passed": False}
                    record["correct"] = False
                    torch.testing.assert_close(output.squeeze(1), expected, atol=0.05, rtol=0.05)
                    record["checks"][phase]["passed"] = True
                    record["correct"] = True
                    record["max_abs_error"] = max(row["max_abs_error"] for row in record["checks"].values())

                check(reference, "eager_positive")
                # Replay with changed query values to catch stale cached outputs.
                query.neg_()
                run()
                torch.cuda.synchronize()
                check(negative_reference, "eager_negative")
                query.neg_()
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                check(reference, "eager_restored")
                graph = None
                if args.timing_mode == "graph":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.repetitions):
                            run()
                    # Captured buffers must observe changed inputs, not just
                    # reproduce the values present during capture.
                    query.neg_()
                    graph.replay()
                    torch.cuda.synchronize()
                    check(negative_reference, "graph_negative")
                    query.neg_()
                    graph.replay()
                    torch.cuda.synchronize()
                    check(reference, "graph_restored")
                record["correct"] = True
                arms[width] = {"run": run, "graph": graph, "record": record, "check": check}
                save()
                deadline()
            # Alternate arm order across trials to reduce warm/thermal bias.
            for trial in range(args.trials):
                offset = trial % len(widths)
                order = widths[offset:] + widths[:offset]
                if (trial // len(widths)) % 2:
                    order = list(reversed(order))
                report["trial_orders"].append({"tokens": num_tokens, "trial": trial, "widths": order})
                for width in order:
                    deadline()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    if arms[width]["graph"] is not None:
                        arms[width]["graph"].replay()
                    else:
                        for _ in range(args.repetitions):
                            arms[width]["run"]()
                    end.record()
                    end.synchronize()
                    arm = arms[width]
                    arm["record"]["trials_us"].append(begin.elapsed_time(end) * 1000 / args.repetitions)
                    arm["check"](reference, f"after_trial_{trial}")
                    save()
            for width in widths:
                arm = arms[width]
                arm["record"]["median_us"] = statistics.median(arm["record"]["trials_us"])
            save()
            print(json.dumps(report["results"][-len(widths):]), flush=True)
        report["status"] = "passed"
        report["gpu"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["decode_hot_cache_tactics"] = [
            {"signature": repr(key), "tactic": repr(value)}
            for key, value in sparse._decode_dsv4_hot_cache.items()
        ]
        libraries = list(jit_env.FLASHINFER_JIT_DIR.glob("*sparse_mla_sm120*/*.so"))
        libraries.extend(jit_env.FLASHINFER_AOT_DIR.glob("*sparse_mla_sm120*/*.so"))
        report["library_artifacts"] = [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in libraries
        ]
        save()


if __name__ == "__main__":
    main()
