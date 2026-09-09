#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded real-weight B12X TC-decode component gate; never serving TPS.

Run only in a disposable container with exclusive GPU access. The API model
must be stopped first. No server settings, weights or dependency files change.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time

TC_ENV = "B12X_W4A16_TC_DECODE"
TILE_ENV = "VLLM_B12X_W4A16_FORCE_TILE_CONFIG"
PINNED_SOURCES = {
    "b12x/integration/tp_moe.py": "49cd151aa80f4fdfa603eafe21b792b51a6483fa6f39452892ed2240fd79da34",
    "b12x/moe/fused/w4a16/kernel.py": "28872ab5e474f13212a9e57b22a06e1ccba8fef012ba183bf69208d8f8c9e677",
    "benchmarks/benchmark_moe.py": "ae213824d7931a51a719e2d1ee0fb67dacb075e6ab8ce65002db67019f617b79",
}


@contextmanager
def tc_mode(enabled):
    original = os.environ.get(TC_ENV)
    os.environ[TC_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(TC_ENV, None)
        else:
            os.environ[TC_ENV] = original


@contextmanager
def tile_mode(value: str):
    """Temporarily select a B12X tile override after the adapter is imported."""
    original = os.environ.get(TILE_ENV)
    os.environ[TILE_ENV] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(TILE_ENV, None)
        else:
            os.environ[TILE_ENV] = original


def validate_dispatch(observed, tokens, enabled, tile_config=None):
    expected = bool(enabled and tokens <= 8)
    if not observed or any(r["tc_decode_fused_sum"] != expected or
                           r["weight_layout"] != "packed" or
                           r["token_count"] != tokens for r in observed):
        raise RuntimeError(f"dispatch mismatch: tokens={tokens}, enabled={enabled}, observed={observed}")
    if tile_config is not None:
        tile_k, tile_n, _cta_threads = (int(part) for part in tile_config.split(","))
        if any(
            r["fc1_tile_k"] != tile_k or r["fc1_tile_n"] != tile_n
            for r in observed
        ):
            raise RuntimeError(
                f"FC1 tile override did not engage: expected={tile_config}, observed={observed}"
            )


def parity_gate(metrics):
    """Apply B12X's TC-decode oracle criterion, not output identity.

    The fused epilogue atomically reduces six expert contributions, whereas the
    control path performs a separate ordered top-k sum. Those are intentionally
    different accumulation orders in bf16. Each result must therefore be
    compared independently with the FP32 W4A16 oracle. This mirrors the
    upstream TC-decode test's SiLU cosine threshold.
    """
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        raise RuntimeError("nonfinite parity metrics")
    if metrics["cos"] < 0.9975:
        raise RuntimeError(f"numerical oracle parity failed: {metrics}")


def compare_to_oracle(actual, expected):
    """Return upstream-compatible per-token cosine plus diagnostic errors."""
    import torch

    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    difference = actual_fp32 - expected_fp32
    actual_rows = actual_fp32.reshape(actual_fp32.shape[0], -1)
    expected_rows = expected_fp32.reshape(expected_fp32.shape[0], -1)
    dot = (actual_rows * expected_rows).sum(dim=1)
    denominator = actual_rows.norm(dim=1) * expected_rows.norm(dim=1)
    both_zero = (actual_rows.norm(dim=1) <= 1e-12) & (expected_rows.norm(dim=1) <= 1e-12)
    row_cosines = torch.where(
        both_zero,
        torch.ones_like(dot),
        torch.where(denominator > 1e-24, dot / denominator, torch.zeros_like(dot)),
    )
    return {
        "max_abs": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "mean_abs": difference.abs().mean().item(),
        "cos": row_cosines.mean().item(),
    }


def real_weight_oracle(reference, x, weights, spec, ids, scores):
    """Independent FP32 W4A16 oracle for the loaded DeepSeek E8M0-K32 shard."""
    import torch

    if any(value is None for value in (
        weights.oracle_w13_weight, weights.oracle_w13_scale,
        weights.oracle_w2_weight, weights.oracle_w2_scale,
    )):
        raise RuntimeError("missing immutable pre-pack E8M0-K32 oracle weights")
    return reference(
        x, weights.oracle_w13_weight, weights.oracle_w13_scale,
        torch.ones(spec.num_experts, device=x.device, dtype=torch.float32),
        weights.oracle_w2_weight, weights.oracle_w2_scale,
        torch.ones(spec.num_experts, device=x.device, dtype=torch.float32),
        ids, scores, spec.num_experts, spec.hidden_size, spec.I_tp,
        activation="silu", swiglu_limit=10.0, w13_layout=weights.w13_layout,
    )


def timing_decision(rows, tokens, candidate="tc_decode"):
    ratios = {}
    for cache in ("cold_l2", "warm_l2"):
        group = {r["variant"]: r for r in rows if r["tokens"] == tokens and r["cache"] == cache}
        control = group["control"]["median_us"]
        candidate_us = group[candidate]["median_us"]
        if not (math.isfinite(control) and math.isfinite(candidate_us) and min(control, candidate_us) > 0):
            raise RuntimeError("invalid component timings")
        ratios[cache] = candidate_us / control
    passed = ratios["cold_l2"] <= (0.90 if tokens == 6 else 1.03) and ratios["warm_l2"] <= 1.03
    return {"tokens": tokens, "candidate_over_control": ratios, "worthwhile_component_gate": passed}


def make_scratch_plan(adapter, spec, x, weights):
    # Frozen B12X scratch contracts compare devices exactly. An unspecified
    # torch.device('cuda') is not equal to the tensor's resolved cuda:0.
    # Derive device and dtype from the actual activation, as serving does.
    return adapter._plan_b12x_moe_fp4_scratch(
        tokens=x.shape[0], weight_E=spec.num_experts, k=spec.hidden_size,
        n=spec.I_tp, topk=spec.top_k, device=x.device, dtype=x.dtype,
        activation='silu', quant_mode='w4a16', source_format=weights.source_format,
        w13_layout=weights.w13_layout, swiglu_limit=10.0)


def make_run_kwargs(x, weights, prepared, unit, ids, scores, out, plan, scratch):
    """Mirror B12xExperts.apply's binding contract with caller-owned buffers."""
    return dict(
        a=x, a1_gscale=unit, w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled, w1_alphas=unit,
        a2_gscale=unit, w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled, w2_alphas=unit,
        output=out, topk_weights=scores, topk_ids=ids,
        apply_router_weight_on_input=False, input_scales_are_reciprocal=True,
        input_scales_static=True, activation='silu', quant_mode='w4a16',
        unit_scale_contract=True, source_format=weights.source_format,
        w13_layout=weights.w13_layout, prepared_w4a16=prepared.w4a16,
        swiglu_limit=10.0, plan=plan, scratch=scratch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--b12x-source", type=Path, default=Path("/opt/b12x"))
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", default="6,1,8,12")
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--budget-seconds", type=int, default=360)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-tile-config",
        help="Optional TILE_K,TILE_N,CTA_THREADS B12X candidate; compared to the default path.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    tokens_list = [int(t) for t in args.tokens.split(",")]
    if tokens_list[0] != 6 or len(set(tokens_list)) != len(tokens_list) or not set(tokens_list) <= {1, 6, 8, 12}:
        parser.error("tokens must start with 6 and be unique members of 1,6,8,12")
    if not 1 <= args.budget_seconds <= 360 or args.layer != 3:
        parser.error("budget must be 1..360 seconds; audited layer is 3")
    if args.candidate_tile_config is not None:
        try:
            tile_values = tuple(int(part.strip()) for part in args.candidate_tile_config.split(","))
        except ValueError as exc:
            parser.error(f"invalid candidate tile config: {exc}")
        if len(tile_values) != 3 or any(value <= 0 for value in tile_values):
            parser.error("candidate tile config must be three positive integers")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier evidence. Later checkpoints update
    # only this invocation's own report and retain every completed result.
    with args.output.open("x") as out:
        out.write("{}\n")
    started = time.monotonic()
    report = {
        "schema_version": 1, "kind": "real_weight_component_not_serving",
        "status": "planned", "tokens": tokens_list, "seed": args.seed,
        "layer": args.layer, "budget_seconds": args.budget_seconds,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [], "decisions": [], "dispatch": [], "progress": [],
        "contract": "TP2 rank0 shard; bf16 activations, MXFP4 E8M0K32 weights, real model router on seeded synthetic activations; routing/compilation/weight load excluded. Independent captured buffers, 4 alternating repeats x20 CUDA events. Each variant requires cosine>=.9975 against the independent FP32 W4A16 oracle; M6 cold gain>=10%, warm and later shapes regress<=3%. Not a serving-TPS prediction.",
        "candidate_tile_config": args.candidate_tile_config,
    }

    def save():
        report["elapsed_s"] = time.monotonic() - started
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        temporary = args.output.with_suffix(args.output.suffix + '.partial')
        temporary.write_text(payload)
        temporary.replace(args.output)

    def gate(stage):
        mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        row = {"stage": stage, "elapsed_s": time.monotonic()-started,
               "available_bytes": mem['MemAvailable']*1024, "swap_free_bytes": mem['SwapFree']*1024}
        report['progress'].append(row)
        save()
        print(json.dumps(row), flush=True)
        if row['available_bytes'] < 8*1024**3 or row['swap_free_bytes'] < 4*1024**3:
            raise RuntimeError("component host memory floor crossed")
        if row['elapsed_s'] > args.budget_seconds:
            raise TimeoutError("component budget exhausted")

    save()
    if args.plan_only:
        print("Planned only; no CUDA imports or requests.")
        return
    def alarm(*_):
        raise TimeoutError("component budget exhausted")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(args.budget_seconds)
    original_dispatch = None
    try:
        report['status'] = 'running'
        gate('before-import')
        if importlib.metadata.version('b12x') != '0.15.3':
            raise RuntimeError('requires audited B12X0.15.3')
        report['source_hashes'] = {name: hashlib.sha256((args.b12x_source/name).read_bytes()).hexdigest() for name in PINNED_SOURCES}
        if report['source_hashes'] != PINNED_SOURCES:
            raise RuntimeError('component source fingerprint mismatch')
        # The adapter installs its selector wrapper at import time only. Seed
        # the candidate once so its control/candidate contexts can toggle the
        # environment dynamically while retaining one real loaded layer.
        if args.candidate_tile_config is not None:
            os.environ[TILE_ENV] = args.candidate_tile_config
        sys.path.insert(0, str(args.b12x_source))
        import torch
        from benchmarks import benchmark_moe as helper
        from b12x.integration import prepare_b12x_fp4_moe_weights
        from b12x.integration import tp_moe
        from b12x.moe.fused.reference import moe_reference_w4a16_fp4_e8m0_k32
        from vllm.model_executor.layers.fused_moe.experts import b12x_mxfp4_moe as adapter
        if torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError('requires SM121')
        profile = helper.MODEL_PROFILES['deepseek-v4-flash']
        spec = helper.build_model_spec(args.model_dir, profile, tp_size_override=2, tp_rank=0)
        if (spec.hidden_size, spec.I_tp, spec.num_experts, spec.top_k) != (4096, 1024, 256, 6):
            raise RuntimeError(f'unexpected0731 TP2 shape: {spec}')
        report['shape'] = asdict(spec)
        report['checkpoint_metadata_sha256'] = {name:hashlib.sha256((args.model_dir/name).read_bytes()).hexdigest() for name in ['config.json','model.safetensors.index.json']}
        report['device'] = {'name':torch.cuda.get_device_name(), 'SMs':torch.cuda.get_device_properties(0).multi_processor_count}
        report['adapter_sha256'] = hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
        gate('before-one-layer-load')
        weights = helper.load_expert_weights(args.model_dir,spec,layer_idx=args.layer,checkpoint_family=profile.checkpoint_family,keep_flashinfer_oracle_copy=True)
        gate('after-one-layer-load')
        unit = torch.ones(spec.num_experts,device='cuda',dtype=torch.float32)
        prepared = prepare_b12x_fp4_moe_weights(source_format=weights.source_format,w13_layout=weights.w13_layout,
            w1_fp4=weights.w13_weight,w1_blockscale=weights.w13_blockscale_swizzled,w1_global_scale=unit,a1_gscale=unit,
            w2_fp4=weights.w2_weight,w2_blockscale=weights.w2_blockscale_swizzled,w2_global_scale=unit,a2_gscale=unit,
            activation='silu',params_dtype=torch.bfloat16,prepare_runtime_alphas=False,prepare_w4a16=True,reuse_input_storage=True)
        report['prepared_layout'] = prepared.w4a16.weight_layout
        report['source_format'] = weights.source_format
        report['w13_layout'] = weights.w13_layout
        if (report['prepared_layout'],weights.source_format,weights.w13_layout) != ('packed','fp4_e8m0_k32','w31'):
            raise RuntimeError('unexpected prepared weight contract')
        gate('after-preparation')
        flush = helper.make_l2_flush_fn(enabled=True)
        original_dispatch = tp_moe._w4a16_preplanned_launches
        observed = []
        def dispatch(*a,**kw):
            launches = original_dispatch(*a,**kw)
            launch = launches[0]
            observed.append({
                'token_count':kw['token_count'], 'weight_layout':kw['weight_layout'],
                'tc_decode_fused_sum':bool(getattr(launch,'tc_decode_fused_sum',False)),
                'fc1_tile_k':getattr(launch,'fc1_tile_k',None),
                'fc1_tile_n':getattr(launch,'fc1_tile_n',None),
                'fc2_tile_k':getattr(launch,'fc2_tile_k',None),
                'fc2_tile_n':getattr(launch,'fc2_tile_n',None),
                'blocks_per_sm':getattr(launch,'blocks_per_sm',None),
            })
            return launches
        tp_moe._w4a16_preplanned_launches = dispatch
        variants = (
            [(False, 'control', ''), (False, 'tile_candidate', args.candidate_tile_config)]
            if args.candidate_tile_config is not None
            else [(False, 'control', ''), (True, 'tc_decode', '')]
        )
        candidate_label = variants[1][1]
        for tokens in tokens_list:
            x,ids,scores = helper.make_profile_routed_inputs(profile,weights,spec,tokens,args.seed,torch.device('cuda'))
            ids = ids.to(torch.int32).contiguous()
            scores = scores.to(torch.float32).contiguous()
            report['dispatch'].append({'tokens':tokens,'route_ids':ids.cpu().tolist(),'variants':[]})
            graphs = []
            for enabled,label,tile_config in variants:
                observed.clear()
                with tc_mode(enabled), tile_mode(tile_config):
                    plan = make_scratch_plan(adapter,spec,x,weights)
                    scratch = torch.empty(adapter._b12x_scratch_nbytes(plan),device='cuda',dtype=torch.uint8)
                    out = torch.empty_like(x)
                    kwargs = make_run_kwargs(x,weights,prepared,unit,ids,scores,out,plan,scratch)
                    def run(kwargs=kwargs):
                        adapter._run_b12x_moe_fp4(**kwargs)
                    run()
                    torch.cuda.synchronize()
                    gate(f'compiled-M{tokens}-{label}')
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    validate_dispatch(observed,tokens,enabled,tile_config or None)
                    report['dispatch'][-1]['variants'].append({'variant':label,'tile_config':tile_config or None,'observed':list(observed),'scratch_bytes':scratch.numel()})
                    graphs.append((label,graph,run,out))
            # Distinct seeded inputs use the same captured addresses; no new
            # allocations occur in replay. Both paths see identical tensors.
            parity = []
            for seed in [args.seed,args.seed+1]:
                new_x,new_ids,new_scores = helper.make_profile_routed_inputs(profile,weights,spec,tokens,seed,torch.device('cuda'))
                x.copy_(new_x); ids.copy_(new_ids); scores.copy_(new_scores)
                expected = real_weight_oracle(moe_reference_w4a16_fp4_e8m0_k32, x, weights, spec, ids, scores)
                for _,graph,_,out in graphs:
                    graph.replay()
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(out).all()):
                        raise RuntimeError('nonfinite component output')
                control_metrics = compare_to_oracle(graphs[0][3], expected)
                candidate_metrics = compare_to_oracle(graphs[1][3], expected)
                parity_gate(control_metrics)
                parity_gate(candidate_metrics)
                parity.append({'seed':seed,'control_oracle':control_metrics,'candidate_oracle':candidate_metrics})
            x0,ids0,scores0 = helper.make_profile_routed_inputs(profile,weights,spec,tokens,args.seed,torch.device('cuda'))
            x.copy_(x0);ids.copy_(ids0);scores.copy_(scores0)
            expected = real_weight_oracle(moe_reference_w4a16_fp4_e8m0_k32, x, weights, spec, ids, scores)
            for cache,l2 in [('cold_l2',flush),('warm_l2',None)]:
                samples = {label:[] for label,*_ in graphs}
                for repeat in range(4):
                    for label,graph,_,out in (graphs if repeat%2==0 else graphs[::-1]):
                        times = helper.bench_events(graph.replay,warmup=3,iters=20,l2_flush=l2)
                        samples[label].append(statistics.median(times)*1000)
                for label,*_ in graphs:
                    row = {'tokens':tokens,'variant':label,'cache':cache,'median_us':statistics.median(samples[label]),'repeat_medians_us':samples[label],'parity':parity}
                    report['rows'].append(row)
                    print(json.dumps(row),flush=True)
                # Repeated replay must not accumulate stale FC2 output or
                # corrupt shared scratch. Inspect the post-timing outputs.
                for _,graph,_,out in graphs:
                    graph.replay()
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(out).all()):
                        raise RuntimeError('nonfinite post-timing output')
                parity_gate(compare_to_oracle(graphs[0][3], expected))
                parity_gate(compare_to_oracle(graphs[1][3], expected))
            decision = timing_decision(report['rows'],tokens,candidate=candidate_label)
            report['decisions'].append(decision)
            gate(f'measured-M{tokens}')
            if not decision['worthwhile_component_gate']:
                report['status'] = 'rejected_component_speed'
                break
            del graphs,graph,run,out,plan,scratch
            gc.collect()
        else:
            report['status'] = 'passed_component_only'
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if original_dispatch is not None:
            tp_moe._w4a16_preplanned_launches = original_dispatch
        signal.alarm(0)
        save()
    print(json.dumps({'status':report['status'],'decisions':report['decisions']}),flush=True)
    if report['status'] != 'passed_component_only':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
