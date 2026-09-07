# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 by FlashInfer team.
# Bounded dequantization reference adapted from pinned FlashInfer
# tests/moe/test_trtllm_cutlass_fused_moe.py (Apache-2.0).
"""Small real-GPU NVFP4/clamp canary, not a checkpoint or speed benchmark."""
import json
import signal
import time


def main():
    import torch
    from torch.nn import functional as F
    from flashinfer import fp4_quantize
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    started = time.monotonic()
    def expired(*_):
        raise TimeoutError("180-second component deadline exceeded")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    report = {"kind": "synthetic_gpu_component_not_model_inference", "rows": []}
    try:
        assert torch.cuda.get_device_capability() == (12, 1), "SM121 canary only"
        torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(5205)
        dtype, device = torch.bfloat16, "cuda"
        one = torch.tensor(1.0, device=device)

        def dequant(packed, scales, global_scale):
            m, pk = packed.shape
            k = pk * 2
            raw = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(m, k)
            table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=device)
            values = table[(raw & 7).long()] * torch.where((raw & 8) != 0, -1., 1.)
            mt, kt = (m + 127) // 128, (k + 63) // 64
            linear = scales.view(torch.float8_e4m3fn).reshape(1, mt, kt, 32, 4, 4)
            linear = linear.permute(0, 1, 4, 3, 2, 5).reshape(mt * 128, kt * 4)[:m, :k // 16]
            return (values.reshape(m, k // 16, 16) *
                    (linear.float() / global_scale).unsqueeze(-1)).reshape(m, k).to(dtype)

        # Small capability gate, then target per-TP matrix dimensions. Expert
        # count is deliberately reduced: this does not validate full TP2 MoE.
        for m, k, n, e, topk in ((4, 128, 128, 8, 2), (6, 4096, 1024, 8, 6)):
            shape_start = time.monotonic()
            print(json.dumps({"phase": "prepare", "m": m, "k": k, "n": n}), flush=True)
            def weights(rows, cols):
                quantized, scales, globals_, restored = [], [], [], []
                for _ in range(e):
                    w = torch.randn(rows, cols, dtype=dtype, device=device) / 5
                    gs = 2688.0 / w.float().abs().amax()
                    q, sf = fp4_quantize(w, gs)
                    quantized.append(q)
                    scales.append(sf)
                    globals_.append(gs)
                    restored.append(dequant(q, sf, gs))
                return torch.stack(quantized), torch.stack(scales), torch.stack(globals_), restored
            w1, sf1, gs1, ref1 = weights(n * 2, k)
            w2, sf2, gs2, ref2 = weights(k, n)
            x = torch.randn(m, k, dtype=dtype, device=device) * 4
            xq, xs = fp4_quantize(x, one)
            xr = dequant(xq, xs, one)
            ids = torch.stack([torch.randperm(e, device=device)[:topk] for _ in range(m)]).int()
            routing = torch.full((m, topk), 1 / topk, dtype=torch.float32, device=device)
            scales = [one, sf1.view(torch.int32), 1 / gs1, one, sf2.view(torch.int32), 1 / gs2]
            output = torch.empty_like(x)
            print(json.dumps({"phase": "kernel", "m": m, "k": k, "n": n}), flush=True)
            cutlass_fused_moe(x, ids, routing, w1.contiguous().view(torch.long),
                             w2.contiguous().view(torch.long), dtype,
                             quant_scales=scales, output=output,
                             activation_type=ActivationType.Swiglu,
                             swiglu_limit=torch.full((e,), 10., device=device))
            torch.cuda.synchronize()
            assert torch.isfinite(output).all(), "nonfinite kernel output"
            reference = torch.zeros_like(x)
            clipped = 0
            for expert in range(e):
                token, slot = torch.where(ids == expert)
                if token.numel() == 0:
                    continue
                projected = xr[token] @ ref1[expert].t()
                up, gate = projected.chunk(2, dim=-1)
                clipped += int(((gate > 10) | (up.abs() > 10)).sum())
                # DeepSeek clamps gate before SILU (not Step-3 post-SILU).
                intermediate = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                iq, isc = fp4_quantize(intermediate, one)
                restored = dequant(iq, isc, one)
                contribution = (restored @ ref2[expert].t()) * routing[token, slot, None].to(dtype)
                reference.index_add_(0, token, contribution)
            assert clipped > 0, "test did not exercise clamp"
            error = (output.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
            cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
            row = {"m": m, "k": k, "n": n, "experts": e, "topk": topk,
                   "clamp": 10, "clipped_elements": clipped,
                   "relative_l2": float(error), "cosine": float(cosine),
                   "elapsed_with_setup_s": time.monotonic() - shape_start,
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            assert float(error) < .08 and float(cosine) > .995, "quantized reference mismatch"
            del w1, sf1, gs1, ref1, w2, sf2, gs2, ref2, output, reference
            torch.cuda.empty_cache()
        report["status"] = "passed_component_only"
    except Exception as exc:
        report.update(status="failed_component", error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report["elapsed_s"] = time.monotonic() - started
        print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
