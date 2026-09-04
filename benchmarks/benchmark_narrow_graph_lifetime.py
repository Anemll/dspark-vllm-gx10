# SPDX-License-Identifier: MIT
"""GPU structural/lifetime canary for actual overlay graph-boundary methods.

Preparation kernels are synthetic tensor operations, not DeepSeek inference.
Timing measures launch overhead only and must not be reported as model speed.
Uses real vLLM graph capture, weak tensor references and multi-stream helpers.
Run only on an idle GPU, with an outer five-minute process deadline.
"""

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import MethodType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    import torch
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture, eager_break_during_capture,
    )
    from vllm.utils.multi_stream_utils import (
        execute_in_parallel, maybe_execute_in_parallel,
    )

    torch.manual_seed(4104)
    source = args.source.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "DeepseekV4Attention")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_prepare_and_attn", "_sparse_indexer_and_attn")]
    scope = dict(torch=torch, eager_break_during_capture=eager_break_during_capture,
                 execute_in_parallel=execute_in_parallel,
                 maybe_execute_in_parallel=maybe_execute_in_parallel,
                 get_forward_context=lambda: SimpleNamespace(attn_metadata={}))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                      names=[ast.alias(name="annotations")]), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(args.source), "exec"), scope)

    class FakeIndexer:
        def __init__(self, owner):
            self.owner = owner
            self.events = [torch.cuda.Event(), torch.cuda.Event()]
            self.cache = torch.zeros_like(owner.scored)

        def __call__(self, hidden, qr, score, weights, pos, rope, *, prepare_only=False):
            def prepare_q():
                q = qr.sin().cos()
                scale = q.abs().mean(dim=1, keepdim=True) + 1
                return q, scale, weights.sigmoid()

            (q, scale, w), _ = maybe_execute_in_parallel(
                prepare_q,
                lambda: self.cache.copy_(score.cos().mean(dim=1)),
                *self.events, self.owner.aux_stream_list[2],
            )
            if prepare_only:
                return (q, scale), w
            self.indexer_op(hidden, (q, scale), None, w)

        def indexer_op(self, hidden, quant, k, weights):
            # Fresh eager metadata must be observed on every graph replay.
            q, scale = quant
            self.owner.scored.copy_((q * scale).mean(dim=1) + weights.mean(dim=1)
                                   + self.cache + self.owner.metadata_offset)

    def owner_for(n, narrow, branch):
        owner = SimpleNamespace(
            _narrow_attention_graph=narrow,
            n_local_heads=32, head_dim=512, rotary_emb=None,
            indexer_rotary_emb=None, metadata_offset=0.0,
            aux_stream_list=[torch.cuda.Stream() for _ in range(3)],
            ln_events=[torch.cuda.Event() for _ in range(3)],
            compressed=torch.zeros(n, device="cuda"),
            scored=torch.zeros(n, device="cuda"),
        )
        projection = torch.randn(128, 32 * 512, device="cuda") / 128
        owner.wq_b = lambda qr: qr @ projection
        owner._fused_qnorm_rope_kv_insert = (
            lambda q, kv, pos, metadata: q + kv[:, None, :] + pos[:, None, None]
        )
        owner.compressor = (lambda score, pos, rope: owner.compressed.copy_(
            (score.sin().cos()).mean(dim=1) + pos
        )) if branch != "swa" else None
        owner.indexer = FakeIndexer(owner) if branch == "c4" else None
        owner.forward_mqa = lambda q, kv, pos, out: out.copy_(
            q + owner.compressed[:, None, None] + owner.scored[:, None, None]
        )
        owner._sparse_indexer_and_attn = MethodType(scope["_sparse_indexer_and_attn"], owner)
        method = MethodType(scope["_prepare_and_attn"], owner)
        owner.run = method if narrow else eager_break_during_capture(method)
        return owner

    started = time.monotonic()
    rows = []
    for narrow in (False, True):
        for branch in ("swa", "c128", "c4"):
            pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()
            entries = []
            for n in (72, 24, 8, 2, 1):
                owner = owner_for(n, narrow, branch)
                inputs = [torch.randn(n, size, device="cuda") for size in
                          (128, 128, 512, 128, 128, 128)]
                inputs += [torch.arange(n, device="cuda", dtype=torch.float32),
                           torch.empty(n, 32, 512, device="cuda")]
                final = torch.empty_like(inputs[-1])
                torch.cuda.synchronize()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        owner.run(*inputs)
                    torch.cuda.synchronize()
                    capture = BreakableCUDAGraphCapture(pool=pool)
                    with capture:
                        # Nonempty before/after segments, like the real layer.
                        qr = inputs[1] + 0
                        owner.run(inputs[0], qr, *inputs[2:])
                        final.copy_(inputs[-1] * 0.5)
                torch.cuda.synchronize()
                assert capture.num_eager_breaks == 1
                entries.append((owner, inputs, final, capture))

            # Shared-pool captures in different orders, changing inputs and
            # metadata, with allocator pressure between replays.
            timings = []
            for repeat in range(4):
                for owner, inputs, final, capture in entries[::(-1 if repeat % 2 else 1)]:
                    for tensor in inputs[:-1]:
                        tensor.add_(0.125)
                    owner.metadata_offset = float(repeat + 3)
                    scope["_prepare_and_attn"](owner, *inputs)
                    expected = (inputs[-1] * 0.5).clone()
                    torch.cuda.synchronize()
                    junk = [torch.full_like(final, -999) for _ in range(4)]
                    del junk
                    gc.collect()
                    with torch.cuda.stream(stream):
                        capture.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(final, expected, rtol=1e-5, atol=1e-5)
            owner, inputs, final, capture = entries[-1]
            for _ in range(5):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(30):
                        capture.replay()
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - begin) * 1e6 / 30)
            rows.append(dict(narrow=narrow, branch=branch, checked_replays=20,
                             replay_us=statistics.median(timings), trials_us=timings))
            print(rows[-1], flush=True)
            del entries, owner, inputs, final, capture
            gc.collect()
            torch.cuda.empty_cache()
    report = dict(kind="synthetic_graph_lifetime_not_model_speed", results=rows,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  device=torch.cuda.get_device_name(),
                  elapsed_seconds=time.monotonic() - started)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
