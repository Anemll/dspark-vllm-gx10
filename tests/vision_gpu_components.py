# SPDX-License-Identifier: MIT
"""Bounded real-CUDA canaries: routing, dense image slots, graph replay."""
from types import SimpleNamespace
import json
import unittest

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
from vllm.models.deepseek_v4.nvidia.vision_router import DeepseekV4VisionRouter
from vllm.models.deepseek_v4.nvidia.vision_attention import ImageVisibilityBuffers


class GPUComponents(unittest.TestCase):
    def test_router_matches_unbiased_reference_and_graph_replay(self):
        torch.manual_seed(4104)
        tokens = torch.tensor([0, 5, 129256, 129257, 129258, 129259,
                               129260, 129261, 129262, 129279], device="cuda")
        logits = torch.randn(10, 256, device="cuda", dtype=torch.float32)
        hidden = torch.empty(10, 32, device="cuda", dtype=torch.bfloat16)
        correction = torch.randn(256, device="cuda")
        bias_vl = torch.randn(256, device="cuda")
        table = torch.randint(0, 256, (129280, 6), device="cuda", dtype=torch.int32)
        for hashed in (False, True):
            original = FusedTopKBiasRouter(6, 256, correction,
                routed_scaling_factor=1.5, scoring_func="sqrtsoftplus",
                hash_indices_table=table if hashed else None)
            router = DeepseekV4VisionRouter(original, correction, bias_vl)
            weights, ids = router.select_experts(hidden, logits, input_ids=tokens)
            scores = torch.sqrt(F.softplus(logits.float()))
            image = (tokens >= 129257) & (tokens < 129262)
            choices = scores + torch.where(image[:, None], bias_vl, correction)
            expected_ids = choices.topk(6, dim=-1).indices
            if hashed:
                expected_ids = torch.where(image[:, None], expected_ids, table[tokens].long())
            expected_weights = scores.gather(1, expected_ids)
            expected_weights *= 1.5 / expected_weights.sum(1, keepdim=True)
            torch.testing.assert_close(ids.long(), expected_ids)
            torch.testing.assert_close(weights, expected_weights, atol=2e-6, rtol=2e-5)
            # Capture actual routing, mutate inputs, and compare replay to eager.
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    router.select_experts(hidden, logits, input_ids=tokens)
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_weights, graph_ids = router.select_experts(hidden, logits, input_ids=tokens)
            logits.add_(0.1)
            graph.replay()
            eager_weights, eager_ids = router.select_experts(hidden, logits, input_ids=tokens)
            torch.testing.assert_close(graph_ids, eager_ids)
            torch.testing.assert_close(graph_weights, eager_weights)

    def test_image_attention_slots_and_no_image_fast_path(self):
        # Decode first; image and text prefills follow. Include invalid padding.
        qlens = [1, 400, 8]
        seqs = [500, 400, 8]
        qsl_cpu = torch.tensor([0, 1, 401, 409], dtype=torch.int32)
        seq_cpu = torch.tensor(seqs, dtype=torch.int32)
        block_table = torch.arange(24, dtype=torch.int32, device="cuda").view(3, 8)
        token_to_req = torch.repeat_interleave(torch.arange(3, device="cuda", dtype=torch.int32),
                                               torch.tensor(qlens, device="cuda"))
        valid = torch.ones(409, device="cuda", dtype=torch.bool)
        valid[-1] = False
        common = SimpleNamespace(num_reqs=3, mm_req_doc_ranges={1: [(3, 386)]},
            seq_lens_cpu_upper_bound=seq_cpu, query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_cpu.cuda(), query_start_loc=qsl_cpu.cuda(),
            block_table_tensor=block_table)
        buffers = ImageVisibilityBuffers(512, 3, 128, 384, "cuda")
        indices, lengths = buffers.build(common, 1, 1, 408, token_to_req, valid, 64)
        expected = []
        for req, qlen in ((1, 400), (2, 8)):
            for pos in range(qlen):
                left = min(pos-3, 383) if req == 1 and 3 <= pos <= 386 else 0
                right = min(386-pos, 384) if req == 1 and 3 <= pos <= 386 else 0
                begin = max(pos-127-max(left-127, 0), 0)
                slots = [req*8*64 + p for p in range(begin, pos+right+1)]
                expected.append(slots)
        expected[-1] = []
        self.assertEqual(lengths.cpu().tolist(), list(map(len, expected)))
        self.assertEqual(indices[:, 0].cpu().tolist(), [row + [-1]*(512-len(row)) for row in expected])
        common.mm_req_doc_ranges = {}
        self.assertIsNone(buffers.build(common, 1, 1, 408, token_to_req, valid, 64))
        common.mm_req_doc_ranges = {1: [(3, 420)]}
        with self.assertRaisesRegex(ValueError, "split"):
            buffers.build(common, 1, 1, 408, token_to_req, valid, 64)


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"cuda_free_before": free, "cuda_total": total}), flush=True)
    if free < 256 * 1024**2:
        raise SystemExit("Insufficient CUDA headroom for bounded canary")
    torch.cuda.set_per_process_memory_fraction(0.01)
    torch.cuda.reset_peak_memory_stats()
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GPUComponents)
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    print(json.dumps({"cuda_peak_allocated": peak}), flush=True)
    if peak > 32 * 1024**2:
        raise SystemExit("Canary exceeded its 32 MiB allocation budget")
    raise SystemExit(not result.wasSuccessful())
