# SPDX-License-Identifier: MIT
"""Exercise the actual overlaid cache-hit method without importing GPU vLLM.

Only the method's dependency boundary is faked. These tests establish prefix
window arithmetic and coordinator/statistics interactions, not GPU numerics.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "overlay/vllm/v1/core/kv_cache_manager.py")


def load_cache_hit_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_computed_blocks")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["get_computed_blocks"]


class Coordinator:
    def __init__(self, available, block_size=256):
        self.available = available
        self.block_size = block_size
        self.calls = []

    def find_longest_cache_hit(self, hashes, limit):
        self.calls.append((hashes, limit))
        hit = min(limit, self.available) // self.block_size * self.block_size
        return [hit // self.block_size], hit


class CacheWindowTests(unittest.TestCase):
    hit_method = staticmethod(load_cache_hit_method())

    def manager(self, window, available=100000, caching=True, stats=False):
        records = []
        return SimpleNamespace(
            enable_caching=caching, dspark_window_size=window,
            empty_kv_cache_blocks=(), coordinator=Coordinator(available),
            log_stats=stats, prefix_cache_stats=SimpleNamespace(record=lambda **kw: records.append(kw)),
            create_kv_cache_blocks=tuple, records=records,
        )

    def request(self, tokens, skip=False):
        return SimpleNamespace(num_tokens=tokens, skip_reading_prefix_cache=skip,
                               block_hashes=("prefix", "suffix"), num_preemptions=0)

    def test_window_and_anchor_survive_aligned_repeated_prefix_hits(self):
        for tokens in (1, 127, 128, 129, 255, 256, 257, 383, 384, 385, 512, 8192, 65536):
            with self.subTest(tokens=tokens):
                manager = self.manager(128)
                _, hit = self.hit_method(manager, self.request(tokens))
                self.assertEqual(hit % 256, 0)
                self.assertGreaterEqual(tokens - hit, min(tokens, 129))
                self.assertEqual(manager.coordinator.calls[0][1], max(tokens - 129, 0))

    def test_no_dspark_preserves_original_last_token_rule(self):
        for window in (None, 0):
            manager = self.manager(window)
            _, hit = self.hit_method(manager, self.request(513))
            self.assertEqual(hit, 512)
            self.assertEqual(manager.coordinator.calls[0][1], 512)

    def test_divergent_suffix_limits_hits_even_with_long_prompt(self):
        manager = self.manager(128, available=512)
        _, hit = self.hit_method(manager, self.request(8192))
        self.assertEqual(hit, 512)

    def test_skip_read_or_disabled_cache_never_queries_coordinator(self):
        for caching, skip in ((False, False), (True, True)):
            manager = self.manager(128, caching=caching)
            self.assertEqual(self.hit_method(manager, self.request(8192, skip)), ((), 0))
            self.assertEqual(manager.coordinator.calls, [])

    def test_stats_report_actual_aligned_hit_not_requested_cap(self):
        manager = self.manager(128, stats=True)
        request = self.request(8192)
        request.num_preemptions = 1
        _, hit = self.hit_method(manager, request)
        self.assertEqual(manager.records, [{"num_tokens": 8192, "num_hits": hit, "preempted": True}])
        self.assertEqual(hit, 7936)


if __name__ == "__main__":
    unittest.main()
