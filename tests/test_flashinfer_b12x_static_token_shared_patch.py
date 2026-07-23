from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from scripts import patch_flashinfer_b12x_shared_input as shared_input_patch
from scripts import patch_flashinfer_b12x_static_token_shared as static_patch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(os.environ.get("DSPARK_REFERENCE_ROOT", ROOT))
UPSTREAM = (
    REFERENCE_ROOT
    / ".build/flashinfer-upstream/flashinfer/fused_moe/cute_dsl/blackwell_sm12x"
)


class StaticTokenSharedPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shared_dispatch = "\n".join(
            (
                static_patch.DISPATCH_ENV_ANCHOR,
                "def _get_static_kernel(",
                static_patch.STATIC_FACTORY_ARGS_ANCHOR,
                "    cache_key = (",
                static_patch.STATIC_FACTORY_CACHE_ANCHOR,
                "    kernel = MoEStaticKernel(",
                static_patch.STATIC_FACTORY_KERNEL_ANCHOR,
                static_patch.DISPATCH_FLAGS_ANCHOR,
                "        compiled, mac = _get_static_kernel(",
                static_patch.DISPATCH_STATIC_CALL_ANCHOR,
                "def _get_dynamic_kernel():\n    pass\n",
            )
        )
        cls.raw_static = "\n".join(
            (
                "class MoEStaticKernel:",
                "    def __init__(",
                static_patch.STATIC_INIT_ARGS_ANCHOR,
                static_patch.STATIC_INIT_ATTRS_ANCHOR,
                "    def __call__(self):",
                "        pair_idx = Int32(bidz)",
                static_patch.STATIC_QUANT_ANCHOR,
                "            pair_idx += Int32(gdim_z)",
                static_patch.STATIC_PACK_BARRIER_ANCHOR,
            )
        )

    def test_pinned_input_hashes_match(self) -> None:
        if not (UPSTREAM / "moe_dispatch.py").is_file():
            self.skipTest("optional pinned FlashInfer reference checkout absent")
        raw_dispatch = (UPSTREAM / "moe_dispatch.py").read_text()
        shared_dispatch = shared_input_patch.patch_dispatch_source(raw_dispatch)
        raw_static = (UPSTREAM / "moe_static_kernel.py").read_text()
        self.assertEqual(
            hashlib.sha256(shared_dispatch.encode()).hexdigest(),
            static_patch.PINNED_SHARED_INPUT_DISPATCH_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(raw_static.encode()).hexdigest(),
            static_patch.PINNED_STATIC_KERNEL_SHA256,
        )

    def test_dispatch_opt_in_is_static_only_and_cache_keyed(self) -> None:
        patched = static_patch.patch_dispatch_source(self.shared_dispatch)
        self.assertIn(
            '_STATIC_TOKEN_SHARED_INPUT_ENV = '
            '"FLASHINFER_B12X_STATIC_TOKEN_SHARED_INPUT"',
            patched,
        )
        self.assertIn("and not use_micro", patched)
        self.assertIn("and input_gs_is_shared", patched)
        self.assertIn("per_token_shared_input=per_token_shared_input", patched)
        self.assertIn("per_token_shared_input,\n        activation,", patched)
        # The optimization must not change the dynamic/prefill factory.
        dynamic = patched[patched.index("def _get_dynamic_kernel(") :]
        self.assertNotIn("per_token_shared_input", dynamic)

    def test_static_kernel_quantizes_first_route_then_fans_out(self) -> None:
        patched = static_patch.patch_static_kernel_source(self.raw_static)
        self.assertIn("self.per_token_shared_input = per_token_shared_input", patched)
        self.assertIn("pair_idx % num_topk == Int32(0)", patched)
        self.assertIn("source_expert_id = topk_ids[token_idx * num_topk]", patched)
        self.assertIn(
            "source_local_expert_id = global_to_local_expert[", patched
        )
        self.assertIn("_ld_global_u64(", patched)
        self.assertEqual(
            patched.count("if cutlass.const_expr(self.per_token_shared_input):"),
            2,
        )
        # Route metadata becomes globally visible before any source/destination
        # lookup, and copied packs become visible before compute.
        first_barrier = patched.index(
            "self._resident_grid_barrier(",
            patched.index("pair_idx = Int32(bidz)"),
        )
        fanout = patched.index(
            "if cutlass.const_expr(self.per_token_shared_input):", first_barrier
        )
        second_barrier = patched.index("self._resident_grid_barrier(", fanout)
        compute = patched.index("gA = cute.local_tile(", fanout)
        self.assertLess(first_barrier, fanout)
        self.assertLess(fanout, second_barrier)
        self.assertLess(second_barrier, compute)

    def test_patcher_rejects_unpinned_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dispatch = Path(directory) / "dispatch.py"
            static = Path(directory) / "static.py"
            dispatch.write_text(self.shared_dispatch + "\n# drift\n")
            static.write_text(self.raw_static)
            with self.assertRaisesRegex(RuntimeError, "dispatcher SHA-256"):
                static_patch._patch_file(
                    dispatch,
                    static_patch.PINNED_SHARED_INPUT_DISPATCH_SHA256,
                    static_patch.patch_dispatch_source,
                    "shared-input dispatcher",
                )

    def test_image_overlay_bakes_patch_and_benchmark(self) -> None:
        dockerfile = (
            ROOT / "docker/Dockerfile.nvfp4-b12x-static-token-shared-overlay"
        ).read_text()
        dockerignore = (
            ROOT
            / "docker/Dockerfile.nvfp4-b12x-static-token-shared-overlay.dockerignore"
        ).read_text()
        self.assertIn("patch_flashinfer_b12x_static_token_shared.py", dockerfile)
        self.assertIn("benchmark_nvfp4_prepared_b12x_sm121.py", dockerfile)
        self.assertIn("nvfp4-b12x-static-token-shared", dockerfile)
        self.assertIn(
            "!scripts/patch_flashinfer_b12x_static_token_shared.py", dockerignore
        )
        self.assertIn(
            "!benchmarks/benchmark_nvfp4_prepared_b12x_sm121.py", dockerignore
        )

    def test_benchmark_exposes_fail_closed_opt_in(self) -> None:
        benchmark = (
            ROOT / "benchmarks/benchmark_nvfp4_prepared_b12x_sm121.py"
        ).read_text()
        self.assertIn("--b12x-static-token-shared", benchmark)
        self.assertIn("b12x_wrapper._dspark_unit_input_scales = True", benchmark)
        self.assertIn(
            "static token-sharing must be selected before importing", benchmark
        )


if __name__ == "__main__":
    unittest.main()
