#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

try:
    import torch
except ImportError:  # The Mac-side static/unit environment intentionally lacks Torch.
    torch = None

from benchmarks.nvfp4_packed_inverse_ops import (
    PACKED_WORDS_PER_TILE,
    expected_packed_shape,
    packed_word_destinations,
    pack_modelopt_reference,
    unpack_w4a16_reference,
)
from benchmarks import nvfp4_packed_inverse_ops as inverse_ops


class PackedInverseLayoutTests(unittest.TestCase):
    def test_triton_kernel_has_no_nonconstexpr_tile_global_captures(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(inverse_ops._get_triton_kernel)))
        kernel = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "inverse_scatter"
        )
        captures = sorted(
            {
                node.id
                for node in ast.walk(kernel)
                if isinstance(node, ast.Name) and node.id.startswith("PACKED_")
            }
        )
        self.assertEqual(captures, [])

    def test_expected_shape_matches_equal_weight_bytes(self) -> None:
        shape = expected_packed_shape(num_experts=3, size_n=128, size_k=32)
        self.assertEqual(shape, (3, 2, 256))
        packed_bytes = 3 * 2 * 256 * 4
        modelopt_bytes = 3 * 128 * (32 // 2)
        self.assertEqual(packed_bytes, modelopt_bytes)

    def test_one_tile_destinations_are_bijective(self) -> None:
        destinations = set()
        nibble_count = 0
        for position in range(PACKED_WORDS_PER_TILE):
            for row, byte_col, low_slot, high_slot in packed_word_destinations(
                packed_position=position,
                n_tile=0,
                k_tile=0,
                size_n=64,
            ):
                self.assertNotIn((row, byte_col), destinations)
                destinations.add((row, byte_col))
                self.assertEqual(high_slot, low_slot + 4)
                nibble_count += 2
        self.assertEqual(len(destinations), 64 * (16 // 2))
        self.assertEqual(nibble_count, 64 * 16)

    @unittest.skipIf(torch is None, "Torch is available in the immutable GPU image")
    def test_roundtrip_without_rotation(self) -> None:
        assert torch is not None
        source = torch.arange(2 * 64 * 8, dtype=torch.int64).to(torch.uint8)
        source = source.reshape(2, 64, 8).contiguous()
        packed = pack_modelopt_reference(source)
        restored = unpack_w4a16_reference(packed, size_n=64, size_k=16)
        self.assertTrue(torch.equal(restored, source))

    @unittest.skipIf(torch is None, "Torch is available in the immutable GPU image")
    def test_roundtrip_restores_w13_half_rotation(self) -> None:
        assert torch is not None
        generator = torch.Generator().manual_seed(4104)
        source = torch.randint(
            0,
            256,
            (1, 128, 16),
            dtype=torch.uint8,
            generator=generator,
        )
        packed = pack_modelopt_reference(source, row_rotation=64)
        restored = unpack_w4a16_reference(
            packed,
            size_n=128,
            size_k=32,
            row_rotation=64,
        )
        self.assertTrue(torch.equal(restored, source))

    @unittest.skipIf(torch is None, "Torch is available in the immutable GPU image")
    def test_rotation_changes_packed_bytes_but_not_roundtrip(self) -> None:
        assert torch is not None
        source = torch.zeros((1, 128, 8), dtype=torch.uint8)
        source[:, :64].fill_(0x12)
        source[:, 64:].fill_(0xAB)
        plain = pack_modelopt_reference(source)
        rotated = pack_modelopt_reference(source, row_rotation=64)
        self.assertFalse(torch.equal(plain, rotated))
        restored = unpack_w4a16_reference(
            rotated, size_n=128, size_k=16, row_rotation=64
        )
        self.assertTrue(torch.equal(restored, source))

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            expected_packed_shape(num_experts=1, size_n=63, size_k=16)
        if torch is not None:
            packed = torch.zeros((1, 1, 128), dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "shape"):
                unpack_w4a16_reference(packed, size_n=128, size_k=16)


if __name__ == "__main__":
    unittest.main()
