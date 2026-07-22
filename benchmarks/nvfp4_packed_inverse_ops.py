#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact W4A16-packed -> ModelOpt NVFP4 inverse layout operations.

B12X's packed W4A16 layout stores each logical 16x64 FP4 tile as 128
``int32`` words.  Every word contains eight nibbles selected for the legacy
tensor-core register fragment.  The SM121 W4A4 kernels instead consume the
checkpoint-native ``[E, N, K/2]`` byte layout.  This module provides a small
benchmark-only Triton adapter that reverses the word/nibble permutation for a
bounded set of experts.  It intentionally does not modify serving dispatch.

The inverse is lossless for weights.  Raw E4M3 K/16 block scales remain
separate because the W4A16 scale transform is not invertible.
"""

from __future__ import annotations

from typing import Any


PACKED_TILE_K = 16
PACKED_TILE_N = 64
PACKED_WORDS_PER_TILE = 128


def expected_packed_shape(
    *, num_experts: int, size_n: int, size_k: int
) -> tuple[int, int, int]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if size_n <= 0 or size_n % PACKED_TILE_N:
        raise ValueError("size_n must be a positive multiple of 64")
    if size_k <= 0 or size_k % PACKED_TILE_K:
        raise ValueError("size_k must be a positive multiple of 16")
    return (
        num_experts,
        size_k // PACKED_TILE_K,
        (size_n // PACKED_TILE_N) * PACKED_WORDS_PER_TILE,
    )


def packed_word_destinations(
    *, packed_position: int, n_tile: int, k_tile: int, size_n: int,
    row_rotation: int = 0,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return four ``(row, byte_col, low_slot, high_slot)`` destinations.

    The mapping is the exact inverse of B12X
    ``w4a16.prepare._repack_4bit_no_perm``.  ``row_rotation`` restores the
    source W13 order when the W4A16 pack folded a half-row rotation into FC1.
    """

    if not 0 <= packed_position < PACKED_WORDS_PER_TILE:
        raise ValueError("packed_position must be in [0, 128)")
    if n_tile < 0 or k_tile < 0:
        raise ValueError("tile indices must be non-negative")
    if size_n <= 0 or size_n % PACKED_TILE_N:
        raise ValueError("size_n must be a positive multiple of 64")
    if not 0 <= row_rotation < size_n:
        raise ValueError("row_rotation must be in [0, size_n)")

    warp_id = packed_position % 4
    thread_id = packed_position // 4
    tensor_col = thread_id // 4
    tensor_row = (thread_id % 4) * 2
    result = []
    for pair in range(4):
        k_high = (pair & 1) * 8
        n_high = (pair >> 1) * 8
        packed_row = n_tile * PACKED_TILE_N + warp_id * 16 + tensor_col + n_high
        source_row = (packed_row + row_rotation) % size_n
        source_k = k_tile * PACKED_TILE_K + tensor_row + k_high
        result.append((source_row, source_k // 2, pair, pair + 4))
    return tuple(result)


def pack_modelopt_reference(
    modelopt: Any,
    *,
    row_rotation: int = 0,
) -> Any:
    """Slow CPU reference packer used only by unit tests."""

    import torch

    if modelopt.device.type != "cpu" or modelopt.dtype != torch.uint8:
        raise TypeError("reference packer requires a CPU uint8 tensor")
    if modelopt.ndim != 3:
        raise ValueError("modelopt must have shape [E, N, K/2]")
    experts, size_n, packed_k = map(int, modelopt.shape)
    size_k = packed_k * 2
    packed_shape = expected_packed_shape(
        num_experts=experts, size_n=size_n, size_k=size_k
    )
    packed = torch.zeros(packed_shape, dtype=torch.int32)
    n_tiles = size_n // PACKED_TILE_N
    for expert in range(experts):
        for k_tile in range(size_k // PACKED_TILE_K):
            for n_tile in range(n_tiles):
                for position in range(PACKED_WORDS_PER_TILE):
                    word = 0
                    for row, byte_col, low_slot, high_slot in packed_word_destinations(
                        packed_position=position,
                        n_tile=n_tile,
                        k_tile=k_tile,
                        size_n=size_n,
                        row_rotation=row_rotation,
                    ):
                        value = int(modelopt[expert, row, byte_col])
                        word |= (value & 0xF) << (4 * low_slot)
                        word |= ((value >> 4) & 0xF) << (4 * high_slot)
                    # Preserve the exact 32 bits when assigning through signed int32.
                    if word >= 1 << 31:
                        word -= 1 << 32
                    packed[
                        expert,
                        k_tile,
                        n_tile * PACKED_WORDS_PER_TILE + position,
                    ] = word
    return packed


def unpack_w4a16_reference(
    packed: Any,
    *,
    size_n: int,
    size_k: int,
    row_rotation: int = 0,
) -> Any:
    """Slow CPU inverse used to pin the Triton adapter's bit contract."""

    import torch

    if packed.device.type != "cpu" or packed.dtype != torch.int32:
        raise TypeError("reference inverse requires a CPU int32 tensor")
    if tuple(packed.shape) != expected_packed_shape(
        num_experts=int(packed.shape[0]), size_n=size_n, size_k=size_k
    ):
        raise ValueError("packed tensor shape does not match E/N/K")
    output = torch.zeros(
        (int(packed.shape[0]), size_n, size_k // 2), dtype=torch.uint8
    )
    n_tiles = size_n // PACKED_TILE_N
    for expert in range(int(packed.shape[0])):
        for k_tile in range(size_k // PACKED_TILE_K):
            for n_tile in range(n_tiles):
                for position in range(PACKED_WORDS_PER_TILE):
                    word = int(
                        packed[
                            expert,
                            k_tile,
                            n_tile * PACKED_WORDS_PER_TILE + position,
                        ]
                    ) & 0xFFFFFFFF
                    for row, byte_col, low_slot, high_slot in packed_word_destinations(
                        packed_position=position,
                        n_tile=n_tile,
                        k_tile=k_tile,
                        size_n=size_n,
                        row_rotation=row_rotation,
                    ):
                        low = (word >> (4 * low_slot)) & 0xF
                        high = (word >> (4 * high_slot)) & 0xF
                        output[expert, row, byte_col] = low | (high << 4)
    return output


_TRITON_KERNEL: Any | None = None
_TRITON_MODULE: Any | None = None


def _get_triton_kernel() -> tuple[Any, Any]:
    global _TRITON_KERNEL, _TRITON_MODULE
    if _TRITON_KERNEL is not None and _TRITON_MODULE is not None:
        return _TRITON_KERNEL, _TRITON_MODULE

    import triton
    import triton.language as tl

    @triton.jit
    def inverse_scatter(
        packed_ptr,
        modelopt_ptr,
        total_words: tl.constexpr,
        size_n: tl.constexpr,
        size_k: tl.constexpr,
        n_tiles: tl.constexpr,
        row_rotation: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        word_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = word_index < total_words
        # Triton 3.6 rejects ordinary Python module globals inside @jit even
        # when their values are immutable.  Keep the packed-tile ABI literal
        # at the kernel boundary; the public constants still pin the CPU-side
        # shape/reference contract above.
        position = word_index % 128
        tile_linear = word_index // 128
        n_tile = tile_linear % n_tiles
        tile_linear = tile_linear // n_tiles
        k_tile = tile_linear % (size_k // 16)
        expert = tile_linear // (size_k // 16)

        word = tl.load(packed_ptr + word_index, mask=valid, other=0).to(tl.uint32)
        warp_id = position % 4
        thread_id = position // 4
        tensor_col = thread_id // 4
        tensor_row = (thread_id % 4) * 2

        for pair in tl.static_range(0, 4):
            n_high = (pair // 2) * 8
            k_high = (pair % 2) * 8
            packed_row = n_tile * 64 + warp_id * 16 + tensor_col + n_high
            source_row = (packed_row + row_rotation) % size_n
            source_k = k_tile * 16 + tensor_row + k_high
            low = (word >> (pair * 4)) & 0xF
            high = (word >> ((pair + 4) * 4)) & 0xF
            value = low | (high << 4)
            destination = (
                (expert * size_n + source_row) * (size_k // 2) + source_k // 2
            )
            tl.store(modelopt_ptr + destination, value.to(tl.uint8), mask=valid)

    _TRITON_KERNEL = inverse_scatter
    _TRITON_MODULE = triton
    return inverse_scatter, triton


def unpack_w4a16_packed(
    *,
    packed: Any,
    output: Any,
    row_rotation: int = 0,
    block: int = 256,
) -> None:
    """Launch the lossless GPU inverse-scatter into caller-owned storage."""

    import torch

    if packed.device.type != "cuda" or output.device.type != "cuda":
        raise ValueError("packed and output must be CUDA tensors")
    if packed.dtype != torch.int32 or output.dtype != torch.uint8:
        raise TypeError("packed must be int32 and output must be uint8")
    if packed.ndim != 3 or output.ndim != 3:
        raise ValueError("expected packed [E,K/16,N/64*128] and output [E,N,K/2]")
    experts, size_n, packed_k = map(int, output.shape)
    size_k = packed_k * 2
    if tuple(packed.shape) != expected_packed_shape(
        num_experts=experts, size_n=size_n, size_k=size_k
    ):
        raise ValueError("packed/output shape mismatch")
    if not packed.is_contiguous() or not output.is_contiguous():
        raise ValueError("packed and output must be contiguous")
    if not 0 <= row_rotation < size_n:
        raise ValueError("row_rotation must be in [0, size_n)")
    if block <= 0 or block & (block - 1):
        raise ValueError("block must be a positive power of two")

    kernel, triton = _get_triton_kernel()
    total_words = int(packed.numel())
    kernel[(triton.cdiv(total_words, block),)](
        packed,
        output,
        total_words=total_words,
        size_n=size_n,
        size_k=size_k,
        n_tiles=size_n // PACKED_TILE_N,
        row_rotation=row_rotation,
        BLOCK=block,
        num_warps=4,
    )


__all__ = [
    "PACKED_TILE_K",
    "PACKED_TILE_N",
    "PACKED_WORDS_PER_TILE",
    "expected_packed_shape",
    "packed_word_destinations",
    "pack_modelopt_reference",
    "unpack_w4a16_packed",
    "unpack_w4a16_reference",
]
