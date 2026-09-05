# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-memory checkpoint demultiplexing, shared with loader lifetime tests."""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
_SENTINEL_WEIGHTS = frozenset(("image_start", "image_pad", "image_newline", "image_end"))


def stream_language_weights(
    weights: Iterable[tuple[str, T]], load_vision: Callable[[str, T], None]
) -> Iterator[tuple[str, T]]:
    """Load image weights inline and yield raw language names exactly once.

    Never accumulate, sort, or replay this iterator. The child owns its mapper
    and target/draft filtering. The callback must not retain source tensors.
    """
    for name, tensor in weights:
        if name.startswith(("vision.", "aligner.")) or name in _SENTINEL_WEIGHTS:
            load_vision(name, tensor)
        else:
            yield name, tensor
