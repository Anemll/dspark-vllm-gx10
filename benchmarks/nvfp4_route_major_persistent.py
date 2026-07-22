#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolated persistent route-major specialization of FlashInfer SM12x MoE.

The pinned FlashInfer tree contains a queue-driven ``MoEDynamicKernel``, but
its micro-batch branch deliberately disables early publication.  For M=4 the
kernel therefore takes a resident-grid barrier after route packing and only
then publishes the partial expert tiles.  That makes the existing path a
useful correctness reference, not a producer/consumer experiment.

This module creates a *runtime-only* specialization of that pinned source.  It
does not modify the installed package or serving dispatcher.  The
specialization makes three narrowly bounded changes:

* enable the already implemented append-only ready-task queue for micro M;
* publish a partial expert tile after the last routed row for that expert has
  completed (release store after a device fence), rather than waiting for 128
  rows that decode can never produce; and
* remove the now-redundant end-of-production partial-tile publication, while
  retaining the release of ``all_work_published``.

The consumer side is unchanged pinned FlashInfer code.  A CTA that runs out of
route-pack work immediately enters the persistent consumer loop, claims only a
release-published expert/slice task, and runs FC1 -> clamped SwiGLU-OAI ->
NVFP4 quantization -> FC2 -> weighted scatter.  Its FC2 TMA producer and MMA
consumer retain the original multi-stage (double-buffered on the DSv4 shape)
pipeline.  There is no route-pack -> compute resident-grid phase boundary.

The source SHA and every textual edit are fail-closed.  This is intentionally
benchmark-only: prefill and the serving backend selector are untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterator, Sequence


PINNED_DYNAMIC_SOURCE_SHA256 = (
    "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
)
RUNTIME_MODULE_NAME = (
    "flashinfer.fused_moe.cute_dsl.blackwell_sm12x."
    "moe_route_major_persistent_runtime"
)


@dataclass(frozen=True)
class PersistentSourceProof:
    source_path: str
    source_sha256: str
    transformed_sha256: str
    class_renames: int
    micro_queue_enables: int
    readiness_rewrites: int
    terminal_flush_rewrites: int
    route_compute_barrier_removed: bool
    release_acquire_queue_retained: bool
    fc2_pipeline_retained: bool
    swiglu_limit_retained: bool
    overlap_observation_marker: int


def sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _replace_exact(source: str, old: str, new: str, *, count: int) -> str:
    observed = source.count(old)
    if observed != count:
        raise RuntimeError(
            "persistent source contract drift for exact rewrite: "
            f"expected {count}, observed {observed}: {old[:96]!r}"
        )
    return source.replace(old, new)


def _replace_between_once(source: str, start: str, end: str, replacement: str) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(
            "persistent source contract drift for bounded block rewrite: "
            f"start={source.count(start)} end={source.count(end)}"
        )
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[:begin] + replacement + source[finish:]


def transform_dynamic_source(
    source: str,
    *,
    expected_sha256: str | None = PINNED_DYNAMIC_SOURCE_SHA256,
) -> tuple[str, PersistentSourceProof]:
    """Return the isolated early-publication kernel source and its proof."""

    source_sha = sha256_text(source)
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise RuntimeError(
            "pinned FlashInfer moe_dynamic_kernel.py SHA drift: "
            f"expected {expected_sha256}, observed {source_sha}"
        )

    transformed = source
    transformed = _replace_exact(
        transformed,
        "class MoEDynamicKernel:",
        "class MoERouteMajorPersistentKernel:",
        count=1,
    )
    transformed = _replace_exact(
        transformed,
        "full_tile_publish_enabled = Int32(0)",
        # Micro decode has at most M rows per expert.  The completion count is
        # compared with row_counts[expert], not tile_m, below.
        "full_tile_publish_enabled = Int32(1)",
        count=1,
    )
    transformed = _replace_exact(
        transformed,
        """        self._setup_attributes(hidden_size=hidden_size)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(""",
        """        self._setup_attributes(hidden_size=hidden_size)
        if cutlass.const_expr(self.ab_stage != 2):
            raise ValueError(
                "persistent route-major gate requires a two-stage FC2 pipeline; "
                f"observed ab_stage={self.ab_stage}"
            )

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(""",
        count=1,
    )

    old_ready = """if completed == Int32(self.tile_shape_mnk[0]):
                                        self._publish_ready_tasks(
                                            task_tail,
                                            task_ready,
                                            task_expert,
                                            task_m_tile,
                                            task_slice_begin,
                                            task_slice_count,
                                            task_valid_rows,
                                            route_gate_tile_cnt,
                                            task_slice_chunk,
                                            expert_id,
                                            phys_tile,
                                            Int32(self.tile_shape_mnk[0]),
                                        )"""
    new_ready = """expected_rows = row_counts[expert_id]
                                    if completed == expected_rows:
                                        self._publish_ready_tasks(
                                            task_tail,
                                            task_ready,
                                            task_expert,
                                            task_m_tile,
                                            task_slice_begin,
                                            task_slice_count,
                                            task_valid_rows,
                                            route_gate_tile_cnt,
                                            task_slice_chunk,
                                            expert_id,
                                            phys_tile,
                                            expected_rows,
                                        )"""
    transformed = _replace_exact(
        transformed,
        old_ready,
        new_ready,
        count=2,
    )

    # Delete the entire deferred-publication branch, including both resident
    # barriers.  It is not enough to make it runtime-unreachable: this probe's
    # source proof must show that no route-pack -> compute boundary survives.
    terminal_replacement = """        if is_cta_leader > Int32(0):
            prev_done = atomic_add_global_i32(
                get_ptr_as_int64(producers_done_count, Int32(0)),
                Int32(1),
            )
            if prev_done == Int32(gdim_z) - Int32(1):
                # Every micro expert was release-published by the row that
                # completed row_counts[expert].  Publishing partials here
                # would enqueue every task twice.
                _threadfence()
                _st_global_release_i32(
                    get_ptr_as_int64(all_work_published, Int32(0)),
                    Int32(1),
                )"""
    transformed = _replace_between_once(
        transformed,
        """        if full_tile_publish_enabled == Int32(0):
            # Micro batches cannot fill a full M tile, so overlap is impossible.
""",
        "\n\n        gA = cute.local_tile(",
        terminal_replacement,
    )
    overlap_anchor = """            if has_task > Int32(0) and full_tile_publish_enabled > Int32(0):
                claimed_slot = _ld_shared_i32(ctrl_base_addr + Int32(28))
                _ld_global_acquire_i32(get_ptr_as_int64(task_ready, claimed_slot))
"""
    overlap_instrumented = overlap_anchor + """                if is_cta_leader > Int32(0):
                    published = _ld_global_acquire_i32(
                        get_ptr_as_int64(all_work_published, Int32(0))
                    )
                    if published == Int32(0):
                        # This tile has already release-published every row,
                        # so its write counter is dead producer state.  Mark
                        # an FC2 consumer that starts before global producer
                        # completion without adding a new runtime ABI tensor.
                        observed_tile = _ld_shared_i32(
                            ctrl_base_addr + Int32(12)
                        )
                        atomic_add_global_i32(
                            get_ptr_as_int64(tile_write_count, observed_tile),
                            Int32(65536),
                        )
"""
    transformed = _replace_exact(
        transformed,
        overlap_anchor,
        overlap_instrumented,
        count=1,
    )
    transformed = _replace_exact(
        transformed,
        '__all__ = ["MoEDynamicKernel"]',
        '__all__ = ["MoERouteMajorPersistentKernel"]',
        count=1,
    )

    # Fail closed on the semantic invariants that make this more than a name
    # change.  Two initialization/histogram barriers are intentionally kept;
    # the forbidden boundary is the micro deferred-publication branch.
    if "Micro batches cannot fill a full M tile, so overlap is impossible." in transformed:
        raise RuntimeError("deferred micro route/compute phase boundary survived")
    release_acquire = all(
        token in transformed
        for token in (
            "_st_global_release_i32(get_ptr_as_int64(task_ready, slot), Int32(1))",
            "_ld_global_acquire_i32(get_ptr_as_int64(task_ready, claimed_slot))",
            "_spin_wait_global_eq_i32(",
        )
    )
    fc2_pipeline = all(
        token in transformed
        for token in (
            "phase2_pipeline = pipeline.PipelineTmaAsync.create(",
            "num_stages=self.ab_stage",
            "phase2_pipeline.producer_acquire(",
            "phase2_pipeline.consumer_wait(",
        )
    )
    clamp = all(
        token in transformed
        for token in (
            "swiglu_limit: float | None = None",
            "limit=self.swiglu_limit",
            "activation=self.activation",
        )
    )
    if not release_acquire or not fc2_pipeline or not clamp:
        raise RuntimeError(
            "persistent transform lost queue, double-buffer pipeline, or clamp semantics"
        )

    proof = PersistentSourceProof(
        source_path="",
        source_sha256=source_sha,
        transformed_sha256=sha256_text(transformed),
        class_renames=1,
        micro_queue_enables=1,
        readiness_rewrites=2,
        terminal_flush_rewrites=1,
        route_compute_barrier_removed=True,
        release_acquire_queue_retained=release_acquire,
        fc2_pipeline_retained=fc2_pipeline,
        swiglu_limit_retained=clamp,
        overlap_observation_marker=65536,
    )
    return transformed, proof


def dynamic_source_path() -> Path:
    spec = importlib.util.find_spec(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
    )
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot resolve installed FlashInfer dynamic kernel source")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"resolved dynamic kernel source is not a file: {path}")
    return path


def load_persistent_kernel_class(
    path: Path | None = None,
) -> tuple[type[Any], PersistentSourceProof, ModuleType]:
    """Load the transformed class without altering the installed package."""

    source_path = (path or dynamic_source_path()).resolve()
    transformed, proof = transform_dynamic_source(source_path.read_text())
    proof = PersistentSourceProof(
        **{**proof.__dict__, "source_path": str(source_path)}
    )
    # CuTeDSL preprocesses the decorated Python source lazily, on the first
    # kernel compile.  Pointing ``__file__`` at the unmodified installed file
    # makes its AST disagree with the transformed code object (and turns
    # compile-time branches into apparent dynamic booleans).  Keep an exact
    # transformed source file alive for the full installation context so the
    # compiler sees the same text that Python executed.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="moe_route_major_persistent_",
        suffix=".py",
        delete=False,
    ) as runtime_file:
        runtime_file.write(transformed)
        runtime_path = Path(runtime_file.name)

    module = ModuleType(RUNTIME_MODULE_NAME)
    module.__file__ = str(runtime_path)
    module.__package__ = RUNTIME_MODULE_NAME.rpartition(".")[0]
    sys.modules[RUNTIME_MODULE_NAME] = module
    try:
        exec(compile(transformed, str(runtime_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(RUNTIME_MODULE_NAME, None)
        runtime_path.unlink(missing_ok=True)
        raise
    cls = getattr(module, "MoERouteMajorPersistentKernel")
    return cls, proof, module


@contextmanager
def install_isolated_persistent_kernel(
    moe_dispatch: Any,
    *,
    source_path: Path | None = None,
) -> Iterator[PersistentSourceProof]:
    """Temporarily route the dynamic compiler to the isolated specialization."""

    cls, proof, module = load_persistent_kernel_class(source_path)
    original_class = moe_dispatch.MoEDynamicKernel
    original_selector = moe_dispatch.select_sm120_moe_backend
    cache = moe_dispatch._DYNAMIC_KERNEL_CACHE
    if cache:
        raise RuntimeError(
            "dynamic kernel cache must be empty before isolated installation"
        )
    moe_dispatch.MoEDynamicKernel = cls
    moe_dispatch.select_sm120_moe_backend = lambda **_: "dynamic"
    try:
        yield proof
    finally:
        cache.clear()
        moe_dispatch.MoEDynamicKernel = original_class
        moe_dispatch.select_sm120_moe_backend = original_selector
        sys.modules.pop(module.__name__, None)
        Path(module.__file__).unlink(missing_ok=True)


def simulate_expert_publication(
    expert_ids: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """CPU proof: return ``(route_index, expert)`` release-publication order."""

    ids = tuple(int(expert) for expert in expert_ids)
    if not ids or any(expert < 0 for expert in ids):
        raise ValueError("expert IDs must be a non-empty non-negative sequence")
    expected: dict[int, int] = {}
    completed: dict[int, int] = {}
    for expert in ids:
        expected[expert] = expected.get(expert, 0) + 1
    published: list[tuple[int, int]] = []
    for route, expert in enumerate(ids):
        completed[expert] = completed.get(expert, 0) + 1
        if completed[expert] == expected[expert]:
            published.append((route, expert))
    if {expert for _, expert in published} != set(ids):
        raise AssertionError("every active expert must publish exactly once")
    return tuple(published)


def readiness_bank(epoch: int) -> int:
    """Two-bank graph/replay workspace selection used by the hardware probe."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return int(epoch) & 1


__all__ = [
    "PINNED_DYNAMIC_SOURCE_SHA256",
    "PersistentSourceProof",
    "dynamic_source_path",
    "install_isolated_persistent_kernel",
    "load_persistent_kernel_class",
    "readiness_bank",
    "sha256_text",
    "simulate_expert_publication",
    "transform_dynamic_source",
]
