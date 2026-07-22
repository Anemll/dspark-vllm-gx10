#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Preplan B12X TC-decode launches for single-copy ModelOpt W4A16.

The kernel-side ModelOpt TC-decode patch admits the native ModelOpt payload,
but vLLM serving uses B12X's frozen TP-MoE arena.  The pinned planner derives
``weight_layout="packed"`` from every source format because its ordinary
serving path always repacks.  The prepared dual backend deliberately retains
the original ModelOpt bytes, so its frozen arena must carry an explicit layout
override from scratch planning through materialization and prewarm.  Keep the
override optional so every existing caller remains packed, and widen only the
TC-decode build/selection policy to accept the already-normalized ``modelopt``
layout.  Kernel math, checkpoint storage, and prefill are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "c2ca5aca4f9efd8ac8afb52909ef18410d1afd455d7e994debcd4e0bc13e019d"
)
PATCHED_SOURCE_SHA256 = (
    "ba980ff1df1df0b9959c274fa255c2fcb538671f0cdd068b0ec7cdf4f434933d"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py"
)


_REPLACEMENTS = (
    (
        """\
    source_format: str = "modelopt_nvfp4"
    w13_layout: str = "w13"
    frozen: bool = True
""",
        """\
    source_format: str = "modelopt_nvfp4"
    w13_layout: str = "w13"
    w4a16_weight_layout: str | None = None
    frozen: bool = True
""",
        "scratch-cap layout field",
    ),
    (
        """\
        object.__setattr__(self, "w13_layout", _normalize_w13_layout(self.w13_layout))
        object.__setattr__(self, "frozen", bool(self.frozen))
""",
        """\
        object.__setattr__(self, "w13_layout", _normalize_w13_layout(self.w13_layout))
        if self.w4a16_weight_layout is not None:
            object.__setattr__(
                self,
                "w4a16_weight_layout",
                _normalize_w4a16_weight_layout(self.w4a16_weight_layout),
            )
        object.__setattr__(self, "frozen", bool(self.frozen))
""",
        "scratch-cap layout normalization",
    ),
    (
        """\
            source_format=self.caps.source_format,
            w13_layout=self.caps.w13_layout,
        )
        if _B12X_TIMING:
""",
        """\
            source_format=self.caps.source_format,
            w13_layout=self.caps.w13_layout,
            w4a16_weight_layout=self.caps.w4a16_weight_layout,
        )
        if _B12X_TIMING:
""",
        "workspace materialization layout propagation",
    ),
    (
        """\
def _plan_core_workspace(
    implementation: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    routed_rows: int,
    max_rows: int,
    activation: str = "silu",
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    apply_router_weight_on_input: bool = False,
""",
        """\
def _plan_core_workspace(
    implementation: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    routed_rows: int,
    max_rows: int,
    activation: str = "silu",
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    w4a16_weight_layout: str | None = None,
    apply_router_weight_on_input: bool = False,
""",
        "core-workspace layout argument",
    ),
    (
        """\
        scale_format = _w4a16_scale_format_for_source(source_format)
        weight_layout = _w4a16_weight_layout_for_source(source_format)
        routed_capacity = max(int(routed_rows), 1)
""",
        """\
        scale_format = _w4a16_scale_format_for_source(source_format)
        weight_layout = (
            _w4a16_weight_layout_for_source(source_format)
            if w4a16_weight_layout is None
            else _normalize_w4a16_weight_layout(w4a16_weight_layout)
        )
        routed_capacity = max(int(routed_rows), 1)
""",
        "core-workspace layout resolution",
    ),
    (
        """\
def plan_tp_moe_arena_layout(
    *,
    max_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device | str,
    dtype: torch.dtype,
    core_token_counts: tuple[int, ...] | None = None,
    route_num_experts: int | None = None,
    route_logits_dtype: torch.dtype | None = None,
    quant_mode: str | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
) -> TPMoEArenaLayout:
""",
        """\
def plan_tp_moe_arena_layout(
    *,
    max_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device | str,
    dtype: torch.dtype,
    core_token_counts: tuple[int, ...] | None = None,
    route_num_experts: int | None = None,
    route_logits_dtype: torch.dtype | None = None,
    quant_mode: str | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    w4a16_weight_layout: str | None = None,
) -> TPMoEArenaLayout:
""",
        "arena-layout argument",
    ),
    (
        """\
            source_format=source_format,
            w13_layout=w13_layout,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
        )
        core_nbytes = max(core_nbytes, _core_workspace_nbytes(core_plan))
""",
        """\
            source_format=source_format,
            w13_layout=w13_layout,
            w4a16_weight_layout=w4a16_weight_layout,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
        )
        core_nbytes = max(core_nbytes, _core_workspace_nbytes(core_plan))
""",
        "arena-layout core propagation",
    ),
    (
        """\
        source_format=caps.source_format,
        w13_layout=caps.w13_layout,
    )
    return TPMoEScratchPlan(
""",
        """\
        source_format=caps.source_format,
        w13_layout=caps.w13_layout,
        w4a16_weight_layout=caps.w4a16_weight_layout,
    )
    return TPMoEScratchPlan(
""",
        "scratch-plan layout propagation",
    ),
    (
        """\
def materialize_tp_moe_arena_workspaces(
    pool: TPMoEWorkspacePool,
    *,
    max_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device | str,
    dtype: torch.dtype,
    core_token_counts: tuple[int, ...] | None = None,
    quant_mode: str | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
) -> None:
""",
        """\
def materialize_tp_moe_arena_workspaces(
    pool: TPMoEWorkspacePool,
    *,
    max_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device | str,
    dtype: torch.dtype,
    core_token_counts: tuple[int, ...] | None = None,
    quant_mode: str | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    w4a16_weight_layout: str | None = None,
) -> None:
""",
        "workspace-materialization argument",
    ),
    (
        """\
    w4a16_scale_format = _w4a16_scale_format_for_source(source_format)
    w4a16_weight_layout = _w4a16_weight_layout_for_source(source_format)

    device = torch.device(device)
""",
        """\
    w4a16_scale_format = _w4a16_scale_format_for_source(source_format)
    w4a16_weight_layout = (
        _w4a16_weight_layout_for_source(source_format)
        if w4a16_weight_layout is None
        else _normalize_w4a16_weight_layout(w4a16_weight_layout)
    )

    device = torch.device(device)
""",
        "workspace-materialization layout resolution",
    ),
    (
        """\
            source_format=source_format,
            w13_layout=w13_layout,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
        )
        required_nbytes = _core_workspace_nbytes(core_plan)
""",
        """\
            source_format=source_format,
            w13_layout=w13_layout,
            w4a16_weight_layout=w4a16_weight_layout,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
        )
        required_nbytes = _core_workspace_nbytes(core_plan)
""",
        "workspace-materialization core propagation",
    ),
    (
        """\
        and weight_layout == "packed"
        and token_count <= _TC_DECODE_MAX_M
""",
        """\
        and weight_layout in {"packed", "modelopt"}
        and token_count <= _TC_DECODE_MAX_M
""",
        "preplanned TC-decode selection layout",
    ),
    (
        """\
        # TC-decode (B12X_W4A16_TC_DECODE) is a packed-layout small-M decode path.
        # Build its fused-sum launch variant only for the supported decode sizes
""",
        """\
        # TC-decode supports packed and single-copy ModelOpt small-M decode.
        # Build its fused-sum launch variant only for the supported decode sizes
""",
        "TC-decode planner contract",
    ),
    (
        """\
            and weight_layout == "packed"
            and element_dtype == "bf16"
""",
        """\
            and weight_layout in {"packed", "modelopt"}
            and element_dtype == "bf16"
""",
        "TC-decode launch prewarm layout",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_source(source: str) -> str:
    for anchor, replacement, label in _REPLACEMENTS:
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected one {label} anchor, found {count}")
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    original = args.target.read_bytes()
    original_sha = _sha256(original)
    if original_sha != PINNED_SOURCE_SHA256:
        raise RuntimeError(
            "pinned B12X TP-MoE planner SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    patched_sha = _sha256(patched)
    if patched_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "deterministic B12X ModelOpt TC-decode planner result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X ModelOpt TC-decode frozen planner: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
