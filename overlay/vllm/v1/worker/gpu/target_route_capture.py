# SPDX-License-Identifier: Apache-2.0
"""Opt-in target-only DeepSeek-V4 route capture for decode diagnostics.

The normal serving path never constructs this object.  When explicitly enabled,
the V2 GPU model runner binds callbacks to the 43 target ``BaseRouter`` objects.
Those callbacks continuously update stable GPU slots, which also makes the
capture work when the target forward is replayed from a CUDA graph.  Only
steady, target-only C=4 decode steps are copied into a bounded GPU history.
The completed history crosses to CPU exactly once and is then persisted as one
rank-local NPY artifact with a JSON manifest.

This is diagnostic instrumentation: its device copies and final synchronization
are deliberately absent unless ``VLLM_DSPARK_TARGET_ROUTE_CAPTURE=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CAPTURE_ENV = "VLLM_DSPARK_TARGET_ROUTE_CAPTURE"
OUTPUT_DIR_ENV = "VLLM_DSPARK_TARGET_ROUTE_CAPTURE_DIR"
STEPS_ENV = "VLLM_DSPARK_TARGET_ROUTE_CAPTURE_STEPS"
WARMUP_STEPS_ENV = "VLLM_DSPARK_TARGET_ROUTE_CAPTURE_WARMUP_STEPS"

SCHEMA_VERSION = 1
EXPECTED_LAYERS = tuple(range(43))
EXPECTED_CONCURRENCY = 4
EXPECTED_TOP_K = 6
EXPECTED_GLOBAL_EXPERTS = 256
MAX_CAPTURE_STEPS = 4096
MAX_WARMUP_STEPS = 4096
_TARGET_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts$")


@dataclass(frozen=True)
class TargetRouteCaptureConfig:
    output_dir: Path
    steps: int
    warmup_steps: int

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "TargetRouteCaptureConfig | None":
        values = os.environ if environment is None else environment
        enabled = values.get(CAPTURE_ENV, "0")
        if enabled not in ("0", "1"):
            raise ValueError(f"{CAPTURE_ENV} must be exactly 0 or 1, got {enabled!r}")
        if enabled == "0":
            return None

        raw_dir = values.get(OUTPUT_DIR_ENV)
        if raw_dir is None or not raw_dir:
            raise ValueError(f"{OUTPUT_DIR_ENV} is required when capture is enabled")
        output_dir = Path(raw_dir)
        if not output_dir.is_absolute():
            raise ValueError(f"{OUTPUT_DIR_ENV} must be absolute, got {raw_dir!r}")
        if not output_dir.is_dir():
            raise FileNotFoundError(f"capture directory does not exist: {output_dir}")

        steps = _bounded_integer(
            values, STEPS_ENV, default=64, maximum=MAX_CAPTURE_STEPS
        )
        warmup_steps = _bounded_integer(
            values,
            WARMUP_STEPS_ENV,
            default=8,
            maximum=MAX_WARMUP_STEPS,
            minimum=0,
        )
        return cls(output_dir=output_dir, steps=steps, warmup_steps=warmup_steps)


def _bounded_integer(
    values: Mapping[str, str],
    name: str,
    *,
    default: int,
    maximum: int,
    minimum: int = 1,
) -> int:
    raw = values.get(name, str(default))
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def validate_target_only_runtime(
    *,
    speculative_config: Any,
    speculator: Any,
    num_speculative_steps: int,
    enable_return_routed_experts: bool,
    tensor_parallel_size: int,
) -> None:
    """Fail closed unless the diagnostic is the intended TP=2 target-only run."""

    if speculative_config is not None or speculator is not None:
        raise RuntimeError("target route capture requires speculator=None (MTP off)")
    if num_speculative_steps != 0:
        raise RuntimeError(
            "target route capture requires num_speculative_steps=0, got "
            f"{num_speculative_steps}"
        )
    if enable_return_routed_experts:
        raise RuntimeError(
            "target route capture is incompatible with enable_return_routed_experts"
        )
    if tensor_parallel_size != 2:
        raise RuntimeError(
            f"target route capture is pinned to TP=2, got TP={tensor_parallel_size}"
        )


def validate_loaded_target_model(model: Any) -> None:
    """Pin the diagnostic to the intended DeepSeek-V4 target architecture."""

    class_name = model.__class__.__name__
    if class_name != "DeepseekV4ForCausalLM":
        raise RuntimeError(
            "target route capture requires DeepseekV4ForCausalLM, got "
            f"{class_name}"
        )
    config = getattr(model, "config", None)
    contract = {
        "num_hidden_layers": len(EXPECTED_LAYERS),
        "n_routed_experts": EXPECTED_GLOBAL_EXPERTS,
        "num_experts_per_tok": EXPECTED_TOP_K,
    }
    for attribute, expected in contract.items():
        actual = getattr(config, attribute, None)
        if actual != expected:
            raise RuntimeError(
                f"target model {attribute} drift: expected {expected}, got {actual}"
            )


def is_steady_target_c4_step(input_batch: Any, *, dummy_run: bool) -> bool:
    """Return true only for four target decode tokens, one per request."""

    if dummy_run or int(input_batch.num_reqs) != EXPECTED_CONCURRENCY:
        return False
    if int(input_batch.num_tokens) != EXPECTED_CONCURRENCY:
        return False
    scheduled = np.asarray(input_batch.num_scheduled_tokens)
    if scheduled.shape != (EXPECTED_CONCURRENCY,) or not np.all(scheduled == 1):
        return False
    if int(input_batch.num_draft_tokens) != 0:
        return False
    draft_per_req = input_batch.num_draft_tokens_per_req
    if draft_per_req is not None and np.any(np.asarray(draft_per_req) != 0):
        return False
    prefilling = np.asarray(input_batch.is_prefilling_np)
    return prefilling.shape == (EXPECTED_CONCURRENCY,) and not bool(prefilling.any())


def target_layer_id(layer_name: str) -> int | None:
    match = _TARGET_LAYER_RE.fullmatch(layer_name)
    if match is None:
        return None
    layer = int(match.group(1))
    return layer if layer in EXPECTED_LAYERS else None


def collect_target_routers(
    static_forward_context: Mapping[str, Any],
    *,
    runner_type: type[Any] | None = None,
    router_type: type[Any] | None = None,
) -> dict[int, Any]:
    """Select and validate exactly the 43 target MoE routers by semantic name."""

    if runner_type is None:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        runner_type = MoERunner
    if router_type is None:
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

        router_type = BaseRouter

    selected: dict[int, Any] = {}
    for context_name, module in static_forward_context.items():
        if not isinstance(module, runner_type):
            continue
        layer_name = str(module.layer_name)
        layer = target_layer_id(layer_name)
        if layer is None:
            continue
        if context_name != layer_name:
            raise RuntimeError(
                f"target route context/name drift: {context_name!r} != {layer_name!r}"
            )
        if int(module.layer_id) != layer:
            raise RuntimeError(
                f"target route layer-id drift for {layer_name}: {module.layer_id}"
            )
        if layer in selected:
            raise RuntimeError(f"duplicate target route layer {layer}: {layer_name}")
        if not isinstance(module.router, router_type):
            raise RuntimeError(f"target layer {layer} does not use BaseRouter")
        if module.router.capture_fn is not None:
            raise RuntimeError(f"target layer {layer} already has a route callback")
        if int(module.router.top_k) != EXPECTED_TOP_K:
            raise RuntimeError(
                f"target layer {layer} top-k drift: {module.router.top_k}"
            )
        if int(module.router.global_num_experts) != EXPECTED_GLOBAL_EXPERTS:
            raise RuntimeError(
                "target layer "
                f"{layer} expert-count drift: {module.router.global_num_experts}"
            )
        selected[layer] = module.router

    if tuple(sorted(selected)) != EXPECTED_LAYERS:
        missing = sorted(set(EXPECTED_LAYERS) - selected.keys())
        extra = sorted(set(selected) - set(EXPECTED_LAYERS))
        raise RuntimeError(
            f"target route layer set drift: missing={missing}, extra={extra}"
        )
    return selected


class TargetRouteCapture:
    """Bounded persistent device capture with a single completed-history D2H."""

    def __init__(
        self,
        config: TargetRouteCaptureConfig,
        *,
        device: torch.device,
        rank: int,
        world_size: int,
        layer_names: list[str],
    ):
        if world_size != 2 or rank not in (0, 1):
            raise RuntimeError(
                "target route rank contract requires TP=2, got "
                f"rank={rank}/{world_size}"
            )
        if len(layer_names) != len(EXPECTED_LAYERS):
            raise RuntimeError(
                f"target route layer-name count drift: {len(layer_names)}"
            )
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.layer_names = tuple(layer_names)
        self._current = torch.full(
            (len(EXPECTED_LAYERS), EXPECTED_CONCURRENCY, EXPECTED_TOP_K),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._history = torch.empty(
            (
                config.steps,
                EXPECTED_CONCURRENCY,
                len(EXPECTED_LAYERS),
                EXPECTED_TOP_K,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._eligible_steps = 0
        self._captured_steps = 0
        self._step_active = False
        self._finalized = False
        self._data_path = config.output_dir / f"target-routes-rank-{rank}.npy"
        self._manifest_path = config.output_dir / f"target-routes-rank-{rank}.json"
        for path in (self._data_path, self._manifest_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite route artifact: {path}")

    def callback(self, layer: int, topk_ids: torch.Tensor) -> None:
        if layer not in EXPECTED_LAYERS:
            raise RuntimeError(f"unexpected target route layer callback: {layer}")
        if topk_ids.ndim != 2 or topk_ids.shape[1] != EXPECTED_TOP_K:
            raise RuntimeError(
                f"target layer {layer} route shape drift: {tuple(topk_ids.shape)}"
            )
        rows = min(int(topk_ids.shape[0]), EXPECTED_CONCURRENCY)
        if rows:
            self._current[layer, :rows].copy_(
                topk_ids[:rows], non_blocking=True
            )

    def begin_step(self, input_batch: Any, *, dummy_run: bool) -> bool:
        if self._finalized or not is_steady_target_c4_step(
            input_batch, dummy_run=dummy_run
        ):
            return False
        if self._step_active:
            raise RuntimeError("target route capture step re-entered")
        self._eligible_steps += 1
        if self._eligible_steps <= self.config.warmup_steps:
            return False
        self._current.fill_(-1)
        self._step_active = True
        return True

    def end_step(self) -> Path | None:
        if not self._step_active:
            return None
        self._history[self._captured_steps].copy_(
            self._current.permute(1, 0, 2), non_blocking=True
        )
        self._captured_steps += 1
        self._step_active = False
        if self._captured_steps != self.config.steps:
            return None
        return self._finalize()

    def _finalize(self) -> Path:
        if self._finalized:
            raise RuntimeError("target route capture finalized twice")
        # This is the sole device-to-host transfer in the capture lifecycle.
        host = self._history.cpu().numpy()
        if host.shape != (
            self.config.steps,
            EXPECTED_CONCURRENCY,
            len(EXPECTED_LAYERS),
            EXPECTED_TOP_K,
        ):
            raise RuntimeError(f"target route host shape drift: {host.shape}")
        invalid = np.argwhere((host < 0) | (host >= EXPECTED_GLOBAL_EXPERTS))
        if invalid.size:
            first = invalid[0].tolist()
            raise RuntimeError(
                "target route history has missing/out-of-range expert id at "
                f"index={first}, value={int(host[tuple(first)])}"
            )

        raw_sha256 = hashlib.sha256(host.tobytes(order="C")).hexdigest()
        _atomic_save_npy(self._data_path, host)
        data_sha256 = _sha256_file(self._data_path)
        layer_name_sha256 = hashlib.sha256(
            "\n".join(self.layer_names).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "rank": self.rank,
            "world_size": self.world_size,
            "shape": list(host.shape),
            "dtype": str(host.dtype),
            "steps": self.config.steps,
            "warmup_steps": self.config.warmup_steps,
            "eligible_steps_seen": self._eligible_steps,
            "concurrency": EXPECTED_CONCURRENCY,
            "tokens_per_request": 1,
            "top_k": EXPECTED_TOP_K,
            "global_experts": EXPECTED_GLOBAL_EXPERTS,
            "layers": list(EXPECTED_LAYERS),
            "layer_names": list(self.layer_names),
            "layer_name_sha256": layer_name_sha256,
            "data_file": self._data_path.name,
            "data_size": self._data_path.stat().st_size,
            "data_sha256": data_sha256,
            "raw_tensor_sha256": raw_sha256,
        }
        _atomic_write_json(self._manifest_path, manifest)
        self._finalized = True
        return self._manifest_path


def bind_target_route_capture(
    *,
    config: TargetRouteCaptureConfig,
    static_forward_context: Mapping[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
) -> TargetRouteCapture:
    routers = collect_target_routers(static_forward_context)
    layer_names = [f"model.layers.{layer}.ffn.experts" for layer in EXPECTED_LAYERS]
    capture = TargetRouteCapture(
        config,
        device=device,
        rank=rank,
        world_size=world_size,
        layer_names=layer_names,
    )
    for layer, router in routers.items():

        def _capture(topk_ids: torch.Tensor, *, _layer: int = layer) -> None:
            capture.callback(_layer, topk_ids)

        router.set_capture_fn(_capture)
    return capture


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError(
                    f"short manifest write: {written} of {len(encoded)} bytes"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
