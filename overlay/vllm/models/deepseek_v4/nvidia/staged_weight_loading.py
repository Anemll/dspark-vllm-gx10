# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded CPU staging for DeepSeek V4 ModelOpt NVFP4 expert weights.

The ordinary RoutedExperts weight loader remains authoritative for TP slicing
and return_success behavior.  This exact deployment lane rejects expert
parallelism/EPLB.  The module only substitutes a CPU-backed destination while
a complete routed layer is read, then copies the eight raw parameters into
their existing CUDA storage once.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

STAGED_LOAD_ENV = "VLLM_DSV4_NVFP4_LAYER_STAGED_LOAD"
ROCE_LOAD_FORMAT = "roce_tp"

EXPECTED_NVFP4_EXPERTS = 256
EXPECTED_NVFP4_LAYERS = 43
EXPECTED_HIDDEN_SIZE = 4_096
EXPECTED_INTERMEDIATE_SIZE_PER_RANK = 1_024
EXPECTED_TENSORS_PER_LAYER = 3_072
EXPECTED_STAGE_BYTES = 1_811_945_472
EXPECTED_COMMIT_CALLS = 8

_CHECKPOINT_SUFFIXES = (
    "weight",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
)
_PARAMETER_ORDER = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
)
_RAW_BYTE_PARAMETERS = frozenset(("w13_weight_scale", "w2_weight_scale"))
_REVIEWED_MAPPING_PREFIXES = {
    "w1": "experts.routed_experts.w13_",
    "w2": "experts.routed_experts.w2_",
    "w3": "experts.routed_experts.w13_",
}


def _expected_parameter_shapes() -> dict[str, tuple[int, ...]]:
    experts = EXPECTED_NVFP4_EXPERTS
    hidden = EXPECTED_HIDDEN_SIZE
    intermediate = EXPECTED_INTERMEDIATE_SIZE_PER_RANK
    return {
        "w13_weight": (experts, 2 * intermediate, hidden // 2),
        "w2_weight": (experts, hidden, intermediate // 2),
        "w13_weight_scale": (experts, 2 * intermediate, hidden // 16),
        "w2_weight_scale": (experts, hidden, intermediate // 16),
        "w13_weight_scale_2": (experts, 2),
        "w2_weight_scale_2": (experts,),
        # ModelOpt global activation scales are expanded across all logical
        # experts.  W13 retains distinct w1/w3 columns; W2 has one value.
        "w13_input_scale": (experts, 2),
        "w2_input_scale": (experts,),
    }


def staged_load_requested(environ: dict[str, str] | None = None) -> bool:
    """Return the strict opt-in state; reject ambiguous truthy values."""

    source = os.environ if environ is None else environ
    value = source.get(STAGED_LOAD_ENV, "0")
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{STAGED_LOAD_ENV} must be exactly '0' or '1'; got {value!r}")


def _device_type(tensor: Any) -> str | None:
    device = getattr(tensor, "device", None)
    return getattr(device, "type", None)


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _expected_parameter_dtypes(torch_module: Any) -> dict[str, Any]:
    return {
        "w13_weight": torch_module.uint8,
        "w2_weight": torch_module.uint8,
        "w13_weight_scale": torch_module.float8_e4m3fn,
        "w2_weight_scale": torch_module.float8_e4m3fn,
        "w13_weight_scale_2": torch_module.float32,
        "w2_weight_scale_2": torch_module.float32,
        "w13_input_scale": torch_module.float32,
        "w2_input_scale": torch_module.float32,
    }


def _expected_checkpoint_dtype(torch_module: Any, suffix: str) -> Any:
    if suffix == "weight":
        return torch_module.uint8
    if suffix == "weight_scale":
        # The pinned NVIDIA main routed checkpoint stores ModelOpt NVFP4 block
        # scales as E4M3.  Staging reinterprets these one-byte values only
        # after validating this exact on-disk dtype.
        return torch_module.float8_e4m3fn
    if suffix in ("weight_scale_2", "input_scale"):
        return torch_module.float32
    raise ValueError(f"Unsupported staged NVFP4 checkpoint suffix: {suffix!r}")


def _expected_checkpoint_shape(
    projection: str,
    suffix: str,
) -> tuple[int, ...]:
    """Return the exact pre-TP source shape proven from the 46 headers."""

    if projection not in ("w1", "w2", "w3"):
        raise ValueError(
            f"Unsupported staged NVFP4 checkpoint projection: {projection!r}"
        )
    if suffix == "weight":
        return (2_048, 2_048) if projection in ("w1", "w3") else (4_096, 1_024)
    if suffix == "weight_scale":
        return (2_048, 256) if projection in ("w1", "w3") else (4_096, 128)
    if suffix in ("weight_scale_2", "input_scale"):
        return ()
    raise ValueError(f"Unsupported staged NVFP4 checkpoint suffix: {suffix!r}")


def _reviewed_parameter_relative_names(
    expert_mapping_index: Any,
) -> dict[str, str]:
    """Derive and validate destinations from vLLM's real mapping output."""

    prefixes: dict[str, set[str]] = {"w1": set(), "w2": set(), "w3": set()}
    for mapping_key, candidates in expert_mapping_index.mappings.items():
        _experts, logical_expert_text, projection, _empty = mapping_key.split(".")
        logical_expert = int(logical_expert_text)
        if len(candidates) != 1:
            raise RuntimeError(
                "NVFP4 staged expert mapping requires exactly one physical "
                f"candidate per logical key; {mapping_key!r} has "
                f"{len(candidates)}"
            )
        for param_name, weight_name, expert_id, shard_id in candidates:
            if (
                weight_name != mapping_key
                or int(expert_id) != logical_expert
                or shard_id != projection
            ):
                raise RuntimeError(
                    "NVFP4 staged expert mapping candidate drifted: "
                    f"key={mapping_key!r}, weight_name={weight_name!r}, "
                    f"expert_id={expert_id!r}, shard_id={shard_id!r}"
                )
            if shard_id not in prefixes:
                raise RuntimeError(
                    f"Unexpected staged NVFP4 expert shard id: {shard_id!r}"
                )
            prefixes[shard_id].add(param_name)

    observed = {
        shard_id: next(iter(values)) if len(values) == 1 else None
        for shard_id, values in prefixes.items()
    }
    if observed != _REVIEWED_MAPPING_PREFIXES:
        raise RuntimeError(
            "NVFP4 staged destination mapping drifted from the reviewed "
            f"fused_moe_make_expert_params_mapping output: {observed!r}"
        )

    result: dict[str, str] = {}
    for basename in _PARAMETER_ORDER:
        family = "w13" if basename.startswith("w13_") else "w2"
        suffix = basename.removeprefix(f"{family}_")
        projection = "w1" if family == "w13" else "w2"
        result[basename] = f"{observed[projection]}{suffix}"
    if len(set(result.values())) != len(_PARAMETER_ORDER):
        raise RuntimeError("NVFP4 staged destination parameters are not unique")
    return result


@dataclass(frozen=True)
class StagedSource:
    """One source tensor currently being dispatched through mapping candidates."""

    name: str
    layer: int
    key: str
    suffix: str
    loaded_weight: Any


@dataclass
class _StagedParameter:
    name: str
    basename: str
    actual: Any
    proxy: Any
    raw_bytes: bool


@dataclass
class _ActiveLayer:
    layer: int
    expected_keys: frozenset[str]
    started_at: float = field(default_factory=time.perf_counter)
    seen: set[str] = field(default_factory=set)
    parameters: dict[str, _StagedParameter] = field(default_factory=dict)


class Nvfp4LayerStager:
    """Stage exactly one official NVFP4 routed layer at a time."""

    def __init__(
        self,
        *,
        torch_module: Any,
        eligible_parameters: dict[int, dict[str, Any]],
        expected_source_keys: frozenset[str],
        expected_stage_bytes: int = EXPECTED_STAGE_BYTES,
        expected_commit_calls: int = EXPECTED_COMMIT_CALLS,
        expected_checkpoint_shapes: dict[tuple[str, str], tuple[int, ...]]
        | None = None,
    ) -> None:
        self._torch = torch_module
        self._eligible_parameters = eligible_parameters
        self._expected_source_keys = expected_source_keys
        self._expected_stage_bytes = expected_stage_bytes
        self._expected_commit_calls = expected_commit_calls
        if expected_checkpoint_shapes is not None:
            expected_shape_keys = {
                (projection, suffix)
                for projection in ("w1", "w2", "w3")
                for suffix in _CHECKPOINT_SUFFIXES
            }
            if set(expected_checkpoint_shapes) != expected_shape_keys:
                raise RuntimeError(
                    "Synthetic NVFP4 checkpoint-shape override is incomplete"
                )
            if expected_stage_bytes == EXPECTED_STAGE_BYTES:
                raise RuntimeError(
                    "Official NVFP4 staging cannot override checkpoint shapes"
                )
        self._expected_checkpoint_shapes = expected_checkpoint_shapes
        self._active: _ActiveLayer | None = None
        self._pending: StagedSource | None = None
        self._completed_layers: set[int] = set()
        self._total_source_tensors = 0
        self._total_commit_calls = 0
        self._started_at = time.perf_counter()

    @property
    def completed_layers(self) -> frozenset[int]:
        return frozenset(self._completed_layers)

    @property
    def total_source_tensors(self) -> int:
        return self._total_source_tensors

    @property
    def total_commit_calls(self) -> int:
        return self._total_commit_calls

    def begin_source(
        self,
        name: str,
        loaded_weight: Any,
        expert_name_match: Any,
    ) -> StagedSource | None:
        """Begin staging a source, or decline a PP-missing layer."""

        layer = int(expert_name_match.layer)
        if layer not in self._eligible_parameters:
            return None
        if self._pending is not None:
            raise RuntimeError(
                "NVFP4 staged loader received a new tensor before completing "
                f"{self._pending.name!r}"
            )
        if _device_type(loaded_weight) != "cpu":
            raise RuntimeError(
                "NVFP4 layer staging supports CPU checkpoint tensors only; "
                f"{name!r} is on {_device_type(loaded_weight)!r}"
            )

        suffix = str(expert_name_match.suffix)
        expected_dtype = _expected_checkpoint_dtype(self._torch, suffix)
        if loaded_weight.dtype != expected_dtype:
            raise RuntimeError(
                f"NVFP4 staged source {name!r} has dtype {loaded_weight.dtype}; "
                f"expected {expected_dtype}"
            )
        projection = str(expert_name_match.projection)
        expected_shape = (
            self._expected_checkpoint_shapes[(projection, suffix)]
            if self._expected_checkpoint_shapes is not None
            else _expected_checkpoint_shape(projection, suffix)
        )
        observed_shape = tuple(loaded_weight.shape)
        if observed_shape != expected_shape:
            raise RuntimeError(
                f"NVFP4 staged source {name!r} has shape {observed_shape}; "
                f"expected {expected_shape} for {projection}.{suffix}"
            )
        key = f"{expert_name_match.mapping_key}{suffix}"
        if key not in self._expected_source_keys:
            raise RuntimeError(f"Unexpected staged NVFP4 source key: {key!r}")

        if self._active is None:
            if layer in self._completed_layers:
                raise RuntimeError(
                    f"NVFP4 layer {layer} appeared again after its staging commit"
                )
            self._active = _ActiveLayer(layer, self._expected_source_keys)
        elif self._active.layer != layer:
            missing = len(self._active.expected_keys - self._active.seen)
            raise RuntimeError(
                "NVFP4 checkpoint interleaves routed layers; bounded staging "
                f"still has layer {self._active.layer} active with {missing} "
                f"missing tensors when layer {layer} appeared"
            )
        if key in self._active.seen:
            raise RuntimeError(f"Duplicate staged NVFP4 source tensor: {name!r}")

        staged_weight = loaded_weight
        if suffix == "weight_scale":
            if int(loaded_weight.element_size()) != 1:
                raise RuntimeError(
                    f"NVFP4 staged block scale {name!r} is not one byte"
                )
            # CPU float8 arithmetic is deliberately outside this path's
            # contract.  Preserve the exact E4M3 payload while the real
            # RoutedExperts loader performs only slicing and copy_ into the
            # uint8 proxy; commit later reinterprets the CUDA destination.
            staged_weight = loaded_weight.view(self._torch.uint8)

        source = StagedSource(name, layer, key, suffix, staged_weight)
        self._pending = source
        return source

    def destination(
        self,
        source: StagedSource,
        mapped_name: str,
        actual_parameter: Any,
    ) -> Any:
        """Return a CPU proxy while retaining the actual loader callable."""

        if source is not self._pending or self._active is None:
            raise RuntimeError("Staged NVFP4 source is not the active source")
        basename = mapped_name.rsplit(".", 1)[-1]
        if basename not in _PARAMETER_ORDER:
            raise RuntimeError(
                f"Unexpected staged NVFP4 destination parameter: {mapped_name!r}"
            )
        reviewed = self._eligible_parameters[source.layer][basename]
        if actual_parameter is not reviewed:
            raise RuntimeError(
                f"Staged NVFP4 destination identity drifted for {mapped_name!r}"
            )

        staged = self._active.parameters.get(basename)
        if staged is None:
            raw_bytes = basename in _RAW_BYTE_PARAMETERS
            dtype = self._torch.uint8 if raw_bytes else actual_parameter.dtype
            host = self._torch.empty(
                tuple(actual_parameter.shape), dtype=dtype, device="cpu"
            )
            proxy = self._torch.nn.Parameter(host, requires_grad=False)
            # RoutedExperts.weight_loader consults these optional attributes.
            # Preserve them without copying hooks or the CUDA storage itself.
            for attr in (
                "is_transposed",
                "use_bitsandbytes_4bit",
                "quant_method",
                "load_full_w2",
            ):
                if hasattr(actual_parameter, attr):
                    setattr(proxy, attr, getattr(actual_parameter, attr))
            staged = _StagedParameter(
                mapped_name, basename, actual_parameter, proxy, raw_bytes
            )
            self._active.parameters[basename] = staged
        elif staged.actual is not actual_parameter:
            raise RuntimeError(
                f"Multiple actual parameters mapped to staged slot {basename!r}"
            )
        return staged.proxy

    def complete_source(self, source: StagedSource) -> None:
        """Record one source and commit only after the complete layer is staged."""

        if source is not self._pending or self._active is None:
            raise RuntimeError("Staged NVFP4 source completion is out of order")
        self._active.seen.add(source.key)
        self._total_source_tensors += 1
        self._pending = None
        if self._active.seen == self._active.expected_keys:
            self._commit_active_layer()

    def _commit_active_layer(self) -> None:
        active = self._active
        if active is None:
            raise RuntimeError("No active NVFP4 layer to commit")
        parameter_names = set(active.parameters)
        expected_names = set(_PARAMETER_ORDER)
        if parameter_names != expected_names:
            missing = sorted(expected_names - parameter_names)
            unexpected = sorted(parameter_names - expected_names)
            raise RuntimeError(
                "NVFP4 staged parameter set is incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        staged_bytes = sum(
            _tensor_bytes(active.parameters[name].proxy)
            for name in _PARAMETER_ORDER
        )
        if staged_bytes != self._expected_stage_bytes:
            raise RuntimeError(
                "NVFP4 staged allocation drifted from its reviewed contract: "
                f"observed {staged_bytes}, expected {self._expected_stage_bytes}"
            )

        commit_started = time.perf_counter()
        commit_calls = 0
        for basename in _PARAMETER_ORDER:
            staged = active.parameters[basename]
            source = staged.proxy.data
            destination = staged.actual.data
            if staged.raw_bytes:
                destination = destination.view(self._torch.uint8)
            # Default blocking semantics keep the CPU buffer alive until the
            # transfer is complete and avoid a second CUDA-sized allocation.
            destination.copy_(source)
            commit_calls += 1
        if commit_calls != self._expected_commit_calls:
            raise RuntimeError(
                "NVFP4 staged commit call count drifted: "
                f"observed {commit_calls}, expected {self._expected_commit_calls}"
            )

        elapsed = time.perf_counter() - active.started_at
        commit_elapsed = time.perf_counter() - commit_started
        logger.info(
            "NVFP4_LAYER_STAGED event=layer_commit layer=%d tensors=%d "
            "bytes=%d copies=%d layer_seconds=%.6f commit_seconds=%.6f",
            active.layer,
            len(active.seen),
            staged_bytes,
            commit_calls,
            elapsed,
            commit_elapsed,
        )
        self._total_commit_calls += commit_calls
        self._completed_layers.add(active.layer)
        self._active = None

    def finish(self) -> None:
        """Fail closed on partial or missing local PP layers."""

        if self._pending is not None:
            raise RuntimeError(
                f"NVFP4 staged source {self._pending.name!r} was not completed"
            )
        if self._active is not None:
            missing = sorted(self._active.expected_keys - self._active.seen)
            raise RuntimeError(
                f"NVFP4 staged layer {self._active.layer} is incomplete; "
                f"missing {len(missing)} tensors, first={missing[:3]}"
            )
        expected_layers = set(self._eligible_parameters)
        if self._completed_layers != expected_layers:
            missing_layers = sorted(expected_layers - self._completed_layers)
            raise RuntimeError(
                f"NVFP4 staged load missed local PP layers: {missing_layers}"
            )
        expected_sources = len(expected_layers) * len(self._expected_source_keys)
        expected_commits = len(expected_layers) * self._expected_commit_calls
        if self._total_source_tensors != expected_sources:
            raise RuntimeError(
                "NVFP4 staged source count drifted: "
                f"observed {self._total_source_tensors}, expected {expected_sources}"
            )
        if self._total_commit_calls != expected_commits:
            raise RuntimeError(
                "NVFP4 staged aggregate commit count drifted: "
                f"observed {self._total_commit_calls}, expected {expected_commits}"
            )
        logger.info(
            "NVFP4_LAYER_STAGED event=complete layers=%d tensors=%d copies=%d "
            "elapsed_seconds=%.6f",
            len(self._completed_layers),
            self._total_source_tensors,
            self._total_commit_calls,
            time.perf_counter() - self._started_at,
        )

    def abort(self) -> None:
        """Release active host staging storage after an outer load failure."""

        if self._active is not None:
            for staged in self._active.parameters.values():
                try:
                    source = staged.proxy.data
                    staged.proxy.data = self._torch.empty(
                        (0,), dtype=source.dtype, device="cpu"
                    )
                except Exception:
                    logger.exception(
                        "Failed to release NVFP4 staged proxy %s during abort",
                        staged.name,
                    )
            self._active.parameters.clear()
            self._active.seen.clear()
        self._pending = None
        self._active = None


class Nvfp4LayerStagedLoadSession:
    """Own one stager across every nested target-model loader invocation."""

    def __init__(self) -> None:
        self._active = False
        self._stager: Nvfp4LayerStager | None = None
        self._staged_requested = False
        self._nested_load_calls = 0
        self._finish_calls = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def nested_load_calls(self) -> int:
        return self._nested_load_calls

    @property
    def finish_calls(self) -> int:
        return self._finish_calls

    def begin(
        self,
        stager: Nvfp4LayerStager | None,
        *,
        staged_requested: bool,
    ) -> None:
        if self._active:
            raise RuntimeError("NVFP4 staged-load session reentry is not allowed")
        if (stager is not None) != staged_requested:
            raise RuntimeError(
                "NVFP4 staged-load session request/stager contract drifted"
            )
        self._active = True
        self._stager = stager
        self._staged_requested = staged_requested
        self._nested_load_calls = 0

    def stager_for_nested_load(
        self,
        *,
        staged_requested: bool,
    ) -> Nvfp4LayerStager | None:
        if not self._active:
            if staged_requested:
                raise RuntimeError(
                    "NVFP4 staged nested load requires an active outer session"
                )
            return None
        if staged_requested != self._staged_requested:
            raise RuntimeError(
                "NVFP4 staged-load opt-in changed during an active session"
            )
        self._nested_load_calls += 1
        return self._stager

    def finish(self) -> None:
        if not self._active:
            raise RuntimeError("No active NVFP4 staged-load session to finish")
        if self._stager is not None:
            self._stager.finish()
        self._finish_calls += 1
        self._reset()

    def abort(self) -> None:
        """Drop any active CPU proxies after an outer load failure."""

        try:
            if self._stager is not None:
                self._stager.abort()
        except Exception:
            logger.exception("Failed to abort the active NVFP4 layer stager")
        finally:
            self._reset()

    def _reset(self) -> None:
        self._active = False
        self._stager = None
        self._staged_requested = False


def maybe_create_nvfp4_layer_stager(
    *,
    torch_module: Any,
    params_dict: dict[str, Any],
    expert_mapping_index: Any,
    start_layer: int,
    end_layer: int,
    num_hidden_layers: int,
    num_routed_experts: int,
    tp_size: int,
    use_mega_moe: bool,
    enable_expert_parallel: bool,
    num_redundant_experts: int,
    load_format: str,
    quant_config: Any,
    environ: dict[str, str] | None = None,
) -> Nvfp4LayerStager | None:
    """Validate the exact reviewed contract before allocating a host buffer."""

    if not staged_load_requested(environ):
        return None
    source = os.environ if environ is None else environ
    # Accept either the configured string or an enum-style ``LoadFormat.X``
    # representation, but compare the effective terminal token exactly.
    effective_load_format = str(load_format).lower().rsplit(".", 1)[-1]
    if (
        effective_load_format == ROCE_LOAD_FORMAT
        or source.get("DSPARK_WEIGHT_LOAD_FORMAT", "auto").lower()
        == ROCE_LOAD_FORMAT
    ):
        raise RuntimeError("NVFP4 layer staging does not support roce_tp loading")
    if use_mega_moe:
        raise RuntimeError("NVFP4 layer staging does not support MegaMoE")
    if enable_expert_parallel or num_redundant_experts != 0:
        raise RuntimeError(
            "NVFP4 layer staging requires EP/EPLB disabled with zero "
            f"redundant experts; got enable_expert_parallel="
            f"{enable_expert_parallel}, num_redundant_experts="
            f"{num_redundant_experts}"
        )
    quant_config_name = (
        quant_config.__class__.__name__ if quant_config is not None else "None"
    )
    expert_dtype = getattr(quant_config, "expert_dtype", None)
    moe_quant_algo = getattr(quant_config, "moe_quant_algo", None)
    target_num_hidden_layers = getattr(
        quant_config, "target_num_hidden_layers", None
    )
    if (
        quant_config_name != "DeepseekV4FP8Config"
        or expert_dtype != "fp4"
        or moe_quant_algo != "NVFP4"
        or target_num_hidden_layers != EXPECTED_NVFP4_LAYERS
    ):
        raise RuntimeError(
            "NVFP4 layer staging requires the DeepseekV4FP8Config target "
            "wrapper with fp4/NVFP4 routed experts and 43 target layers; "
            f"got {quant_config_name}/{expert_dtype}/{moe_quant_algo}/"
            f"{target_num_hidden_layers}"
        )
    if num_hidden_layers != EXPECTED_NVFP4_LAYERS:
        raise RuntimeError(
            f"NVFP4 staged contract requires {EXPECTED_NVFP4_LAYERS} layers; "
            f"got {num_hidden_layers}"
        )
    if num_routed_experts != EXPECTED_NVFP4_EXPERTS:
        raise RuntimeError(
            f"NVFP4 staged contract requires {EXPECTED_NVFP4_EXPERTS} experts; "
            f"got {num_routed_experts}"
        )
    if tp_size != 2:
        raise RuntimeError(f"NVFP4 staged contract requires TP=2; got TP={tp_size}")
    if not bool(expert_mapping_index.safe):
        raise RuntimeError("NVFP4 staged load requires a safe expert mapping index")

    mapping_keys = tuple(expert_mapping_index.mappings)
    expected_mapping_keys = frozenset(
        f"experts.{expert}.{projection}."
        for expert in range(EXPECTED_NVFP4_EXPERTS)
        for projection in ("w1", "w2", "w3")
    )
    observed_mapping_keys = frozenset(mapping_keys)
    if observed_mapping_keys != expected_mapping_keys:
        missing = sorted(expected_mapping_keys - observed_mapping_keys)
        unexpected = sorted(observed_mapping_keys - expected_mapping_keys)
        raise RuntimeError(
            "NVFP4 staged mapping-key set drifted: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    expected_source_keys = frozenset(
        f"{mapping_key}{suffix}"
        for mapping_key in mapping_keys
        for suffix in _CHECKPOINT_SUFFIXES
    )
    if len(expected_source_keys) != EXPECTED_TENSORS_PER_LAYER:
        raise RuntimeError(
            "NVFP4 staged source contract drifted: "
            f"observed {len(expected_source_keys)}, "
            f"expected {EXPECTED_TENSORS_PER_LAYER}"
        )

    expected_dtypes = _expected_parameter_dtypes(torch_module)
    parameter_relative_names = _reviewed_parameter_relative_names(
        expert_mapping_index
    )
    expected_shapes = _expected_parameter_shapes()
    eligible_parameters: dict[int, dict[str, Any]] = {}
    for layer in range(start_layer, end_layer):
        parameters: dict[str, Any] = {}
        for basename in _PARAMETER_ORDER:
            name = f"layers.{layer}.ffn.{parameter_relative_names[basename]}"
            if name not in params_dict:
                raise RuntimeError(f"Missing reviewed staged NVFP4 parameter {name!r}")
            parameter = params_dict[name]
            if _device_type(parameter) != "cuda":
                raise RuntimeError(
                    f"Staged NVFP4 parameter {name!r} must be on CUDA; "
                    f"got {_device_type(parameter)!r}"
                )
            observed_shape = tuple(parameter.shape)
            if observed_shape != expected_shapes[basename]:
                raise RuntimeError(
                    f"Staged NVFP4 parameter {name!r} has shape "
                    f"{observed_shape}; expected {expected_shapes[basename]}"
                )
            if parameter.dtype != expected_dtypes[basename]:
                raise RuntimeError(
                    f"Staged NVFP4 parameter {name!r} has dtype "
                    f"{parameter.dtype}; expected {expected_dtypes[basename]}"
                )
            if (
                basename in _RAW_BYTE_PARAMETERS
                and int(parameter.element_size()) != 1
            ):
                raise RuntimeError(
                    f"Staged NVFP4 raw-byte parameter {name!r} must use "
                    "one-byte E4M3 storage"
                )
            parameters[basename] = parameter
        observed_bytes = sum(_tensor_bytes(p) for p in parameters.values())
        if observed_bytes != EXPECTED_STAGE_BYTES:
            raise RuntimeError(
                f"NVFP4 layer {layer} parameter bytes drifted: observed "
                f"{observed_bytes}, expected {EXPECTED_STAGE_BYTES}"
            )
        eligible_parameters[layer] = parameters
    if not eligible_parameters:
        raise RuntimeError("NVFP4 staged load has no local PP layers")

    logger.info(
        "NVFP4_LAYER_STAGED event=enabled local_layers=%d "
        "tensors_per_layer=%d bytes_per_layer=%d copies_per_layer=%d",
        len(eligible_parameters),
        len(expected_source_keys),
        EXPECTED_STAGE_BYTES,
        EXPECTED_COMMIT_CALLS,
    )
    return Nvfp4LayerStager(
        torch_module=torch_module,
        eligible_parameters=eligible_parameters,
        expected_source_keys=expected_source_keys,
    )
