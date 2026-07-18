#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-model-load parity probe for DeepSeek V4 NVFP4 CPU staging.

The default CPU probe is intended to run inside the assembled candidate image
while serving remains live.  It binds the pinned vLLM ``RoutedExperts`` loader
to a tiny, deterministic ModelOpt-like harness, then dispatches the same source
tensors directly and through ``Nvfp4LayerStager`` for TP ranks zero and one.

No checkpoint or model configuration is opened.  ``--device cuda`` is an
optional raw-commit check; the mandatory proof is CPU-only and uses a few KiB.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import textwrap
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Sequence


SCHEMA_VERSION = 1
TP_SIZE = 2
EXPERTS = 2
HIDDEN = 8
INTERMEDIATE_PER_RANK = 4
PACKED_INTERMEDIATE_PER_RANK = 3
W13_SCALE_COLUMNS = 3
W2_SCALE_COLUMNS_PER_RANK = 2
LAYER = 0
OFFICIAL_LAYERS = 43
OFFICIAL_EXPERTS = 256
OFFICIAL_TENSORS_PER_LAYER = 3_072
OFFICIAL_STAGE_BYTES = 1_811_945_472

PARAMETER_ORDER = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
)
RAW_BLOCK_SCALE_PARAMETERS = frozenset(
    ("w13_weight_scale", "w2_weight_scale")
)
PACKED_PARAMETERS = frozenset(("w13_weight", "w2_weight"))
FP32_PARAMETERS = frozenset(
    (
        "w13_weight_scale_2",
        "w2_weight_scale_2",
        "w13_input_scale",
        "w2_input_scale",
    )
)
SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")
PROJECTIONS = ("w1", "w2", "w3")

# Exact main-target checkpoint header contract, proven independently from all
# 46 official shards.  MTP uses a different representation and is deliberately
# outside this probe.
MAIN_TARGET_SOURCE_CONTRACT = {
    "weight": {"dtype": "torch.uint8", "rank": 2},
    "weight_scale": {"dtype": "torch.float8_e4m3fn", "rank": 2},
    "weight_scale_2": {"dtype": "torch.float32", "rank": 0, "shape": []},
    "input_scale": {"dtype": "torch.float32", "rank": 0, "shape": []},
}
OFFICIAL_CHECKPOINT_SOURCE_SHAPES = {
    (projection, suffix): (
        (2_048, 2_048)
        if suffix == "weight" and projection in ("w1", "w3")
        else (4_096, 1_024)
        if suffix == "weight"
        else (2_048, 256)
        if suffix == "weight_scale" and projection in ("w1", "w3")
        else (4_096, 128)
        if suffix == "weight_scale"
        else ()
    )
    for projection in PROJECTIONS
    for suffix in SUFFIXES
}
CHECKPOINT_LAYOUT_CONTRACT = {
    "main_target": {
        "checkpoint_shards": 46,
        "layers": OFFICIAL_LAYERS,
        "tensors_per_layer": OFFICIAL_TENSORS_PER_LAYER,
        "layers_contiguous": True,
        "each_routed_layer_wholly_in_one_shard": True,
        "layer_to_shard": {
            str(layer): f"model-{layer + 2:05d}-of-00046.safetensors"
            for layer in range(OFFICIAL_LAYERS)
        },
        "source_tensors": MAIN_TARGET_SOURCE_CONTRACT,
    },
    "mtp_excluded": {
        "total_tensors": 1_575,
        "routed_tensors": 1_536,
        "shard": "model-00046-of-00046.safetensors",
        "weight_dtype": "torch.int8",
        "weight_scale_dtype": "torch.float8_e8m0fnu",
        "weight_shapes": {
            "w1_w3": [2_048, 2_048],
            "w2": [4_096, 1_024],
        },
        "weight_scale_shapes": {
            "w1_w3": [2_048, 128],
            "w2": [4_096, 64],
        },
    },
}

# Normalized inspect-source SHA-256 values from pinned vLLM 752a3a504485790a
# and the reviewed staging helper.  Any source drift requires a fresh review.
EXPECTED_ROUTED_SOURCE_SHA256 = {
    "weight_loader": "2f20f887e1293711c148bde475691568eed0c5c4e95f37d53e43a0cb1ebc4811",
    "_map_global_expert_id_to_local_expert_id": "5b4f0d08bd2b86c8e70eaffb707859afe6e74ee9dd337ef1d878df294dc3e0fc",
    "_to_scalar": "598d082dfc003aa9136780e0ab450c36a9a34db7b71132253f5aaf8c13834d95",
    "_load_per_tensor_weight_scale": "9ccf24fb3ba6d482ad1b5ecbc4880c0e289858da7a2179d31cdd50e2c41cd6ec",
    "_load_model_weight_or_group_weight_scale": "60b36428571f3a280ae6193ae7fcf474793917d5b9062060ddb44ef54d15ceb7",
    "_load_w13": "184c71b2290ee5c438c5ad43ea999b8a551bdebc9c56725e309d16f960234d68",
    "_load_w2": "5de40ccc3e921e49eab35c5d7f6a0823eed94f6cdea95315829777f99090b6c2",
    "_load_single_value": "e90c3096e9456ede054d52fc9cf42048127b983e419d2f070f206ab9e3b9c5cb",
    "_get_hidden_dim": "50e0b2112d51993e5b41f305f27055426f5551af287362572b6c4a25639ee6ed",
    "_narrow_expert_data_for_padding": "5560edc5321ed0115dfaa4f1cbded5029bc19b1ab13dd08267af7e7773ce4997",
}
EXPECTED_STAGER_SOURCE_SHA256 = {
    "__init__": "3355f7700c9f57be16beef0b956e4a11e0ba33c1223e891ebb6e75e9eb9d60b0",
    "begin_source": "2e21bd619b7e52513e398a52890e8950c52bc5fd2aba918ec4843c1d3360c492",
    "destination": "fac2ea53e6ebb5cf1ed5ae3a6e896d20527e78d47a08bb102ee8394b87c39642",
    "complete_source": "897eab1b9737b51cba4103f9e641422bf995322527220b6183f110a58fd17883",
    "_commit_active_layer": "3a17a7c696c56a10566df91ff5859c091b5abeaa5445e1504193402f4e46d3b7",
    "finish": "53c071f9a95d43b45a6ce857f35e699e3563c8ac08047409965abe94753c9739",
    "abort": "edb1f76910dceee5f6660c4fa421d82a947862632d4fc4f21153efbdf326df21",
}
EXPECTED_SESSION_SOURCE_SHA256 = {
    "__init__": "b954f631e4e929cb679ccf719744d9f3839990ae1cede4467d8d80078bdf4713",
    "begin": "e129a8991c917921cdf7ec1fe1c165f09fc50e41b1154f2ffa591f82ea8422c4",
    "stager_for_nested_load": "d09928906bab1872ca146b18e3c6d2f6d869ab7b000f0487ca916b4e4ffdc297",
    "finish": "bd5516ee2742562400a8b2c654a65dcc9edc9d7c4a6c279516fa06d464ef336d",
    "abort": "b7363e52f72050478b7ddc06b179da09cd2f4ee66d71fe003d1799cec2faceb5",
    "_reset": "32bd9307b217e908e0aeca112de5bd55a41918aa44d506965215443f25dfd0df",
}
EXPECTED_MODEL_SOURCE_SHA256 = {
    "__init__": "8842833cffdb66beee2f9ec88af270f39e467db5ffc7b8927aee70d951aa5f67",
    "begin_nvfp4_layer_staged_load": "db4c41b733d7cfeb999a34cccf216bdfd099647beb9368d334f8bb3b549349b9",
    "finish_nvfp4_layer_staged_load": "153c4b7ae5e912382457f4dbf8f13a21e6fbe1f5fa8ac9a7d7e0cfa4cc6598ab",
    "abort_nvfp4_layer_staged_load": "db2a8e4f45e565e330cb42f39824e6c47590a266d65a10d9658b1b66a563d153",
    "load_weights": "7dbff6a7bc0a4ecfb3d1b84a2014bb59478956e30d46337a0bef0c8480a58941",
}
EXPECTED_CAUSAL_MODEL_SOURCE_SHA256 = {
    "load_weights": "6db76fea1575c92e6baa571433d36ca5da715da1d47b97ac0b7604dbc1d8863c",
}
EXPECTED_STAGER_FACTORY_SOURCE_SHA256 = (
    "e0f14123b9d205d7b9239671b949808e74a36cbe0c4a0244317bec2dada42a50"
)
EXPECTED_STAGER_HELPER_SOURCE_SHA256 = {
    "_expected_checkpoint_shape": (
        "7fe6dd6c9c4479a36393277e5f4a39d73f2d3e50e4610f7e391e28456e6e79df"
    ),
}
EXPECTED_DRAFT_BACKEND_SOURCE_SHA256 = {
    "_scope_prepared_nvfp4_target_backend": (
        "44cc99e153381347a1705accb603f5d915b60f1bf779397639387e45550dbd8a"
    ),
    "_get_priority_backends": (
        "e1e98d64b484cd682455102d2ecb1f816623904538c74784887d3504d9014183"
    ),
    "convert_weight_to_mxfp4_moe_kernel_format": (
        "1a63cdb9aa0209ce35b1fc0104df0087c0881e3ef4127aadca9ad6ea5b220094"
    ),
}
EXPECTED_DRAFT_METHOD_SOURCE_SHA256 = {
    "_setup_kernel": (
        "54890a5ec8d8613675f6a45c786801bb1365e9a22115a661437a7a2acc1236fe"
    ),
    "process_weights_after_loading": (
        "b9a6f5f980e63031e9c05ea36e941291ed088f08ba5975e94f42f90290324994"
    ),
}
EXPECTED_PARAM_ATTRIBUTE_CHAINS = {
    "weight_loader": [
        "param.data",
        "param.data.device",
        "param.data.ndim",
        "param.data.shape",
    ],
    "_map_global_expert_id_to_local_expert_id": [],
    "_to_scalar": [],
    "_load_per_tensor_weight_scale": ["param.data"],
    "_load_model_weight_or_group_weight_scale": [],
    "_load_w13": [],
    "_load_w2": [],
    "_load_single_value": ["param.data"],
    "_get_hidden_dim": [],
    "_narrow_expert_data_for_padding": [],
}
EXPECTED_OPTIONAL_PARAM_ATTRS = [
    "is_transposed",
    "load_full_w2",
    "quant_method",
    "use_bitsandbytes_4bit",
]
EXPECTED_PROXY_COPIED_ATTRS = [
    "is_transposed",
    "load_full_w2",
    "quant_method",
    "use_bitsandbytes_4bit",
]
EXPECTED_DEVICE_BRANCHES = ["loaded_weight.to(param.data.device)"]


@dataclass(frozen=True)
class _ExpertMatch:
    layer: int
    mapping_key: str
    suffix: str
    projection: str


@dataclass(frozen=True)
class _ShapeOnlyCudaDevice:
    type: str = "cuda"


@dataclass(frozen=True)
class _ShapeOnlyCudaParameter:
    """Allocation-free descriptor for the real factory's metadata checks."""

    shape: tuple[int, ...]
    dtype: Any
    element_bytes: int
    device: Any = _ShapeOnlyCudaDevice()

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result

    def element_size(self) -> int:
        return self.element_bytes


class _ParityQuantConfig:
    @staticmethod
    def get_name() -> str:
        return "deepseek_v4_nvfp4_parity_probe"


class _ModelOptNvFp4ParityMethod:
    """Minimal state read by the real ModelOpt branch of weight_loader."""

    use_global_sf = True

    @staticmethod
    def uses_weight_scale_2_pattern() -> bool:
        return True


class _IdentityExpertMap:
    @staticmethod
    def map_global_to_local(expert_id: int) -> int:
        return expert_id if 0 <= expert_id < EXPERTS else -1


def _normalized_callable_source(callable_object: Any) -> str:
    raw = textwrap.dedent(inspect.getsource(callable_object))
    tree = ast.parse(raw)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        raise RuntimeError("inspected callable did not contain exactly one function")
    segment = ast.get_source_segment(raw, functions[0])
    if segment is None:
        raise RuntimeError("could not recover normalized inspected source")
    return textwrap.dedent(segment).strip() + "\n"


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _attribute_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join((node.id, *reversed(parts)))


def _param_source_contract(source: str) -> dict[str, list[str]]:
    tree = ast.parse(source)
    chains = sorted(
        {
            chain
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and (chain := _attribute_chain(node)) is not None
            and chain.startswith("param.")
        }
    )
    optional: set[str] = set()
    device_branches: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "param"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            optional.add(node.args[1].value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _attribute_chain(node.func) == "loaded_weight.to"
            and any(
                _attribute_chain(candidate) == "param.data.device"
                for argument in node.args
                for candidate in ast.walk(argument)
                if isinstance(candidate, ast.Attribute)
            )
        ):
            device_branches.add(ast.unparse(node))
    return {
        "attribute_chains": chains,
        "optional_attrs": sorted(optional),
        "device_branches": sorted(device_branches),
    }


def _proxy_copied_attrs(source: str) -> list[str]:
    tree = ast.parse(source)
    copied: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        loop_name = node.target.id
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        values = [
            item.value
            for item in node.iter.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        if len(values) != len(node.iter.elts):
            continue
        copies_loop_attr = any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "setattr"
            and len(candidate.args) >= 2
            and isinstance(candidate.args[1], ast.Name)
            and candidate.args[1].id == loop_name
            for statement in node.body
            for candidate in ast.walk(statement)
        )
        if copies_loop_attr:
            copied.update(values)
    return sorted(copied)


def _audit_runtime_sources(
    routed_experts_class: Any,
    stager_class: Any,
    staged_session_class: Any,
    stager_factory: Any,
    expected_checkpoint_shape: Any,
    model_class: Any,
    causal_model_class: Any,
    prepared_backend_scope: Any,
    draft_backend_priority: Any,
    draft_converter: Any,
    draft_method_class: Any,
) -> dict[str, Any]:
    routed: dict[str, Any] = {}
    all_param_contracts: dict[str, Any] = {}
    for name, expected_sha in EXPECTED_ROUTED_SOURCE_SHA256.items():
        source = _normalized_callable_source(getattr(routed_experts_class, name))
        observed_sha = _source_sha256(source)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"pinned RoutedExperts.{name} source drifted: "
                f"observed {observed_sha}, expected {expected_sha}"
            )
        contract = _param_source_contract(source)
        expected_chains = EXPECTED_PARAM_ATTRIBUTE_CHAINS[name]
        if contract["attribute_chains"] != expected_chains:
            raise RuntimeError(
                f"RoutedExperts.{name} param attribute contract drifted: "
                f"{contract['attribute_chains']!r} != {expected_chains!r}"
            )
        routed[name] = {"sha256": observed_sha, **contract}
        all_param_contracts[name] = contract["attribute_chains"]

    loader_contract = routed["weight_loader"]
    if loader_contract["optional_attrs"] != EXPECTED_OPTIONAL_PARAM_ATTRS:
        raise RuntimeError(
            "RoutedExperts.weight_loader optional param attributes drifted: "
            f"{loader_contract['optional_attrs']!r}"
        )
    if loader_contract["device_branches"] != EXPECTED_DEVICE_BRANCHES:
        raise RuntimeError(
            "RoutedExperts.weight_loader device branch drifted: "
            f"{loader_contract['device_branches']!r}"
        )

    staged: dict[str, Any] = {}
    destination_source = ""
    for name, expected_sha in EXPECTED_STAGER_SOURCE_SHA256.items():
        source = _normalized_callable_source(getattr(stager_class, name))
        observed_sha = _source_sha256(source)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"Nvfp4LayerStager.{name} source drifted: "
                f"observed {observed_sha}, expected {expected_sha}"
            )
        staged[name] = {"sha256": observed_sha}
        if name == "destination":
            destination_source = source

    copied_attrs = _proxy_copied_attrs(destination_source)
    if copied_attrs != EXPECTED_PROXY_COPIED_ATTRS:
        raise RuntimeError(
            "Nvfp4LayerStager proxy attribute contract drifted: "
            f"{copied_attrs!r} != {EXPECTED_PROXY_COPIED_ATTRS!r}"
        )
    factory_source = _normalized_callable_source(stager_factory)
    factory_sha = _source_sha256(factory_source)
    if factory_sha != EXPECTED_STAGER_FACTORY_SOURCE_SHA256:
        raise RuntimeError(
            "maybe_create_nvfp4_layer_stager source drifted: "
            f"observed {factory_sha}, expected "
            f"{EXPECTED_STAGER_FACTORY_SOURCE_SHA256}"
        )
    helper_source = _normalized_callable_source(expected_checkpoint_shape)
    helper_sha = _source_sha256(helper_source)
    expected_helper_sha = EXPECTED_STAGER_HELPER_SOURCE_SHA256[
        "_expected_checkpoint_shape"
    ]
    if helper_sha != expected_helper_sha:
        raise RuntimeError(
            "_expected_checkpoint_shape source drifted: observed "
            f"{helper_sha}, expected {expected_helper_sha}"
        )
    observed_checkpoint_shapes = {
        (projection, suffix): tuple(expected_checkpoint_shape(projection, suffix))
        for projection in PROJECTIONS
        for suffix in SUFFIXES
    }
    if observed_checkpoint_shapes != OFFICIAL_CHECKPOINT_SOURCE_SHAPES:
        raise RuntimeError(
            "official NVFP4 checkpoint source-shape contract drifted: "
            f"{observed_checkpoint_shapes!r}"
        )
    session_methods: dict[str, Any] = {}
    for name, expected_sha in EXPECTED_SESSION_SOURCE_SHA256.items():
        source = _normalized_callable_source(getattr(staged_session_class, name))
        observed_sha = _source_sha256(source)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"Nvfp4LayerStagedLoadSession.{name} source drifted: "
                f"observed {observed_sha}, expected {expected_sha}"
            )
        session_methods[name] = {"sha256": observed_sha}

    model_methods: dict[str, Any] = {}
    for name, expected_sha in EXPECTED_MODEL_SOURCE_SHA256.items():
        source = _normalized_callable_source(getattr(model_class, name))
        observed_sha = _source_sha256(source)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"DeepseekV4Model.{name} source drifted: observed "
                f"{observed_sha}, expected {expected_sha}"
            )
        model_methods[name] = {"sha256": observed_sha}

    causal_model_methods: dict[str, Any] = {}
    for name, expected_sha in EXPECTED_CAUSAL_MODEL_SOURCE_SHA256.items():
        source = _normalized_callable_source(getattr(causal_model_class, name))
        observed_sha = _source_sha256(source)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"DeepseekV4ForCausalLM.{name} source drifted: observed "
                f"{observed_sha}, expected {expected_sha}"
            )
        causal_model_methods[name] = {"sha256": observed_sha}

    draft_backend_sources: dict[str, Any] = {}
    for name, candidate in (
        ("_scope_prepared_nvfp4_target_backend", prepared_backend_scope),
        ("_get_priority_backends", draft_backend_priority),
        ("convert_weight_to_mxfp4_moe_kernel_format", draft_converter),
    ):
        source = _normalized_callable_source(candidate)
        observed_sha = _source_sha256(source)
        expected_sha = EXPECTED_DRAFT_BACKEND_SOURCE_SHA256[name]
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"DeepSeek-V4 draft backend source drifted for {name}: "
                f"observed {observed_sha}, expected {expected_sha}"
            )
        draft_backend_sources[name] = {"sha256": observed_sha}
    for name, expected_sha in EXPECTED_DRAFT_METHOD_SOURCE_SHA256.items():
        source = _normalized_callable_source(getattr(draft_method_class, name))
        observed_sha = _source_sha256(source)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"Mxfp4MoEMethod.{name} source drifted: observed "
                f"{observed_sha}, expected {expected_sha}"
            )
        draft_backend_sources[f"Mxfp4MoEMethod.{name}"] = {
            "sha256": observed_sha
        }
    return {
        "passed": True,
        "routed_experts_module": routed_experts_class.__module__,
        "routed_experts_file": str(inspect.getsourcefile(routed_experts_class)),
        "stager_module": stager_class.__module__,
        "stager_file": str(inspect.getsourcefile(stager_class)),
        "routed_methods": routed,
        "stager_methods": staged,
        "staged_session_methods": session_methods,
        "model_lifecycle_methods": model_methods,
        "causal_model_lifecycle_methods": causal_model_methods,
        "draft_backend_sources": draft_backend_sources,
        "stager_factory": {
            "name": stager_factory.__name__,
            "sha256": factory_sha,
        },
        "stager_helpers": {
            "_expected_checkpoint_shape": {
                "sha256": helper_sha,
                "shapes": {
                    f"{projection}.{suffix}": list(shape)
                    for (projection, suffix), shape in sorted(
                        observed_checkpoint_shapes.items()
                    )
                },
            }
        },
        "param_attribute_chains": all_param_contracts,
        "optional_param_attrs": loader_contract["optional_attrs"],
        "device_branches": loader_contract["device_branches"],
        "proxy_copied_attrs": copied_attrs,
        "proxy_inherent_attrs": ["data"],
        "all_optional_param_attrs_available": (
            copied_attrs == loader_contract["optional_attrs"]
        ),
    }


def _run_draft_postload_backend_proof(
    torch: Any,
    *,
    prepared_backend_scope: Any,
    draft_backend_priority: Any,
    draft_converter: Any,
    backend_enum: Any,
    draft_method_class: Any,
) -> dict[str, Any]:
    """Exercise the real native-MXFP4 conversion branch on tiny CPU tensors.

    FlashInfer's permutation helpers are GPU-only implementation details, so
    this CPU gate replaces only those two primitives with shape-preserving
    deterministic equivalents.  The actual pinned vLLM converter still owns
    backend dispatch, w1/w3 reordering, tensor views, reshape contracts, and
    the old unsupported-CUTLASS rejection.  This is the free pre-outage proof
    for the exact failure boundary observed in Phase C.
    """

    target = SimpleNamespace(moe_config=SimpleNamespace(moe_backend="auto"))
    draft = SimpleNamespace(moe_config=SimpleNamespace(moe_backend="auto"))
    flag = "VLLM_DSV4_NVFP4_CUTLASS_PREPARED_LOAD"
    old_flag = os.environ.get(flag)
    os.environ[flag] = "1"
    try:
        runner_backend = prepared_backend_scope(target)
    finally:
        if old_flag is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = old_flag
    if runner_backend != "auto":
        raise RuntimeError("prepared target scope did not observe runner auto")
    if target.moe_config.moe_backend != "flashinfer_cutlass":
        raise RuntimeError("prepared target was not scoped to CUTLASS")
    if draft.moe_config.moe_backend != "auto":
        raise RuntimeError("native draft runner backend was mutated")

    priority = list(draft_backend_priority())
    intended = backend_enum.FLASHINFER_TRTLLM_MXFP4_MXFP8
    if not priority or priority[0] != intended:
        raise RuntimeError(
            "native DeepSeek-V4 MXFP4 auto priority no longer begins with "
            "FLASHINFER_TRTLLM_MXFP4_MXFP8"
        )

    fp4_quantization = importlib.import_module("flashinfer.fp4_quantization")
    fused_moe_core = importlib.import_module("flashinfer.fused_moe.core")
    real_interleave = fp4_quantization.nvfp4_block_scale_interleave
    real_permute = fused_moe_core.get_w2_permute_indices_with_cache

    def cpu_interleave(value: Any) -> Any:
        return value.contiguous()

    def cpu_permute(
        cache: dict[Any, Any],
        value: Any,
        epilogue_tile_m: int,
        num_elts_per_sf: int | None = None,
    ) -> Any:
        key = (tuple(value.shape), epilogue_tile_m, num_elts_per_sf)
        if key not in cache:
            cache[key] = torch.arange(value.shape[0], device=value.device)
        return cache[key]

    experts = 1
    hidden = 32
    intermediate = 32
    w1 = torch.full(
        (experts, intermediate, hidden // 2),
        0x11,
        dtype=torch.uint8,
    )
    w3 = torch.full_like(w1, 0x33)
    w13 = torch.cat((w1, w3), dim=1)
    w2 = torch.full(
        (experts, hidden, intermediate // 2),
        0x55,
        dtype=torch.uint8,
    )
    s1 = torch.full((experts, intermediate, 1), 0x22, dtype=torch.uint8)
    s3 = torch.full_like(s1, 0x44)
    w13_scale = torch.cat((s1, s3), dim=1)
    w2_scale = torch.full((experts, hidden, 1), 0x66, dtype=torch.uint8)

    fp4_quantization.nvfp4_block_scale_interleave = cpu_interleave
    fused_moe_core.get_w2_permute_indices_with_cache = cpu_permute
    try:
        draft_layer = torch.nn.Module()
        for name, value in (
            ("w13_weight", w13),
            ("w2_weight", w2),
            ("w13_weight_scale", w13_scale),
            ("w2_weight_scale", w2_scale),
        ):
            draft_layer.register_parameter(
                name, torch.nn.Parameter(value.clone(), requires_grad=False)
            )
        draft_method = draft_method_class.__new__(draft_method_class)
        draft_method.num_experts = experts
        draft_method.intermediate_size = intermediate
        draft_method.hidden_size = hidden
        draft_method.mxfp4_backend = intended
        draft_method.experts_cls = None
        draft_method._cache_permute_indices = {}
        draft_method.moe_quant_config = None
        draft_method.moe_kernel = None
        draft_method.w13_precision_config = None
        draft_method.w2_precision_config = None
        draft_method.process_weights_after_loading(draft_layer)
    finally:
        fp4_quantization.nvfp4_block_scale_interleave = real_interleave
        fused_moe_core.get_w2_permute_indices_with_cache = real_permute

    converted_w13 = draft_layer.w13_weight.data
    converted_w2 = draft_layer.w2_weight.data
    converted_s13 = draft_layer.w13_weight_scale.data
    converted_s2 = draft_layer.w2_weight_scale.data
    raw_s13 = converted_s13.view(torch.uint8)
    raw_s2 = converted_s2.view(torch.uint8)
    checks = {
        "w13_w3_rows_first": bool(
            torch.equal(converted_w13[:, 0::2], torch.full_like(w1, 0x33))
        ),
        "w13_w1_rows_second": bool(
            torch.equal(converted_w13[:, 1::2], torch.full_like(w1, 0x11))
        ),
        "w2_preserved": bool(torch.equal(converted_w2, w2)),
        "w13_scale_w3_rows_first": bool(
            torch.equal(raw_s13[:, 0::2], torch.full_like(s1, 0x44))
        ),
        "w13_scale_w1_rows_second": bool(
            torch.equal(raw_s13[:, 1::2], torch.full_like(s1, 0x22))
        ),
        "w2_scale_preserved": bool(torch.equal(raw_s2, w2_scale)),
        "quant_config_created": draft_method.moe_quant_config is not None,
        "kernel_skipped_for_cpu_probe": draft_method.moe_kernel is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"native draft TRTLLM conversion drifted: {checks!r}")

    old_failure = backend_enum.FLASHINFER_CUTLASS_MXFP4_MXFP8
    try:
        draft_converter(
            mxfp4_backend=old_failure,
            layer=SimpleNamespace(),
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            _cache_permute_indices={},
        )
    except ValueError as error:
        if "Unsupported mxfp4_backend for Mxfp4MoEMethod" not in str(error):
            raise RuntimeError("old draft CUTLASS rejection message drifted") from error
        old_cutlass_rejected = True
    else:
        raise RuntimeError("old unsupported draft CUTLASS path was accepted")

    return {
        "passed": True,
        "runner_backend": runner_backend,
        "prepared_target_backend": target.moe_config.moe_backend,
        "native_draft_backend": draft.moe_config.moe_backend,
        "native_draft_priority": [backend.value for backend in priority],
        "intended_native_draft_backend": intended.value,
        "converter": draft_converter.__name__,
        "converter_executed": True,
        "postload_method": (
            f"{draft_method_class.__name__}.process_weights_after_loading"
        ),
        "postload_method_executed": True,
        "cpu_flashinfer_primitives": "deterministic_shape_preserving_stubs",
        "checks": checks,
        "old_global_cutlass_backend": old_failure.value,
        "old_global_cutlass_rejected": old_cutlass_rejected,
    }


def _official_factory_parameter_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "w13_weight": (OFFICIAL_EXPERTS, 2 * 1_024, 4_096 // 2),
        "w2_weight": (OFFICIAL_EXPERTS, 4_096, 1_024 // 2),
        "w13_weight_scale": (OFFICIAL_EXPERTS, 2 * 1_024, 4_096 // 16),
        "w2_weight_scale": (OFFICIAL_EXPERTS, 4_096, 1_024 // 16),
        "w13_weight_scale_2": (OFFICIAL_EXPERTS, 2),
        "w2_weight_scale_2": (OFFICIAL_EXPERTS,),
        "w13_input_scale": (OFFICIAL_EXPERTS, 2),
        "w2_input_scale": (OFFICIAL_EXPERTS,),
    }


def _run_factory_preflight(
    torch: Any,
    stager_class: Any,
    stager_factory: Any,
) -> dict[str, Any]:
    """Exercise the real 43-layer factory without allocating tensor storage."""

    shapes = _official_factory_parameter_shapes()
    dtypes = {
        "w13_weight": (torch.uint8, 1),
        "w2_weight": (torch.uint8, 1),
        "w13_weight_scale": (torch.float8_e4m3fn, 1),
        "w2_weight_scale": (torch.float8_e4m3fn, 1),
        "w13_weight_scale_2": (torch.float32, 4),
        "w2_weight_scale_2": (torch.float32, 4),
        "w13_input_scale": (torch.float32, 4),
        "w2_input_scale": (torch.float32, 4),
    }
    expected_layer_bytes = sum(
        _ShapeOnlyCudaParameter(shape, dtypes[name][0], dtypes[name][1]).numel()
        * dtypes[name][1]
        for name, shape in shapes.items()
    )
    if expected_layer_bytes != OFFICIAL_STAGE_BYTES:
        raise RuntimeError(
            "shape-only factory byte contract drifted: "
            f"{expected_layer_bytes} != {OFFICIAL_STAGE_BYTES}"
        )

    params: dict[str, _ShapeOnlyCudaParameter] = {}
    for layer in range(OFFICIAL_LAYERS):
        for basename, shape in shapes.items():
            dtype, element_bytes = dtypes[basename]
            name = f"layers.{layer}.ffn.experts.routed_experts.{basename}"
            params[name] = _ShapeOnlyCudaParameter(shape, dtype, element_bytes)

    prefixes = {
        "w1": "experts.routed_experts.w13_",
        "w2": "experts.routed_experts.w2_",
        "w3": "experts.routed_experts.w13_",
    }
    mappings: dict[str, list[tuple[str, str, int, str]]] = {}
    for expert in range(OFFICIAL_EXPERTS):
        for projection, prefix in prefixes.items():
            key = f"experts.{expert}.{projection}."
            mappings[key] = [(prefix, key, expert, projection)]
    if len(mappings) != OFFICIAL_EXPERTS * len(PROJECTIONS):
        raise RuntimeError("shape-only factory mapping count drifted")

    class DeepseekV4FP8Config:
        expert_dtype = "fp4"
        moe_quant_algo = "NVFP4"
        target_num_hidden_layers = OFFICIAL_LAYERS

    created = stager_factory(
        torch_module=torch,
        params_dict=params,
        expert_mapping_index=SimpleNamespace(mappings=mappings, safe=True),
        start_layer=0,
        end_layer=OFFICIAL_LAYERS,
        num_hidden_layers=OFFICIAL_LAYERS,
        num_routed_experts=OFFICIAL_EXPERTS,
        tp_size=TP_SIZE,
        use_mega_moe=False,
        enable_expert_parallel=False,
        num_redundant_experts=0,
        load_format="auto",
        quant_config=DeepseekV4FP8Config(),
        environ={
            "VLLM_DSV4_NVFP4_LAYER_STAGED_LOAD": "1",
            "DSPARK_WEIGHT_LOAD_FORMAT": "auto",
        },
    )
    if created is None or not isinstance(created, stager_class):
        raise RuntimeError("real staging factory did not return Nvfp4LayerStager")
    eligible = getattr(created, "_eligible_parameters", None)
    if not isinstance(eligible, dict) or len(eligible) != OFFICIAL_LAYERS:
        raise RuntimeError("real staging factory did not retain all 43 layers")
    if any(
        not isinstance(parameter, _ShapeOnlyCudaParameter)
        for layer in eligible.values()
        for parameter in layer.values()
    ):
        raise RuntimeError("real staging factory replaced shape-only descriptors")
    if getattr(created, "_expected_stage_bytes", None) != OFFICIAL_STAGE_BYTES:
        raise RuntimeError("real staging factory stage-byte contract drifted")
    if getattr(created, "_expected_commit_calls", None) != len(PARAMETER_ORDER):
        raise RuntimeError("real staging factory commit-count contract drifted")
    source_keys = getattr(created, "_expected_source_keys", None)
    if not isinstance(source_keys, frozenset) or len(source_keys) != (
        OFFICIAL_TENSORS_PER_LAYER
    ):
        raise RuntimeError("real staging factory source-key contract drifted")
    return {
        "passed": True,
        "factory": stager_factory.__name__,
        "returned_type": type(created).__name__,
        "shape_only_cuda_descriptors": True,
        "tensor_storage_allocated": False,
        "layers": len(eligible),
        "experts": OFFICIAL_EXPERTS,
        "mapping_keys": len(mappings),
        "source_keys_per_layer": len(source_keys),
        "parameter_descriptors": len(params),
        "virtual_bytes_per_layer": expected_layer_bytes,
        "tp_size": TP_SIZE,
        "load_format": "auto",
        "expert_parallel": False,
        "redundant_experts": 0,
        "mega_moe": False,
    }


def _parameter_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "w13_weight": (EXPERTS, 2 * INTERMEDIATE_PER_RANK, HIDDEN // 2),
        "w2_weight": (EXPERTS, HIDDEN, PACKED_INTERMEDIATE_PER_RANK),
        "w13_weight_scale": (
            EXPERTS,
            2 * INTERMEDIATE_PER_RANK,
            W13_SCALE_COLUMNS,
        ),
        "w2_weight_scale": (EXPERTS, HIDDEN, W2_SCALE_COLUMNS_PER_RANK),
        "w13_weight_scale_2": (EXPERTS, 2),
        "w2_weight_scale_2": (EXPERTS,),
        "w13_input_scale": (EXPERTS, 2),
        "w2_input_scale": (EXPERTS,),
    }


def _mapped_name(basename: str) -> str:
    return f"layers.{LAYER}.ffn.experts.routed_experts.{basename}"


def _basename(projection: str, suffix: str) -> str:
    family = "w13" if projection in ("w1", "w3") else "w2"
    return f"{family}_{suffix}"


def _make_parameters(torch: Any, device: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for basename, shape in _parameter_shapes().items():
        if basename in RAW_BLOCK_SCALE_PARAMETERS:
            # ModelOptNvFp4FusedMoE's actual block-scale storage is E4M3.
            # Seed it through a uint8 view so setup itself performs no float8
            # arithmetic or accidental numeric conversion.
            data = torch.full(shape, 0xD3, dtype=torch.uint8, device=device).view(
                torch.float8_e4m3fn
            )
        elif basename in FP32_PARAMETERS:
            data = torch.full(shape, -777.0, dtype=torch.float32, device=device)
        else:
            data = torch.full(shape, 0xD3, dtype=torch.uint8, device=device)
        parameter = torch.nn.Parameter(
            data,
            requires_grad=False,
        )
        parameter.is_transposed = False
        parameter.use_bitsandbytes_4bit = False
        parameter.quant_method = (
            "block" if basename in RAW_BLOCK_SCALE_PARAMETERS else None
        )
        parameters[basename] = parameter
    return parameters


def _make_oracle(torch: Any) -> dict[str, Any]:
    oracle: dict[str, Any] = {}
    for basename, shape in _parameter_shapes().items():
        dtype = torch.float32 if basename in FP32_PARAMETERS else torch.uint8
        fill = -777.0 if dtype == torch.float32 else 0xD3
        oracle[basename] = torch.full(shape, fill, dtype=dtype, device="cpu")
    return oracle


def _source_tensor(
    torch: Any,
    *,
    expert: int,
    projection: str,
    suffix: str,
) -> Any:
    projection_lane = PROJECTIONS.index(projection)
    seed = 17 + expert * 71 + projection_lane * 23 + SUFFIXES.index(suffix) * 11
    if suffix == "weight":
        shape = _tiny_checkpoint_shape(projection, suffix)
        count = 1
        for size in shape:
            count *= size
        return (
            torch.arange(count, dtype=torch.int64)
            .mul(13)
            .add(seed)
            .remainder(251)
            .to(torch.uint8)
            .reshape(shape)
        )
    if suffix == "weight_scale":
        shape = _tiny_checkpoint_shape(projection, suffix)
        count = 1
        for size in shape:
            count *= size
        # Build the exact checkpoint dtype from deterministic, finite raw
        # E4M3 bytes.  The explicit oracle views these bytes as uint8 before
        # applying TP slicing, so a numeric float8 conversion cannot pass.
        raw = (
            torch.arange(count, dtype=torch.int64)
            .mul(7)
            .add(0x71 + seed)
            .remainder(126)
            .add(1)
            .to(torch.uint8)
            .reshape(shape)
        )
        return raw.view(torch.float8_e4m3fn)
    value = expert * 100.0 + projection_lane * 10.0 + (
        0.125 if suffix == "weight_scale_2" else 1.5
    )
    return torch.tensor(value, dtype=torch.float32)


def _tiny_checkpoint_shape(projection: str, suffix: str) -> tuple[int, ...]:
    if suffix == "weight":
        return (
            (TP_SIZE * INTERMEDIATE_PER_RANK, HIDDEN // 2)
            if projection in ("w1", "w3")
            else (HIDDEN, TP_SIZE * PACKED_INTERMEDIATE_PER_RANK)
        )
    if suffix == "weight_scale":
        return (
            (TP_SIZE * INTERMEDIATE_PER_RANK, W13_SCALE_COLUMNS)
            if projection in ("w1", "w3")
            else (HIDDEN, TP_SIZE * W2_SCALE_COLUMNS_PER_RANK)
        )
    if suffix in ("weight_scale_2", "input_scale"):
        return ()
    raise ValueError(f"unsupported tiny checkpoint suffix {suffix!r}")


def _oracle_load(
    destination: Any,
    loaded_weight: Any,
    *,
    projection: str,
    expert: int,
    tp_rank: int,
    raw_bytes: bool,
) -> None:
    if raw_bytes:
        if int(loaded_weight.element_size()) != 1:
            raise RuntimeError("raw-byte oracle requires one-byte source storage")
        loaded_weight = loaded_weight.contiguous().view(destination.dtype)
    if loaded_weight.ndim == 0:
        if projection == "w1":
            destination[expert][0] = loaded_weight
        elif projection == "w3":
            destination[expert][1] = loaded_weight
        else:
            destination[expert] = loaded_weight
        return
    if projection in ("w1", "w3"):
        per_rank = loaded_weight.shape[0] // TP_SIZE
        shard = loaded_weight.narrow(0, tp_rank * per_rank, per_rank)
        offset = 0 if projection == "w1" else per_rank
        destination[expert].narrow(0, offset, per_rank).copy_(shard)
    else:
        per_rank = loaded_weight.shape[1] // TP_SIZE
        shard = loaded_weight.narrow(1, tp_rank * per_rank, per_rank)
        destination[expert].copy_(shard)


def _raw_bytes(torch: Any, tensor: Any) -> Any:
    # Flatten first because PyTorch cannot dtype-view a rank-0 FP32 tensor
    # directly.  This keeps the probe's checkpoint scalar inputs genuinely
    # rank zero while still exposing their four storage bytes for comparison.
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()


def _storage_fingerprint(torch: Any, tensor: Any) -> dict[str, Any]:
    raw = _raw_bytes(torch, tensor).flatten()
    payload = bytes(raw.tolist())
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _make_moe_config(tp_rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=TP_SIZE,
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=TP_SIZE),
    )


def _make_bound_loader(routed_experts_class: Any, tp_rank: int) -> tuple[Any, Any]:
    routed = routed_experts_class.__new__(routed_experts_class)
    object.__setattr__(routed, "quant_config", _ParityQuantConfig())
    object.__setattr__(routed, "quant_method", _ModelOptNvFp4ParityMethod())
    object.__setattr__(routed, "expert_map_manager", _IdentityExpertMap())
    object.__setattr__(
        routed,
        "moe_config",
        _make_moe_config(tp_rank),
    )
    loader = routed.weight_loader
    if (
        loader.__self__ is not routed
        or loader.__func__ is not routed_experts_class.weight_loader
    ):
        raise RuntimeError("RoutedExperts.weight_loader is not the actual bound method")
    return routed, loader


def _run_rank(
    torch: Any,
    routed_experts_class: Any,
    stager_class: Any,
    staged_session_class: Any,
    *,
    tp_rank: int,
    device: str,
) -> dict[str, Any]:
    _routed, loader = _make_bound_loader(routed_experts_class, tp_rank)
    direct = _make_parameters(torch, device)
    staged = _make_parameters(torch, device)
    oracle = _make_oracle(torch)
    expected_keys = frozenset(
        f"experts.{expert}.{projection}.{suffix}"
        for expert in range(EXPERTS)
        for projection in PROJECTIONS
        for suffix in SUFFIXES
    )
    staged_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in staged.values()
    )
    if staged_bytes >= 1 << 20:
        raise RuntimeError(f"tiny parity probe unexpectedly uses {staged_bytes} bytes")
    stager = stager_class(
        torch_module=torch,
        eligible_parameters={LAYER: staged},
        expected_source_keys=expected_keys,
        expected_stage_bytes=staged_bytes,
        expected_commit_calls=len(PARAMETER_ORDER),
        expected_checkpoint_shapes={
            (projection, suffix): _tiny_checkpoint_shape(projection, suffix)
            for projection in PROJECTIONS
            for suffix in SUFFIXES
        },
    )
    staged_session = staged_session_class()
    staged_session.begin(stager, staged_requested=True)
    first_nested_stager = staged_session.stager_for_nested_load(
        staged_requested=True
    )
    if first_nested_stager is not stager:
        raise RuntimeError("first nested load did not receive the shared stager")

    loader_calls = 0
    observed_source_contract: dict[str, dict[str, Any]] = {
        suffix: {"count": 0, "dtypes": set(), "ranks": set(), "shapes": set()}
        for suffix in SUFFIXES
    }
    staged_dispatch_contract: dict[str, dict[str, Any]] = {
        suffix: {"count": 0, "dtypes": set(), "raw_identity": True}
        for suffix in SUFFIXES
    }
    for expert in range(EXPERTS):
        for projection in PROJECTIONS:
            mapping_key = f"experts.{expert}.{projection}."
            for suffix in SUFFIXES:
                loaded = _source_tensor(
                    torch,
                    expert=expert,
                    projection=projection,
                    suffix=suffix,
                )
                expected_source = MAIN_TARGET_SOURCE_CONTRACT[suffix]
                if str(loaded.dtype) != expected_source["dtype"]:
                    raise RuntimeError(
                        f"source dtype drift for {mapping_key}{suffix}: "
                        f"{loaded.dtype} != {expected_source['dtype']}"
                    )
                if int(loaded.ndim) != expected_source["rank"]:
                    raise RuntimeError(
                        f"source rank drift for {mapping_key}{suffix}: "
                        f"{loaded.ndim} != {expected_source['rank']}"
                    )
                if (
                    "shape" in expected_source
                    and list(loaded.shape) != expected_source["shape"]
                ):
                    raise RuntimeError(
                        f"source shape drift for {mapping_key}{suffix}: "
                        f"{list(loaded.shape)!r} != {expected_source['shape']!r}"
                    )
                observed = observed_source_contract[suffix]
                observed["count"] += 1
                observed["dtypes"].add(str(loaded.dtype))
                observed["ranks"].add(int(loaded.ndim))
                observed["shapes"].add(tuple(int(size) for size in loaded.shape))
                basename = _basename(projection, suffix)
                mapped_name = _mapped_name(basename)
                direct_success = loader(
                    direct[basename],
                    loaded,
                    mapped_name,
                    shard_id=projection,
                    expert_id=expert,
                    return_success=True,
                )
                if direct_success is not True:
                    raise RuntimeError(
                        f"direct loader rejected {mapping_key}{suffix}"
                    )
                source = first_nested_stager.begin_source(
                    f"layers.{LAYER}.ffn.{mapping_key}{suffix}",
                    loaded,
                    _ExpertMatch(LAYER, mapping_key, suffix, projection),
                )
                if source is None:
                    raise RuntimeError(f"stager declined {mapping_key}{suffix}")
                expected_dispatch_dtype = (
                    torch.uint8 if suffix == "weight_scale" else loaded.dtype
                )
                if source.loaded_weight.dtype != expected_dispatch_dtype:
                    raise RuntimeError(
                        f"stager dispatch dtype drift for {mapping_key}{suffix}: "
                        f"{source.loaded_weight.dtype} != {expected_dispatch_dtype}"
                    )
                raw_identity = bool(
                    torch.equal(
                        _raw_bytes(torch, source.loaded_weight),
                        _raw_bytes(torch, loaded),
                    )
                )
                if not raw_identity:
                    raise RuntimeError(
                        f"stager changed source bytes for {mapping_key}{suffix}"
                    )
                dispatched = staged_dispatch_contract[suffix]
                dispatched["count"] += 1
                dispatched["dtypes"].add(str(source.loaded_weight.dtype))
                dispatched["raw_identity"] = (
                    dispatched["raw_identity"] and raw_identity
                )
                proxy = first_nested_stager.destination(
                    source, mapped_name, staged[basename]
                )
                staged_success = loader(
                    proxy,
                    source.loaded_weight,
                    mapped_name,
                    shard_id=projection,
                    expert_id=expert,
                    return_success=True,
                )
                if staged_success is not True:
                    raise RuntimeError(
                        f"staged loader rejected {mapping_key}{suffix}"
                    )
                _oracle_load(
                    oracle[basename],
                    loaded,
                    projection=projection,
                    expert=expert,
                    tp_rank=tp_rank,
                    raw_bytes=suffix == "weight_scale",
                )
                first_nested_stager.complete_source(source)
                loader_calls += 2
    second_nested_stager = staged_session.stager_for_nested_load(
        staged_requested=True
    )
    if second_nested_stager is not first_nested_stager:
        raise RuntimeError("second nested load did not reuse the shared stager")
    duplicate_expert_rejected = False
    try:
        second_nested_stager.begin_source(
            "layers.0.ffn.experts.0.w1.weight",
            _source_tensor(
                torch,
                expert=0,
                projection="w1",
                suffix="weight",
            ),
            _ExpertMatch(LAYER, "experts.0.w1.", "weight", "w1"),
        )
    except RuntimeError as exc:
        if "appeared again after its staging commit" not in str(exc):
            raise
        duplicate_expert_rejected = True
    if not duplicate_expert_rejected:
        raise RuntimeError("second nested load accepted a repeated expert tensor")
    nested_load_calls = staged_session.nested_load_calls
    staged_session.finish()
    if staged_session.finish_calls != 1 or staged_session.active:
        raise RuntimeError("staged session did not finish exactly once")
    if device == "cuda":
        torch.cuda.synchronize()

    storages: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for basename in PARAMETER_ORDER:
        staged_raw = _raw_bytes(torch, staged[basename])
        direct_raw = _raw_bytes(torch, direct[basename])
        oracle_raw = _raw_bytes(torch, oracle[basename])
        staged_matches_oracle = bool(torch.equal(staged_raw, oracle_raw))
        direct_matches_oracle = bool(torch.equal(direct_raw, oracle_raw))
        direct_comparison_required = True
        passed = staged_matches_oracle and direct_matches_oracle
        row = {
            "family": (
                "block_scale_raw_bytes"
                if basename in RAW_BLOCK_SCALE_PARAMETERS
                else "packed_weight"
                if basename in PACKED_PARAMETERS
                else "fp32_global"
            ),
            "direct_comparison_required": direct_comparison_required,
            "staged_matches_explicit_oracle": staged_matches_oracle,
            "direct_matches_explicit_oracle": direct_matches_oracle,
            "passed": passed,
            "staged": _storage_fingerprint(torch, staged[basename]),
            "direct_raw_control": _storage_fingerprint(torch, direct[basename]),
            "explicit_oracle": _storage_fingerprint(torch, oracle[basename]),
        }
        storages[basename] = row
        if not passed:
            failures.append(
                {
                    "kind": "raw_storage_parity",
                    "parameter": basename,
                    "staged_matches_oracle": staged_matches_oracle,
                    "direct_matches_oracle": direct_matches_oracle,
                    "direct_comparison_required": direct_comparison_required,
                }
            )

    expected_sources = EXPERTS * len(PROJECTIONS) * len(SUFFIXES)
    if stager.total_source_tensors != expected_sources:
        failures.append(
            {
                "kind": "source_count",
                "observed": stager.total_source_tensors,
                "expected": expected_sources,
            }
        )
    if stager.total_commit_calls != len(PARAMETER_ORDER):
        failures.append(
            {
                "kind": "commit_count",
                "observed": stager.total_commit_calls,
                "expected": len(PARAMETER_ORDER),
            }
        )

    serialized_source_contract = {
        suffix: {
            "count": row["count"],
            "dtypes": sorted(row["dtypes"]),
            "ranks": sorted(row["ranks"]),
            "shapes": [list(shape) for shape in sorted(row["shapes"])],
        }
        for suffix, row in observed_source_contract.items()
    }
    serialized_dispatch_contract = {
        suffix: {
            "count": row["count"],
            "dtypes": sorted(row["dtypes"]),
            "raw_identity": row["raw_identity"],
        }
        for suffix, row in staged_dispatch_contract.items()
    }
    return {
        "tp_rank": tp_rank,
        "device": device,
        "passed": not failures,
        "failures": failures,
        "loader_calls": loader_calls,
        "source_tensors": stager.total_source_tensors,
        "commit_calls": stager.total_commit_calls,
        "multi_invocation_lifecycle": {
            "nested_load_calls": nested_load_calls,
            "shared_stager_identity": second_nested_stager is first_nested_stager,
            "duplicate_expert_rejected": duplicate_expert_rejected,
            "finish_calls": staged_session.finish_calls,
            "active_after_finish": staged_session.active,
        },
        "staged_destination_bytes": staged_bytes,
        "checkpoint_source_contract": serialized_source_contract,
        "stager_dispatch_contract": serialized_dispatch_contract,
        "storages": storages,
    }


def _run_auto_loader_grouping_proof(
    torch: Any,
    auto_weights_loader_class: Any,
    mapper_factory: Any,
    parse_expert_name: Any,
) -> dict[str, Any]:
    """Prove the exact target root re-entry and MTP filtering semantics."""

    mtp_names = [f"mtp.0.synthetic.{index}.weight" for index in range(1_575)]
    original_names = [
        "layers.0.ffn.experts.0.w1.weight",
        "head.weight",
        "norm.weight",
        *mtp_names,
    ]
    mapper = mapper_factory("fp4")
    mapped = list(mapper.apply((name, None) for name in original_names))
    mapped_mtp = [name for name, _value in mapped if "mtp." in name]
    if len(mapped_mtp) != 1_575 or any(
        not name.startswith("model.mtp.") for name in mapped_mtp
    ):
        raise RuntimeError("target mapper MTP contract drifted")

    events: list[str] = []

    class _TargetModelSpy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[list[str]] = []

        def load_weights(self, weights: Any) -> set[str]:
            names = [name for name, _value in weights]
            self.calls.append(names)
            events.append("model")
            return set(names)

    class _CausalModelSpy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _TargetModelSpy()

            class _LmHeadSpy(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.zeros(1, dtype=torch.float32),
                        requires_grad=False,
                    )

                    def load_lm_head(param: Any, loaded_weight: Any) -> None:
                        events.append("lm_head")
                        param.data.copy_(loaded_weight)

                    self.weight.weight_loader = load_lm_head

            self.lm_head = _LmHeadSpy()

    target = _CausalModelSpy()
    loader = auto_weights_loader_class(target, skip_substrs=["mtp."])
    loaded = loader.load_weights(
        ((name, torch.zeros(1, dtype=torch.float32)) for name in original_names),
        mapper=mapper,
    )
    if events != ["model", "lm_head", "model"]:
        raise RuntimeError(
            f"target AutoWeightsLoader invocation order drifted: {events!r}"
        )
    if len(target.model.calls) != 2:
        raise RuntimeError("target child was not invoked exactly twice")
    retained_names = {name for call in target.model.calls for name in call}
    retained_names.update(loaded)
    if any("mtp." in name for name in retained_names):
        raise RuntimeError("target AutoWeightsLoader retained an MTP tensor")
    mtp_parse_names = (
        "mtp.0.ffn.experts.0.w1.weight",
        "model.mtp.0.ffn.experts.0.w1.weight",
        "mtp.0.ffn.experts.0.w1.scale",
        "model.mtp.0.ffn.experts.0.w1.weight_scale",
    )
    if any(parse_expert_name(name) is not None for name in mtp_parse_names):
        raise RuntimeError("target expert-name parser claimed an MTP tensor")
    return {
        "passed": True,
        "root_runs": events,
        "nested_model_invocations": len(target.model.calls),
        "nested_model_names": target.model.calls,
        "mapped_mtp_tensors": len(mapped_mtp),
        "retained_mtp_tensors": 0,
        "mtp_parser_rejections": list(mtp_parse_names),
    }


def run_probe(device: str = "cpu") -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        _get_priority_backends,
        convert_weight_to_mxfp4_moe_kernel_format,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.deepseek_v4.nvidia.model import (
        DeepseekV4ForCausalLM,
        DeepseekV4Model,
        _make_deepseek_v4_weights_mapper,
    )
    from vllm.models.deepseek_v4.nvidia.staged_weight_loading import (
        Nvfp4LayerStager,
        Nvfp4LayerStagedLoadSession,
        _expected_checkpoint_shape,
        maybe_create_nvfp4_layer_stager,
    )
    from vllm.models.deepseek_v4.nvidia.weight_loading import parse_expert_name
    from vllm.models.deepseek_v4.quant_config import (
        _scope_prepared_nvfp4_target_backend,
    )

    if device not in ("cpu", "cuda"):
        raise ValueError(f"unsupported probe device {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    source_audit = _audit_runtime_sources(
        RoutedExperts,
        Nvfp4LayerStager,
        Nvfp4LayerStagedLoadSession,
        maybe_create_nvfp4_layer_stager,
        _expected_checkpoint_shape,
        DeepseekV4Model,
        DeepseekV4ForCausalLM,
        _scope_prepared_nvfp4_target_backend,
        _get_priority_backends,
        convert_weight_to_mxfp4_moe_kernel_format,
        Mxfp4MoEMethod,
    )
    auto_loader_grouping = _run_auto_loader_grouping_proof(
        torch,
        AutoWeightsLoader,
        _make_deepseek_v4_weights_mapper,
        parse_expert_name,
    )
    factory_preflight = _run_factory_preflight(
        torch,
        Nvfp4LayerStager,
        maybe_create_nvfp4_layer_stager,
    )
    draft_backend_proof = _run_draft_postload_backend_proof(
        torch,
        prepared_backend_scope=_scope_prepared_nvfp4_target_backend,
        draft_backend_priority=_get_priority_backends,
        draft_converter=convert_weight_to_mxfp4_moe_kernel_format,
        backend_enum=Mxfp4MoeBackend,
        draft_method_class=Mxfp4MoEMethod,
    )
    ranks = [
        _run_rank(
            torch,
            RoutedExperts,
            Nvfp4LayerStager,
            Nvfp4LayerStagedLoadSession,
            tp_rank=tp_rank,
            device=device,
        )
        for tp_rank in range(TP_SIZE)
    ]
    failures = [
        {"kind": "rank", "tp_rank": row["tp_rank"], "failures": row["failures"]}
        for row in ranks
        if not row["passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ok": (
            source_audit["passed"]
            and factory_preflight["passed"]
            and draft_backend_proof["passed"]
            and not failures
        ),
        "probe": "real_routed_experts_nvfp4_layer_stager_parity",
        "model_loaded": False,
        "checkpoint_opened": False,
        "default_cpu_proof": device == "cpu",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "source_audit": source_audit,
        "auto_loader_grouping": auto_loader_grouping,
        "factory_preflight": factory_preflight,
        "draft_backend_proof": draft_backend_proof,
        "settings": {
            "tp_size": TP_SIZE,
            "experts": EXPERTS,
            "source_tensors_per_rank": EXPERTS
            * len(PROJECTIONS)
            * len(SUFFIXES),
            "destination_storages": list(PARAMETER_ORDER),
            "block_scale_oracle": "explicit raw-byte TP slice",
            "checkpoint_layout_contract": CHECKPOINT_LAYOUT_CONTRACT,
        },
        "ranks": ranks,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is mandatory/default; CUDA optionally checks raw commit copies",
    )
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def _emit(report: dict[str, Any], output: pathlib.Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_probe(args.device)
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "probe": "real_routed_experts_nvfp4_layer_stager_parity",
            "model_loaded": False,
            "checkpoint_opened": False,
            "default_cpu_proof": args.device == "cpu",
            "device": args.device,
            "settings": {
                "checkpoint_layout_contract": CHECKPOINT_LAYOUT_CONTRACT,
            },
            "failures": [
                {
                    "kind": "exception",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                }
            ],
        }
    _emit(report, args.output)
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
