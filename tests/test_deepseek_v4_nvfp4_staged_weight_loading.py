# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT / "overlay/vllm/models/deepseek_v4/nvidia/staged_weight_loading.py"
)
MODEL_PATH = ROOT / "overlay/vllm/models/deepseek_v4/nvidia/model.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_deepseek_v4_nvfp4_staged_weight_loading_under_test", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type


class _FakeStorage:
    def __init__(self):
        self.copy_calls = 0
        self.payload: list[tuple[int, str, int]] = []
        self.byte_regions: dict[tuple[int, str, str], bytes] = {}
        self.source_bytes = b""
        self.last_source_dtype = None


class _FakeTensor:
    _ELEMENT_SIZES = {
        "uint8": 1,
        "float8_e4m3fn": 1,
        "float8_e8m0fnu": 1,
        "float32": 4,
    }

    def __init__(
        self,
        shape,
        *,
        dtype: str,
        device: str,
        storage: _FakeStorage | None = None,
        source_bytes: bytes = b"",
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _FakeDevice(device)
        self._storage = storage or _FakeStorage()
        if source_bytes:
            self._storage.source_bytes = source_bytes

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self._ELEMENT_SIZES[self.dtype]

    def view(self, dtype):
        return _FakeTensor(
            self.shape,
            dtype=dtype,
            device=self.device.type,
            storage=self._storage,
        )

    def copy_(self, source):
        self._storage.copy_calls += 1
        self._storage.payload = list(source._storage.payload)
        self._storage.byte_regions = dict(source._storage.byte_regions)
        self._storage.last_source_dtype = source.dtype
        return self


class _FakeParameter:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    def numel(self):
        return self.data.numel()

    def element_size(self):
        return self.data.element_size()


class _FakeNN:
    Parameter = _FakeParameter


class _FakeTorch:
    uint8 = "uint8"
    float8_e4m3fn = "float8_e4m3fn"
    float8_e8m0fnu = "float8_e8m0fnu"
    float32 = "float32"
    nn = _FakeNN()

    @staticmethod
    def empty(shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)


class DeepseekV4FP8Config:
    expert_dtype = "fp4"
    moe_quant_algo = "NVFP4"
    target_num_hidden_layers = 43


_PARAMETER_SHAPES = {
    "w13_weight": (256, 2_048, 2_048),
    "w2_weight": (256, 4_096, 512),
    "w13_weight_scale": (256, 2_048, 256),
    "w2_weight_scale": (256, 4_096, 64),
    "w13_weight_scale_2": (256, 2),
    "w2_weight_scale_2": (256,),
    "w13_input_scale": (256, 2),
    "w2_input_scale": (256,),
}
_PARAMETER_DTYPES = {
    "w13_weight": _FakeTorch.uint8,
    "w2_weight": _FakeTorch.uint8,
    "w13_weight_scale": _FakeTorch.uint8,
    "w2_weight_scale": _FakeTorch.uint8,
    "w13_weight_scale_2": _FakeTorch.float32,
    "w2_weight_scale_2": _FakeTorch.float32,
    "w13_input_scale": _FakeTorch.float32,
    "w2_input_scale": _FakeTorch.float32,
}


def _parameter_name(layer: int, basename: str) -> str:
    return f"layers.{layer}.ffn.experts.routed_experts.{basename}"


def _make_params(*layers: int):
    return {
        _parameter_name(layer, basename): _FakeParameter(
            _FakeTensor(
                shape,
                dtype=_PARAMETER_DTYPES[basename],
                device="cuda",
            )
        )
        for layer in layers
        for basename, shape in _PARAMETER_SHAPES.items()
    }


def _make_index():
    mappings = {
        f"experts.{expert}.{projection}.": (
            (
                "experts.routed_experts."
                + ("w13_" if projection in ("w1", "w3") else "w2_"),
                f"experts.{expert}.{projection}.",
                expert,
                projection,
            ),
        )
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
    }
    return SimpleNamespace(safe=True, mappings=mappings)


def _match(layer: int, mapping_key: str, suffix: str):
    projection = mapping_key.removesuffix(".").rsplit(".", 1)[-1]
    return SimpleNamespace(
        layer=layer,
        mapping_key=mapping_key,
        suffix=suffix,
        projection=projection,
    )


def _source_dtype(suffix: str):
    return {
        "weight": _FakeTorch.uint8,
        "weight_scale": _FakeTorch.uint8,
        "weight_scale_2": _FakeTorch.float32,
        "input_scale": _FakeTorch.float32,
    }[suffix]


def _destination_basename(projection: str, suffix: str) -> str:
    prefix = "w13" if projection in ("w1", "w3") else "w2"
    return f"{prefix}_{suffix}"


def _fake_routed_weight_loader(
    param,
    loaded_weight,
    mapped_name: str,
    *,
    shard_id: str,
    expert_id: int,
    return_success: bool,
    tp_rank: int,
):
    """Small deterministic model of RoutedExperts TP=2 destination slicing."""

    suffix = mapped_name.rsplit("_", 1)[-1]
    if mapped_name.endswith("weight_scale_2"):
        suffix = "weight_scale_2"
    elif mapped_name.endswith("input_scale"):
        suffix = "input_scale"
    elif mapped_name.endswith("weight_scale"):
        suffix = "weight_scale"
    raw = loaded_weight._storage.source_bytes
    if suffix in ("weight", "weight_scale"):
        if len(raw) % 2:
            raise AssertionError("TP=2 fake source must split evenly")
        width = len(raw) // 2
        raw = raw[tp_rank * width : (tp_rank + 1) * width]
    param.data._storage.byte_regions[(expert_id, shard_id, suffix)] = raw
    return bool(return_success)


def _factory(helper, params, *, start=0, end=1, environ=None, index=None):
    return helper.maybe_create_nvfp4_layer_stager(
        torch_module=_FakeTorch,
        params_dict=params,
        expert_mapping_index=index or _make_index(),
        start_layer=start,
        end_layer=end,
        num_hidden_layers=43,
        num_routed_experts=256,
        tp_size=2,
        use_mega_moe=False,
        enable_expert_parallel=False,
        num_redundant_experts=0,
        load_format="auto",
        quant_config=DeepseekV4FP8Config(),
        environ=environ
        or {
            helper.STAGED_LOAD_ENV: "1",
            "DSPARK_WEIGHT_LOAD_FORMAT": "auto",
        },
    )


class DeepseekV4NvFp4StagedWeightLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = _load_helper()

    def test_default_is_off_and_invalid_values_fail(self):
        self.assertFalse(self.helper.staged_load_requested({}))
        self.assertFalse(
            self.helper.staged_load_requested({self.helper.STAGED_LOAD_ENV: "0"})
        )
        self.assertTrue(
            self.helper.staged_load_requested({self.helper.STAGED_LOAD_ENV: "1"})
        )
        for value in ("true", "yes", "2", ""):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "exactly '0' or '1'"
            ):
                self.helper.staged_load_requested(
                    {self.helper.STAGED_LOAD_ENV: value}
                )

    def test_factory_default_off_does_not_inspect_model(self):
        self.assertIsNone(
            self.helper.maybe_create_nvfp4_layer_stager(
                torch_module=_FakeTorch,
                params_dict={},
                expert_mapping_index=None,
                start_layer=0,
                end_layer=43,
                num_hidden_layers=0,
                num_routed_experts=0,
                tp_size=0,
                use_mega_moe=True,
                enable_expert_parallel=True,
                num_redundant_experts=1,
                load_format="roce_tp",
                quant_config=object(),
                environ={},
            )
        )

    def test_factory_names_match_real_fused_moe_mapping_output(self):
        # This is the default output grammar of
        # RoutedExperts.make_expert_params_mapping, reached through
        # fused_moe_make_expert_params_mapping in DeepseekV4Model.
        index = _make_index()
        relative = self.helper._reviewed_parameter_relative_names(index)
        self.assertEqual(
            relative,
            {
                basename: f"experts.routed_experts.{basename}"
                for basename in _PARAMETER_SHAPES
            },
        )
        for projection in ("w1", "w2", "w3"):
            mapping_key = f"experts.0.{projection}."
            param_name, weight_name, _expert_id, _shard_id = (
                index.mappings[mapping_key][0]
            )
            for suffix in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            ):
                checkpoint_name = f"layers.0.ffn.{mapping_key}{suffix}"
                mapped_name = checkpoint_name.replace(weight_name, param_name)
                basename = _destination_basename(projection, suffix)
                self.assertEqual(mapped_name, _parameter_name(0, basename))
        self.assertIsNotNone(_factory(self.helper, _make_params(0), index=index))

    def test_same_cardinality_mapping_drift_is_rejected(self):
        mappings = dict(_make_index().mappings)
        mappings.pop("experts.255.w3.")
        mappings["experts.999.w3."] = (
            (
                "experts.routed_experts.w13_",
                "experts.999.w3.",
                999,
                "w3",
            ),
        )
        same_cardinality = SimpleNamespace(safe=True, mappings=mappings)
        with self.assertRaisesRegex(RuntimeError, "mapping-key set drifted"):
            _factory(self.helper, _make_params(0), index=same_cardinality)

        mappings = dict(_make_index().mappings)
        param_name, weight_name, _expert_id, shard_id = mappings[
            "experts.0.w1."
        ][0]
        mappings["experts.0.w1."] = (
            (param_name, weight_name, 1, shard_id),
        )
        wrong_candidate = SimpleNamespace(safe=True, mappings=mappings)
        with self.assertRaisesRegex(RuntimeError, "mapping candidate drifted"):
            _factory(self.helper, _make_params(0), index=wrong_candidate)

    def test_post_e8m0_reinterpret_is_the_only_scale_source_contract(self):
        stager = _factory(self.helper, _make_params(0))
        raw_e8m0 = _FakeTensor(
            (8,), dtype=_FakeTorch.float8_e8m0fnu, device="cpu"
        )
        post_reinterpret = raw_e8m0.view(_FakeTorch.uint8)
        source = stager.begin_source(
            "layers.0.ffn.experts.0.w1.weight_scale",
            post_reinterpret,
            _match(0, "experts.0.w1.", "weight_scale"),
        )
        self.assertIs(source.loaded_weight._storage, raw_e8m0._storage)
        self.assertEqual(source.loaded_weight.dtype, _FakeTorch.uint8)

        other = _factory(self.helper, _make_params(0))
        with self.assertRaisesRegex(RuntimeError, "has dtype"):
            other.begin_source(
                "unconverted-e8m0",
                raw_e8m0,
                _match(0, "experts.0.w1.", "weight_scale"),
            )

    def test_exact_full_layer_contract_and_eight_bulk_commits(self):
        params = _make_params(0)
        stager = _factory(self.helper, params)
        self.assertIsNotNone(stager)
        index = _make_index()

        for mapping_key in index.mappings:
            projection = mapping_key.removesuffix(".").rsplit(".", 1)[-1]
            expert = int(mapping_key.split(".")[1])
            for suffix in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            ):
                match = _match(0, mapping_key, suffix)
                loaded = _FakeTensor(
                    (1,), dtype=_source_dtype(suffix), device="cpu"
                )
                name = f"layers.0.ffn.{mapping_key}{suffix}"
                source = stager.begin_source(name, loaded, match)
                self.assertIsNotNone(source)
                basename = _destination_basename(projection, suffix)
                mapped_name = _parameter_name(0, basename)
                proxy = stager.destination(source, mapped_name, params[mapped_name])

                # Symbolic fake of RoutedExperts.weight_loader.  Crucially,
                # w1 is the first raw W13 half and w3 is the second.
                region = 0 if projection == "w1" else 1 if projection == "w3" else 0
                proxy.data._storage.payload.append((region, suffix, expert))
                if suffix == "weight_scale":
                    self.assertEqual(source.loaded_weight.dtype, _FakeTorch.uint8)
                stager.complete_source(source)

        stager.finish()
        self.assertEqual(stager.total_source_tensors, 3_072)
        self.assertEqual(stager.total_commit_calls, 8)
        self.assertEqual(stager.completed_layers, frozenset({0}))
        for param in params.values():
            self.assertEqual(param.data._storage.copy_calls, 1)
        self.assertEqual(
            sum(param.numel() * param.element_size() for param in params.values()),
            1_811_945_472,
        )
        w13_payload = params[_parameter_name(0, "w13_weight")].data._storage.payload
        self.assertTrue(any(region == 0 for region, _, _ in w13_payload))
        self.assertTrue(any(region == 1 for region, _, _ in w13_payload))
        scale = params[_parameter_name(0, "w13_weight_scale")]
        self.assertEqual(scale.data._storage.last_source_dtype, _FakeTorch.uint8)

    def test_staged_and_direct_fake_routed_loader_match_for_both_tp_ranks(self):
        suffixes = (
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale",
        )
        mapping_keys = tuple(
            f"experts.{expert}.{projection}."
            for expert in range(2)
            for projection in ("w1", "w2", "w3")
        )
        expected_keys = frozenset(
            f"{mapping_key}{suffix}"
            for mapping_key in mapping_keys
            for suffix in suffixes
        )

        for tp_rank in (0, 1):
            with self.subTest(tp_rank=tp_rank):
                direct_params = _make_params(0)
                staged_params = _make_params(0)
                eligible = {
                    0: {
                        basename: staged_params[_parameter_name(0, basename)]
                        for basename in _PARAMETER_SHAPES
                    }
                }
                staged_bytes = sum(
                    param.numel() * param.element_size()
                    for param in eligible[0].values()
                )
                stager = self.helper.Nvfp4LayerStager(
                    torch_module=_FakeTorch,
                    eligible_parameters=eligible,
                    expected_source_keys=expected_keys,
                    expected_stage_bytes=staged_bytes,
                )
                for mapping_key in mapping_keys:
                    expert = int(mapping_key.split(".")[1])
                    projection = mapping_key.removesuffix(".").rsplit(".", 1)[-1]
                    for suffix in suffixes:
                        length = 16 if suffix in ("weight", "weight_scale") else 4
                        seed = expert * 37 + int(projection[-1]) * 11 + len(suffix)
                        loaded = _FakeTensor(
                            (length,),
                            dtype=_source_dtype(suffix),
                            device="cpu",
                            source_bytes=bytes(
                                (seed + offset) % 256 for offset in range(length)
                            ),
                        )
                        basename = _destination_basename(projection, suffix)
                        mapped_name = _parameter_name(0, basename)
                        direct_ok = _fake_routed_weight_loader(
                            direct_params[mapped_name],
                            loaded,
                            mapped_name,
                            shard_id=projection,
                            expert_id=expert,
                            return_success=True,
                            tp_rank=tp_rank,
                        )
                        source = stager.begin_source(
                            f"layers.0.ffn.{mapping_key}{suffix}",
                            loaded,
                            _match(0, mapping_key, suffix),
                        )
                        proxy = stager.destination(
                            source, mapped_name, staged_params[mapped_name]
                        )
                        staged_ok = _fake_routed_weight_loader(
                            proxy,
                            source.loaded_weight,
                            mapped_name,
                            shard_id=projection,
                            expert_id=expert,
                            return_success=True,
                            tp_rank=tp_rank,
                        )
                        self.assertEqual(staged_ok, direct_ok)
                        stager.complete_source(source)
                stager.finish()
                for name in direct_params:
                    self.assertEqual(
                        staged_params[name].data._storage.byte_regions,
                        direct_params[name].data._storage.byte_regions,
                    )

    def test_pp_missing_layer_is_not_claimed(self):
        stager = _factory(self.helper, _make_params(0))
        match = _match(1, "experts.0.w1.", "weight")
        source = stager.begin_source(
            "layers.1.ffn.experts.0.w1.weight",
            _FakeTensor((1,), dtype=_FakeTorch.uint8, device="cpu"),
            match,
        )
        self.assertIsNone(source)

    def test_duplicate_interleaved_and_incomplete_layers_fail_closed(self):
        params = _make_params(0, 1)
        stager = _factory(self.helper, params, start=0, end=2)
        loaded = _FakeTensor((1,), dtype=_FakeTorch.uint8, device="cpu")
        first_match = _match(0, "experts.0.w1.", "weight")
        first = stager.begin_source(
            "layers.0.ffn.experts.0.w1.weight", loaded, first_match
        )
        stager.complete_source(first)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            stager.begin_source(
                "layers.0.ffn.experts.0.w1.weight", loaded, first_match
            )
        with self.assertRaisesRegex(RuntimeError, "interleaves routed layers"):
            stager.begin_source(
                "layers.1.ffn.experts.0.w1.weight",
                loaded,
                _match(1, "experts.0.w1.", "weight"),
            )
        with self.assertRaisesRegex(RuntimeError, "is incomplete"):
            stager.finish()
        self.assertEqual(
            sum(param.data._storage.copy_calls for param in params.values()), 0
        )

    def test_source_device_dtype_and_pending_contracts_fail_closed(self):
        stager = _factory(self.helper, _make_params(0))
        match = _match(0, "experts.0.w1.", "weight")
        with self.assertRaisesRegex(RuntimeError, "CPU checkpoint tensors only"):
            stager.begin_source(
                "gpu", _FakeTensor((1,), dtype=_FakeTorch.uint8, device="cuda"), match
            )
        with self.assertRaisesRegex(RuntimeError, "has dtype"):
            stager.begin_source(
                "wrong-dtype",
                _FakeTensor((1,), dtype=_FakeTorch.float32, device="cpu"),
                match,
            )
        source = stager.begin_source(
            "good", _FakeTensor((1,), dtype=_FakeTorch.uint8, device="cpu"), match
        )
        with self.assertRaisesRegex(RuntimeError, "before completing"):
            stager.begin_source(
                "second",
                _FakeTensor((1,), dtype=_FakeTorch.uint8, device="cpu"),
                _match(0, "experts.0.w2.", "weight"),
            )
        with self.assertRaisesRegex(RuntimeError, "was not completed"):
            stager.finish()
        self.assertIsNotNone(source)

    def test_roce_mega_wrong_quant_and_unsafe_mapping_are_rejected(self):
        params = _make_params(0)
        base = {
            self.helper.STAGED_LOAD_ENV: "1",
            "DSPARK_WEIGHT_LOAD_FORMAT": "auto",
        }
        with self.assertRaisesRegex(RuntimeError, "does not support roce_tp"):
            _factory(
                self.helper,
                params,
                environ=base | {"DSPARK_WEIGHT_LOAD_FORMAT": "roce_tp"},
            )

        kwargs = dict(
            torch_module=_FakeTorch,
            params_dict=params,
            expert_mapping_index=_make_index(),
            start_layer=0,
            end_layer=1,
            num_hidden_layers=43,
            num_routed_experts=256,
            tp_size=2,
            enable_expert_parallel=False,
            num_redundant_experts=0,
            load_format="auto",
            quant_config=DeepseekV4FP8Config(),
            environ=base,
        )
        with self.assertRaisesRegex(RuntimeError, "does not support MegaMoE"):
            self.helper.maybe_create_nvfp4_layer_stager(
                **(kwargs | {"use_mega_moe": True})
            )
        with self.assertRaisesRegex(RuntimeError, "requires EP/EPLB disabled"):
            self.helper.maybe_create_nvfp4_layer_stager(
                **(
                    kwargs
                    | {
                        "use_mega_moe": False,
                        "enable_expert_parallel": True,
                    }
                )
            )
        with self.assertRaisesRegex(RuntimeError, "requires EP/EPLB disabled"):
            self.helper.maybe_create_nvfp4_layer_stager(
                **(
                    kwargs
                    | {
                        "use_mega_moe": False,
                        "num_redundant_experts": 1,
                    }
                )
            )
        with self.assertRaisesRegex(RuntimeError, "does not support roce_tp"):
            self.helper.maybe_create_nvfp4_layer_stager(
                **(
                    kwargs
                    | {
                        "use_mega_moe": False,
                        "load_format": "ROCE_TP",
                    }
                )
            )
        with self.assertRaisesRegex(RuntimeError, "requires the DeepseekV4FP8Config"):
            self.helper.maybe_create_nvfp4_layer_stager(
                **(
                    kwargs
                    | {
                        "use_mega_moe": False,
                        "quant_config": SimpleNamespace(
                            expert_dtype="fp4",
                            moe_quant_algo="W4A16_NVFP4",
                            target_num_hidden_layers=43,
                        ),
                    }
                )
            )
        unsafe = SimpleNamespace(safe=False, mappings=_make_index().mappings)
        with self.assertRaisesRegex(RuntimeError, "safe expert mapping index"):
            self.helper.maybe_create_nvfp4_layer_stager(
                **(
                    kwargs
                    | {
                        "use_mega_moe": False,
                        "expert_mapping_index": unsafe,
                    }
                )
            )

    def test_parameter_schema_and_exact_bytes_are_preflighted(self):
        params = _make_params(0)
        weight = params[_parameter_name(0, "w13_weight")]
        weight.data.dtype = _FakeTorch.float32
        with self.assertRaisesRegex(RuntimeError, "has dtype"):
            _factory(self.helper, params)

        params = _make_params(0)
        scale = params[_parameter_name(0, "w13_weight_scale")]
        scale.data.dtype = _FakeTorch.float32
        with self.assertRaisesRegex(RuntimeError, "raw-byte parameter"):
            _factory(self.helper, params)

        params = _make_params(0)
        for basename in ("w13_weight_scale", "w2_weight_scale"):
            params[_parameter_name(0, basename)].data.dtype = (
                _FakeTorch.float8_e4m3fn
            )
        self.assertIsNotNone(_factory(self.helper, params))

        params = _make_params(0)
        weight = params[_parameter_name(0, "w13_weight")]
        weight.data.shape = (255, 2_048, 2_048)
        with self.assertRaisesRegex(RuntimeError, "has shape"):
            _factory(self.helper, params)

    def test_shape_collision_cannot_pass_on_equal_total_bytes(self):
        params = _make_params(0)
        original_bytes = sum(
            param.numel() * param.element_size() for param in params.values()
        )
        collision_shapes = {
            "w13_weight": (128, 4_096, 2_048),
            "w2_weight": (128, 4_096, 1_024),
            "w13_weight_scale": (128, 4_096, 256),
            "w2_weight_scale": (128, 4_096, 128),
        }
        for basename, shape in collision_shapes.items():
            params[_parameter_name(0, basename)].data.shape = shape
        collision_bytes = sum(
            param.numel() * param.element_size() for param in params.values()
        )
        self.assertEqual(collision_bytes, original_bytes)
        with self.assertRaisesRegex(RuntimeError, "has shape"):
            _factory(self.helper, params)

    def test_model_hook_keeps_actual_loader_and_return_success_contract(self):
        source = MODEL_PATH.read_text()
        model_start = source.index("class DeepseekV4Model")
        start = source.index("    def load_weights(", model_start)
        end = source.index("    def _pad_shared_expert_weight", start)
        body = source[start:end]
        self.assertIn("actual_param = params_dict[name_mapped]", body)
        self.assertIn("expert_stager.destination(", body)
        self.assertIn("actual_param.weight_loader", body)
        self.assertIn("return_success=True", body)
        self.assertIn("name_mapped: str | None = None", body)
        self.assertIn("mapping_succeeded = False", body)
        self.assertIn("mapping_succeeded = True", body)
        failure_gate = body.index("if not mapping_succeeded:")
        completion = body.index("expert_stager.complete_source(staged_source)")
        proxy_release = body.index("del param", completion)
        self.assertLess(failure_gate, completion)
        self.assertLess(completion, proxy_release)
        self.assertIn("expert_stager.finish()", body)
        reinterpret = body.index("loaded_weight = loaded_weight.view(torch.uint8)")
        staged_begin = body.index("expert_stager.begin_source(")
        self.assertLess(reinterpret, staged_begin)


if __name__ == "__main__":
    unittest.main()
