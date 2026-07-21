# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT / "overlay/vllm/models/deepseek_v4/nvidia/prepared_weight_loading.py"
)
MODEL_PATH = ROOT / "overlay/vllm/models/deepseek_v4/nvidia/model.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_deepseek_v4_nvfp4_prepared_weight_loading_under_test", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Device:
    def __init__(self, kind: str):
        self.type = kind


class _Tensor:
    _SIZES = {"uint8": 1, "float8_e4m3fn": 1, "float32": 4}

    def __init__(self, shape, *, dtype, device, storage=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _Device(device)
        self.storage = storage if storage is not None else {"copies": 0}

    @property
    def data(self):
        return self

    def __getitem__(self, index):
        if not isinstance(index, int) or not self.shape:
            raise IndexError(index)
        return _Tensor(
            self.shape[1:],
            dtype=self.dtype,
            device=self.device.type,
            storage=self.storage,
        )

    def is_contiguous(self):
        return True

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self._SIZES[self.dtype]

    def copy_(self, source):
        if self.shape != source.shape or self.dtype != source.dtype:
            raise AssertionError("fake copy contract drifted")
        self.storage["copies"] += 1
        return self


class _FakeTorch:
    uint8 = "uint8"
    float8_e4m3fn = "float8_e4m3fn"
    float32 = "float32"

    @staticmethod
    def empty(shape, *, dtype, device):
        kind = device.type if hasattr(device, "type") else str(device)
        return _Tensor(shape, dtype=dtype, device=kind)


class _SafeOpenHandle:
    def __init__(self, tensors, metadata):
        self.tensors = tensors
        self._metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return list(self.tensors)

    def metadata(self):
        return self._metadata

    def get_tensor(self, name):
        return self.tensors[name]


class PreparedNvfp4WeightLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = _load_helper()

    def _write_tiny_prepared_layer(self, path: Path, *, layer: int = 0):
        helper = self.helper
        shapes = {
            family: (2, 2) for family in helper.PREPARED_FAMILY_ORDER
        }
        header = {
            "__metadata__": helper._expected_prepared_header_metadata(layer)
        }
        payload = bytearray()
        expected_by_rank = {0: {}, 1: {}}
        for family_index, family in enumerate(helper.PREPARED_FAMILY_ORDER):
            dtype, element_size = helper._PREPARED_DTYPE_TOKENS[family]
            rank_bytes = 2 * element_size
            start = len(payload)
            for rank in range(2):
                value = 1 + family_index * 2 + rank
                rank_payload = bytes([value]) * rank_bytes
                expected_by_rank[rank][family] = rank_payload
                payload.extend(rank_payload)
            header[
                f"{helper.PREPARED_NAMESPACE}.layers.{layer}.experts.{family}"
            ] = {
                "dtype": dtype,
                "shape": [2, 2],
                "data_offsets": [start, len(payload)],
            }
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
        path.write_bytes(
            len(raw_header).to_bytes(8, byteorder="little")
            + raw_header
            + payload
        )
        return shapes, expected_by_rank

    def _write_checkpoint_contract(self, root: Path) -> str:
        helper = self.helper
        config = {
            "model_type": "deepseek_v4",
            "dspark_nvfp4_prepared": helper._expected_config_marker(),
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        weight_map = {}
        files = []
        dtype_names = {
            "w13.weight": "U8",
            "w2.weight": "U8",
            "w13.weight_scale": "F8_E4M3",
            "w2.weight_scale": "F8_E4M3",
            "a1_gscale": "F32",
            "a2_gscale": "F32",
            "g1_alphas": "F32",
            "g2_alphas": "F32",
        }
        source_shapes = helper._source_shapes()
        for layer in range(helper.EXPECTED_LAYERS):
            filename = f"model-layer-{layer:05d}.safetensors"
            (root / filename).write_bytes(b"x")
            tensor_rows = []
            for family in helper.PREPARED_FAMILY_ORDER:
                name = (
                    f"{helper.PREPARED_NAMESPACE}.layers.{layer}.experts."
                    f"{family}"
                )
                weight_map[name] = filename
                tensor_rows.append(
                    {
                        "name": name,
                        "family": family,
                        "kind": "tp2_rank_major_cutlass_prepared",
                        "dtype": dtype_names[family],
                        "shape": list(source_shapes[family]),
                        "bytes": 1,
                        "sha256": "1" * 64,
                    }
                )
            files.append(
                {
                    "path": filename,
                    "size": 1,
                    "sha256": "0" * 64,
                    "payload_bytes": len(tensor_rows),
                    "tensor_count": len(tensor_rows),
                    "tensors": tensor_rows,
                }
            )
        index = {
            "metadata": {
                **helper._expected_index_metadata(),
                "total_size": len(weight_map),
            },
            "weight_map": weight_map,
        }
        index_path = root / helper.INDEX_NAME
        index_path.write_text(
            json.dumps(index, sort_keys=True), encoding="utf-8"
        )
        index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
        identity = {
            "implementation": (
                "scripts.repack_deepseek_v4_nvfp4_tp2._cpu_prepare_rank"
            ),
            "engine": helper.PREPARED_ENGINE,
            "backend": helper.PREPARED_BACKEND,
            "vllm_layout_pin": helper.VLLM_LAYOUT_PIN,
            "flashinfer_layout_pin": helper.FLASHINFER_LAYOUT_PIN,
            "numpy_version": "2.3.0",
            "transform_spec_sha256": "2" * 64,
            "repacker_script_path": "/immutable/repacker.py",
            "repacker_script_sha256": "3" * 64,
            "source_revision": "4" * 40,
            "pinned_preparation_source_sha256": dict(
                helper.PINNED_PREPARATION_SOURCE_SHA256
            ),
            "is_act_and_mul": True,
        }
        layer_rank_proofs = {
            str(layer): [
                {
                    "rank": rank,
                    "engine": helper.PREPARED_ENGINE,
                    "w13_scale_2_columns_bitwise_equal": True,
                    "a13_global_scale": 1.0,
                    "a2_global_scale": 1.0,
                    "transform_spec_sha256": "2" * 64,
                }
                for rank in range(helper.EXPECTED_TP_SIZE)
            ]
            for layer in range(helper.EXPECTED_LAYERS)
        }
        manifest = {
            "schema_version": helper.PREPARED_SCHEMA_VERSION,
            "format": helper.PREPARED_SCHEMA,
            "loader": helper._expected_loader_contract(),
            "preparation": {
                "identity": identity,
                "layer_rank_proofs": layer_rank_proofs,
            },
            "source": {
                "config_sha256": helper.SOURCE_CONFIG_SHA256,
                "index_sha256": helper.SOURCE_INDEX_SHA256,
            },
            "output": {
                "config_sha256": config_sha,
                "index_sha256": index_sha,
                "payload_bytes": len(weight_map),
                "tensor_count": len(weight_map),
                "layer_file_count": helper.EXPECTED_LAYERS,
                "files": files,
            },
            "integrity": dict(helper.REQUIRED_INTEGRITY),
        }
        manifest_path = root / helper.MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (root / helper.MANIFEST_DIGEST_NAME).write_text(
            f"{digest}  {helper.MANIFEST_NAME}\n", encoding="ascii"
        )
        return digest

    def test_checkpoint_requires_self_marker_env_and_exact_manifest_pin(self):
        helper = self.helper
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = self._write_checkpoint_contract(root)
            environ = {
                helper.PREPARED_LOAD_ENV: "1",
                helper.PREPARED_MANIFEST_SHA256_ENV: digest,
            }
            contract = helper.inspect_prepared_checkpoint(root, environ=environ)
            self.assertIsNotNone(contract)
            self.assertEqual(contract.manifest_sha256, digest)
            self.assertEqual(len(contract.layer_files), helper.EXPECTED_LAYERS)

            with self.assertRaisesRegex(RuntimeError, "declaration mismatch"):
                helper.inspect_prepared_checkpoint(root, environ={})
            with self.assertRaisesRegex(RuntimeError, "sidecar/env pin"):
                helper.inspect_prepared_checkpoint(
                    root,
                    environ={
                        helper.PREPARED_LOAD_ENV: "1",
                        helper.PREPARED_MANIFEST_SHA256_ENV: "f" * 64,
                    },
                )
            config = json.loads((root / "config.json").read_text())
            config["dspark_nvfp4_prepared"]["tp_size"] = 4
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "config marker"):
                helper.inspect_prepared_checkpoint(root, environ=environ)

    def test_ordinary_checkpoint_without_index_metadata_is_unchanged(self):
        helper = self.helper
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
            )
            (root / helper.INDEX_NAME).write_text(
                json.dumps({"weight_map": {}}), encoding="utf-8"
            )
            self.assertIsNone(
                helper.inspect_prepared_checkpoint(root, environ={})
            )

    def test_direct_read_flag_defaults_on_and_has_strict_opt_out(self):
        helper = self.helper
        self.assertTrue(helper.prepared_direct_read_requested({}))
        self.assertTrue(
            helper.prepared_direct_read_requested(
                {helper.PREPARED_DIRECT_READ_ENV: "1"}
            )
        )
        self.assertFalse(
            helper.prepared_direct_read_requested(
                {helper.PREPARED_DIRECT_READ_ENV: "0"}
            )
        )
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            helper.prepared_direct_read_requested(
                {helper.PREPARED_DIRECT_READ_ENV: "true"}
            )

    def test_direct_reader_reads_only_selected_rank_ranges(self):
        helper = self.helper
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = "model-layer-00000.safetensors"
            shapes, expected = self._write_tiny_prepared_layer(root / filename)
            contract = helper.PreparedCheckpointContract(
                checkpoint=root,
                manifest_sha256="a" * 64,
                output_index_sha256="b" * 64,
                layer_files=(filename,),
            )
            copied = {}

            def copy_bytes(destination, buffer, nbytes):
                copied[destination] = bytes(memoryview(buffer)[:nbytes])

            reader = helper.PreparedSafetensorsDirectReader(
                torch_module=_FakeTorch,
                contract=contract,
                tp_rank=1,
                source_shapes=shapes,
                copy_bytes_fn=copy_bytes,
            )
            for family in helper.PREPARED_FAMILY_ORDER:
                reader.copy_into(
                    layer=0,
                    family=family,
                    destination=family,
                )
            stats = reader.layer_stats(0)
            self.assertEqual(copied, expected[1])
            self.assertEqual(stats["ranges"], len(helper.PREPARED_FAMILY_ORDER))
            self.assertEqual(stats["syscalls"], len(helper.PREPARED_FAMILY_ORDER))
            self.assertEqual(stats["bytes"], sum(map(len, expected[1].values())))
            self.assertEqual(reader.summary()["ranges"], 8)
            self.assertEqual(reader.summary()["bytes"], stats["bytes"])
            reader.finish()

    def test_direct_reader_header_and_short_read_fail_closed(self):
        helper = self.helper
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-layer-00000.safetensors"
            shapes, _ = self._write_tiny_prepared_layer(path)
            fd = os.open(path, os.O_RDONLY)
            try:
                ranges = helper._parse_prepared_rank_ranges(
                    fd,
                    path=path,
                    layer=0,
                    tp_rank=0,
                    source_shapes=shapes,
                )
            finally:
                os.close(fd)
            self.assertEqual(set(ranges), set(helper.PREPARED_FAMILY_ORDER))

            data = bytearray(8)
            calls = []

            def partial(_fd, views, offset):
                calls.append(offset)
                if len(calls) == 1:
                    views[0][:3] = b"abc"
                    return 3
                return 0

            with self.assertRaisesRegex(RuntimeError, "ended before"):
                helper._preadv_exact_into(
                    1, memoryview(data), 11, preadv_fn=partial
                )
            self.assertEqual(calls, [11, 14])

            raw = bytearray(path.read_bytes())
            header_size = int.from_bytes(raw[:8], "little")
            header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
            first_name = next(
                name for name in header if name != "__metadata__"
            )
            header[first_name]["dtype"] = "F16"
            replacement = json.dumps(header, separators=(",", ":")).encode("utf-8")
            replacement += b" " * (header_size - len(replacement))
            self.assertEqual(len(replacement), header_size)
            raw[8 : 8 + header_size] = replacement
            path.write_bytes(raw)
            fd = os.open(path, os.O_RDONLY)
            try:
                with self.assertRaisesRegex(RuntimeError, "metadata drifted"):
                    helper._parse_prepared_rank_ranges(
                        fd,
                        path=path,
                        layer=0,
                        tp_rank=0,
                        source_shapes=shapes,
                    )
            finally:
                os.close(fd)

    def test_manifest_schema_provenance_integrity_and_output_sha_tamper_fail(self):
        helper = self.helper
        cases = (
            (
                "schema",
                lambda manifest: manifest.__setitem__("schema_version", 2),
                "manifest schema",
            ),
            (
                "preparation",
                lambda manifest: manifest["preparation"]["identity"].__setitem__(
                    "engine", "drifted"
                ),
                "identity 'engine' drifted",
            ),
            (
                "integrity",
                lambda manifest: manifest["integrity"].__setitem__(
                    "output_files_hashed", False
                ),
                "integrity contract",
            ),
            (
                "output-config",
                lambda manifest: manifest["output"].__setitem__(
                    "config_sha256", "0" * 64
                ),
                "output config digest",
            ),
            (
                "output-index",
                lambda manifest: manifest["output"].__setitem__(
                    "index_sha256", "0" * 64
                ),
                "output index digest",
            ),
        )
        for label, mutate, pattern in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_checkpoint_contract(root)
                manifest_path = root / helper.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                (root / helper.MANIFEST_DIGEST_NAME).write_text(
                    f"{digest}  {helper.MANIFEST_NAME}\n", encoding="ascii"
                )
                environ = {
                    helper.PREPARED_LOAD_ENV: "1",
                    helper.PREPARED_MANIFEST_SHA256_ENV: digest,
                }
                with self.assertRaisesRegex(RuntimeError, pattern):
                    helper.inspect_prepared_checkpoint(root, environ=environ)

    def test_physical_layer0_validator_uses_exact_loader_contract_and_fingerprints(self):
        helper = self.helper
        shapes = helper._source_shapes()
        dtypes = helper._family_dtypes(_FakeTorch)
        tensors = {
            (
                f"{helper.PREPARED_NAMESPACE}.layers.0.experts.{family}"
            ): _Tensor(shapes[family], dtype=dtypes[family], device="cpu")
            for family in helper.PREPARED_FAMILY_ORDER
        }
        calls = []

        def safe_open(path, *, framework, device):
            calls.append((path, framework, device))
            return _SafeOpenHandle(
                tensors, helper._expected_prepared_header_metadata(0)
            )

        def fingerprint(_tensor, family, _torch):
            return helper.LAYER0_RANK0_REFERENCE_FINGERPRINTS[family]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model-layer-00000.safetensors"
            path.write_bytes(b"physical-prepared-layer")
            evidence = helper.validate_prepared_layer_file(
                path,
                torch_module=_FakeTorch,
                safe_open_fn=safe_open,
                fingerprint_fn=fingerprint,
            )
            self.assertTrue(evidence["ok"])
            self.assertEqual(evidence["tensor_count"], 8)
            self.assertEqual(
                evidence["rank_bytes"],
                {
                    "0": helper.EXPECTED_RANK_BYTES,
                    "1": helper.EXPECTED_RANK_BYTES,
                },
            )
            self.assertEqual(
                evidence["rank0_fingerprints"],
                helper.LAYER0_RANK0_REFERENCE_FINGERPRINTS,
            )
            json.dumps(evidence)
            self.assertEqual(calls, [(str(path.resolve()), "pt", "cpu")])

            with self.assertRaisesRegex(RuntimeError, "fingerprints drifted"):
                helper.validate_prepared_layer_file(
                    path,
                    torch_module=_FakeTorch,
                    safe_open_fn=safe_open,
                    fingerprint_fn=lambda *_args: "0" * 64,
                )
            extra = dict(tensors)
            extra["unexpected"] = next(iter(tensors.values()))
            with self.assertRaisesRegex(RuntimeError, "tensor names drifted"):
                helper.validate_prepared_layer_file(
                    path,
                    torch_module=_FakeTorch,
                    safe_open_fn=lambda *_args, **_kwargs: _SafeOpenHandle(
                        extra, helper._expected_prepared_header_metadata(0)
                    ),
                    fingerprint_fn=fingerprint,
                )
            with self.assertRaisesRegex(RuntimeError, "header metadata drifted"):
                helper.validate_prepared_layer_file(
                    path,
                    torch_module=_FakeTorch,
                    safe_open_fn=lambda *_args, **_kwargs: _SafeOpenHandle(
                        tensors, {"format": "pt"}
                    ),
                    fingerprint_fn=fingerprint,
                )
            real_path = path.with_name("physical-layer-real.safetensors")
            path.rename(real_path)
            path.symlink_to(real_path)
            with self.assertRaisesRegex(RuntimeError, "direct regular file"):
                helper.validate_prepared_layer_file(
                    path,
                    torch_module=_FakeTorch,
                    safe_open_fn=safe_open,
                    fingerprint_fn=fingerprint,
                )

    def test_eight_direct_rank_copies_complete_one_layer(self):
        helper = self.helper
        source_shapes = {
            family: (2, 2) for family in helper.PREPARED_FAMILY_ORDER
        }
        dtypes = helper._family_dtypes(_FakeTorch)
        parameters = {
            0: {
                family: _Tensor((2,), dtype=dtypes[family], device="cuda")
                for family in helper.PREPARED_FAMILY_ORDER
            }
        }
        state = helper.PreparedPostloadState(0)
        loader = helper.Nvfp4PreparedLayerLoader(
            torch_module=_FakeTorch,
            tp_rank=1,
            parameters=parameters,
            states={0: state},
            expected_source_shapes=source_shapes,
            expected_rank_bytes=40,
        )
        loaded_names = []
        for family in helper.PREPARED_FAMILY_ORDER:
            name = f"{helper.PREPARED_NAMESPACE}.layers.0.experts.{family}"
            result = loader.consume(
                name,
                _Tensor(source_shapes[family], dtype=dtypes[family], device="cpu"),
            )
            loaded_names.append(result)
        loader.finish()
        self.assertTrue(state.loaded)
        self.assertEqual(loader.total_h2d_calls, 8)
        self.assertEqual(loader.completed_layers, frozenset({0}))
        self.assertEqual(len(set(loaded_names)), 8)
        self.assertTrue(
            all(
                tensor.storage["copies"] == 1
                for tensor in parameters[0].values()
            )
        )

    def test_second_nested_load_after_completion_is_verified_noop(self):
        helper = self.helper
        source_shapes = {
            family: (2, 2) for family in helper.PREPARED_FAMILY_ORDER
        }
        dtypes = helper._family_dtypes(_FakeTorch)
        parameters = {
            0: {
                family: _Tensor((2,), dtype=dtypes[family], device="cuda")
                for family in helper.PREPARED_FAMILY_ORDER
            }
        }
        state = helper.PreparedPostloadState(0)
        loader = helper.Nvfp4PreparedLayerLoader(
            torch_module=_FakeTorch,
            tp_rank=0,
            parameters=parameters,
            states={0: state},
            expected_source_shapes=source_shapes,
            expected_rank_bytes=40,
        )
        session = helper.Nvfp4PreparedLoadSession()
        session.begin(loader, prepared_requested=True)
        first = session.loader_for_nested_load(prepared_requested=True)
        self.assertIs(first, loader)
        for family in helper.PREPARED_FAMILY_ORDER:
            first.consume(
                f"{helper.PREPARED_NAMESPACE}.layers.0.experts.{family}",
                _Tensor(
                    source_shapes[family], dtype=dtypes[family], device="cpu"
                ),
            )
        second = session.loader_for_nested_load(prepared_requested=True)
        self.assertIsNone(second)
        self.assertEqual(session.nested_load_calls, 2)
        self.assertEqual(session.completed_noop_calls, 1)
        session.finish()
        self.assertEqual(loader.total_h2d_calls, 8)
        self.assertTrue(state.loaded)

    def test_prepared_postload_binds_final_scales_and_builds_kernel_once(self):
        helper = self.helper
        sentinel_backend = object()
        tensors = {
            name: object()
            for name in (
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
                "w13_weight_scale",
                "w2_weight_scale",
            )
        }
        routed = SimpleNamespace(
            **tensors,
            swiglu_limit=10.0,
            _expert_routing_tables=lambda: ("r0", "r1", "r2"),
        )
        quant_method = SimpleNamespace(
            nvfp4_backend=sentinel_backend,
            moe_quant_config=None,
            moe_kernel=None,
            moe="moe-config",
            experts_cls=object,
        )
        calls = {"quant": 0, "kernel": 0}

        def quant_factory(**kwargs):
            calls["quant"] += 1
            self.assertIs(kwargs["g1_alphas"], tensors["w13_weight_scale_2"])
            self.assertIs(kwargs["g2_alphas"], tensors["w2_weight_scale_2"])
            self.assertIs(kwargs["a1_gscale"], tensors["w13_input_scale"])
            self.assertIs(kwargs["a2_gscale"], tensors["w2_input_scale"])
            self.assertTrue(kwargs["is_scale_swizzled"])
            return "final-quant-config"

        def kernel_factory(**kwargs):
            calls["kernel"] += 1
            self.assertEqual(kwargs["moe_quant_config"], "final-quant-config")
            return "final-kernel"

        state = helper.PreparedPostloadState(0, loaded=True)
        helper._finalize_prepared_cutlass(
            quant_method,
            routed,
            state,
            quant_config_factory=quant_factory,
            kernel_factory=kernel_factory,
            expected_backend=sentinel_backend,
        )
        self.assertTrue(state.finalized)
        self.assertEqual(quant_method.moe_kernel, "final-kernel")
        self.assertEqual(calls, {"quant": 1, "kernel": 1})
        with self.assertRaisesRegex(RuntimeError, "loaded/unfinalized"):
            helper._finalize_prepared_cutlass(
                quant_method,
                routed,
                state,
                quant_config_factory=quant_factory,
                kernel_factory=kernel_factory,
                expected_backend=sentinel_backend,
            )

    def test_prepared_b12x_recovers_raw_scalars_then_runs_kernel_postload(self):
        helper = self.helper
        sentinel_backend = object()

        class ScalarTensor:
            def __init__(self, values):
                self.values = list(values)
                self.shape = (len(self.values),)

            @property
            def data(self):
                return self

            def mul_(self, other):
                self.values = [
                    left * right
                    for left, right in zip(self.values, other.values, strict=True)
                ]
                return self

        g1 = ScalarTensor([8.0, 15.0])
        g2 = ScalarTensor([12.0, 21.0])
        a1 = ScalarTensor([0.25, 0.2])
        a2 = ScalarTensor([0.5, 1.0 / 3.0])
        tensors = {
            "w13_weight_scale_2": g1,
            "w2_weight_scale_2": g2,
            "w13_input_scale": a1,
            "w2_input_scale": a2,
            "w13_weight_scale": object(),
            "w2_weight_scale": object(),
        }
        routed = SimpleNamespace(
            **tensors,
            swiglu_limit=10.0,
            _expert_routing_tables=lambda: ("r0", "r1", "r2"),
        )
        quant_method = SimpleNamespace(
            nvfp4_backend=sentinel_backend,
            moe_quant_config=None,
            moe_kernel=None,
            moe="moe-config",
            experts_cls=object,
        )
        calls = []

        def quant_factory(**kwargs):
            self.assertIs(kwargs["g1_alphas"], g1)
            self.assertIs(kwargs["g2_alphas"], g2)
            self.assertTrue(kwargs["is_scale_swizzled"])
            return "b12x-quant-config"

        kernel = SimpleNamespace()

        def postload(layer):
            self.assertIs(layer, routed)
            self.assertEqual(g1.values, [2.0, 3.0])
            self.assertEqual(g2.values, [6.0, 7.0])
            calls.append("postload")

        kernel.process_weights_after_loading = postload

        def kernel_factory(**kwargs):
            self.assertEqual(kwargs["moe_quant_config"], "b12x-quant-config")
            self.assertIs(kwargs["backend"], sentinel_backend)
            return kernel

        state = helper.PreparedPostloadState(0, loaded=True)
        helper._finalize_prepared_b12x(
            quant_method,
            routed,
            state,
            quant_config_factory=quant_factory,
            kernel_factory=kernel_factory,
            expected_backend=sentinel_backend,
        )
        self.assertTrue(state.finalized)
        self.assertEqual(calls, ["postload"])
        self.assertIs(quant_method.moe_kernel, kernel)

    def test_installed_hook_is_per_method_and_calls_only_prepared_finalizer(self):
        helper = self.helper

        def original(self, layer):
            raise AssertionError("native ModelOpt post-load must be bypassed")

        original.__module__ = "vllm.model_executor.layers.quantization.modelopt"
        original.__qualname__ = (
            "ModelOptNvFp4FusedMoE.process_weights_after_loading"
        )
        experts_cls = type("FlashInferExperts", (), {})
        experts_cls.__module__ = (
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe"
        )
        method_cls = type(
            "ModelOptNvFp4FusedMoE",
            (),
            {"process_weights_after_loading": original},
        )
        method = method_cls()
        method.nvfp4_backend = SimpleNamespace(value=helper.PREPARED_BACKEND)
        method.experts_cls = experts_cls
        routed = SimpleNamespace(quant_method=method)
        state = helper.PreparedPostloadState(0, loaded=True)
        calls = []
        with mock.patch.object(
            helper,
            "_finalize_prepared_cutlass",
            side_effect=lambda qm, layer, observed: calls.append(
                (qm, layer, observed)
            ),
        ):
            helper._install_prepared_postload_hook(routed, state)
            method.process_weights_after_loading(routed)
        self.assertEqual(calls, [(method, routed, state)])
        self.assertTrue(callable(method._dspark_prepared_original_postload))
        with self.assertRaisesRegex(RuntimeError, "installed twice"):
            helper._install_prepared_postload_hook(routed, state)

    def test_bypassed_runtime_transform_sources_are_sha_pinned(self):
        helper = self.helper
        method_cls = type("ModelOptNvFp4FusedMoE", (), {})
        method_cls.__module__ = (
            "vllm.model_executor.layers.quantization.modelopt"
        )
        experts_cls = type("FlashInferExperts", (), {})
        experts_cls.__module__ = (
            "vllm.model_executor.layers.fused_moe.experts."
            "flashinfer_cutlass_moe"
        )
        method = method_cls()
        method.nvfp4_backend = SimpleNamespace(value=helper.PREPARED_BACKEND)
        method.experts_cls = experts_cls
        routed = SimpleNamespace(quant_method=method)
        paths = ["/pinned/modelopt.py", "/pinned/flashinfer_cutlass_moe.py"]
        hashes = [
            helper.PINNED_PREPARATION_SOURCE_SHA256["modelopt"],
            helper.PINNED_PREPARATION_SOURCE_SHA256["flashinfer_experts"],
        ]
        with (
            mock.patch.object(helper.inspect, "getsourcefile", side_effect=paths),
            mock.patch.object(helper, "_sha256_file", side_effect=hashes),
        ):
            helper._validate_runtime_transform_sources(routed)

    def test_b12x_runtime_transform_source_is_sha_pinned(self):
        helper = self.helper
        method_cls = type("ModelOptNvFp4FusedMoE", (), {})
        method_cls.__module__ = (
            "vllm.model_executor.layers.quantization.modelopt"
        )
        experts_cls = type("FlashInferB12xExperts", (), {})
        experts_cls.__module__ = (
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe"
        )
        method = method_cls()
        method.nvfp4_backend = SimpleNamespace(
            value=helper.PREPARED_B12X_BACKEND
        )
        method.experts_cls = experts_cls
        routed = SimpleNamespace(quant_method=method)
        paths = ["/pinned/modelopt.py", "/pinned/flashinfer_b12x_moe.py"]
        hashes = [
            helper.PINNED_PREPARATION_SOURCE_SHA256["modelopt"],
            helper.PINNED_PREPARATION_SOURCE_SHA256[
                "flashinfer_b12x_experts"
            ],
        ]
        with (
            mock.patch.object(helper.inspect, "getsourcefile", side_effect=paths),
            mock.patch.object(helper, "_sha256_file", side_effect=hashes),
        ):
            helper._validate_runtime_transform_sources(routed)
        with (
            mock.patch.object(helper.inspect, "getsourcefile", side_effect=paths),
            mock.patch.object(helper, "_sha256_file", return_value="0" * 64),
            self.assertRaisesRegex(RuntimeError, "source digest drifted"),
        ):
            helper._validate_runtime_transform_sources(routed)

    def test_factory_full_lifecycle_is_43_layers_344_copies_and_one_finalize(self):
        helper = self.helper

        def original(self, layer):
            raise AssertionError("ordinary ModelOpt post-load must not run")

        original.__module__ = "vllm.model_executor.layers.quantization.modelopt"
        original.__qualname__ = (
            "ModelOptNvFp4FusedMoE.process_weights_after_loading"
        )
        experts_cls = type("FlashInferExperts", (), {})
        experts_cls.__module__ = (
            "vllm.model_executor.layers.fused_moe.experts."
            "flashinfer_cutlass_moe"
        )
        method_cls = type(
            "ModelOptNvFp4FusedMoE",
            (),
            {"process_weights_after_loading": original},
        )
        destination_shapes = helper._destination_shapes()
        dtypes = helper._family_dtypes(_FakeTorch)
        routed_layers = {}
        for layer in range(helper.EXPECTED_LAYERS):
            method = method_cls()
            method.nvfp4_backend = SimpleNamespace(
                value=helper.PREPARED_BACKEND
            )
            method.experts_cls = experts_cls
            method.moe_quant_config = None
            method.moe_kernel = None
            routed = SimpleNamespace(quant_method=method)
            for family, basename in helper._FAMILY_TO_PARAMETER.items():
                shape = destination_shapes[family]
                if basename in ("w13_weight_scale_2", "w13_input_scale"):
                    shape = (helper.EXPECTED_EXPERTS, 2)
                setattr(
                    routed,
                    basename,
                    _Tensor(shape, dtype=dtypes[family], device="cuda"),
                )
            routed_layers[layer] = routed

        quant_config_cls = type("DeepseekV4FP8Config", (), {})
        quant_config = quant_config_cls()
        quant_config.expert_dtype = "fp4"
        quant_config.moe_quant_algo = "NVFP4"
        replacements = []

        def replace_parameter(module, name, value):
            replacements.append((module, name, value))
            setattr(module, name, value)

        contract = helper.PreparedCheckpointContract(
            checkpoint=Path("/prepared"),
            manifest_sha256="a" * 64,
            output_index_sha256="b" * 64,
            layer_files=tuple(
                f"model-layer-{layer:05d}.safetensors"
                for layer in range(helper.EXPECTED_LAYERS)
            ),
        )
        with (
            mock.patch.object(
                helper, "inspect_prepared_checkpoint", return_value=contract
            ),
            mock.patch.object(helper, "_validate_runtime_transform_sources"),
        ):
            loader = helper.maybe_create_nvfp4_prepared_loader(
                torch_module=_FakeTorch,
                checkpoint="/prepared",
                routed_layers=routed_layers,
                start_layer=0,
                end_layer=helper.EXPECTED_LAYERS,
                num_hidden_layers=helper.EXPECTED_LAYERS,
                num_routed_experts=helper.EXPECTED_EXPERTS,
                tp_size=helper.EXPECTED_TP_SIZE,
                tp_rank=1,
                use_mega_moe=False,
                enable_expert_parallel=False,
                num_redundant_experts=0,
                load_format="auto",
                quant_config=quant_config,
                environ={helper.PREPARED_DIRECT_READ_ENV: "0"},
                replace_parameter_fn=replace_parameter,
            )
        self.assertIsNotNone(loader)
        self.assertEqual(len(replacements), helper.EXPECTED_LAYERS * 2)
        self.assertEqual(
            len({id(routed.quant_method) for routed in routed_layers.values()}),
            helper.EXPECTED_LAYERS,
        )
        for routed in routed_layers.values():
            self.assertEqual(
                routed.w13_weight_scale_2.shape,
                (helper.EXPECTED_EXPERTS,),
            )
            self.assertEqual(
                routed.w13_input_scale.shape,
                (helper.EXPECTED_EXPERTS,),
            )

        session = helper.Nvfp4PreparedLoadSession()
        session.begin(loader, prepared_requested=True)
        active = session.loader_for_nested_load(prepared_requested=True)
        self.assertIs(active, loader)
        source_shapes = helper._source_shapes()
        for layer in range(helper.EXPECTED_LAYERS):
            for family in helper.PREPARED_FAMILY_ORDER:
                active.consume(
                    (
                        f"{helper.PREPARED_NAMESPACE}.layers.{layer}.experts."
                        f"{family}"
                    ),
                    _Tensor(
                        source_shapes[family],
                        dtype=dtypes[family],
                        device="cpu",
                    ),
                )

        finalized = []

        def finalize(method, routed, state):
            self.assertTrue(state.loaded)
            self.assertFalse(state.finalized)
            state.finalized = True
            finalized.append((method, routed, state.layer))

        with mock.patch.object(
            helper, "_finalize_prepared_cutlass", side_effect=finalize
        ):
            for routed in routed_layers.values():
                routed.quant_method.process_weights_after_loading(routed)
        self.assertEqual(len(finalized), helper.EXPECTED_LAYERS)
        self.assertIsNone(
            session.loader_for_nested_load(prepared_requested=True)
        )
        session.finish()
        self.assertEqual(
            loader.total_h2d_calls,
            helper.EXPECTED_LAYERS * helper.EXPECTED_H2D_CALLS_PER_LAYER,
        )
        self.assertEqual(session.nested_load_calls, 2)
        self.assertEqual(session.completed_noop_calls, 1)

    def test_model_routes_custom_namespace_before_ordinary_expert_mapping(self):
        source = MODEL_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'f"{PREPARED_NAMESPACE}.": f"model.{PREPARED_NAMESPACE}.",', source
        )
        load_start = source.index("    def load_weights(", source.index("class DeepseekV4Model"))
        load_end = source.index("    def _pad_shared_expert_weight", load_start)
        body = source[load_start:load_end]
        self.assertLess(
            body.index("prepared_loader.consume(name, loaded_weight)"),
            body.index('if ".experts." in name:'),
        )
        prepared_source = HELPER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "_validate_runtime_transform_sources(routed_layers[0])",
            prepared_source,
        )
        self.assertNotIn("prepare_nvfp4_moe_layer_for_fi_or_cutlass", prepared_source)
        self.assertNotIn("fused_experts.process_weights_after_loading", prepared_source)


if __name__ == "__main__":
    unittest.main()
