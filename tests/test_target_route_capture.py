# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "overlay/vllm/v1/worker/gpu/target_route_capture.py"
)
SPEC = importlib.util.spec_from_file_location("target_route_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)

ANALYZER_PATH = ROOT / "scripts/analyze_target_route_capture.py"
DOCKERFILE_PATH = ROOT / "docker/Dockerfile.target-route-capture-overlay"
DOCKERIGNORE_PATH = (
    ROOT / "docker/Dockerfile.target-route-capture-overlay.dockerignore"
)
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "analyze_target_route_capture", ANALYZER_PATH
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
analyzer = importlib.util.module_from_spec(ANALYZER_SPEC)
sys.modules[ANALYZER_SPEC.name] = analyzer
ANALYZER_SPEC.loader.exec_module(analyzer)


class FakeRouter:
    def __init__(self) -> None:
        self.top_k = 6
        self.global_num_experts = 256
        self.capture_fn = None

    def set_capture_fn(self, callback) -> None:
        self.capture_fn = callback


class FakeRunner:
    def __init__(self, layer: int) -> None:
        self.layer_name = f"model.layers.{layer}.ffn.experts"
        self.layer_id = layer
        self.router = FakeRouter()


def make_batch(**overrides):
    values = {
        "num_reqs": 4,
        "num_tokens": 4,
        "num_scheduled_tokens": np.ones(4, dtype=np.int32),
        "num_draft_tokens": 0,
        "num_draft_tokens_per_req": None,
        "is_prefilling_np": np.zeros(4, dtype=np.bool_),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TargetRouteCaptureTests(unittest.TestCase):
    def test_dedicated_image_bakes_exact_route_runner(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text()
        dockerignore = DOCKERIGNORE_PATH.read_text()
        runner_path = ROOT / "overlay/vllm/v1/worker/gpu/model_runner.py"
        helper_path = (
            ROOT / "overlay/vllm/v1/worker/gpu/target_route_capture.py"
        )
        runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
        helper_sha = hashlib.sha256(helper_path.read_bytes()).hexdigest()

        self.assertIn(
            "ARG BASE_V2_MODEL_RUNNER_SHA256="
            "58f45c58969cdd9cba707863e82fefda818002de45c621032b58b6eb364deedf",
            dockerfile,
        )
        self.assertIn(
            f"ARG ROUTE_V2_MODEL_RUNNER_SHA256={runner_sha}", dockerfile
        )
        self.assertIn(
            f"ARG TARGET_ROUTE_CAPTURE_SHA256={helper_sha}", dockerfile
        )
        base_hash_index = dockerfile.index(
            'echo "${BASE_V2_MODEL_RUNNER_SHA256}'
        )
        runner_copy_index = dockerfile.index(
            "COPY overlay/vllm/v1/worker/gpu/model_runner.py"
        )
        helper_copy_index = dockerfile.index(
            "COPY overlay/vllm/v1/worker/gpu/target_route_capture.py"
        )
        final_hash_index = dockerfile.index(
            'echo "${ROUTE_V2_MODEL_RUNNER_SHA256}'
        )
        self.assertLess(base_hash_index, runner_copy_index)
        self.assertLess(runner_copy_index, helper_copy_index)
        self.assertLess(helper_copy_index, final_hash_index)
        pycompile_index = dockerfile.index("python3 -m py_compile")
        self.assertIn(
            "/vllm/v1/worker/gpu/target_route_capture.py",
            dockerfile[pycompile_index:],
        )
        self.assertIn(
            "!overlay/vllm/v1/worker/gpu/model_runner.py", dockerignore
        )
        self.assertIn(
            "!overlay/vllm/v1/worker/gpu/target_route_capture.py",
            dockerignore,
        )

    def test_runner_keeps_helper_imports_out_of_module_scope(self) -> None:
        runner_path = ROOT / "overlay/vllm/v1/worker/gpu/model_runner.py"
        tree = ast.parse(runner_path.read_text())
        top_level_imports = [
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        self.assertNotIn(
            "vllm.v1.worker.gpu.target_route_capture", top_level_imports
        )
        source = runner_path.read_text()
        self.assertIn("bind_target_route_capture", source)
        self.assertIn("target_route_capture.begin_step", source)
        self.assertIn("target_route_capture.end_step", source)

    def test_compose_propagates_only_opt_in_capture_settings(self) -> None:
        source = (ROOT / "docker-compose.yml").read_text()
        for name in (
            capture.CAPTURE_ENV,
            capture.OUTPUT_DIR_ENV,
            capture.STEPS_ENV,
            capture.WARMUP_STEPS_ENV,
        ):
            self.assertIn(f'{name}: "${{{name}:-', source)

    def test_disabled_environment_is_inert(self) -> None:
        self.assertIsNone(
            capture.TargetRouteCaptureConfig.from_environment(
                {
                    capture.CAPTURE_ENV: "0",
                    capture.STEPS_ENV: "not-an-integer",
                    capture.OUTPUT_DIR_ENV: "relative",
                }
            )
        )

    def test_enabled_environment_is_strict_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = capture.TargetRouteCaptureConfig.from_environment(
                {
                    capture.CAPTURE_ENV: "1",
                    capture.OUTPUT_DIR_ENV: directory,
                    capture.STEPS_ENV: "3",
                    capture.WARMUP_STEPS_ENV: "2",
                }
            )
            assert config is not None
            self.assertEqual(config.steps, 3)
            self.assertEqual(config.warmup_steps, 2)
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            capture.TargetRouteCaptureConfig.from_environment(
                {capture.CAPTURE_ENV: "true"}
            )

    def test_target_only_runtime_rejects_draft_and_return_routes(self) -> None:
        valid = dict(
            speculative_config=None,
            speculator=None,
            num_speculative_steps=0,
            enable_return_routed_experts=False,
            tensor_parallel_size=2,
        )
        capture.validate_target_only_runtime(**valid)
        for key, value in (
            ("speculative_config", object()),
            ("speculator", object()),
            ("num_speculative_steps", 1),
            ("enable_return_routed_experts", True),
            ("tensor_parallel_size", 1),
        ):
            broken = dict(valid)
            broken[key] = value
            with self.assertRaises(RuntimeError, msg=key):
                capture.validate_target_only_runtime(**broken)

    def test_loaded_target_model_contract_is_exact(self) -> None:
        model_type = type("DeepseekV4ForCausalLM", (), {})
        model = model_type()
        model.config = SimpleNamespace(
            num_hidden_layers=43,
            n_routed_experts=256,
            num_experts_per_tok=6,
        )
        capture.validate_loaded_target_model(model)
        model.config.num_experts_per_tok = 5
        with self.assertRaisesRegex(RuntimeError, "num_experts_per_tok drift"):
            capture.validate_loaded_target_model(model)
        with self.assertRaisesRegex(RuntimeError, "requires DeepseekV4ForCausalLM"):
            capture.validate_loaded_target_model(SimpleNamespace())

    def test_only_real_target_c4_one_token_steps_qualify(self) -> None:
        self.assertTrue(capture.is_steady_target_c4_step(make_batch(), dummy_run=False))
        rejected = (
            (make_batch(), True),
            (make_batch(num_reqs=3), False),
            (make_batch(num_tokens=8), False),
            (make_batch(num_scheduled_tokens=np.array([1, 1, 1, 2])), False),
            (make_batch(num_draft_tokens=1), False),
            (make_batch(num_draft_tokens_per_req=np.array([0, 0, 1, 0])), False),
            (make_batch(is_prefilling_np=np.array([False, False, True, False])), False),
        )
        for batch, dummy_run in rejected:
            self.assertFalse(
                capture.is_steady_target_c4_step(batch, dummy_run=dummy_run)
            )

    def test_router_binding_requires_exact_target_layer_set(self) -> None:
        runners = {
            f"model.layers.{layer}.ffn.experts": FakeRunner(layer)
            for layer in capture.EXPECTED_LAYERS
        }
        draft = FakeRunner(0)
        draft.layer_name = "model.mtp.0.ffn.experts"
        runners[draft.layer_name] = draft
        selected = capture.collect_target_routers(
            runners, runner_type=FakeRunner, router_type=FakeRouter
        )
        self.assertEqual(tuple(sorted(selected)), capture.EXPECTED_LAYERS)

        missing = dict(runners)
        del missing["model.layers.42.ffn.experts"]
        with self.assertRaisesRegex(RuntimeError, r"missing=\[42\]"):
            capture.collect_target_routers(
                missing, runner_type=FakeRunner, router_type=FakeRouter
            )

        runners["model.layers.0.ffn.experts"].router.capture_fn = object()
        with self.assertRaisesRegex(RuntimeError, "already has"):
            capture.collect_target_routers(
                runners, runner_type=FakeRunner, router_type=FakeRouter
            )

    def test_bounded_reverse_order_capture_writes_rank_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = capture.TargetRouteCaptureConfig(
                output_dir=Path(directory), steps=2, warmup_steps=1
            )
            collector = capture.TargetRouteCapture(
                config,
                device=torch.device("cpu"),
                rank=1,
                world_size=2,
                layer_names=[
                    f"model.layers.{layer}.ffn.experts"
                    for layer in capture.EXPECTED_LAYERS
                ],
            )
            batch = make_batch()
            self.assertFalse(collector.begin_step(batch, dummy_run=False))

            expected_steps = []
            manifest_path = None
            for step in range(2):
                self.assertTrue(collector.begin_step(batch, dummy_run=False))
                expected = np.empty((4, 43, 6), dtype=np.int32)
                for layer in reversed(capture.EXPECTED_LAYERS):
                    values = (
                        np.arange(24, dtype=np.int32).reshape(4, 6)
                        + layer * 7
                        + step
                    ) % 256
                    expected[:, layer, :] = values
                    collector.callback(layer, torch.from_numpy(values))
                expected_steps.append(expected)
                manifest_path = collector.end_step()

            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            data_path = Path(directory) / "target-routes-rank-1.npy"
            actual = np.load(data_path, allow_pickle=False)
            np.testing.assert_array_equal(actual, np.stack(expected_steps))
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["shape"], [2, 4, 43, 6])
            self.assertEqual(manifest["rank"], 1)
            self.assertEqual(manifest["world_size"], 2)
            self.assertEqual(manifest["eligible_steps_seen"], 3)
            self.assertEqual(
                manifest["data_sha256"],
                hashlib.sha256(data_path.read_bytes()).hexdigest(),
            )

    def test_shape_and_missing_callback_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collector = capture.TargetRouteCapture(
                capture.TargetRouteCaptureConfig(
                    output_dir=Path(directory), steps=1, warmup_steps=0
                ),
                device=torch.device("cpu"),
                rank=0,
                world_size=2,
                layer_names=[
                    f"model.layers.{layer}.ffn.experts"
                    for layer in capture.EXPECTED_LAYERS
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "shape drift"):
                collector.callback(0, torch.zeros((4, 5), dtype=torch.int32))
            self.assertTrue(collector.begin_step(make_batch(), dummy_run=False))
            values = torch.zeros((4, 6), dtype=torch.int32)
            for layer in capture.EXPECTED_LAYERS[:-1]:
                collector.callback(layer, values)
            with self.assertRaisesRegex(RuntimeError, "missing/out-of-range"):
                collector.end_step()
            self.assertFalse(
                (Path(directory) / "target-routes-rank-0.npy").exists()
            )

    def test_rank_pair_analysis_proves_equality_and_collision_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            values = torch.tensor(
                [[0, 1, 2, 3, 4, 5]] * 4, dtype=torch.int32
            )
            for rank in (0, 1):
                rank_dir = root / f"rank{rank}"
                rank_dir.mkdir()
                collector = capture.TargetRouteCapture(
                    capture.TargetRouteCaptureConfig(
                        output_dir=rank_dir, steps=1, warmup_steps=0
                    ),
                    device=torch.device("cpu"),
                    rank=rank,
                    world_size=2,
                    layer_names=[
                        f"model.layers.{layer}.ffn.experts"
                        for layer in capture.EXPECTED_LAYERS
                    ],
                )
                self.assertTrue(
                    collector.begin_step(make_batch(), dummy_run=False)
                )
                for layer in capture.EXPECTED_LAYERS:
                    collector.callback(layer, values)
                manifest = collector.end_step()
                assert manifest is not None
                manifests.append(manifest)

            result = analyzer.analyze_rank_pair(manifests[0], manifests[1])
            self.assertTrue(result["rank_pair_equal"])
            self.assertEqual(result["active_experts"]["mean"], 6.0)
            self.assertEqual(result["cross_token_collisions"]["mean"], 18.0)


if __name__ == "__main__":
    unittest.main()
