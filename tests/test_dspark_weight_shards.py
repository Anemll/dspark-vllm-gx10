# SPDX-License-Identifier: MIT
"""Unit coverage for DSpark's local checkpoint shard selection."""

import ast
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/vllm/models/deepseek_v4/nvidia/dspark.py"
UPSTREAM = ROOT / ".build/vllm-upstream/vllm/models/deepseek_v4/nvidia/dspark.py"


def load_selector():
    tree = ast.parse(SOURCE.read_text())
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_local_mtp_safetensors_patterns"
    )
    logger = SimpleNamespace(info_once=lambda *args, **kwargs: None,
                             warning_once=lambda *args, **kwargs: None)
    namespace = {"json": json, "Path": Path, "logger": logger}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[copy.deepcopy(selector)], type_ignores=[])
            ),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace[selector.name]


class DSparkWeightShardTests(unittest.TestCase):
    def setUp(self):
        self.selector = load_selector()

    def write_index(self, directory, weight_map):
        (Path(directory) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    def test_only_mtp_shards_are_selected(self):
        with TemporaryDirectory() as directory:
            self.write_index(
                directory,
                {
                    "model.layers.0.weight": "model-00001-of-00048.safetensors",
                    "mtp.0.main_proj.weight": "model-00046-of-00048.safetensors",
                    "mtp.1.experts.3.w1.weight": "model-00047-of-00048.safetensors",
                    "mtp.2.markov_head.bias": "model-00048-of-00048.safetensors",
                    "mtp.2.duplicate": "model-00048-of-00048.safetensors",
                },
            )
            self.assertEqual(
                self.selector(directory),
                ["model-0004[678]-of-00048.safetensors"],
            )

    def test_missing_non_mtp_and_invalid_indexes_fail_open(self):
        with TemporaryDirectory() as directory:
            self.assertIsNone(self.selector(directory))
            self.write_index(directory, {"model.layers.0.weight": "model.safetensors"})
            self.assertIsNone(self.selector(directory))
            (Path(directory) / "model.safetensors.index.json").write_text("not json")
            self.assertIsNone(self.selector(directory))

    def test_selector_is_wired_through_the_standard_loader_hook(self):
        source = SOURCE.read_text()
        self.assertIn("self.allow_patterns_overrides = _local_mtp_safetensors_patterns(", source)
        self.assertIn("allow_patterns_overrides=getattr(model", (
            ROOT / ".build/vllm-upstream/vllm/model_executor/model_loader/default_loader.py"
        ).read_text())

    def test_pinned_source_did_not_select_draft_shards(self):
        if not UPSTREAM.exists():
            self.skipTest("optional pristine upstream source absent")
        self.assertNotIn("_local_mtp_safetensors_patterns", UPSTREAM.read_text())

    def test_component_image_copies_the_dspark_model_overlay(self):
        dockerfile = (ROOT / "docker/Dockerfile.nvfp4-cutlass").read_text()
        self.assertIn(
            "COPY overlay/vllm/models/deepseek_v4/nvidia/dspark.py "
            "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
