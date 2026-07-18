# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "verify_nvfp4_dspark_split_contract.py"
SPEC = importlib.util.spec_from_file_location("split_contract", PATH)
assert SPEC and SPEC.loader
split = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = split
SPEC.loader.exec_module(split)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def write_safetensors(path: Path, rows: dict[str, tuple[str, list[int], int]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape, size) in rows.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


class SplitContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.draft = self.root / "draft"
        self.target.mkdir()
        self.draft.mkdir()
        common = {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "hidden_size": 4096,
            "vocab_size": 129280,
            "num_attention_heads": 64,
            "num_hidden_layers": 43,
            "n_routed_experts": 256,
            "moe_intermediate_size": 2048,
            "expert_dtype": "fp4",
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "rms_norm_eps": 1e-6,
            "same_extra_field": "pinned",
        }
        marker = {
            "schema": split.PREPARED_SCHEMA,
            "loader_contract": split.PREPARED_LOADER_CONTRACT,
            "required_backend": split.PREPARED_BACKEND,
            "tp_size": 2,
            "vllm_layout_pin": split.PINNED_VLLM,
            "manifest": split.PREPARED_MANIFEST,
            "manifest_digest": split.PREPARED_MANIFEST_DIGEST,
        }
        target_config = {
            **common,
            "compress_ratios": list(range(44)),
            "quantization_config": {"moe_quant_algo": "NVFP4"},
            "dspark_nvfp4_prepared": marker,
        }
        draft_config = {
            **common,
            "compress_ratios": [*range(44), 0, 0],
            "quantization_config": {"quant_method": "fp8"},
            **split.EXPECTED_DSPARK_FIELDS,
        }
        write_json(self.target / "config.json", target_config)
        write_json(
            self.target / "model.safetensors.index.json",
            {"metadata": {"total_size": 1}, "weight_map": {"x": "target.safetensors"}},
        )
        manifest = {
            "format": split.PREPARED_SCHEMA,
            "integrity": {
                "output_files_hashed": True,
                "output_tensors_hashed": True,
            },
            "loader": {
                "required_backend": split.PREPARED_BACKEND,
                "required_runtime_transforms": [],
                "runtime_h2d_calls_per_layer": 8,
                "runtime_source_reads_per_layer": 8,
            },
            "output": {
                "config_sha256": split.sha256(self.target / "config.json"),
                "index_sha256": split.sha256(
                    self.target / "model.safetensors.index.json"
                ),
                "tensor_count": 1,
                "layer_file_count": 43,
            },
        }
        write_json(self.target / split.PREPARED_MANIFEST, manifest)
        manifest_sha = split.sha256(self.target / split.PREPARED_MANIFEST)
        (self.target / split.PREPARED_MANIFEST_DIGEST).write_text(
            f"{manifest_sha}  {split.PREPARED_MANIFEST}\n", encoding="ascii"
        )

        weight_map = {"target.weight": "model-00001-of-00048.safetensors"}
        for stage, shard in split.EXPECTED_STAGE_SHARDS.items():
            rows = {
                f"mtp.{stage}.ffn.experts.0.w1.weight": ("I8", [2, 2], 4),
                f"mtp.{stage}.ffn.experts.0.w1.scale": ("F8_E8M0", [2], 2),
                f"mtp.{stage}.norm.weight": ("BF16", [2], 4),
            }
            write_safetensors(self.draft / shard, rows)
            weight_map.update({name: shard for name in rows})
        write_json(self.draft / "config.json", draft_config)
        write_json(
            self.draft / "model.safetensors.index.json",
            {"metadata": {"total_size": 30}, "weight_map": weight_map},
        )
        self.args = argparse.Namespace(
            target_dir=self.target,
            draft_dir=self.draft,
            expected_target_manifest_sha256=manifest_sha,
            expected_draft_config_sha256=split.sha256(self.draft / "config.json"),
            expected_draft_index_sha256=split.sha256(
                self.draft / "model.safetensors.index.json"
            ),
            tp_size=2,
            num_speculative_tokens=5,
            usable_memory_gib_per_rank=121.0,
            target_only_model_gib=78.11,
            observed_target_only_kv_gib=10.8,
            loader_overhead_fraction=0.15,
            graph_workspace_reserve_gib=4.0,
            system_safety_reserve_gib=2.0,
            minimum_kv_gib=30.0,
        )
        self.patches = (
            mock.patch.object(split, "EXPECTED_DRAFT_TOTAL_SIZE", 30),
            mock.patch.object(split, "EXPECTED_DRAFT_TENSOR_COUNT", 10),
            mock.patch.object(split, "EXPECTED_STAGE_COUNTS", {0: 3, 1: 3, 2: 3}),
            mock.patch.object(split, "EXPECTED_EXPERT_TENSORS_PER_STAGE", 2),
            mock.patch.object(
                split,
                "EXPECTED_EXPERT_SUFFIX_COUNTS",
                split.Counter({"weight": 1, "scale": 1}),
            ),
            mock.patch.object(
                split,
                "EXPECTED_DRAFT_SHARD_SHA256",
                {
                    stage: split.sha256(self.draft / shard)
                    for stage, shard in split.EXPECTED_STAGE_SHARDS.items()
                },
            ),
        )
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def test_split_contract_and_memory_projection_pass(self) -> None:
        report = split.verify(self.args)
        self.assertTrue(report["ok"])
        self.assertTrue(report["compatibility"]["passed"])
        self.assertTrue(report["memory"]["passed"])
        self.assertEqual(report["draft"]["mtp_payload_bytes"], 30)
        self.assertEqual(report["draft"]["expert_payload_bytes"], 18)
        self.assertEqual(report["draft"]["nonexpert_payload_bytes"], 12)
        self.assertEqual(report["memory"]["rank_parameter_bytes"], 21)
        self.assertTrue(report["memory"]["configuration_retune_required"])
        self.assertGreater(report["memory"]["projected_remaining_kv_gib"], 30.0)
        self.assertEqual(
            report["memory"]["recommended_kv_cache_memory_bytes"], 30 * (1 << 30)
        )
        self.assertEqual(
            report["memory"]["recommended_allocation_mode"],
            "explicit --kv-cache-memory-bytes",
        )
        self.assertEqual(report["runtime"]["num_speculative_tokens"], 5)
        self.assertEqual(report["runtime"]["native_mtp_stage_count"], 3)

    def test_unapproved_config_difference_fails(self) -> None:
        path = self.draft / "config.json"
        config = json.loads(path.read_text())
        config["hidden_size"] = 8192
        write_json(path, config)
        self.args.expected_draft_config_sha256 = split.sha256(path)
        with self.assertRaisesRegex(split.ContractError, "incompatible config field"):
            split.verify(self.args)

    def test_draft_identity_drift_fails(self) -> None:
        self.args.expected_draft_index_sha256 = "0" * 64
        with self.assertRaisesRegex(split.ContractError, "draft index SHA-256"):
            split.verify(self.args)

    def test_manifest_digest_filename_drift_fails(self) -> None:
        digest = self.args.expected_target_manifest_sha256
        (self.target / split.PREPARED_MANIFEST_DIGEST).write_text(
            f"{digest}  wrong.json\n", encoding="ascii"
        )
        with self.assertRaisesRegex(split.ContractError, "digest filename drifted"):
            split.verify(self.args)

    def test_kv_reserve_fails_closed(self) -> None:
        self.args.usable_memory_gib_per_rank = 110.0
        self.args.minimum_kv_gib = 30.0
        with self.assertRaisesRegex(split.ContractError, "projected KV reserve"):
            split.verify(self.args)


if __name__ == "__main__":
    unittest.main()
