# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_hybrid_nvfp4_dspark_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("hybrid_checkpoint_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_safetensors(path: Path, tensor_names: list[str]) -> int:
    offset = 0
    header: dict[str, dict] = {}
    for tensor_name in tensor_names:
        header[tensor_name] = {
            "dtype": "U8",
            "shape": [4],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return offset


class HybridCheckpointBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.nvidia = self.root / "nvidia"
        self.dspark = self.root / "dspark"
        self.nvidia.mkdir()
        self.dspark.mkdir()
        self.contract = builder.HybridContract(
            nvidia=builder.IndexContract(
                total_size=8,
                tensor_count=2,
                non_mtp_count=1,
                mtp_counts=((0, 1),),
                shard_count=2,
                stage_shards=((0, 2),),
            ),
            dspark=builder.IndexContract(
                total_size=16,
                tensor_count=4,
                non_mtp_count=1,
                mtp_counts=((0, 1), (1, 1), (2, 1)),
                shard_count=4,
                stage_shards=((0, 2), (1, 3), (2, 4)),
            ),
            num_hidden_layers=1,
            nvidia_compress_ratios=(0, 0),
            dspark_compress_ratios=(0, 0, 0, 0),
            dspark_fields={
                "dspark_block_size": 5,
                "dspark_noise_token_id": 99,
                "dspark_target_layer_ids": [0, 0, 0],
                "dspark_markov_rank": 8,
            },
            expected_nvidia_revision="synthetic-revision",
        )
        self._write_sources()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _base_config(self) -> dict:
        return {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "num_hash_layers": 3,
            "num_nextn_predict_layers": 1,
            "expert_dtype": "fp4",
        }

    def _write_sources(self) -> None:
        nvidia_config = {
            **self._base_config(),
            "compress_ratios": [0, 0],
            "quantization_config": {
                "moe_quant_algo": "NVFP4",
                "quant_algo": "MIXED_PRECISION",
                "group_size": 16,
                "producer": {
                    "name": "modelopt",
                    "version": "dsv4-nvfp4-experts",
                },
                "ignore": ["mtp.*"],
                "quantized_layers": {
                    "layers.0.ffn.experts": {
                        "group_size": 16,
                        "quant_algo": "NVFP4",
                    }
                },
            },
        }
        dspark_config = {
            **self._base_config(),
            "compress_ratios": [0, 0, 0, 0],
            "quantization_config": {"quant_method": "fp8"},
            **self.contract.dspark_fields,
        }
        _write_json(self.nvidia / "config.json", nvidia_config)
        _write_json(self.dspark / "config.json", dspark_config)

        nvidia_index = {
            "metadata": {"total_size": 8},
            "weight_map": {
                "layers.0.ffn.experts.weight": "model-00001-of-00002.safetensors",
                "mtp.0.draft.weight": "model-00002-of-00002.safetensors",
            },
        }
        dspark_index = {
            "metadata": {"total_size": 16},
            "weight_map": {
                "layers.0.ffn.experts.weight": "model-00001-of-00004.safetensors",
                "mtp.0.draft.weight": "model-00002-of-00004.safetensors",
                "mtp.1.draft.weight": "model-00003-of-00004.safetensors",
                "mtp.2.draft.weight": "model-00004-of-00004.safetensors",
            },
        }
        _write_json(self.nvidia / "model.safetensors.index.json", nvidia_index)
        _write_json(self.dspark / "model.safetensors.index.json", dspark_index)
        _write_safetensors(
            self.nvidia / "model-00001-of-00002.safetensors",
            ["layers.0.ffn.experts.weight"],
        )
        _write_safetensors(
            self.nvidia / "model-00002-of-00002.safetensors",
            ["mtp.0.draft.weight"],
        )
        for stage, shard in enumerate((2, 3, 4)):
            _write_safetensors(
                self.dspark / f"model-{shard:05d}-of-00004.safetensors",
                [f"mtp.{stage}.draft.weight"],
            )
        for name in builder.RUNTIME_METADATA_FILES:
            _write_json(self.nvidia / name, {"source": "nvidia", "name": name})

        cache = self.nvidia / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True)
        (cache / "config.json.metadata").write_text(
            "synthetic-revision\nconfig-etag\n", encoding="utf-8"
        )
        (cache / "model-00001-of-00002.safetensors.metadata").write_text(
            "synthetic-revision\nshard-etag\n", encoding="utf-8"
        )

    def test_validate_and_build_symlink_view(self) -> None:
        sources = builder.validate_sources(
            self.nvidia, self.dspark, contract=self.contract
        )
        self.assertEqual(sources.merged_tensor_count, 4)
        self.assertEqual(sources.merged_total_size, 16)

        output = self.root / "hybrid"
        builder.build_hybrid_view(sources, output, mode="symlink")

        target = output / "model-00001-of-00004.safetensors"
        self.assertTrue(target.is_symlink())
        self.assertEqual(
            target.resolve(),
            (self.nvidia / "model-00001-of-00002.safetensors").resolve(),
        )
        for shard in (2, 3, 4):
            self.assertTrue(
                (output / f"model-{shard:05d}-of-00004.safetensors").is_symlink()
            )

        merged_index = json.loads(
            (output / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(merged_index["metadata"]["total_size"], 16)
        self.assertEqual(len(merged_index["weight_map"]), 4)
        self.assertEqual(
            merged_index["weight_map"]["layers.0.ffn.experts.weight"],
            "model-00001-of-00004.safetensors",
        )

        merged_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        nvidia_config = json.loads(
            (self.nvidia / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            merged_config["quantization_config"],
            nvidia_config["quantization_config"],
        )
        self.assertEqual(merged_config["dspark_block_size"], 5)
        self.assertEqual(merged_config["compress_ratios"], [0, 0, 0, 0])

        provenance = json.loads(
            (output / "checkpoint.provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["sources"]["nvidia"]["observed_revision"], "synthetic-revision")
        self.assertEqual(provenance["shards"][0]["hub_etag"], "shard-etag")
        self.assertIsNone(provenance["shards"][0]["sha256"])

    def test_rejects_nvidia_one_stage_source_as_dspark(self) -> None:
        nvidia_index = json.loads(
            (self.nvidia / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        _write_json(self.dspark / "model.safetensors.index.json", nvidia_index)
        with self.assertRaisesRegex(builder.ContractError, "NVIDIA-only one-stage"):
            builder.validate_sources(self.nvidia, self.dspark, contract=self.contract)

    def test_nonempty_output_requires_explicit_force(self) -> None:
        sources = builder.validate_sources(
            self.nvidia, self.dspark, contract=self.contract
        )
        output = self.root / "hybrid"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("do not overwrite", encoding="utf-8")

        with self.assertRaisesRegex(builder.ContractError, "--force"):
            builder.build_hybrid_view(sources, output, mode="symlink")
        self.assertEqual(marker.read_text(encoding="utf-8"), "do not overwrite")

        builder.build_hybrid_view(sources, output, mode="symlink", force=True)
        self.assertFalse(marker.exists())
        self.assertTrue((output / "checkpoint.provenance.json").is_file())

    def test_output_symlink_is_rejected_without_touching_target(self) -> None:
        sources = builder.validate_sources(
            self.nvidia, self.dspark, contract=self.contract
        )
        target = self.root / "valuable-directory"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("preserve me", encoding="utf-8")
        output_link = self.root / "hybrid"
        output_link.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(builder.ContractError, "symbolic-link output"):
            builder.build_hybrid_view(
                sources, output_link, mode="symlink", force=True
            )
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
        self.assertTrue(output_link.is_symlink())

    def test_wrong_nvidia_revision_is_rejected(self) -> None:
        metadata = (
            self.nvidia
            / ".cache"
            / "huggingface"
            / "download"
            / "config.json.metadata"
        )
        metadata.write_text("wrong-revision\nconfig-etag\n", encoding="utf-8")
        with self.assertRaisesRegex(builder.ContractError, "source revision"):
            builder.validate_sources(self.nvidia, self.dspark, contract=self.contract)

    def test_missing_revision_requires_immutable_hash_identity(self) -> None:
        metadata = (
            self.nvidia
            / ".cache"
            / "huggingface"
            / "download"
            / "config.json.metadata"
        )
        metadata.unlink()
        with self.assertRaisesRegex(builder.ContractError, "immutable config/index"):
            builder.validate_sources(self.nvidia, self.dspark, contract=self.contract)

    def test_pinned_hashes_identify_copy_without_hub_metadata(self) -> None:
        metadata = (
            self.nvidia
            / ".cache"
            / "huggingface"
            / "download"
            / "config.json.metadata"
        )
        metadata.unlink()
        copied_contract = replace(
            self.contract,
            expected_nvidia_config_sha256=builder._sha256(
                self.nvidia / "config.json"
            ),
            expected_nvidia_index_sha256=builder._sha256(
                self.nvidia / "model.safetensors.index.json"
            ),
        )
        sources = builder.validate_sources(
            self.nvidia, self.dspark, contract=copied_contract
        )
        self.assertIsNone(sources.nvidia_revision)

    def test_manifest_output_is_deterministic_and_has_no_shards(self) -> None:
        sources = builder.validate_sources(
            self.nvidia, self.dspark, contract=self.contract
        )
        first = self.root / "first"
        second = self.root / "second"
        builder.build_hybrid_view(sources, first, mode="manifest")
        builder.build_hybrid_view(sources, second, mode="manifest")

        for name in (
            "config.json",
            "model.safetensors.index.json",
            "checkpoint.provenance.json",
        ):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertEqual(list(first.glob("*.safetensors")), [])
        provenance = json.loads(
            (first / "checkpoint.provenance.json").read_text(encoding="utf-8")
        )
        self.assertFalse(provenance["runnable_view"])


if __name__ == "__main__":
    unittest.main()
