# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "verify_hybrid_nvfp4_dspark_checkpoint.py"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location("hybrid_checkpoint_verifier", SCRIPT)
    assert SPEC is not None and SPEC.loader is not None
    verifier = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = verifier
    SPEC.loader.exec_module(verifier)
finally:
    sys.path.pop(0)

builder = verifier.builder


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridCheckpointVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runnable = self.root / "runnable"
        self.runnable.mkdir()
        self.source_contract = builder.HybridContract(
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
            expected_nvidia_config_sha256="a" * 64,
            expected_nvidia_index_sha256="b" * 64,
        )
        self.merged_contract = builder.IndexContract(
            total_size=16,
            tensor_count=4,
            non_mtp_count=1,
            mtp_counts=((0, 1), (1, 1), (2, 1)),
            shard_count=4,
            stage_shards=((0, 2), (1, 3), (2, 4)),
        )
        self.contract = self._write_fixture(self.runnable)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self) -> dict:
        return {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "num_hash_layers": 3,
            "num_nextn_predict_layers": 1,
            "expert_dtype": "fp4",
            "compress_ratios": [0, 0, 0, 0],
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
            **self.source_contract.dspark_fields,
        }

    def _index(self) -> dict:
        return {
            "metadata": {"total_size": 16},
            "weight_map": {
                "layers.0.ffn.experts.weight": "model-00001-of-00004.safetensors",
                "mtp.0.draft.weight": "model-00002-of-00004.safetensors",
                "mtp.1.draft.weight": "model-00003-of-00004.safetensors",
                "mtp.2.draft.weight": "model-00004-of-00004.safetensors",
            },
        }

    def _write_fixture(
        self, directory: Path, hash_shards: bool = False
    ) -> verifier.VerificationContract:
        _write_json(directory / builder.CONFIG_NAME, self._config())
        _write_json(directory / builder.INDEX_NAME, self._index())
        for name in builder.RUNTIME_METADATA_FILES:
            _write_json(directory / name, {"fixture": name})

        artifacts = {
            name: {
                "sha256": _sha256(directory / name),
                "size": (directory / name).stat().st_size,
            }
            for name in verifier.ARTIFACT_FILES
        }
        shard_records: list[dict] = []
        for number in range(1, 5):
            destination = builder._shard_name(number, 4)
            payload = bytes([number]) * (number + 2)
            (directory / destination).write_bytes(payload)
            if number == 1:
                role = "nvidia-target"
                source_name = builder._shard_name(number, 2)
                hub_etag = "fixture-etag"
                hub_revision = "synthetic-revision"
            else:
                role = f"dspark-mtp.{number - 2}"
                source_name = destination
                hub_etag = None
                hub_revision = None
            shard_records.append(
                {
                    "destination": destination,
                    "role": role,
                    "source_name": source_name,
                    "source_size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest() if hash_shards else None,
                    "hub_etag": hub_etag,
                    "hub_revision": hub_revision,
                }
            )

        provenance = {
            "schema": "anemll.hybrid-checkpoint-provenance.v1",
            "builder": "scripts/build_hybrid_nvfp4_dspark_checkpoint.py",
            "materialization": "copy",
            "runnable_view": True,
            "shard_sha256_computed": hash_shards,
            "sources": {
                "nvidia": {
                    "path": "/fixture/nvidia",
                    "expected_revision": "synthetic-revision",
                    "observed_revision": "synthetic-revision",
                    "config_sha256": "a" * 64,
                    "index_sha256": "b" * 64,
                    "contract": verifier._expected_summary(self.source_contract.nvidia),
                },
                "dspark": {
                    "path": "/fixture/dspark",
                    "config_sha256": "c" * 64,
                    "index_sha256": "d" * 64,
                    "contract": verifier._expected_summary(self.source_contract.dspark),
                },
            },
            "merged": {
                "artifacts": artifacts,
                "config_base": "nvidia",
                "copied_dspark_config_fields": list(builder.DSPARK_CONFIG_FIELDS),
                "draft_stages": [0, 1, 2],
                "index_total_size": 16,
                "tensor_count": 4,
                "shard_count": 4,
                "target_layers": [0, 0],
                "target_quantization": "NVFP4 W4A4",
                "draft_quantization": "native MXFP4",
            },
            "shards": shard_records,
        }
        _write_json(directory / builder.PROVENANCE_NAME, provenance)
        return verifier.VerificationContract(
            sources=self.source_contract,
            merged_index=self.merged_contract,
            expected_merged_config_sha256=artifacts[builder.CONFIG_NAME]["sha256"],
            expected_merged_index_sha256=artifacts[builder.INDEX_NAME]["sha256"],
            expected_dspark_config_sha256="c" * 64,
            expected_dspark_index_sha256="d" * 64,
            expected_artifacts=artifacts,
        )

    def _metadata_only_copy(self, name: str = "metadata") -> Path:
        destination = self.root / name
        destination.mkdir()
        for filename in verifier.METADATA_FILES:
            (destination / filename).write_bytes((self.runnable / filename).read_bytes())
        return destination

    def test_runnable_fixture_is_valid(self) -> None:
        result = verifier.verify_checkpoint(
            self.runnable, "runnable", contract=self.contract
        )
        self.assertEqual(result["tensor_count"], 4)
        self.assertEqual(result["index_total_size"], 16)
        self.assertEqual(result["materialized_shard_count"], 4)
        self.assertEqual(result["payload_sha256_verified"], 0)

    def test_metadata_only_fixture_matches_runnable_reference(self) -> None:
        metadata = self._metadata_only_copy()
        result = verifier.verify_checkpoint(
            metadata,
            "metadata-only",
            reference=self.runnable,
            contract=self.contract,
        )
        self.assertEqual(result["metadata_file_count"], 7)
        self.assertEqual(result["materialized_shard_count"], 0)
        self.assertIsNotNone(result["reference_sha256"])

    def test_metadata_only_rejects_shard_extra_and_symlink(self) -> None:
        for case in ("shard", "extra", "symlink"):
            with self.subTest(case=case):
                metadata = self._metadata_only_copy(case)
                if case == "shard":
                    (metadata / "model-00001-of-00004.safetensors").write_bytes(b"x")
                elif case == "extra":
                    (metadata / "notes.txt").write_text("extra", encoding="utf-8")
                else:
                    target = metadata / builder.CONFIG_NAME
                    target.unlink()
                    target.symlink_to(self.runnable / builder.CONFIG_NAME)
                with self.assertRaises(builder.ContractError):
                    verifier.verify_checkpoint(
                        metadata, "metadata-only", contract=self.contract
                    )

    def test_runnable_rejects_symlink_and_wrong_size_shards(self) -> None:
        shard = self.runnable / "model-00001-of-00004.safetensors"
        original = shard.read_bytes()
        shard.unlink()
        target = self.root / "payload"
        target.write_bytes(original)
        shard.symlink_to(target)
        with self.assertRaisesRegex(builder.ContractError, "must not be a symlink"):
            verifier.verify_checkpoint(
                self.runnable, "runnable", contract=self.contract
            )

        shard.unlink()
        shard.write_bytes(original + b"x")
        with self.assertRaisesRegex(builder.ContractError, "size does not match"):
            verifier.verify_checkpoint(
                self.runnable, "runnable", contract=self.contract
            )

    def test_payload_sha256_is_verified_when_present(self) -> None:
        hashed = self.root / "hashed"
        hashed.mkdir()
        contract = self._write_fixture(hashed, hash_shards=True)
        result = verifier.verify_checkpoint(hashed, "runnable", contract=contract)
        self.assertEqual(result["payload_sha256_verified"], 4)

        shard = hashed / "model-00001-of-00004.safetensors"
        data = shard.read_bytes()
        shard.write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])
        with self.assertRaisesRegex(builder.ContractError, "SHA-256 does not match"):
            verifier.verify_checkpoint(hashed, "runnable", contract=contract)

    def test_metadata_artifact_tamper_is_rejected(self) -> None:
        path = self.runnable / "generation_config.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(builder.ContractError, "size does not match"):
            verifier.verify_checkpoint(
                self.runnable, "runnable", contract=self.contract
            )

    def test_reference_metadata_mismatch_is_rejected(self) -> None:
        metadata = self._metadata_only_copy()
        reference = self.root / "reference"
        reference.mkdir()
        for filename in verifier.METADATA_FILES:
            (reference / filename).write_bytes((self.runnable / filename).read_bytes())
        provenance = json.loads(
            (reference / builder.PROVENANCE_NAME).read_text(encoding="utf-8")
        )
        provenance["materialization"] = "hardlink"
        _write_json(reference / builder.PROVENANCE_NAME, provenance)
        with self.assertRaisesRegex(builder.ContractError, "does not match reference"):
            verifier.verify_checkpoint(
                metadata,
                "metadata-only",
                reference=reference,
                contract=self.contract,
            )

    def test_default_contract_pins_production_counts_and_hashes(self) -> None:
        contract = verifier.PINNED_VERIFICATION_CONTRACT
        self.assertEqual(contract.merged_index.shard_count, 48)
        self.assertEqual(contract.merged_index.tensor_count, 138_365)
        self.assertEqual(contract.merged_index.total_size, 175_535_844_088)
        self.assertEqual(len(contract.expected_artifacts), 6)


if __name__ == "__main__":
    unittest.main()
