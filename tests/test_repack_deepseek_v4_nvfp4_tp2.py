# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import repack_deepseek_v4_nvfp4_tp2 as repack  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_f32(value: int) -> bytes:
    return struct.pack("<f", float(value))


def _write_safetensors(
    path: pathlib.Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _read_tensors(
    path: pathlib.Path,
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length))
        payload_base = 8 + header_length
        result: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            handle.seek(payload_base + start)
            result[name] = (
                entry["dtype"],
                tuple(entry["shape"]),
                handle.read(end - start),
            )
        return result


def _matrix(rows: int, columns: int, seed: int) -> bytes:
    return bytes(
        (seed + row * columns + column) % 251
        for row in range(rows)
        for column in range(columns)
    )


class SyntheticCheckpoint:
    hidden = 16
    intermediate = 32
    experts = 2
    layers = 2

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.root.mkdir()
        config = {
            "model_type": "deepseek_v4",
            "expert_dtype": "fp4",
            "hidden_size": self.hidden,
            "moe_intermediate_size": self.intermediate,
            "n_routed_experts": self.experts,
            "num_hidden_layers": self.layers,
            "quantization_config": {
                "group_size": 16,
                "moe_quant_algo": "NVFP4",
                "producer": {"name": "modelopt", "version": "test"},
            },
        }
        (root / "config.json").write_text(
            json.dumps(config, sort_keys=True), encoding="utf-8"
        )
        (root / "tokenizer.json").write_bytes(b'{"bitwise":"metadata"}\n')

        all_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
        self.source_payloads: dict[str, bytes] = {}
        for layer in range(self.layers):
            layer_prefix = f"layers.{layer}"
            layer_nonexpert = f"{layer_prefix}.input_layernorm.weight"
            all_tensors[layer_nonexpert] = (
                "U8",
                (3,),
                bytes((90 + layer, 91 + layer, 92 + layer)),
            )
            for expert in range(self.experts):
                prefix = f"{layer_prefix}.ffn.experts.{expert}"
                for projection_index, projection in enumerate(("w1", "w3", "w2")):
                    seed = 10 * layer + 3 * expert + projection_index
                    if projection == "w2":
                        weight_shape = (self.hidden, self.intermediate // 2)
                        scale_shape = (self.hidden, self.intermediate // 16)
                    else:
                        weight_shape = (self.intermediate, self.hidden // 2)
                        scale_shape = (self.intermediate, self.hidden // 16)
                    values = {
                        "weight": (
                            "U8",
                            weight_shape,
                            _matrix(*weight_shape, seed=seed),
                        ),
                        "weight_scale": (
                            "U8",
                            scale_shape,
                            _matrix(*scale_shape, seed=100 + seed),
                        ),
                        "weight_scale_2": (
                            "F32",
                            (1,),
                            _encode_f32(200 + seed),
                        ),
                        "input_scale": (
                            "F32",
                            (1,),
                            _encode_f32(300 + seed),
                        ),
                    }
                    for suffix, row in values.items():
                        name = f"{prefix}.{projection}.{suffix}"
                        all_tensors[name] = row
                        self.source_payloads[name] = row[2]

        all_tensors["embed_tokens.weight"] = ("U8", (5,), b"abcde")
        # Scatter every other tensor across two files.  The output must not
        # depend on this deliberately poor source locality.
        shard_rows = ({}, {})
        weight_map: dict[str, str] = {}
        for index, (name, row) in enumerate(all_tensors.items()):
            shard = index % 2
            shard_rows[shard][name] = row
            weight_map[name] = f"model-{shard + 1:05d}-of-00002.safetensors"
        for shard, rows in enumerate(shard_rows, start=1):
            _write_safetensors(
                root / f"model-{shard:05d}-of-00002.safetensors", rows
            )
        total_size = sum(len(row[2]) for row in all_tensors.values())
        (root / repack.INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.total_size = total_size
        self.config_sha256 = _sha256(root / "config.json")
        self.index_sha256 = _sha256(root / repack.INDEX_NAME)

    def build(self, output: pathlib.Path) -> dict[str, object]:
        return repack.build_repacked_checkpoint(
            self.root,
            output,
            namespace=repack.NAMESPACE,
            expected_config_sha256=self.config_sha256,
            expected_index_sha256=self.index_sha256,
            chunk_bytes=97,
        )


class DeepSeekV4Nvfp4Tp2RepackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = pathlib.Path(self.temporary.name)
        self.source = SyntheticCheckpoint(self.base / "source")

    def test_build_fuses_exactly_eight_families_and_preserves_payload_bytes(
        self,
    ) -> None:
        output = self.base / "output"
        result = self.source.build(output)

        self.assertTrue(result["ok"])
        self.assertEqual(result["source_payload_bytes"], self.source.total_size)
        self.assertEqual(result["output_payload_bytes"], self.source.total_size)
        self.assertEqual(result["layer_file_count"], self.source.layers)
        self.assertTrue((output / "model-layer-00000.safetensors").is_file())
        self.assertTrue((output / "model-layer-00001.safetensors").is_file())
        self.assertTrue((output / "model-nonlayer.safetensors").is_file())
        self.assertEqual(
            (output / "config.json").read_bytes(),
            (self.source.root / "config.json").read_bytes(),
        )
        self.assertEqual(
            (output / "tokenizer.json").read_bytes(),
            (self.source.root / "tokenizer.json").read_bytes(),
        )

        index = json.loads((output / repack.INDEX_NAME).read_text())
        self.assertEqual(index["metadata"]["total_size"], self.source.total_size)
        for layer in range(self.source.layers):
            prefix = f"{repack.NAMESPACE}.layers.{layer}.experts."
            families = {
                name[len(prefix) :]
                for name in index["weight_map"]
                if name.startswith(prefix)
            }
            self.assertEqual(families, set(repack.FAMILY_ORDER))
        self.assertFalse(
            any(repack.EXPERT_RE.fullmatch(name) for name in index["weight_map"])
        )

        manifest = json.loads((output / repack.MANIFEST_NAME).read_text())
        self.assertFalse(manifest["loader"]["standard_vllm_compatible"])
        self.assertTrue(
            manifest["loader"]["fail_closed_without_exact_loader_contract"]
        )
        self.assertEqual(
            manifest["loader"]["w13_raw_projection_order"], ["w1", "w3"]
        )
        self.assertEqual(manifest["loader"]["serving_postload_swap_count"], 1)
        self.assertEqual(manifest["loader"]["payload_stage"], repack.PAYLOAD_STAGE)
        self.assertEqual(
            manifest["loader"]["required_backend"], repack.REQUIRED_BACKEND
        )
        self.assertFalse(manifest["loader"]["cutlass_serving_layout_ready"])
        prepared = manifest["loader"]["reserved_payload_stages"][
            repack.RESERVED_PREPARED_STAGE
        ]
        self.assertFalse(prepared["implemented"])
        self.assertEqual(prepared["required_final_projection_order"], ["w3", "w1"])
        self.assertEqual(prepared["required_runtime_w13_reorder_count"], 0)
        self.assertTrue(manifest["output"]["payload_bytes_preserved"])
        self.assertEqual(len(manifest["source"]["shards"]), 2)
        self.assertTrue(repack.verify_repacked_checkpoint(output)["ok"])

    def test_rank_major_matrix_layout_and_raw_w13_order_are_exact(self) -> None:
        output = self.base / "output"
        self.source.build(output)
        tensors = _read_tensors(output / "model-layer-00000.safetensors")
        prefix = f"{repack.NAMESPACE}.layers.0.experts"

        w13_name = f"{prefix}.w13.weight"
        dtype, shape, observed_w13 = tensors[w13_name]
        self.assertEqual(dtype, "U8")
        self.assertEqual(shape, (2, 2, 32, 8))
        expected_w13 = bytearray()
        rank_rows = self.source.intermediate // 2
        row_bytes = self.source.hidden // 2
        for rank in range(2):
            for expert in range(2):
                for projection in ("w1", "w3"):
                    name = f"layers.0.ffn.experts.{expert}.{projection}.weight"
                    payload = self.source.source_payloads[name]
                    start = rank * rank_rows * row_bytes
                    expected_w13.extend(payload[start : start + rank_rows * row_bytes])
        self.assertEqual(observed_w13, bytes(expected_w13))

        w2_name = f"{prefix}.w2.weight"
        _, w2_shape, observed_w2 = tensors[w2_name]
        self.assertEqual(w2_shape, (2, 2, 16, 8))
        expected_w2 = bytearray()
        full_columns = self.source.intermediate // 2
        rank_columns = full_columns // 2
        for rank in range(2):
            for expert in range(2):
                name = f"layers.0.ffn.experts.{expert}.w2.weight"
                payload = self.source.source_payloads[name]
                for row in range(self.source.hidden):
                    start = row * full_columns + rank * rank_columns
                    expected_w2.extend(payload[start : start + rank_columns])
        self.assertEqual(observed_w2, bytes(expected_w2))

        w13_scale2 = tensors[f"{prefix}.w13.weight_scale_2"]
        self.assertEqual(w13_scale2[1], (2, 2))
        self.assertEqual(
            w13_scale2[2],
            b"".join(
                self.source.source_payloads[
                    f"layers.0.ffn.experts.{expert}.{projection}.weight_scale_2"
                ]
                for expert in range(2)
                for projection in ("w1", "w3")
            ),
        )

    def test_nonexpert_tensor_payloads_are_bitwise_identical(self) -> None:
        output = self.base / "output"
        self.source.build(output)
        layer = _read_tensors(output / "model-layer-00000.safetensors")
        residual = _read_tensors(output / "model-nonlayer.safetensors")
        self.assertEqual(
            layer["layers.0.input_layernorm.weight"],
            ("U8", (3,), bytes((90, 91, 92))),
        )
        self.assertEqual(residual["embed_tokens.weight"], ("U8", (5,), b"abcde"))

    def test_build_rejects_wrong_namespace_and_digest_without_publishing(self) -> None:
        wrong_namespace = self.base / "wrong-namespace"
        with self.assertRaisesRegex(repack.ContractError, "namespace"):
            repack.build_repacked_checkpoint(
                self.source.root,
                wrong_namespace,
                namespace="custom-but-unreviewed",
                expected_config_sha256=self.source.config_sha256,
                expected_index_sha256=self.source.index_sha256,
            )
        self.assertFalse(wrong_namespace.exists())

        wrong_digest = self.base / "wrong-digest"
        with self.assertRaisesRegex(repack.ContractError, "index digest mismatch"):
            repack.build_repacked_checkpoint(
                self.source.root,
                wrong_digest,
                namespace=repack.NAMESPACE,
                expected_config_sha256=self.source.config_sha256,
                expected_index_sha256="0" * 64,
            )
        self.assertFalse(wrong_digest.exists())

    def test_build_rejects_missing_expert_family_and_existing_output(self) -> None:
        index_path = self.source.root / repack.INDEX_NAME
        index = json.loads(index_path.read_text())
        missing = "layers.0.ffn.experts.0.w1.input_scale"
        shard_name = index["weight_map"].pop(missing)
        # Rebuild that shard without the missing tensor so index and headers stay
        # internally consistent; the semantic layer contract must still reject it.
        shard_path = self.source.root / shard_name
        rows = _read_tensors(shard_path)
        rows.pop(missing)
        _write_safetensors(shard_path, rows)
        index["metadata"]["total_size"] -= 4
        index_path.write_text(json.dumps(index, sort_keys=True))
        config_digest = _sha256(self.source.root / "config.json")
        index_digest = _sha256(index_path)
        output = self.base / "missing"
        with self.assertRaisesRegex(repack.ContractError, "expected .* routed expert"):
            repack.build_repacked_checkpoint(
                self.source.root,
                output,
                namespace=repack.NAMESPACE,
                expected_config_sha256=config_digest,
                expected_index_sha256=index_digest,
            )
        self.assertFalse(output.exists())

        existing = self.base / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(repack.ContractError, "must not already exist"):
            repack.build_repacked_checkpoint(
                self.source.root,
                existing,
                namespace=repack.NAMESPACE,
                expected_config_sha256=config_digest,
                expected_index_sha256=index_digest,
            )

    def test_verify_fails_closed_after_payload_corruption(self) -> None:
        output = self.base / "output"
        self.source.build(output)
        layer_path = output / "model-layer-00000.safetensors"
        data = bytearray(layer_path.read_bytes())
        data[-1] ^= 0xFF
        layer_path.write_bytes(data)
        with self.assertRaisesRegex(repack.ContractError, "file digest mismatch"):
            repack.verify_repacked_checkpoint(output)

    def test_cli_build_and_verify(self) -> None:
        output = self.base / "cli-output"
        script = ROOT / "scripts" / "repack_deepseek_v4_nvfp4_tp2.py"
        build = subprocess.run(
            [
                sys.executable,
                str(script),
                "build",
                "--source",
                str(self.source.root),
                "--output",
                str(output),
                "--namespace",
                repack.NAMESPACE,
                "--expected-config-sha256",
                self.source.config_sha256,
                "--expected-index-sha256",
                self.source.index_sha256,
                "--chunk-mib",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertTrue(json.loads(build.stdout)["ok"])
        verify = subprocess.run(
            [
                sys.executable,
                str(script),
                "verify",
                "--checkpoint",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertTrue(json.loads(verify.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
