# SPDX-License-Identifier: MIT

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assemble-nvidia-0731-vision-hybrid.py"
SPEC = importlib.util.spec_from_file_location("assemble_hybrid", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_tensor_file(path: Path, tensors: dict[str, bytes]) -> dict[str, dict]:
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, payload in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors.values())
    )
    return header


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridAssemblyTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        nvidia = root / "nvidia"
        vision = root / "vision"
        output = root / "hybrid"
        nvidia.mkdir()
        vision.mkdir()
        common = {
            "hidden_size": 4,
            "num_hidden_layers": 43,
            "n_routed_experts": 2,
            "vocab_size": 8,
            "hc_mult": 2,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [41, 42, 43],
        }
        nvidia_config = {
            **common,
            "rms_norm_eps": 1e-6,
            "num_nextn_predict_layers": 1,
            "quantization_config": {"moe_quant_algo": "NVFP4"},
        }
        vision_config = {
            **common,
            "rms_norm_eps": 1e-20,
            "num_nextn_predict_layers": 3,
            **{field: index + 1 for index, field in enumerate(MODULE.VISION_CONFIG_FIELDS)},
        }
        (nvidia / "config.json").write_text(json.dumps(nvidia_config))
        (vision / "config.json").write_text(json.dumps(vision_config))

        nvidia_map = {}
        for i in range(48):
            filename = f"model-{i + 1:05d}-of-00048.safetensors"
            tensor = f"layers.{i}.weight"
            write_tensor_file(nvidia / filename, {tensor: bytes([i])})
            nvidia_map[tensor] = filename
        additions = {
            "vision.patch_embed.weight": b"vision",
            "aligner.w1.weight": b"aligner",
            "image_start": b"start",
            "layers.0.ffn.gate.bias": b"bias",
            "layers.42.ffn.gate.bias_vl": b"vl",
            "mtp.2.ffn.gate.bias_vl": b"mtp",
        }
        donor_name = "model-00001-of-00048.safetensors"
        write_tensor_file(vision / donor_name, additions)
        vision_map = {name: donor_name for name in additions}
        nvidia_index = {"metadata": {"total_size": 48}, "weight_map": nvidia_map}
        vision_index = {"metadata": {"total_size": sum(map(len, additions.values()))}, "weight_map": vision_map}
        (nvidia / "model.safetensors.index.json").write_text(json.dumps(nvidia_index))
        (vision / "model.safetensors.index.json").write_text(json.dumps(vision_index))
        (nvidia / "tokenizer.json").write_text("nvidia-tokenizer")
        names = sorted(additions)
        pins = MODULE.Pins(
            nvidia_config=sha(nvidia / "config.json"),
            nvidia_index=sha(nvidia / "model.safetensors.index.json"),
            vision_config=sha(vision / "config.json"),
            vision_index=sha(vision / "model.safetensors.index.json"),
            tensor_count=len(names),
            tensor_names=MODULE.tensor_names_digest(names),
            payload_bytes=sum(map(len, additions.values())),
        )
        return nvidia, vision, output, pins, additions

    def test_dry_run_validates_without_writing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            nvidia, vision, output, pins, additions = self.make_fixture(Path(temporary))
            result = MODULE.assemble(nvidia, vision, output, pins, dry_run=True)
            self.assertEqual(result["status"], "validated")
            self.assertEqual(result["vision_tensor_count"], len(additions))
            self.assertFalse(output.exists())

    def test_assembly_hardlinks_target_and_extracts_only_vision(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            nvidia, vision, output, pins, additions = self.make_fixture(Path(temporary))
            result = MODULE.assemble(nvidia, vision, output, pins)
            self.assertEqual(result["status"], "assembled")
            self.assertEqual(
                (nvidia / "model-00001-of-00048.safetensors").stat().st_ino,
                (output / "model-00001-of-00048.safetensors").stat().st_ino,
            )
            header, _ = MODULE.read_safetensors_header(output / MODULE.DONOR_SHARD)
            self.assertEqual(set(header) - {"__metadata__"}, set(additions))
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["rms_norm_eps"], 1e-6)
            self.assertEqual(config["num_nextn_predict_layers"], 1)
            self.assertEqual(config["vision_n_layers"], 8)
            self.assertEqual((output / "tokenizer.json").read_text(), "nvidia-tokenizer")

    def test_existing_output_and_manifest_drift_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            nvidia, vision, output, pins, _ = self.make_fixture(Path(temporary))
            output.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.assemble(nvidia, vision, output, pins)
            bad_pins = MODULE.Pins(**{**pins.__dict__, "tensor_count": pins.tensor_count + 1})
            with self.assertRaisesRegex(ValueError, "tensor count"):
                MODULE.assemble(nvidia, vision, output / "other", bad_pins, dry_run=True)


if __name__ == "__main__":
    unittest.main()
