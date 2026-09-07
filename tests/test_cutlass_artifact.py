# SPDX-License-Identifier: MIT
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extract_cutlass_test", ROOT / "scripts/extract-cutlass-module.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CutlassArtifactTests(unittest.TestCase):
    def fixture(self, root, arch=True):
        root = Path(root)
        python_root = root / "python"
        (python_root / "fused_moe").mkdir(parents=True)
        (python_root / "fused_moe/core.py").write_text("pinned source")
        binary = b"\x7fELF\x02\x01" + b"\x00" * 12 + (b"\xb7\x00" if arch else b"\x3e\x00") + b"\x00" * 44
        wheel = root / "test.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("cache/fused_moe_120.so", binary)
            archive.writestr("package/METADATA", "Version: 0.6.15+cu130\n")
            archive.writestr("cache/other_text_kernel.so", "must not extract")
        lock = {"module_member": "cache/fused_moe_120.so", "metadata_member": "package/METADATA",
                "wheel_version": "0.6.15+cu130", "wheel_sha256": MODULE.sha256(wheel),
                "module_bytes": len(binary), "module_sha256": hashlib.sha256(binary).hexdigest(),
                "python_core_sha256": MODULE.sha256(python_root / "fused_moe/core.py")}
        return wheel, lock, root / "out", "a" * 40, python_root

    def test_extracts_only_pinned_module_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            MODULE.extract(*args)
            self.assertEqual({p.name for p in args[2].iterdir()}, {"manifest.json", "fused_moe_120.so"})
            self.assertEqual(json.loads((args[2] / "manifest.json").read_text())["source_revision"], "a" * 40)
            with self.assertRaises(FileExistsError): MODULE.extract(*args)

    def test_bad_hashes_or_version_fail_before_output(self):
        for key in ("wheel_sha256", "module_sha256", "python_core_sha256", "wheel_version"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                args = self.fixture(root)
                args[1][key] = "wrong"
                with self.assertRaises(ValueError): MODULE.extract(*args)
                self.assertFalse(args[2].exists())

    def test_wrong_architecture_and_source_revision_fail(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root, arch=False)
            with self.assertRaisesRegex(ValueError, "ARM64"): MODULE.extract(*args)
            with self.assertRaisesRegex(ValueError, "full source commit"):
                MODULE.extract(*args[:3], "short", args[4])

    def test_docker_repair_does_not_reinstall_entire_cache(self):
        source = (ROOT / "docker/Dockerfile.nvfp4-cutlass").read_text()
        self.assertNotIn("pip install", source)
        self.assertEqual(source.count("COPY --from=extract /cutlass-output/fused_moe_120.so"), 1)
        self.assertIn("--source-revision", source)


if __name__ == "__main__":
    unittest.main()
