# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks import probe_nvfp4_prepared_layer as probe


class PreparedLayerProbeTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, str, str]:
        layer = root / probe.LAYER_FILENAME
        layer.write_bytes(b"immutable prepared physical layer")
        layer_sha = probe._sha256_file(layer)
        state = {
            "contract": {
                "schema_version": probe.SCHEMA_VERSION,
                "format": probe.PREPARED_SCHEMA,
            },
            "files": {
                probe.LAYER_FILENAME: {
                    "path": probe.LAYER_FILENAME,
                    "layer": probe.LAYER,
                    "size": layer.stat().st_size,
                    "sha256": layer_sha,
                }
            },
        }
        state_path = root / probe.BUILD_STATE_FILENAME
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return layer, state_path, layer_sha, probe._sha256_file(state_path)

    @staticmethod
    def _physical_evidence(path: Path, *, layer: int):
        return {
            "ok": True,
            "path": str(path.resolve()),
            "layer": layer,
            "tensor_count": 8,
            "fingerprints_match": True,
        }

    def test_probe_pins_file_and_state_before_after_and_cross_checks_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer, state, layer_sha, state_sha = self._write_fixture(root)
            calls = []

            def validator(path, *, layer):
                calls.append((path, layer))
                return self._physical_evidence(path, layer=layer)

            report = probe.run_probe(
                layer,
                expected_file_sha256=layer_sha,
                build_state=state,
                expected_build_state_sha256=state_sha,
                validator_fn=validator,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(calls, [(layer.resolve(), 0)])
            self.assertTrue(report["layer_file"]["unchanged"])
            self.assertTrue(report["build_state"]["unchanged"])
            self.assertEqual(
                report["build_state"]["layer0_row"]["sha256"], layer_sha
            )
            json.dumps(report)

    def test_expected_digest_mismatch_fails_before_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            layer, _state, _layer_sha, _state_sha = self._write_fixture(
                Path(temporary)
            )
            with self.assertRaisesRegex(RuntimeError, "expected pin"):
                probe.run_probe(
                    layer,
                    expected_file_sha256="0" * 64,
                    validator_fn=lambda *_args, **_kwargs: self.fail(
                        "validator must not run after identity failure"
                    ),
                )

    def test_build_state_row_mismatch_fails_before_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer, state, _layer_sha, _state_sha = self._write_fixture(root)
            value = json.loads(state.read_text())
            value["files"][probe.LAYER_FILENAME]["sha256"] = "0" * 64
            state.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "layer0 row drifted"):
                probe.run_probe(
                    layer,
                    build_state=state,
                    validator_fn=lambda *_args, **_kwargs: self.fail(
                        "validator must not run after state-row failure"
                    ),
                )

    def test_read_only_post_snapshot_detects_validator_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer, _state, _layer_sha, _state_sha = self._write_fixture(root)

            def mutate(path, *, layer):
                path.write_bytes(b"mutated")
                return self._physical_evidence(path, layer=layer)

            with self.assertRaisesRegex(RuntimeError, "changed"):
                probe.run_probe(layer, validator_fn=mutate)

    def test_user_supplied_layer_and_state_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer, state, _layer_sha, _state_sha = self._write_fixture(root)
            real_layer = root / "layer-real.safetensors"
            layer.rename(real_layer)
            layer.symlink_to(real_layer)
            with self.assertRaisesRegex(RuntimeError, "direct regular file"):
                probe.run_probe(
                    layer,
                    build_state=state,
                    validator_fn=self._physical_evidence,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer, state, _layer_sha, _state_sha = self._write_fixture(root)
            real_state = root / "build-state-real.json"
            state.rename(real_state)
            state.symlink_to(real_state)
            with self.assertRaisesRegex(RuntimeError, "direct regular file"):
                probe.run_probe(
                    layer,
                    build_state=state,
                    validator_fn=self._physical_evidence,
                )

    def test_cross_directory_build_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            layer, _state, _layer_sha, _state_sha = self._write_fixture(first)
            _other_layer, other_state, _other_sha, _other_state_sha = (
                self._write_fixture(second)
            )
            with self.assertRaisesRegex(RuntimeError, "exactly beside"):
                probe.run_probe(
                    layer,
                    build_state=other_state,
                    validator_fn=self._physical_evidence,
                )

    def test_main_emits_failure_json_and_returns_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = probe.main(
                    [
                        "--layer-file",
                        str(Path(temporary) / probe.LAYER_FILENAME),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(rc, 2)
            report = json.loads(output.read_text())
            self.assertFalse(report["ok"])
            self.assertIn("failures", report)
            self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_probe_is_copied_into_candidate_image(self):
        root = Path(__file__).parents[1]
        dockerfile = (
            root / "docker" / "Dockerfile.nvfp4-aot-overlay"
        ).read_text()
        copy_contract = (
            "COPY benchmarks/probe_nvfp4_prepared_layer.py "
            "/usr/local/bin/dspark-probe-nvfp4-prepared-layer"
        )
        self.assertEqual(dockerfile.count(copy_contract), 1)
        dockerignore = (
            root / "docker" / "Dockerfile.nvfp4-aot-overlay.dockerignore"
        ).read_text().splitlines()
        self.assertIn("!benchmarks/probe_nvfp4_prepared_layer.py", dockerignore)


if __name__ == "__main__":
    unittest.main()
