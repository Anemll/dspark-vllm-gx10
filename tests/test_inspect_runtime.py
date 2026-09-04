# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect-runtime.py"
SPEC = importlib.util.spec_from_file_location("inspect_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeInspectionTests(unittest.TestCase):
    def test_dispatch_is_parsed_without_executing_module(self):
        source = 'raise RuntimeError("must not execute")\nSHAPES = frozenset({(32, 128), (32, 512)})'
        self.assertEqual(runtime.declared_constant(source, "SHAPES"), [(32, 128), (32, 512)])
        self.assertIsNone(runtime.declared_constant("SHAPES = arbitrary_call()", "SHAPES"))

    def test_request_flags_exclude_credentials(self):
        result = runtime.requested_flags([
            "vllm", "serve", "model", "--api-key", "secret", "--max-num-seqs=12",
            "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}',
            "--enable-prefix-caching", "--trust-remote-code",
            "--kv-cache-memory-bytes", "10737418240",
        ])
        self.assertNotIn("api-key", result)
        self.assertEqual(result["speculative-config"]["num_speculative_tokens"], 5)
        self.assertIs(result["enable-prefix-caching"], True)
        self.assertEqual(result["kv-cache-memory-bytes"], "10737418240")

    def test_experiment_policy_provenance_is_allowlisted(self):
        self.assertIn("DSPARK_NARROW_ATTN_GRAPH", runtime.ENV_NAMES)
        self.assertIn("VLLM_DSV4_NATIVE_SPARSE_WIDTHS", runtime.ENV_NAMES)
        self.assertEqual(runtime.SOURCE_FILES["attention_preparation"],
                         "vllm/models/deepseek_v4/attention.py")

    def test_source_roots_respect_overlay_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            relative = runtime.SOURCE_FILES["sparse_wrapper"]
            for parent, width in ((old, 512), (new, 256)):
                path = parent / relative
                path.parent.mkdir(parents=True)
                path.write_text(f"_FLASHINFER_DSV4_DECODE_TOPKS = (128, {width}, 1024)\n")
            result = runtime.inspect_sources([new, old])
            self.assertEqual(result["sparse_wrapper"]["declared_decode_topks"], (128, 256, 1024))
            self.assertFalse(result["prefix_manager"]["available"])

    def test_log_evidence_does_not_invent_selection(self):
        self.assertEqual(runtime.observed_log_evidence([]), {"provided": False, "matches": [], "unreadable": []})

    def test_missing_model_does_not_look_like_valid_empty_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            result = runtime.model_manifest(Path(directory) / "absent")
            self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
