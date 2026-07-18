from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_flashinfer_runtime.py"
SPEC = importlib.util.spec_from_file_location("verify_flashinfer_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class VerifyFlashInferRuntimeTests(unittest.TestCase):
    def versions(self, values: dict[str, str]):
        def get_version(name: str) -> str:
            try:
                return values[name]
            except KeyError as exc:
                raise importlib.metadata.PackageNotFoundError(name) from exc

        return get_version

    def validate(self, values: dict[str, str], **kwargs):
        return verify.validate_version_contract(
            expected_version="0.6.15",
            expected_cuda_suffix="cu130",
            require_jit_cache=kwargs.pop("require_jit_cache", True),
            require_cubin=kwargs.pop("require_cubin", True),
            get_version=self.versions(values),
            version_check_override=kwargs.pop("version_check_override", ""),
            **kwargs,
        )

    def test_matching_binary_artifacts_pass(self) -> None:
        versions = self.validate(
            {
                "flashinfer-python": "0.6.15",
                "flashinfer-jit-cache": "0.6.15+cu130",
                "flashinfer-cubin": "0.6.15",
            }
        )
        self.assertEqual(versions["flashinfer-jit-cache"], "0.6.15+cu130")

    def test_exact_observed_stale_jit_cache_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 0.6.15\\+cu130"):
            self.validate(
                {
                    "flashinfer-python": "0.6.15",
                    "flashinfer-jit-cache": "0.6.13+cu130",
                    "flashinfer-cubin": "0.6.15",
                }
            )

    def test_stale_cubin_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "flashinfer-cubin mismatch"):
            self.validate(
                {
                    "flashinfer-python": "0.6.15",
                    "flashinfer-jit-cache": "0.6.15+cu130",
                    "flashinfer-cubin": "0.6.13",
                }
            )

    def test_missing_optional_artifacts_are_allowed_when_not_required(self) -> None:
        versions = self.validate(
            {"flashinfer-python": "0.6.15"},
            require_jit_cache=False,
            require_cubin=False,
        )
        self.assertIsNone(versions["flashinfer-jit-cache"])
        self.assertIsNone(versions["flashinfer-cubin"])

    def test_version_check_bypass_is_rejected(self) -> None:
        values = {
            "flashinfer-python": "0.6.15",
            "flashinfer-jit-cache": "0.6.15+cu130",
            "flashinfer-cubin": "0.6.15",
        }
        for override in ("1", "0", "false"):
            with self.subTest(override=override):
                with self.assertRaisesRegex(RuntimeError, "must be unset or empty"):
                    self.validate(values, version_check_override=override)

    def test_fused_moe_probe_calls_exact_nvfp4_eight_argument_abi(self) -> None:
        class FakeTorch:
            bfloat16 = object()
            int64 = object()

        class Runner:
            pass

        class Module:
            def __init__(self) -> None:
                self.calls = []

            def init(self, *args):
                self.calls.append(args)
                return Runner()

        module = Module()
        results = verify._exercise_fused_moe_init(module, FakeTorch)
        common = (
            FakeTorch.bfloat16,
            FakeTorch.int64,
            FakeTorch.bfloat16,
            False,
            False,
            False,
            False,
        )
        self.assertEqual(module.calls, [common + (True,), common + (False,)])
        self.assertEqual(results, {"true": "Runner", "false": "Runner"})


if __name__ == "__main__":
    unittest.main()
