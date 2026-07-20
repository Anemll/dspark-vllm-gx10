# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "run_t8_confidence_capture.py"
SPEC = importlib.util.spec_from_file_location("run_t8_confidence_capture", PATH)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        super().__init__(payload)
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeTransport:
    def __init__(self) -> None:
        self.health_sent_at: float | None = None
        self.post_times: list[float] = []
        self.metrics_requested_after_posts: int | None = None

    def urlopen(self, request: object, *, timeout: float) -> FakeResponse:
        del timeout
        method = request.get_method()  # type: ignore[attr-defined]
        url = request.full_url  # type: ignore[attr-defined]
        if method == "GET" and url.endswith("/health"):
            self.health_sent_at = time.monotonic()
            return FakeResponse(b"")
        if method == "GET" and url.endswith("/metrics"):
            self.metrics_requested_after_posts = len(self.post_times)
            return FakeResponse(b"fixture_metric 1\n")
        if method == "POST" and url.endswith("/v1/chat/completions"):
            json.loads(request.data)  # type: ignore[attr-defined]
            self.post_times.append(time.monotonic())
            return FakeResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": "fixture output"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 1,
                        },
                    }
                ).encode("utf-8")
            )
        raise AssertionError(f"unexpected request: {method} {url}")


class T8PromptCaptureTests(unittest.TestCase):
    def run_fixture(self, output_dir: Path, budget: float = 1.0) -> dict:
        transport = FakeTransport()
        with mock.patch.object(
            capture.urllib.request, "urlopen", side_effect=transport.urlopen
        ):
            result = capture.run_capture(
                base_url="http://fixture.invalid",
                model="fixture-model",
                output_dir=output_dir,
                readiness_timeout_seconds=1.0,
                readiness_poll_seconds=0.01,
                readiness_request_timeout_seconds=0.5,
                ready_to_first_prompt_budget_seconds=budget,
                prompt_timeout_seconds=1.0,
                max_tokens=1,
            )
            self.assertEqual(len(transport.post_times), 5)
            self.assertEqual(transport.metrics_requested_after_posts, 5)
            assert transport.health_sent_at is not None
            observed_delay = transport.post_times[0] - transport.health_sent_at
            self.assertLess(observed_delay, budget)
            return result

    def test_dry_run_starts_first_prompt_without_privileged_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = self.run_fixture(output_dir)
            self.assertLess(result["ready_to_first_prompt_s"], 1.0)
            self.assertEqual(result["pre_prompt_privileged_commands"], 0)
            self.assertTrue((output_dir / "readiness.json").is_file())
            self.assertTrue((output_dir / "prompts.json").is_file())
            self.assertEqual(
                (output_dir / "metrics-after-prompts.txt").read_text(
                    encoding="utf-8"
                ),
                "fixture_metric 1\n",
            )
            self.assertFalse((output_dir / "RESTORE_REQUIRED.json").exists())

    def test_runner_has_no_shell_or_privilege_command_execution(self) -> None:
        source = PATH.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("sudo", source)

    def test_slow_minimal_bank_fails_closed_before_first_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            transport = FakeTransport()
            real_atomic_json = capture._atomic_json

            def slow_ready_write(path: Path, payload: object) -> None:
                real_atomic_json(path, payload)
                if path.name == "readiness.json":
                    time.sleep(0.04)

            with (
                mock.patch.object(
                    capture.urllib.request,
                    "urlopen",
                    side_effect=transport.urlopen,
                ),
                mock.patch.object(
                    capture, "_atomic_json", side_effect=slow_ready_write
                ),
            ):
                with self.assertRaisesRegex(
                    capture.RestoreRequired, "readiness-to-first-prompt delay"
                ):
                    capture.run_capture(
                        base_url="http://fixture.invalid",
                        model="fixture-model",
                        output_dir=output_dir,
                        readiness_timeout_seconds=1.0,
                        readiness_poll_seconds=0.01,
                        readiness_request_timeout_seconds=0.5,
                        ready_to_first_prompt_budget_seconds=0.01,
                        prompt_timeout_seconds=1.0,
                        max_tokens=1,
                    )
            self.assertEqual(transport.post_times, [])
            marker = json.loads(
                (output_dir / "RESTORE_REQUIRED.json").read_text(encoding="utf-8")
            )
            self.assertTrue(marker["restore_required"])
            self.assertIn("readiness-to-first-prompt delay", marker["reason"])

    def test_readiness_timeout_requires_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                mock.patch.object(
                    capture.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.URLError("not ready"),
                ),
                self.assertRaisesRegex(
                    capture.RestoreRequired, "readiness deadline"
                ),
            ):
                capture.run_capture(
                    base_url="http://fixture.invalid",
                    model="fixture-model",
                    output_dir=output_dir,
                    readiness_timeout_seconds=0.03,
                    readiness_poll_seconds=0.01,
                    readiness_request_timeout_seconds=0.01,
                    ready_to_first_prompt_budget_seconds=1.0,
                    prompt_timeout_seconds=1.0,
                    max_tokens=1,
                )
            self.assertTrue((output_dir / "RESTORE_REQUIRED.json").is_file())


if __name__ == "__main__":
    unittest.main()
