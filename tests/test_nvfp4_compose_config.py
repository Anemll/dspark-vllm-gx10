# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from pathlib import Path
import os
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class NvFp4ComposeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = (REPO_ROOT / "docker-compose.yml").read_text()

    def test_moe_backend_is_a_reversible_compose_setting(self) -> None:
        self.assertIn(
            "--moe-backend ${DSPARK_MOE_BACKEND:-flashinfer_b12x}",
            self.compose,
        )

    def test_flashinfer_version_check_cannot_be_disabled_by_host_env(self) -> None:
        self.assertIn('FLASHINFER_DISABLE_VERSION_CHECK: ""', self.compose)
        self.assertNotIn("${FLASHINFER_DISABLE_VERSION_CHECK", self.compose)

    def test_rank_examples_pin_the_same_backend(self) -> None:
        values = []
        for relative_path in ("config/head.env.example", "config/worker.env.example"):
            lines = (REPO_ROOT / relative_path).read_text().splitlines()
            values.extend(
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("DSPARK_MOE_BACKEND=")
            )
        self.assertEqual(values, ["flashinfer_b12x", "flashinfer_b12x"])

    def test_staged_nvfp4_loader_is_reversible_and_defaults_off(self) -> None:
        self.assertIn(
            'VLLM_DSV4_NVFP4_LAYER_STAGED_LOAD: '
            '"${VLLM_DSV4_NVFP4_LAYER_STAGED_LOAD:-0}"',
            self.compose,
        )
        values = []
        for relative_path in ("config/head.env.example", "config/worker.env.example"):
            lines = (REPO_ROOT / relative_path).read_text().splitlines()
            values.extend(
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("VLLM_DSV4_NVFP4_LAYER_STAGED_LOAD=")
            )
        self.assertEqual(values, ["0", "0"])

    def test_speculation_mode_is_propagated_with_a_dspark_default(self) -> None:
        self.assertIn(
            'DSPARK_SPECULATION_MODE: "${DSPARK_SPECULATION_MODE:-dspark}"',
            self.compose,
        )

    def test_vllm_command_consumes_conditional_speculation_outputs(self) -> None:
        self.assertIn('"$${SPECULATIVE_ARGS[@]}"', self.compose)
        self.assertIn(
            '--max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}"',
            self.compose,
        )
        self.assertNotIn(
            '\n        --speculative-config "$${SPECULATIVE_CONFIG}"',
            self.compose,
        )

    def _run_speculation_setup(
        self,
        mode: str | None,
        *,
        max_num_seqs: int = 12,
        mtp_num_tokens: int = 5,
    ) -> subprocess.CompletedProcess[str]:
        start = self.compose.index("        SPECULATIVE_ARGS=();")
        end = self.compose.index("        exec /usr/local/bin/vllm serve", start)
        setup = self.compose[start:end]
        # Compose turns each doubled dollar into one literal dollar before the
        # command reaches bash. Reproduce only that interpolation step here.
        setup = setup.replace("$$", "$")
        script = setup + "\n" + """
printf 'capture=%s\\n' "$MAX_CUDAGRAPH_CAPTURE_SIZE"
printf 'argc=%s\\n' "${#SPECULATIVE_ARGS[@]}"
for arg in "${SPECULATIVE_ARGS[@]}"; do
    printf 'arg=<%s>\\n' "$arg"
done
"""
        env = {
            **os.environ,
            "MAX_NUM_SEQS": str(max_num_seqs),
            "MTP_NUM_TOKENS": str(mtp_num_tokens),
        }
        if mode is None:
            env.pop("DSPARK_SPECULATION_MODE", None)
        else:
            env["DSPARK_SPECULATION_MODE"] = mode
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_dspark_mode_emits_speculative_args_and_expands_graph_size(self) -> None:
        result = self._run_speculation_setup("dspark")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capture=72\n", result.stdout)
        self.assertIn("argc=2\n", result.stdout)
        self.assertIn("arg=<--speculative-config>\n", result.stdout)
        self.assertIn('"method":"dspark"', result.stdout)
        self.assertIn('"num_speculative_tokens":5', result.stdout)

    def test_unset_mode_defaults_to_dspark(self) -> None:
        result = self._run_speculation_setup(None, max_num_seqs=6, mtp_num_tokens=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capture=24\n", result.stdout)
        self.assertIn("argc=2\n", result.stdout)
        self.assertIn('"method":"dspark"', result.stdout)

    def test_off_mode_omits_speculative_args_and_uses_target_graph_size(self) -> None:
        result = self._run_speculation_setup("off")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "capture=12\nargc=0\n")

    def test_invalid_speculation_mode_fails_closed(self) -> None:
        result = self._run_speculation_setup("disabled")
        self.assertEqual(result.returncode, 64)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid DSPARK_SPECULATION_MODE=disabled", result.stderr)
        self.assertIn("expected dspark or off", result.stderr)

    def test_rank_examples_pin_the_same_speculation_mode(self) -> None:
        values = []
        for relative_path in ("config/head.env.example", "config/worker.env.example"):
            lines = (REPO_ROOT / relative_path).read_text().splitlines()
            values.extend(
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("DSPARK_SPECULATION_MODE=")
            )
        self.assertEqual(values, ["dspark", "dspark"])


if __name__ == "__main__":
    unittest.main()
