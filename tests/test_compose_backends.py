# SPDX-License-Identifier: MIT
"""Exercise the actual Compose shell fragment without Docker or GPU imports."""
import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = (ROOT / "docker-compose.yml").read_text()


def speculative_fragment(tokens=3):
    # Extract only the literal shell fragment under command, then apply the
    # two relevant Compose substitutions. Actual Compose rendering is a
    # separate integration check; this does not emulate a YAML/Compose parser.
    fragment = COMPOSE.split('        SPECULATIVE_CONFIG=', 1)[1]
    fragment = 'SPECULATIVE_CONFIG=' + fragment.split(
        '        exec /usr/local/bin/vllm', 1)[0]
    return fragment.replace('${MTP_NUM_TOKENS:-3}', str(tokens)).replace('$$', '$')


def run_fragment(draft=None, tokens=3):
    env = os.environ.copy()
    env.pop('DSPARK_MOE_BACKEND', None)
    if draft is not None:
        env['DSPARK_MOE_BACKEND'] = draft
    return subprocess.run(
        ['bash', '-c', speculative_fragment(tokens)
         + '\nprintf "%s" "$SPECULATIVE_CONFIG"'],
        env=env, capture_output=True, text=True, timeout=10, check=True,
    ).stdout


class ComposeBackendTests(unittest.TestCase):
    def test_original_default_json_is_byte_identical(self):
        expected = ('{"method":"dspark","num_speculative_tokens":3,'
                    '"draft_sample_method":"probabilistic"}')
        self.assertEqual(run_fragment(), expected)
        self.assertEqual(run_fragment(''), expected)

    def test_original_five_token_control_omits_override(self):
        cfg = json.loads(run_fragment(tokens=5))
        self.assertEqual(cfg, {'method': 'dspark', 'num_speculative_tokens': 5,
                              'draft_sample_method': 'probabilistic'})

    def test_explicit_draft_override_preserves_sampling(self):
        cfg = json.loads(run_fragment('flashinfer_b12x', tokens=5))
        self.assertEqual(cfg, {'method': 'dspark', 'num_speculative_tokens': 5,
                              'draft_sample_method': 'probabilistic',
                              'moe_backend': 'flashinfer_b12x'})

    def test_override_is_json_data_not_shell_source(self):
        # Invalid backend names are for vLLM to reject. The launcher must not
        # execute or splice them into additional JSON fields or shell commands.
        value = '\" , \"num_speculative_tokens\":99, \"x\":\"$(printf INJECTED)`printf BAD`'
        cfg = json.loads(run_fragment(value, tokens=5))
        self.assertEqual(cfg['moe_backend'], value)
        self.assertEqual(cfg['num_speculative_tokens'], 5)
        self.assertNotIn('x', cfg)

    def test_backend_environment_defaults_and_quoted_cli(self):
        self.assertIn('TARGET_MOE_BACKEND: "${TARGET_MOE_BACKEND:-flashinfer_b12x}"',
                      COMPOSE)
        self.assertIn('DSPARK_MOE_BACKEND: "${DSPARK_MOE_BACKEND:-}"', COMPOSE)
        self.assertIn('--moe-backend "$${TARGET_MOE_BACKEND}"', COMPOSE)
        self.assertEqual(COMPOSE.count('--moe-backend '), 1)

    def test_json_builder_failure_aborts_launch(self):
        fragment = speculative_fragment().replace('python3 -c', 'false -c')
        result = subprocess.run(
            ['bash', '-c', fragment + '\nprintf UNEXPECTED_LAUNCH'],
            env={**os.environ, 'DSPARK_MOE_BACKEND': 'flashinfer_b12x'},
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('UNEXPECTED_LAUNCH', result.stdout)


if __name__ == '__main__':
    unittest.main()
