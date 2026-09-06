# SPDX-License-Identifier: MIT
"""Execute the actual config guard without loading vLLM or GPU libraries."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_guard(path):
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.unparse(node.test) == 'n_predict is not None']
    if len(matches) != 1:
        raise AssertionError('Expected one pinned n_predict guard')
    function = ast.parse('def validate(self, n_predict):\n    pass').body[0]
    function.body = [copy.deepcopy(matches[0])]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['validate']


guard = load_guard(ROOT / 'overlay/vllm/config/speculative.py')


class DSparkConfigTests(unittest.TestCase):
    def config(self, k, method='dspark', arch='DSparkDraftModel'):
        return SimpleNamespace(num_speculative_tokens=k, method=method,
                               draft_model_config=SimpleNamespace(architectures=[arch]))

    def test_vision_trained_five_token_block_with_three_stages(self):
        config = self.config(5)
        guard(config, 3)
        self.assertEqual(config.num_speculative_tokens, 5)

    def test_mtp_divisibility_is_not_relaxed(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            guard(self.config(5, method='mtp', arch='DeepSeekV4MTPModel'), 3)

    def test_other_dspark_architectures_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            guard(self.config(5, arch='Qwen3DSparkModel'), 3)

    def test_0731_five_tokens_one_stage_unchanged(self):
        config = self.config(5)
        guard(config, 1)
        self.assertEqual(config.num_speculative_tokens, 5)

    def test_existing_three_and_six_token_configs_remain_valid(self):
        for tokens in (3, 6):
            config = self.config(tokens)
            guard(config, 3)
            self.assertEqual(config.num_speculative_tokens, tokens)

    def test_default_and_no_stage_config_unchanged(self):
        config = self.config(None)
        guard(config, 3)
        self.assertEqual(config.num_speculative_tokens, 3)
        config = self.config(5)
        guard(config, None)
        self.assertEqual(config.num_speculative_tokens, 5)

    def test_pinned_source_reproduces_old_rejection_when_available(self):
        path = ROOT / '.build/vllm-upstream/vllm/config/speculative.py'
        if not path.exists():
            self.skipTest('optional pristine reference is absent')
        with self.assertRaisesRegex(ValueError, 'divisible'):
            load_guard(path)(self.config(5), 3)


if __name__ == '__main__':
    unittest.main()
