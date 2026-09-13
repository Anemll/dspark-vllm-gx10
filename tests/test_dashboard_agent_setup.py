# SPDX-License-Identifier: MIT
"""Contract checks for the Pi/Droid dashboard configuration cards."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "dashboard" / "index.html").read_text()


def isolated_setup(**overrides):
    env = {
        **os.environ,
        "DASHBOARD_PUBLIC_API_BASE_URL": "http://spark.test:8888/v1/",
        "DASHBOARD_AGENT_MODEL": "deepseek-v4-flash-vision-exp-dspark",
        **overrides,
    }
    code = (
        "import json; from dashboard.server import agent_setup; "
        "print(json.dumps(agent_setup('metrics-model'), sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=10, check=True,
    )
    return json.loads(result.stdout)


class DashboardAgentSetupTests(unittest.TestCase):
    def test_server_publishes_350k_context_and_separate_output_limit(self):
        setup = isolated_setup()
        self.assertEqual(setup["apiBaseUrl"], "http://spark.test:8888/v1")
        self.assertEqual(setup["contextWindow"], 350000)
        self.assertEqual(setup["maxOutputTokens"], 32768)
        self.assertLess(setup["maxOutputTokens"], setup["contextWindow"])

    def test_output_limit_never_exceeds_context_window(self):
        setup = isolated_setup(
            DASHBOARD_AGENT_CONTEXT_WINDOW="8192",
            DASHBOARD_AGENT_MAX_OUTPUT_TOKENS="32768",
        )
        self.assertEqual(setup["contextWindow"], 8192)
        self.assertEqual(setup["maxOutputTokens"], 8192)

    def test_pi_card_uses_chat_completions_and_explicit_max_tokens(self):
        self.assertIn('api: "openai-completions"', INDEX)
        self.assertIn('maxTokens: agentMaxOutputTokens', INDEX)
        self.assertIn('contextWindow: agentContextWindow', INDEX)
        self.assertIn('maxTokensField: "max_tokens"', INDEX)
        self.assertIn("~/.pi/agent/models.json", INDEX)

    def test_droid_card_uses_chat_completion_provider_and_output_cap(self):
        self.assertIn('provider: "generic-chat-completion-api"', INDEX)
        self.assertIn("noImageSupport: false", INDEX)
        self.assertIn("maxContextLimit: agentContextWindow", INDEX)
        self.assertIn('maxOutputTokens: agentMaxOutputTokens', INDEX)
        self.assertIn("~/.factory/settings.json", INDEX)

    def test_agent_card_explains_model_reload_and_image_attachment(self):
        self.assertIn('id="agentContextWindow">350,000</span>', INDEX)
        self.assertIn('id="agentOutputLimit">32,768</span>', INDEX)
        self.assertIn("machine running your client", INDEX)
        self.assertIn("/model</code> with no arguments", INDEX)
        self.assertIn("reselect this model", INDEX)
        self.assertIn('input: ["text", "image"]', INDEX)
        self.assertIn("images.blockImages", INDEX)
        self.assertIn("Read tool", INDEX)
        self.assertIn("@/path/to/image.png", INDEX)
        self.assertIn("noImageSupport: false", INDEX)
        self.assertIn("remove older duplicate entries", INDEX)
        self.assertIn("fully exit Droid", INDEX)
        self.assertIn("start a new session", INDEX)
        self.assertIn("reselect this exact custom model", INDEX)
        self.assertIn("paste an image with Ctrl+V", INDEX)

    def test_dashboard_does_not_embed_a_private_lan_address(self):
        self.assertNotRegex(INDEX, r"192\.168\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
