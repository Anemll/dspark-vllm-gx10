# SPDX-License-Identifier: MIT
"""Deployment contract checks requiring neither Docker nor a running model."""

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render-dsv41-1017.py"
spec = importlib.util.spec_from_file_location("profile1017", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def config(rank=0):
    return {
        "IMAGE": f"{profile.IMAGE_REPOSITORY}@{profile.IMAGE_DIGEST}",
        "MODEL_DIR": "/srv/models/1017", "CACHE_DIR": "/srv/cache/1017",
        "MARKOV_FILE": "/srv/assets/markov_top_tokens.json", "NODE_RANK": str(rank),
        "MASTER_ADDR": "10.200.0.1", "SELF_ADDR": "10.200.0.1" if rank == 0 else "10.200.0.0",
        "FABRIC_INTERFACE": "fabric0", "NCCL_IB_HCA": "mlx5_0,mlx5_1",
        "NCCL_IB_GID_INDEX": "3", "EXL3_RDMA_DEVICES": "mlx5_0,mlx5_1",
    }


def value_after(argv, name):
    return argv[argv.index(name) + 1]


class ProfileTests(unittest.TestCase):
    def test_both_client_aliases_and_native_vision_survive_render(self):
        rendered = profile.render(config())
        argv = rendered["docker_argv"]
        index = argv.index("--served-model-name")
        self.assertEqual(argv[index + 1:index + 3],
                         ["DeepSeek-v4.1-Flash-EXL3", "gemini-3-flash-preview"])
        self.assertEqual(json.loads(value_after(argv, "--limit-mm-per-prompt")), {"image": 4})
        self.assertEqual(value_after(argv, "--tool-call-parser"), "deepseek_v41")
        self.assertEqual(value_after(argv, "--reasoning-parser"), "deepseek_v41")
        self.assertEqual(value_after(argv, "--max-model-len"), "240000")
        self.assertEqual(value_after(argv, "--max-num-seqs"), "4")
        self.assertEqual(value_after(argv, "--port"), "8888")
        self.assertNotIn("--headless", argv)
        self.assertNotIn("--max-tokens", argv)  # Output limit belongs to clients.

    def test_worker_has_own_fabric_and_is_headless(self):
        data = config(1) | {"FABRIC_INTERFACE": "fabric1", "NCCL_IB_HCA": "mlx5_8",
                            "EXL3_RDMA_DEVICES": "mlx5_8", "NCCL_IB_GID_INDEX": "5"}
        argv = profile.render(data)["docker_argv"]
        self.assertIn("--headless", argv)
        self.assertIn("VLLM_HOST_IP=10.200.0.0", argv)
        self.assertIn("NCCL_IB_GID_INDEX=5", argv)
        self.assertIn("NCCL_IB_HCA=mlx5_8", argv)
        self.assertIn("GLOO_SOCKET_IFNAME=fabric1", argv)
        self.assertEqual(value_after(argv, "--node-rank"), "1")
        self.assertEqual(value_after(argv, "--restart"), "no")
        self.assertNotIn("--privileged", argv)

    def test_stale_alias_parser_or_text_only_override_is_rejected(self):
        for name, value in [("SERVED_MODEL_NAME", "deepseek-v4-flash-0731-dspark"),
                            ("TOOL_CALL_PARSER", "deepseek_v4"), ("TEXT_ONLY", "1"),
                            ("EXTRA", "--limit-mm-per-prompt '{\"image\":0}'")]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsupported"):
                profile.render(config() | {name: value})

    def test_unpinned_or_wrong_runtime_rejected(self):
        for image in [profile.IMAGE_REPOSITORY + ":v24", profile.IMAGE_REPOSITORY + ":latest",
                      profile.IMAGE_REPOSITORY + "@sha256:" + "f" * 64,
                      "other/image@" + profile.IMAGE_DIGEST]:
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "IMAGE"):
                profile.render(config() | {"IMAGE": image})

    def test_transferred_image_runs_by_exact_id_without_registry_lookup(self):
        for stage in ("jit", "serve"):
            with self.subTest(stage=stage):
                rendered = profile.render(config(1), stage)
                self.assertIn(profile.IMAGE_ID, rendered["docker_argv"])
                self.assertNotIn(config()["IMAGE"], rendered["docker_argv"])
                self.assertEqual(rendered["preflight_argv"][-1][-1], profile.IMAGE_ID)
                self.assertEqual(rendered["source"]["image_digest"], profile.IMAGE_DIGEST)
                self.assertEqual(rendered["source"]["image_id"], profile.IMAGE_ID)

    def test_invalid_resource_and_vision_limits_rejected(self):
        cases = {"NODE_RANK": "2", "MAX_NUM_SEQS": "8", "MAX_MODEL_LEN": "32768",
                 "MAX_NUM_BATCHED_TOKENS": "-1", "IMAGE_LIMIT": "0", "SPECULATIVE_TOKENS": "3",
                 "GPU_MEMORY_UTILIZATION": "nan", "PORT": "65536", "NCCL_IB_GID_INDEX": "-1"}
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                profile.render(config() | {name: value})
        with self.assertRaisesRegex(ValueError, "foreground"):
            profile.render(config() | {"LONG_PREFILL_TOKEN_THRESHOLD": "2048"})

    def test_target_only_removes_speculation_flag_without_disabling_vision(self):
        argv = profile.render(config() | {"SPECULATIVE_TOKENS": "0"})["docker_argv"]
        self.assertNotIn("--speculative-config", argv)
        self.assertEqual(json.loads(value_after(argv, "--limit-mm-per-prompt"))["image"], 4)
        speculative = profile.render(config())["docker_argv"]
        self.assertEqual(json.loads(value_after(speculative, "--speculative-config")),
                         {"method": "dspark", "num_speculative_tokens": 5,
                          "num_speculative_tokens_per_batch_size": [[1, 4, 5]]})

    def test_jit_stage_never_mounts_weights_or_loads_model(self):
        rendered = profile.render(config(), "jit")
        argv = rendered["docker_argv"]
        self.assertEqual(value_after(argv, "--entrypoint"), "python3")
        self.assertIn("EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1", argv)
        self.assertNotIn("serve", argv)
        self.assertFalse(any("/model" in part for part in argv))

    def test_missing_fabric_and_wrong_rank_addresses_are_rejected(self):
        data = config()
        del data["NCCL_IB_HCA"]
        with self.assertRaisesRegex(ValueError, "required"):
            profile.render(data)
        for rank, address in [(0, "10.200.0.0"), (1, "10.200.0.1")]:
            with self.subTest(rank=rank), self.assertRaisesRegex(ValueError, "SELF_ADDR"):
                profile.render(config(rank) | {"SELF_ADDR": address})

    def test_shell_render_preserves_quoted_paths_as_one_argument(self):
        with tempfile.TemporaryDirectory() as temp:
            env_file = Path(temp) / "rank.env"
            data = config() | {"MODEL_DIR": "/srv/models/a model's ; literal"}
            env_file.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in data.items()))
            result = subprocess.run([sys.executable, str(SCRIPT), str(env_file), "--format", "shell"],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(shlex.split(result.stdout.splitlines()[-1])[1:],
                             profile.render(data)["docker_argv"])
            subprocess.run(["bash", "-n"], input=result.stdout, text=True, check=True)

    def test_example_requires_site_specific_values(self):
        with self.assertRaisesRegex(ValueError, "concrete"):
            profile.render(profile.read_env(ROOT / "config/dsv41-1017.env.example"))

    def test_renderer_and_provenance_lock_describe_same_runtime(self):
        lock = json.loads((ROOT / "config/dsv41-1017.lock.json").read_text())
        result = profile.render(config())
        self.assertEqual(lock["runtime"]["image"], config()["IMAGE"])
        self.assertEqual(lock["runtime"]["launcher_commit"], profile.LAUNCHER_REVISION)
        self.assertEqual(lock["runtime"]["recipe_commit"], profile.RECIPE_REVISION)
        self.assertEqual(lock["runtime"]["markov_sha256"], profile.MARKOV_SHA256)
        self.assertEqual(lock["initial_profile"]["served_model_names"], list(profile.ALIASES))
        self.assertEqual(lock["initial_profile"]["max_model_len"], result["client_contract"]["context"])

    def test_markov_file_content_is_checked_before_start(self):
        rendered = profile.render(config())
        check = next(command for command in rendered["preflight_argv"] if command[0] == "python3")
        with tempfile.TemporaryDirectory() as temp:
            bad_asset = Path(temp) / "markov.json"
            bad_asset.write_text("[]")
            check[-2] = str(bad_asset)
            result = subprocess.run(check, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("hash mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
