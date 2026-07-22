# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import pathlib
import re
import unittest

from scripts import patch_b12x_w4a16_e8m0_k32_scale_reuse as k32_reuse
from scripts import patch_b12x_w4a16_e8m0_scale_fast as scale_fast
from scripts import patch_b12x_w4a16_modelopt_tc_decode as tc_decode
from scripts import patch_b12x_w4a16_modelopt_tc_planner as tc_planner
from scripts import patch_b12x_w4a16_modelopt_vector_load as vector_load
from scripts import patch_nvfp4_dual_uniform_decode as uniform_decode


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker/Dockerfile.nvfp4-w4a16-dual-decode-overlay"
DOCKERIGNORE = pathlib.Path(f"{DOCKERFILE}.dockerignore")
HOTFIX_DOCKERFILE = ROOT / "docker/Dockerfile.nvfp4-dual-dispatch-hotfix"
HOTFIX_DOCKERIGNORE = pathlib.Path(f"{HOTFIX_DOCKERFILE}.dockerignore")

PATCH_SOURCES = (
    "scripts/patch_b12x_w4a16_modelopt_tc_decode.py",
    "scripts/patch_b12x_w4a16_e8m0_scale_fast.py",
    "scripts/patch_b12x_w4a16_e8m0_k32_scale_reuse.py",
    "scripts/patch_b12x_w4a16_modelopt_vector_load.py",
    "scripts/patch_b12x_w4a16_modelopt_tc_planner.py",
    "scripts/patch_nvfp4_dual_uniform_decode.py",
)
OVERLAY_SOURCES = (
    "overlay/vllm/envs.py",
    "overlay/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py",
    "overlay/vllm/model_executor/layers/fused_moe/experts/"
    "nvfp4_dual_decode_moe.py",
    "overlay/vllm/model_executor/layers/fused_moe/experts/"
    "nvfp4_dual_decode_policy.py",
    "overlay/vllm/model_executor/layers/fused_moe/experts/"
    "b12x_mxfp4_moe.py",
    "overlay/vllm/models/deepseek_v4/nvidia/prepared_weight_loading.py",
)
CONTEXT_FILES = PATCH_SOURCES + OVERLAY_SOURCES

EXPECTED_DOCKERIGNORE_LINES = (
    "**",
    "!scripts/",
    *(f"!{path}" for path in PATCH_SOURCES),
    "!overlay/",
    "!overlay/vllm/",
    "!overlay/vllm/envs.py",
    "!overlay/vllm/model_executor/",
    "!overlay/vllm/model_executor/layers/",
    "!overlay/vllm/model_executor/layers/fused_moe/",
    "!overlay/vllm/model_executor/layers/fused_moe/oracle/",
    "!overlay/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py",
    "!overlay/vllm/model_executor/layers/fused_moe/experts/",
    "!overlay/vllm/model_executor/layers/fused_moe/experts/"
    "nvfp4_dual_decode_moe.py",
    "!overlay/vllm/model_executor/layers/fused_moe/experts/"
    "nvfp4_dual_decode_policy.py",
    "!overlay/vllm/model_executor/layers/fused_moe/experts/"
    "b12x_mxfp4_moe.py",
    "!overlay/vllm/models/",
    "!overlay/vllm/models/deepseek_v4/",
    "!overlay/vllm/models/deepseek_v4/nvidia/",
    "!overlay/vllm/models/deepseek_v4/nvidia/prepared_weight_loading.py",
)

PRIVATE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"/Users/[^ /]+"),
    re.compile(r"/home/anemll"),
    re.compile(r"192\.168\.[0-9]+\.[0-9]+"),
    re.compile(
        r"(?i)(?:password|passwd|api[_-]?key)\s*=\s*"
        r"(?![$<{])[^\s]+"
    ),
)
PRIVATE_NAMES = (
    ".env",
    ".pem",
    ".key",
    "id_rsa",
    "id_ed25519",
)


class NvFp4W4A16DualImageTests(unittest.TestCase):
    def test_patch_chain_and_final_hashes_are_exact(self) -> None:
        self.assertEqual(
            tc_decode.PATCHED_SOURCE_SHA256,
            scale_fast.PINNED_SOURCE_SHA256,
        )
        self.assertEqual(
            scale_fast.PATCHED_SOURCE_SHA256,
            k32_reuse.PINNED_SOURCE_SHA256,
        )
        self.assertEqual(
            k32_reuse.PATCHED_SOURCE_SHA256,
            vector_load.PINNED_SOURCE_SHA256,
        )
        self.assertEqual(
            vector_load.PATCHED_SOURCE_SHA256,
            "e23cccd7e135071f1393184132ec0ad7f277faf7851bfaba2b2a7b15e5a3a7dd",
        )
        self.assertEqual(
            tc_planner.PINNED_SOURCE_SHA256,
            "c2ca5aca4f9efd8ac8afb52909ef18410d1afd455d7e994debcd4e0bc13e019d",
        )
        self.assertEqual(
            tc_planner.PATCHED_SOURCE_SHA256,
            "ba980ff1df1df0b9959c274fa255c2fcb538671f0cdd068b0ec7cdf4f434933d",
        )
        self.assertEqual(
            uniform_decode.MODEL_RUNNER_PATCHED_SHA256,
            "61befb32cdc06e1c58383f9481e805d3b86637c84736f79b904c04a474df34e4",
        )
        self.assertEqual(
            uniform_decode.CUDAGRAPH_UTILS_PATCHED_SHA256,
            "56031f4d39147bc4cb8ee9cf7d1914d6811c677d15b8735a7d292862cba5da4c",
        )

    def test_dockerfile_applies_exact_chain_before_overlay(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        commands = (
            "python3 /usr/local/bin/dspark-patch-b12x-w4a16-modelopt-tc-decode",
            "python3 /usr/local/bin/dspark-patch-b12x-w4a16-e8m0-scale-fast",
            "python3 /usr/local/bin/dspark-patch-b12x-w4a16-e8m0-k32-scale-reuse",
            "python3 /usr/local/bin/dspark-patch-b12x-w4a16-modelopt-vector-load",
            "python3 /usr/local/bin/dspark-patch-b12x-w4a16-modelopt-tc-planner",
            "python3 /usr/local/bin/dspark-patch-nvfp4-dual-uniform-decode",
        )
        offsets = []
        for command in commands:
            self.assertEqual(text.count(command), 1, command)
            offsets.append(text.index(command))
        self.assertEqual(offsets, sorted(offsets))
        self.assertLess(
            offsets[-1],
            text.index("COPY overlay/vllm/envs.py"),
        )
        self.assertNotIn("COPY overlay/vllm/ ", text)
        for source in CONTEXT_FILES:
            self.assertIn(source, text)
        self.assertIn(vector_load.PATCHED_SOURCE_SHA256, text)
        self.assertIn(tc_planner.PATCHED_SOURCE_SHA256, text)
        self.assertIn(uniform_decode.MODEL_RUNNER_PATCHED_SHA256, text)
        self.assertIn(uniform_decode.CUDAGRAPH_UTILS_PATCHED_SHA256, text)
        self.assertIn("sha256sum -c -", text)

    def test_dockerfile_requires_immutable_base_and_revision_labels(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^ARG BASE_IMAGE$")
        self.assertIn("FROM ${BASE_IMAGE}", text)
        self.assertRegex(text, r"(?m)^ARG BASE_IMAGE_ID$")
        self.assertRegex(text, r"(?m)^ARG SOURCE_REVISION$")
        self.assertNotIn("SOURCE_REVISION=unknown", text)
        self.assertIn('r"[0-9a-f]{40}"', text)
        self.assertIn('r"sha256:[0-9a-f]{64}"', text)
        self.assertIn(
            'org.opencontainers.image.revision="${SOURCE_REVISION}"', text
        )
        self.assertIn(
            'org.opencontainers.image.base.name="${BASE_IMAGE}"', text
        )
        self.assertIn(
            'org.opencontainers.image.base.digest="${BASE_IMAGE_ID}"', text
        )

    def test_build_context_is_an_exact_private_safe_allowlist(self) -> None:
        lines = tuple(
            line.strip()
            for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        self.assertEqual(lines, EXPECTED_DOCKERIGNORE_LINES)
        self.assertEqual(lines[0], "**")
        self.assertNotIn("!.local/", lines)
        self.assertNotIn("!config/", lines)
        self.assertNotIn("!benchmarks/", lines)
        self.assertNotIn("!tests/", lines)

        for relative in CONTEXT_FILES:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertFalse(path.is_symlink(), relative)
            lowered_name = path.name.lower()
            self.assertFalse(
                any(
                    lowered_name == token
                    or lowered_name.endswith(token)
                    or lowered_name.startswith(token)
                    for token in PRIVATE_NAMES
                ),
                relative,
            )
            text = path.read_text(encoding="utf-8")
            for pattern in PRIVATE_PATTERNS:
                self.assertIsNone(
                    pattern.search(text), f"{relative}: {pattern.pattern}"
                )

    def test_dispatch_hotfix_is_small_content_addressed_overlay(self) -> None:
        text = HOTFIX_DOCKERFILE.read_text(encoding="utf-8")
        lines = tuple(
            line.strip()
            for line in HOTFIX_DOCKERIGNORE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        self.assertEqual(lines[0], "**")
        self.assertIn("!scripts/patch_nvfp4_dual_uniform_decode.py", lines)
        self.assertIn(
            "!overlay/vllm/model_executor/layers/fused_moe/experts/"
            "nvfp4_dual_decode_moe.py",
            lines,
        )
        self.assertIn(
            "!overlay/vllm/models/deepseek_v4/nvidia/"
            "prepared_weight_loading.py",
            lines,
        )
        self.assertNotIn("patch_b12x_w4a16_modelopt_tc_decode", text)
        self.assertEqual(len(re.findall(r"(?m)^ARG BASE_IMAGE$", text)), 2)
        self.assertEqual(
            text.count("dspark-patch-nvfp4-dual-uniform-decode"), 2
        )
        for digest in (
            "sha256:c018a6b967af6a6d6d2e415fc6fe54b9f4eecf4d72d95dc956bffbca7f88f848",
            uniform_decode.MODEL_RUNNER_SOURCE_SHA256,
            uniform_decode.CUDAGRAPH_UTILS_SOURCE_SHA256,
            uniform_decode.MODEL_RUNNER_PATCHED_SHA256,
            uniform_decode.CUDAGRAPH_UTILS_PATCHED_SHA256,
            "7d3beacd52ae30978be04ed21a0a8cafdcc740f63026f60d4f573f76e9d6dcb8",
            "60a56e0bdb7edcc9ebf6274e3f4de6e9af105dfffe9306258e6957989774e226",
        ):
            self.assertIn(digest, text)
        self.assertNotIn("!config/", lines)
        self.assertNotIn("!benchmarks/", lines)


if __name__ == "__main__":
    unittest.main()
