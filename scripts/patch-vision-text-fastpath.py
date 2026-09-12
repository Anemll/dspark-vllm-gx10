# SPDX-License-Identifier: MIT
"""Patch the known DeepSeek-V4 Vision wrapper text-only hot path.

This is intentionally hash-guarded because the B12X candidate uses a newer,
matched vLLM tree than this repository's July overlay.  Refuse to modify an
unknown wrapper instead of silently applying a stale textual patch.
"""

from __future__ import annotations

import argparse
import hashlib
import sysconfig
from pathlib import Path


DEFAULT_TARGET = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm/models/deepseek_v4/nvidia/vl_model.py"
)

INSERT_AFTER = """        inputs_embeds = self.language_model.embed_input_ids(input_ids)\n"""

FAST_PATH = """
        # Text-only requests and decode steps do not carry encoder embeddings.
        # Return before building the image-sentinel mask/table so the Vision
        # wrapper has the same embedding hot path as the text model.  DeepSeek
        # V4 image spans are scheduled atomically, therefore a scheduled image
        # prefill always supplies at least one multimodal embedding here.
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
"""

LATE_RETURN = """
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    source = args.target.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != args.expected_sha256:
        raise SystemExit(
            f"refusing to patch unknown Vision wrapper: {actual} "
            f"(expected {args.expected_sha256})"
        )

    text = source.decode()
    if text.count(INSERT_AFTER) != 1:
        raise SystemExit("expected one embedding lookup insertion point")
    if text.count(LATE_RETURN) != 1:
        raise SystemExit("expected one late text-only return")

    text = text.replace(LATE_RETURN, "", 1)
    text = text.replace(INSERT_AFTER, INSERT_AFTER + FAST_PATH, 1)
    args.target.write_text(text)
    print(hashlib.sha256(args.target.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
