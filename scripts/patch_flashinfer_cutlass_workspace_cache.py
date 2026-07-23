#!/usr/bin/env python3
"""Add a runner-owned reusable workspace to FlashInfer 0.6.15 CUTLASS MoE."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = (
    "1e17d8e42cbfda1b631813b9ac7281dcf12ed496d5f5199a120a509ed744b192"
)

MEMBER_BEFORE = """\
  Tensor mProfileWorkspace;

  bool mUseDeepSeekFP8BlockScaling = false;
"""

MEMBER_AFTER = """\
  Tensor mProfileWorkspace;
  // runMoe holds mMutex for the complete launch, so this runner-owned arena
  // is non-reentrant and safe to reuse across sequential routed layers.
  Tensor mRuntimeWorkspace;
  size_t mRuntimeWorkspaceBytes{0};
  int mRuntimeWorkspaceDevice{-1};

  bool mUseDeepSeekFP8BlockScaling = false;
"""

ALLOC_BEFORE = """\
    WorkspaceInfo info{};
    int device_id;
    cudaGetDevice(&device_id);
    info.workspace = alloc_tensor({static_cast<int64_t>(total_workspace_size)}, dl_int8,
                                  DLDevice{kDLCUDA, device_id});
    info.src_to_dest_map = common::nextWorkspacePtr(static_cast<int8_t*>(info.workspace.data_ptr()),
                                                    moe_workspace_size);
"""

ALLOC_AFTER = """\
    WorkspaceInfo info{};
    int device_id;
    cudaGetDevice(&device_id);
    if (mRuntimeWorkspaceBytes < total_workspace_size ||
        mRuntimeWorkspaceDevice != device_id) {
      mRuntimeWorkspace =
          alloc_tensor({static_cast<int64_t>(total_workspace_size)}, dl_int8,
                       DLDevice{kDLCUDA, device_id});
      mRuntimeWorkspaceBytes = total_workspace_size;
      mRuntimeWorkspaceDevice = device_id;
    }
    info.workspace = mRuntimeWorkspace;
    info.src_to_dest_map =
        common::nextWorkspacePtr(static_cast<int8_t*>(info.workspace.data_ptr()),
                                 moe_workspace_size);
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_source(source: str) -> str:
    if source.count(MEMBER_BEFORE) != 1:
        raise RuntimeError("FlashInfer runner member anchor drifted")
    if source.count(ALLOC_BEFORE) != 1:
        raise RuntimeError("FlashInfer workspace allocation anchor drifted")
    result = source.replace(MEMBER_BEFORE, MEMBER_AFTER)
    result = result.replace(ALLOC_BEFORE, ALLOC_AFTER)
    if result.count("mRuntimeWorkspaceBytes < total_workspace_size") != 1:
        raise RuntimeError("workspace cache insertion failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    raw = args.target.read_bytes()
    observed = sha256_bytes(raw)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(
            f"FlashInfer 0.6.15 binding mismatch: {observed} != {EXPECTED_SHA256}"
        )
    patched = patch_source(raw.decode("utf-8")).encode("utf-8")
    args.target.write_bytes(patched)
    print(f"INPUT_SHA256={observed}")
    print(f"OUTPUT_SHA256={sha256_bytes(patched)}")


if __name__ == "__main__":
    main()
