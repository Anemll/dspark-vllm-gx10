# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.variable_verifier import (
    complete_async_copy_if_needed,
    compact_scheduler_output_for_variable_drafts,
    trim_invalid_draft_tail,
)


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self.scheduler_requires_draft_tokens = False
        self.copy_wait_fallbacks = 0
        # Per-execute evidence carried to the engine process with
        # ModelRunnerOutput. Emitting Prometheus observations only inside the
        # GPU worker made the live API physical-row series disappear.
        self.last_physical_target_rows: list[int] | None = None
        self.last_d2h_copy_fallback: bool | None = None
        self.variable_draft_lengths = (
            os.environ.get("VLLM_DSPARK_CONFIDENCE_SCHEDULER", "off") == "on"
        )

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        self.scheduler_requires_draft_tokens = (
            input_batch.has_structured_output_reqs
        )
        if (
            not input_batch.has_structured_output_reqs
            and not self.variable_draft_lengths
        ):
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if (
            self.draft_tokens_np is not None
            and self.scheduler_requires_draft_tokens
        ):
            self.copy_wait_fallbacks += int(
                complete_async_copy_if_needed(self.copy_event)
            )
            draft_token_ids = self.draft_tokens_np.tolist()
            if self.variable_draft_lengths:
                draft_token_ids = [
                    trim_invalid_draft_tail(token_ids)
                    for token_ids in draft_token_ids
                ]
        else:
            # The normal unstructured async path intentionally returns the
            # fixed reservation without waiting. Its already-enqueued D2H copy
            # is consumed one engine turn later by physical compaction.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)

    def compact_scheduler_output(self, scheduler_output) -> dict[str, int]:
        """Apply the previous DSpark proposal length before target dispatch."""

        # Do not leak evidence from a prior execute through an early return.
        self.last_physical_target_rows = None
        self.last_d2h_copy_fallback = None
        if not self.variable_draft_lengths:
            return {}
        if not scheduler_output.scheduled_spec_decode_tokens:
            return {}
        if self.draft_tokens_np is None or not self.req_ids:
            raise RuntimeError(
                "DSpark confidence scheduling has target draft rows but no "
                "completed prior proposal transfer"
            )
        fallback_wait = complete_async_copy_if_needed(self.copy_event)
        self.copy_wait_fallbacks += int(fallback_wait)
        invalid = compact_scheduler_output_for_variable_drafts(
            scheduler_output,
            self.req_ids,
            self.draft_tokens_np.tolist(),
        )
        physical_rows = [
            int(scheduler_output.num_scheduled_tokens[req_id])
            for req_id in self.req_ids
            if req_id in scheduler_output.num_scheduled_tokens
        ]
        if not physical_rows:
            raise RuntimeError(
                "DSpark confidence scheduling compacted a proposal but did not "
                "dispatch any physical target rows"
            )
        if any(not 1 <= rows <= 6 for rows in physical_rows):
            raise RuntimeError(
                "DSpark confidence scheduling produced invalid physical target "
                f"rows: {physical_rows}"
            )
        self.last_physical_target_rows = physical_rows
        self.last_d2h_copy_fallback = fallback_wait
        return invalid

    def get_last_compaction_telemetry(
        self,
    ) -> tuple[list[int] | None, bool | None]:
        """Return evidence for the immediately preceding execute call."""

        rows = self.last_physical_target_rows
        return (None if rows is None else list(rows), self.last_d2h_copy_fallback)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
