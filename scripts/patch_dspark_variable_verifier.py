#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install the DSpark physical variable-verifier hooks into pinned vLLM.

The overlay intentionally patches four small integration seams instead of
forking thousands of lines of the pinned model runner and scheduler.  Every
replacement is exact and fail-closed; the Dockerfile separately pins the
complete pre-patch file hashes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_model_runner(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        """    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if not dummy_run:\n            # Update the request states.\n""",
        """    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        confidence_invalid_spec_tokens: dict[str, int] = {}\n        if not dummy_run:\n            # The confidence prefix belongs to the proposal produced by the\n            # prior step. Compact before request updates and CUDA-graph dispatch\n            # so target attention/MoE kernels execute only physical rows.\n            confidence_invalid_spec_tokens = (\n                self.draft_tokens_handler.compact_scheduler_output(\n                    scheduler_output\n                )\n            )\n            # Update the request states.\n""",
        label="model_runner execute hook",
    )
    source = replace_once(
        source,
        """            aux_hidden_states=aux_hidden_states,\n            finished_req_ids=finished_req_ids,\n        )\n""",
        """            aux_hidden_states=aux_hidden_states,\n            finished_req_ids=finished_req_ids,\n            confidence_invalid_spec_tokens=confidence_invalid_spec_tokens,\n        )\n""",
        label="model_runner execute state evidence",
    )
    source = replace_once(
        source,
        """        aux_hidden_states = self.execute_model_state.aux_hidden_states\n        finished_req_ids = self.execute_model_state.finished_req_ids\n        self.execute_model_state = None\n""",
        """        aux_hidden_states = self.execute_model_state.aux_hidden_states\n        finished_req_ids = self.execute_model_state.finished_req_ids\n        confidence_invalid_spec_tokens = (\n            self.execute_model_state.confidence_invalid_spec_tokens\n        )\n        self.execute_model_state = None\n""",
        label="model_runner sample state handoff",
    )
    source = replace_once(
        source,
        """            sampled_token_ids=None,  # type: ignore\n            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]\n        )\n""",
        """            sampled_token_ids=None,  # type: ignore\n            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]\n            confidence_invalid_spec_tokens=(\n                confidence_invalid_spec_tokens or None\n            ),\n        )\n""",
        label="model_runner output evidence",
    )
    source = replace_once(
        source,
        """    aux_hidden_states: list[torch.Tensor] | None\n    finished_req_ids: set[str]\n""",
        """    aux_hidden_states: list[torch.Tensor] | None\n    finished_req_ids: set[str]\n    # Physical proposal slots omitted before target CUDA-graph dispatch.\n    # V2 executes the target and sampling in separate calls, so this evidence\n    # must cross that asynchronous boundary with the other per-step state.\n    confidence_invalid_spec_tokens: dict[str, int]\n""",
        label="model_runner execute state field",
    )
    path.write_text(source, encoding="utf-8")


def patch_outputs(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        """    # req_id -> num_nans_in_logits\n    num_nans_in_logits: dict[str, int] | None = None\n\n    # information related to cudagraph execution\n""",
        """    # req_id -> num_nans_in_logits\n    num_nans_in_logits: dict[str, int] | None = None\n\n    # req_id -> confidence-invalid draft rows physically omitted by the worker.\n    # The scheduler uses this only to correct speculative metrics; its existing\n    # rejection bookkeeping still rolls back all optimistic async placeholders.\n    confidence_invalid_spec_tokens: dict[str, int] | None = None\n\n    # information related to cudagraph execution\n""",
        label="model output field",
    )
    path.write_text(source, encoding="utf-8")


def patch_async_scheduler(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        """    def _update_request_with_output(\n        self, request: Request, new_token_ids: list[int]\n    ) -> tuple[list[int], bool]:\n""",
        """    def update_from_output(self, scheduler_output, model_runner_output):\n        physical_invalid = (\n            model_runner_output.confidence_invalid_spec_tokens or {}\n        )\n        if physical_invalid:\n            merged = dict(scheduler_output.num_invalid_spec_tokens or {})\n            scheduled = scheduler_output.scheduled_spec_decode_tokens\n            for req_id, count in physical_invalid.items():\n                if count <= 0:\n                    raise RuntimeError(\n                        f\"invalid DSpark physical trim for {req_id!r}: {count}\"\n                    )\n                # Grammar validation sees the same confidence-shortened prefix\n                # and may remove more tail rows. The counts overlap, so keep\n                # the stricter prefix instead of double-counting invalid rows.\n                total = max(merged.get(req_id, 0), count)\n                logical_width = len(scheduled.get(req_id, ()))\n                if total > logical_width:\n                    raise RuntimeError(\n                        \"DSpark physical/grammar invalid count exceeds the \"\n                        f\"logical proposal for {req_id!r}: \"\n                        f\"invalid={total}, logical={logical_width}\"\n                    )\n                merged[req_id] = total\n            scheduler_output.num_invalid_spec_tokens = merged\n        return super().update_from_output(scheduler_output, model_runner_output)\n\n    def _update_request_with_output(\n        self, request: Request, new_token_ids: list[int]\n    ) -> tuple[list[int], bool]:\n""",
        label="async scheduler metrics hook",
    )
    path.write_text(source, encoding="utf-8")


def patch_cudagraph_utils(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        """from collections import defaultdict\n""",
        """import os\n\nfrom collections import defaultdict\n""",
        label="cudagraph environment import",
    )
    source = replace_once(
        source,
        """        speculative_config = self.vllm_config.speculative_config\n        if (\n            speculative_config\n            and speculative_config.uses_dynamic_speculative_decoding()\n        ):\n""",
        """        speculative_config = self.vllm_config.speculative_config\n        variable_dspark = (\n            speculative_config is not None\n            and speculative_config.use_dspark()\n            and os.environ.get(\n                \"VLLM_DSPARK_CONFIDENCE_SCHEDULER\", \"off\"\n            ) == \"on\"\n        )\n        if variable_dspark:\n            # Confidence scheduling can physically shorten a five-token DSpark\n            # proposal. Capture exact C=1 target shapes (bonus token plus the\n            # retained prefix) so the shorter verifier does not round back up.\n            decode_query_lens = list(range(1, self.decode_query_len + 1))\n        elif (\n            speculative_config\n            and speculative_config.uses_dynamic_speculative_decoding()\n        ):\n""",
        label="cudagraph variable DSpark lengths",
    )
    source = replace_once(
        source,
        """                    rounded_num_tokens = round_up(num_tokens, decode_query_len)\n                    rounded_num_reqs = rounded_num_tokens // decode_query_len\n\n                    if (\n""",
        """                    rounded_num_tokens = round_up(num_tokens, decode_query_len)\n                    rounded_num_reqs = rounded_num_tokens // decode_query_len\n\n                    # Only the normal full-width graph fans out across request\n                    # counts. The five shortened lengths are exact C=1 verifier\n                    # specializations, bounding the additional graph footprint.\n                    if (\n                        variable_dspark\n                        and decode_query_len != self.decode_query_len\n                        and rounded_num_reqs > 1\n                    ):\n                        continue\n\n                    if (\n""",
        label="cudagraph C1 capture bound",
    )
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root
    patch_model_runner(root / "vllm/v1/worker/gpu/model_runner.py")
    patch_outputs(root / "vllm/v1/outputs.py")
    patch_async_scheduler(root / "vllm/v1/core/sched/async_scheduler.py")
    patch_cudagraph_utils(root / "vllm/v1/worker/gpu/cudagraph_utils.py")


if __name__ == "__main__":
    main()
