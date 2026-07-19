# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure bookkeeping for DSpark physical variable-length verification."""

from __future__ import annotations


def trim_invalid_draft_tail(token_ids: list[int]) -> list[int]:
    """Trim a ``-1`` confidence tail and reject holes or invalid negatives."""

    if any(token_id < -1 for token_id in token_ids):
        raise ValueError(f"invalid negative draft token id: {token_ids}")
    try:
        first_invalid = token_ids.index(-1)
    except ValueError:
        return token_ids
    if any(token_id != -1 for token_id in token_ids[first_invalid:]):
        raise ValueError(f"non-contiguous confidence draft prefix: {token_ids}")
    return token_ids[:first_invalid]


def compact_scheduler_output_for_variable_drafts(
    scheduler_output,
    req_ids: list[str],
    draft_token_ids: list[list[int]],
) -> dict[str, int]:
    """Physically remove confidence-invalid verifier rows.

    Async scheduling reserves the configured speculative width before the
    DSpark proposal is available. The proposal arrives one step later in the
    worker, where this function replaces those optimistic reservations with
    the real contiguous confidence prefix. Existing target input and model
    kernels then execute ``1 + prefix_length`` rows instead of fixed width.

    The returned per-request counts are sent back in ``ModelRunnerOutput`` so
    scheduler-side acceptance metrics exclude rows that were never verified.
    """

    scheduled = scheduler_output.scheduled_spec_decode_tokens
    if not scheduled:
        return {}
    if len(req_ids) != len(draft_token_ids):
        raise RuntimeError(
            "DSpark variable verifier request/token row mismatch: "
            f"{len(req_ids)} request ids vs {len(draft_token_ids)} rows"
        )

    # Length equality is checked above; avoid ``zip(strict=...)`` so the pure
    # bookkeeping tests can also run under the repository's Python 3.9 tools.
    actual_by_req = dict(zip(req_ids, draft_token_ids))
    invalid: dict[str, int] = {}
    for req_id in tuple(scheduled):
        if req_id not in actual_by_req:
            raise RuntimeError(
                "DSpark variable verifier is missing the prior proposal for "
                f"scheduled request {req_id!r}"
            )
        optimistic_k = len(scheduled[req_id])
        physical_tokens = trim_invalid_draft_tail(actual_by_req[req_id])
        # Chunking or an end-of-request token budget can reserve fewer rows than
        # the drafter produced. It is safe to keep only the reserved prefix.
        physical_tokens = physical_tokens[:optimistic_k]
        physical_k = len(physical_tokens)
        removed = optimistic_k - physical_k
        if removed:
            remaining = scheduler_output.num_scheduled_tokens[req_id] - removed
            if remaining < 1:
                raise RuntimeError(
                    "DSpark variable verifier removed the target bonus row for "
                    f"request {req_id!r}: scheduled={optimistic_k}, "
                    f"physical={physical_k}, remaining={remaining}"
                )
            scheduler_output.num_scheduled_tokens[req_id] = remaining
            scheduler_output.total_num_scheduled_tokens -= removed
            invalid[req_id] = removed

        if physical_tokens:
            scheduled[req_id] = physical_tokens
        else:
            scheduled.pop(req_id)

    expected_total = sum(scheduler_output.num_scheduled_tokens.values())
    if scheduler_output.total_num_scheduled_tokens != expected_total:
        raise RuntimeError(
            "DSpark variable verifier total-token accounting drift: "
            f"reported={scheduler_output.total_num_scheduled_tokens}, "
            f"summed={expected_total}"
        )
    return invalid
