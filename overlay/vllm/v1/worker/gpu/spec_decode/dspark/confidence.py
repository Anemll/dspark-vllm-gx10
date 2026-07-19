# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed DSpark confidence scheduler configuration and tensor policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os

import torch


SCHEDULER_ENV = "VLLM_DSPARK_CONFIDENCE_SCHEDULER"
THRESHOLD_ENV = "VLLM_DSPARK_CONFIDENCE_THRESHOLD"
VALID_SCHEDULERS = frozenset(("off", "on"))


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


@dataclass(frozen=True)
class DSparkConfidenceConfig:
    scheduler: str
    threshold: float

    @property
    def enabled(self) -> bool:
        return self.scheduler == "on"

    def as_dict(self) -> dict[str, object]:
        return {
            "scheduler": self.scheduler,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "threshold_domain": "sigmoid_probability_[0,1]",
        }


def parse_confidence_config(
    environment: Mapping[str, str] | None = None,
) -> DSparkConfidenceConfig:
    """Parse the public DSpark environment contract without permissive aliases."""

    values = os.environ if environment is None else environment
    scheduler = values.get(SCHEDULER_ENV, "off")
    if scheduler not in VALID_SCHEDULERS:
        raise ValueError(
            f"{SCHEDULER_ENV} must be one of {sorted(VALID_SCHEDULERS)}, "
            f"got {scheduler!r}"
        )

    raw_threshold = values.get(THRESHOLD_ENV, "0.0")
    try:
        threshold = float(raw_threshold)
    except ValueError as error:
        raise ValueError(
            f"{THRESHOLD_ENV} must be a finite float in [0.0, 1.0], "
            f"got {raw_threshold!r}"
        ) from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"{THRESHOLD_ENV} must be a finite float in [0.0, 1.0], "
            f"got {raw_threshold!r}"
        )
    if scheduler == "off" and threshold != 0.0:
        raise ValueError(
            f"{THRESHOLD_ENV} must be 0.0 when {SCHEDULER_ENV}=off, "
            f"got {raw_threshold!r}"
        )
    return DSparkConfidenceConfig(scheduler=scheduler, threshold=threshold)


def mask_draft_tokens_by_confidence(
    draft_tokens: torch.Tensor,
    confidence_logits: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the prefix through scores >= threshold and mark its tail invalid.

    The head emits unbounded logits. Matching DeepSpec, this policy applies a
    sigmoid and excludes the first below-threshold position and everything
    after it. ``-1`` is vLLM's non-token placeholder and cannot collide with a
    vocabulary id.
    """

    if draft_tokens.ndim != 2 or confidence_logits.shape != draft_tokens.shape:
        raise ValueError(
            "draft tokens and confidence logits must have the same [batch, steps] "
            f"shape, got {tuple(draft_tokens.shape)} and "
            f"{tuple(confidence_logits.shape)}"
        )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be finite and in [0.0, 1.0], got {threshold}"
        )

    if threshold == 0.0:
        lengths = torch.full(
            (draft_tokens.shape[0],),
            draft_tokens.shape[1],
            dtype=torch.int32,
            device=draft_tokens.device,
        )
        return draft_tokens, lengths

    keep = confidence_logits.sigmoid().ge(threshold)
    prefix = keep.to(torch.int32).cumprod(dim=1).to(torch.bool)
    lengths = prefix.sum(dim=1, dtype=torch.int32)
    invalid = torch.full_like(draft_tokens, -1)
    return torch.where(prefix, draft_tokens, invalid), lengths
