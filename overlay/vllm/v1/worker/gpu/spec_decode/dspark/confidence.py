# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed DSpark confidence scheduler configuration and tensor policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from threading import Lock
from typing import Any

import torch

from vllm.v1.worker.gpu.spec_decode.dspark.variable_verifier import (
    compact_scheduler_output_for_variable_drafts,
    trim_invalid_draft_tail,
)


SCHEDULER_ENV = "VLLM_DSPARK_CONFIDENCE_SCHEDULER"
THRESHOLD_ENV = "VLLM_DSPARK_CONFIDENCE_THRESHOLD"
VALID_SCHEDULERS = frozenset(("off", "on"))


_metrics_lock = Lock()
_metrics_instance: DSparkConfidenceMetrics | None = None


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

    _, _, prefix, lengths = confidence_probability_policy(
        confidence_logits,
        threshold=threshold,
    )
    invalid = torch.full_like(draft_tokens, -1)
    return torch.where(prefix, draft_tokens, invalid), lengths


def confidence_probability_policy(
    confidence_logits: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply DeepSpec's sigmoid-probability contiguous-prefix policy.

    Returns ``(probabilities, below_threshold, prefix_mask, prefix_lengths)``.
    Keeping this calculation shared by inference and telemetry prevents score
    logging from silently using a different threshold domain.
    """

    if confidence_logits.ndim != 2:
        raise ValueError(
            "confidence logits must have [batch, steps] shape, got "
            f"{tuple(confidence_logits.shape)}"
        )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be finite and in [0.0, 1.0], got {threshold}"
        )

    probabilities = confidence_logits.sigmoid()
    below_threshold = probabilities.lt(threshold)
    prefix = (
        below_threshold.logical_not()
        .to(torch.int32)
        .cumprod(dim=1)
        .to(torch.bool)
    )
    prefix_lengths = prefix.sum(dim=1, dtype=torch.int32)
    return probabilities, below_threshold, prefix, prefix_lengths


class DSparkConfidenceMetrics:
    """Prometheus observations for real DSpark confidence-head outputs.

    The worker records sigmoid probabilities rather than raw logits because
    that is the threshold domain used by DeepSpec and by the masking policy.
    Position and threshold labels make the resulting distribution auditable.
    """

    def __init__(self, threshold: float, *, registry: Any | None = None):
        from prometheus_client import Counter, Histogram

        self.threshold = threshold
        self.threshold_label = format(threshold, ".6g")
        metric_kwargs = {} if registry is None else {"registry": registry}
        probability_buckets = tuple(i / 100.0 for i in range(101))

        self.probability = Histogram(
            "vllm:dspark_confidence_probability",
            "DSpark confidence-head sigmoid probability by draft position.",
            ("position", "threshold"),
            buckets=probability_buckets,
            **metric_kwargs,
        )
        self.below_threshold = Counter(
            "vllm:dspark_confidence_below_threshold",
            "DSpark confidence positions below the active probability threshold.",
            ("position", "threshold"),
            **metric_kwargs,
        )
        self.prefix_length = Histogram(
            "vllm:dspark_confidence_prefix_length",
            "Logical DSpark proposal prefix length after confidence masking.",
            ("threshold",),
            buckets=tuple(range(6)),
            **metric_kwargs,
        )
        self.position_exposed = Counter(
            "vllm:dspark_confidence_position_exposed",
            "DSpark draft positions retained by the confidence prefix policy.",
            ("position", "threshold"),
            **metric_kwargs,
        )
        self.physical_target_rows = Histogram(
            "vllm:dspark_confidence_physical_target_rows",
            "Physical target verifier rows after confidence compaction.",
            ("threshold",),
            buckets=tuple(range(1, 7)),
            **metric_kwargs,
        )
        self.d2h_copy_completion = Counter(
            "vllm:dspark_confidence_d2h_copy_completion",
            "DSpark proposal D2H copies ready at compaction or requiring a wait.",
            ("result", "threshold"),
            **metric_kwargs,
        )
        self.dropped_batches = Counter(
            "vllm:dspark_confidence_telemetry_dropped_batches",
            "Confidence telemetry batches dropped because both D2H slots were busy.",
            ("threshold",),
            **metric_kwargs,
        )

    def observe(self, confidence_logits: torch.Tensor) -> dict[str, object]:
        """Record one CPU-resident ``[batch, steps]`` logits tensor."""

        if confidence_logits.device.type != "cpu":
            raise ValueError("confidence telemetry must observe CPU-resident logits")
        probabilities, below, prefix, prefix_lengths = (
            confidence_probability_policy(
                confidence_logits,
                threshold=self.threshold,
            )
        )
        probability_rows = probabilities.tolist()
        below_counts = below.sum(dim=0, dtype=torch.int64).tolist()
        exposed_counts = prefix.sum(dim=0, dtype=torch.int64).tolist()
        prefix_values = prefix_lengths.tolist()

        for position in range(probabilities.shape[1]):
            labels = {
                "position": str(position),
                "threshold": self.threshold_label,
            }
            histogram = self.probability.labels(**labels)
            for row in probability_rows:
                histogram.observe(float(row[position]))
            count = int(below_counts[position])
            # inc(0) intentionally materializes a zero-valued series. An absent
            # position must not be mistaken for missing telemetry.
            self.below_threshold.labels(**labels).inc(count)
            self.position_exposed.labels(**labels).inc(
                int(exposed_counts[position])
            )

        prefix_histogram = self.prefix_length.labels(
            threshold=self.threshold_label
        )
        for length in prefix_values:
            prefix_histogram.observe(int(length))

        return {
            "probabilities": probability_rows,
            "below_threshold_per_position": [int(value) for value in below_counts],
            "exposed_per_position": [int(value) for value in exposed_counts],
            "prefix_lengths": [int(value) for value in prefix_values],
        }

    def observe_physical_target_rows(self, rows) -> None:
        histogram = self.physical_target_rows.labels(
            threshold=self.threshold_label
        )
        for value in rows:
            value = int(value)
            if not 1 <= value <= 6:
                raise ValueError(
                    f"DSpark physical target rows must be in [1, 6], got {value}"
                )
            histogram.observe(value)

    def observe_d2h_copy_completion(self, *, fallback_wait: bool) -> None:
        result = "fallback_wait" if fallback_wait else "ready"
        self.d2h_copy_completion.labels(
            result=result,
            threshold=self.threshold_label,
        ).inc()

    def observe_dropped_batch(self) -> None:
        self.dropped_batches.labels(threshold=self.threshold_label).inc()


def get_confidence_metrics(threshold: float) -> DSparkConfidenceMetrics:
    """Return the process-global collector for the worker's fixed threshold."""

    global _metrics_instance
    key = format(threshold, ".6g")
    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = DSparkConfidenceMetrics(threshold)
        elif _metrics_instance.threshold_label != key:
            raise RuntimeError(
                "one vLLM worker cannot register multiple DSpark confidence "
                f"thresholds: {_metrics_instance.threshold_label} and {key}"
            )
        return _metrics_instance


def observe_engine_compaction_telemetry(
    physical_rows: list[int] | None,
    d2h_fallback: bool | None,
) -> None:
    """Export worker compaction evidence after the engine result boundary."""

    if (physical_rows is None) != (d2h_fallback is None):
        raise RuntimeError(
            "incomplete DSpark physical-row telemetry handoff: "
            f"rows={physical_rows}, d2h_fallback={d2h_fallback}"
        )
    if physical_rows is None:
        return
    metrics = get_confidence_metrics(parse_confidence_config().threshold)
    metrics.observe_physical_target_rows(physical_rows)
    metrics.observe_d2h_copy_completion(fallback_wait=d2h_fallback)
