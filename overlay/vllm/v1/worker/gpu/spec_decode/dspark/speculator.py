# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import json
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.confidence import (
    DSparkConfidenceMetrics,
    get_confidence_metrics,
    mask_draft_tokens_by_confidence,
    parse_confidence_config,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


logger = init_logger(__name__)


class _DSparkConfidenceTelemetry:
    """Non-blocking real-score capture outside the DSpark CUDA graph.

    Two tiny pinned buffers allow the CPU to consume one batch while the next
    D2H copy is in flight. If both are unexpectedly busy, telemetry drops that
    batch and records the loss instead of stalling decode.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        num_steps: int,
        threshold: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.metrics: DSparkConfidenceMetrics = get_confidence_metrics(threshold)
        self.host_logits = [
            torch.empty(
                max_num_reqs,
                num_steps,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.pending_rows = [0, 0]

    def _drain_ready(self) -> None:
        for slot, rows in enumerate(self.pending_rows):
            if rows and self.events[slot].query():
                self.metrics.observe(self.host_logits[slot][:rows])
                self.pending_rows[slot] = 0

    def capture(self, confidence_logits: torch.Tensor) -> None:
        self._drain_ready()
        try:
            slot = self.pending_rows.index(0)
        except ValueError:
            self.metrics.observe_dropped_batch()
            return

        rows = confidence_logits.shape[0]
        # Enqueue the tiny copy on the current stream. This preserves ordering
        # with the persistent logits buffer without a host-side synchronize.
        self.host_logits[slot][:rows].copy_(confidence_logits, non_blocking=True)
        self.events[slot].record(torch.cuda.current_stream(self.device))
        self.pending_rows[slot] = rows


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.confidence_config = parse_confidence_config()
        super().__init__(vllm_config, device)

        # Anchor-as-first (N slots) unless the checkpoint uses the 1+N fill-in
        # block, where the anchor is a separate bonus token.
        self.sample_from_anchor = not getattr(
            self.draft_model_config.hf_config, "dspark_bonus_anchor", False
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self.dflash_causal = False

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        self.confidence_logits = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.confidence_prefix_lengths = torch.full(
            (self.max_num_reqs,),
            self.num_speculative_steps,
            dtype=torch.int32,
            device=device,
        )
        self.confidence_head_ready = False
        self._confidence_telemetry = (
            _DSparkConfidenceTelemetry(
                max_num_reqs=self.max_num_reqs,
                num_steps=self.num_speculative_steps,
                threshold=self.confidence_config.threshold,
                device=device,
            )
            if self.confidence_config.enabled
            else None
        )

        logger.info_once(
            "DSpark confidence scheduler config: %s",
            json.dumps(self.confidence_config.as_dict(), sort_keys=True),
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        has_confidence_api = hasattr(model, "compute_confidence_logits")
        confidence_head_loaded = bool(
            getattr(model, "confidence_head_loaded", False)
        )
        if has_confidence_api and not confidence_head_loaded:
            raise RuntimeError(
                "DSpark model exposes confidence logits but its confidence head "
                "weight was not loaded"
            )
        if self.confidence_config.enabled and not (
            has_confidence_api and confidence_head_loaded
        ):
            raise RuntimeError(
                "DSpark confidence scheduler is enabled for a draft model without "
                "a loaded confidence head"
            )
        self.confidence_head_ready = has_confidence_api and confidence_head_loaded
        logger.info_once(
            "DSpark confidence scheduler startup proof: %s",
            json.dumps(self.confidence_contract(), sort_keys=True),
        )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def confidence_contract(self) -> dict[str, object]:
        return {
            **self.confidence_config.as_dict(),
            "confidence_head_loaded": self.confidence_head_ready,
            "sentinel_token_id": -1,
            "prefix_policy": "first_sigmoid_probability_below_threshold",
        }

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        sample_hidden = sample_hidden.view(num_reqs, n_spec, -1)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.confidence_config.enabled:
                self.confidence_logits[:num_reqs, i].copy_(
                    self.model.compute_confidence_logits(
                        sample_hidden[:, i], markov_embed
                    )
                )
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.confidence_config.enabled:
            scores = self.confidence_logits[:num_reqs]
            masked_tokens, prefix_lengths = mask_draft_tokens_by_confidence(
                self.draft_tokens[:num_reqs],
                scores,
                threshold=self.confidence_config.threshold,
            )
            self.draft_tokens[:num_reqs].copy_(masked_tokens)
            self.confidence_prefix_lengths[:num_reqs].copy_(prefix_lengths)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        draft_tokens = super().propose(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )
        if self._confidence_telemetry is not None and not dummy_run:
            self._confidence_telemetry.capture(
                self.confidence_logits[: input_batch.num_reqs]
            )
        return draft_tokens
