# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "audit_nvfp4_dspark_pinned_upstream.py"
SPEC = importlib.util.spec_from_file_location("upstream_audit", PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class PinnedUpstreamAuditTests(unittest.TestCase):
    def test_explicit_dspark_model_builds_separate_config(self) -> None:
        source = '''
class SpeculativeConfig:
    model: str | None = None
    def __post_init__(self):
        if self.model is None and self.method == "dspark":
            self.model = self.target_model_config.model
        self.draft_model_config = ModelConfig(model=self.model)
'''
        result = audit.audit_speculative_config(source)
        self.assertTrue(result["explicit_model_builds_separate_draft_model_config"])

    def test_dspark_model_loader_uses_draft_config(self) -> None:
        source = '''
def load_dspark_model(target_model, vllm_config):
    draft_model_config = vllm_config.speculative_config.draft_model_config
    return get_model(vllm_config=vllm_config, model_config=draft_model_config)
'''
        self.assertTrue(
            audit.audit_dspark_loader(source)["get_model_receives_draft_model_config"]
        )

    def test_dspark_weight_loader_filters_non_mtp(self) -> None:
        source = r'''
class DSparkDeepseekV4ForCausalLM:
    def load_weights(self, weights):
        for name, value in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
    def _remap_dspark_name(self, name):
        match = re.match(r"mtp\.(\d+)\.(.*)", name)
        if match is None:
            return None
        return match.group(2)
'''
        self.assertTrue(audit.audit_dspark_weights(source)["non_mtp_weights_are_skipped"])

    def test_acceptance_metric_source_is_complete(self) -> None:
        source = '''
METRICS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "position",
)
'''
        result = audit.audit_metrics(source)
        self.assertIn("accepted_per_position", result["prometheus_counters"])

    def test_non_eplb_tp_experts_are_partitioned_exactly(self) -> None:
        source = '''
class DeepseekV4MoE:
    def _init_fused_moe_experts(self):
        self.n_local_physical_experts = self.n_physical_experts // self.tp_size
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = self.tp_rank * self.n_local_experts
'''
        result = audit.audit_tp_expert_partition(source)
        self.assertTrue(result["non_eplb_physical_experts_divided_by_tp_size"])

    def test_tp_expert_replication_drift_fails(self) -> None:
        source = '''
class DeepseekV4MoE:
    def _init_fused_moe_experts(self):
        self.n_local_physical_experts = self.n_physical_experts
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = 0
'''
        with self.assertRaises(audit.AuditError):
            audit.audit_tp_expert_partition(source)

    def test_missing_per_position_metric_fails(self) -> None:
        with self.assertRaises(audit.AuditError):
            audit.audit_metrics('VALUES = ("vllm:spec_decode_num_drafts",)')


if __name__ == "__main__":
    unittest.main()
