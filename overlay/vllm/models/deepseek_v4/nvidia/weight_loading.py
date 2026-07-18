# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast, compatibility-preserving DeepSeek V4 expert-name resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ExpertMapping = tuple[str, str, int, str]

_DEEPSEEK_V4_EXPERT_NAME_RE = re.compile(
    r"^(?P<prefix>(?:model\.)?layers\.(?P<layer>[0-9]+)\.ffn\.)"
    r"(?P<mapping_key>"
    r"experts\.(?P<logical_expert>[0-9]+)\."
    r"(?P<projection>w[123])\."
    r"(?:(?P<lora_base>base_layer)\.)?"
    r")"
    r"(?P<suffix>weight|weight_scale|weight_scale_2|input_scale)$"
)
_DEEPSEEK_V4_MAPPING_KEY_RE = re.compile(
    r"^experts\.[0-9]+\.w[123]\.(?:base_layer\.)?$"
)
_KNOWN_FUSED_MAPPING_KEY_RE = re.compile(r"^experts\.(?:w13|w2)$")


@dataclass(frozen=True)
class ExpertNameMatch:
    """Parsed checkpoint expert name with an exact mapping-table key."""

    prefix: str
    mapping_key: str
    suffix: str
    layer: int
    logical_expert: int
    projection: str
    lora_base: bool

    def map_parameter_name(self, param_name: str) -> str:
        """Replace the checkpoint mapping key without substring scanning."""

        return f"{self.prefix}{param_name}{self.suffix}"


@dataclass(frozen=True)
class ExpertMappingIndex:
    """Ordered fast candidates plus proof that indexing preserves scan order."""

    mappings: dict[str, tuple[ExpertMapping, ...]]
    safe: bool


def parse_expert_name(name: str) -> ExpertNameMatch | None:
    """Parse the NVIDIA DeepSeek V4 per-expert checkpoint grammar.

    Names outside the validated target grammar deliberately return ``None`` so
    callers can use the existing full-scan compatibility path.
    """

    match = _DEEPSEEK_V4_EXPERT_NAME_RE.fullmatch(name)
    if match is None:
        return None
    return ExpertNameMatch(
        prefix=match.group("prefix"),
        mapping_key=match.group("mapping_key"),
        suffix=match.group("suffix"),
        layer=int(match.group("layer")),
        logical_expert=int(match.group("logical_expert")),
        projection=match.group("projection"),
        lora_base=match.group("lora_base") is not None,
    )


def build_expert_mapping_index(
    expert_mapping: Iterable[ExpertMapping],
) -> ExpertMappingIndex:
    """Index the authoritative vLLM mapping while preserving tuple order.

    A logical checkpoint key can map to multiple physical experts under EPLB.
    Values therefore remain ordered tuples instead of collapsing to one entry.
    """

    mutable_index: dict[str, list[ExpertMapping]] = {}
    safe = True
    for mapping in expert_mapping:
        key = mapping[1]
        if _DEEPSEEK_V4_MAPPING_KEY_RE.fullmatch(key) is not None:
            mutable_index.setdefault(key, []).append(mapping)
        elif _KNOWN_FUSED_MAPPING_KEY_RE.fullmatch(key) is None:
            # Unknown mapping grammars must preserve the full legacy scan.
            safe = False

    indexed_keys = tuple(mutable_index)
    for index, key in enumerate(indexed_keys):
        for other_key in indexed_keys[index + 1 :]:
            if key in other_key or other_key in key:
                # Plain and LoRA keys can overlap. Legacy order is authoritative.
                safe = False

    return ExpertMappingIndex(
        mappings={key: tuple(value) for key, value in mutable_index.items()},
        safe=safe,
    )


def select_expert_mappings(
    name: str,
    expert_mapping: Sequence[ExpertMapping],
    expert_mapping_index: ExpertMappingIndex,
) -> tuple[ExpertNameMatch | None, Sequence[ExpertMapping]]:
    """Return indexed candidates, or the exact legacy scan as a fallback."""

    if not expert_mapping_index.safe:
        return None, expert_mapping
    match = parse_expert_name(name)
    if match is None:
        return None, expert_mapping
    candidates = expert_mapping_index.mappings.get(match.mapping_key)
    if candidates is None:
        return None, expert_mapping
    return match, candidates


def map_expert_parameter_name(
    name: str,
    param_name: str,
    weight_name: str,
    match: ExpertNameMatch | None,
) -> str:
    """Map a checkpoint name using the fast parse or legacy replacement."""

    if match is not None and match.mapping_key == weight_name:
        return match.map_parameter_name(param_name)
    return name.replace(weight_name, param_name)
