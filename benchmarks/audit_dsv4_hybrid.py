#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only, bounded metadata/weight-sample audit; never a vision quality test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import struct
import time

SENTINELS = {'image_start', 'image_end', 'image_newline', 'image_pad'}
STRUCTURAL_FIELDS = (
    'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
    'head_dim', 'vocab_size', 'n_routed_experts', 'num_experts_per_tok',
    'moe_intermediate_size', 'hc_mult', 'compress_ratios', 'dspark_block_size',
    'dspark_target_layer_ids',
)
SAMPLES = ['embed.weight', 'norm.weight', 'head.weight'] + [
    f'layers.{layer}.{suffix}' for layer in (3, 21, 42)
    for suffix in ('hc_attn_base', 'attn_norm.weight', 'ffn_norm.weight',
                   'ffn.gate.weight', 'ffn.experts.0.w1.weight')
]
TARGET_EXPERT_RE = re.compile(
    r'^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.'
    r'(?P<projection>w[123])\.(?P<component>[^.]+)$'
)
NVFP4_EXPERT_COMPONENTS = {
    'weight', 'weight_scale', 'weight_scale_2', 'input_scale'
}


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_ranges(length, chunk=4096):
    if length < 0 or chunk <= 0:
        raise ValueError('invalid sampling range')
    # Merge overlaps so small tensors are hashed completely, exactly once.
    ranges = sorted((start, min(length, start + chunk)) for start in
                    {0, max(0, length//2 - chunk//2), max(0, length-chunk)})
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = read_json(self.directory/'config.json')
        self.index = read_json(self.directory/'model.safetensors.index.json')
        self.weights = self.index['weight_map']
        self.headers = {}
        self.payload_read = 0

    def shard_path(self, name):
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('index shard must be a safetensors basename')
        return self.directory/name

    def header(self, shard):
        if shard not in self.headers:
            path = self.shard_path(shard)
            with path.open('rb') as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError('truncated safetensors header prefix')
                size, = struct.unpack('<Q', prefix)
                if not 2 <= size <= 16*1024**2:
                    raise ValueError('safetensors header outside audit size limit')
                header = stream.read(size)
                if len(header) != size:
                    raise ValueError('truncated safetensors header')
            self.headers[shard] = (json.loads(header), 8+size, path.stat().st_size)
        return self.headers[shard]

    def sample(self, name):
        shard = self.weights[name]
        header, base, file_size = self.header(shard)
        meta = header[name]
        start, end = meta['data_offsets']
        if not 0 <= start <= end or base+end > file_size:
            raise ValueError('tensor payload outside shard')
        ranges = sample_ranges(end-start)
        read_count = sum(b-a for a, b in ranges)
        if self.payload_read+read_count > 16*1024**2:
            raise ValueError('16 MiB per-checkpoint payload budget exceeded')
        digest = hashlib.sha256()
        with self.shard_path(shard).open('rb') as stream:
            for a, b in ranges:
                stream.seek(base+start+a)
                data = stream.read(b-a)
                if len(data) != b-a:
                    raise ValueError('short tensor payload read')
                digest.update(struct.pack('<QQ', a, b))
                digest.update(data)
        self.payload_read += read_count
        return {'dtype': meta['dtype'], 'shape': meta['shape'],
                'tensor_bytes': end-start, 'sampled_bytes': read_count,
                'ranges': ranges, 'sample_sha256': digest.hexdigest(),
                'complete_tensor': read_count == end-start}

    def tensor_hashes(self, names, *, chunk_size=8*1024**2):
        """Hash complete tensors, opening each shard only once.

        This intentionally bypasses the bounded sample payload counter and is
        called only by the explicit full-streaming CLI mode.
        """
        if chunk_size <= 0:
            raise ValueError('invalid full-hash chunk size')
        by_shard = {}
        for name in names:
            by_shard.setdefault(self.weights[name], []).append(name)
        results = {}
        payload_bytes = 0
        for shard in sorted(by_shard):
            header, base, file_size = self.header(shard)
            entries = []
            for name in by_shard[shard]:
                meta = header[name]
                start, end = meta['data_offsets']
                if not 0 <= start <= end or base+end > file_size:
                    raise ValueError(f'{name}: tensor payload outside shard')
                entries.append((start, end, name, meta))
            with self.shard_path(shard).open('rb') as stream:
                for start, end, name, meta in sorted(entries):
                    stream.seek(base+start)
                    remaining = end-start
                    digest = hashlib.sha256()
                    while remaining:
                        data = stream.read(min(remaining, chunk_size))
                        if not data:
                            raise ValueError(f'{name}: short tensor payload read')
                        digest.update(data)
                        remaining -= len(data)
                    length = end-start
                    payload_bytes += length
                    results[name] = {
                        'dtype': meta['dtype'], 'shape': meta['shape'],
                        'tensor_bytes': length, 'sha256': digest.hexdigest(),
                    }
        return results, payload_bytes

    def identity(self):
        shards = sorted(set(self.weights.values()))
        missing = [name for name in shards if not self.shard_path(name).is_file()]
        revisions = set()
        for name in shards:
            metadata = self.directory/'.cache/huggingface/download'/(name+'.metadata')
            if metadata.exists():
                revisions.add(metadata.read_text().splitlines()[0])
        return {
            'config_sha256': sha(self.directory/'config.json'),
            'index_sha256': sha(self.directory/'model.safetensors.index.json'),
            'tensor_count': len(self.weights), 'shard_count': len(shards),
            'missing_shards': missing,
            'present_shard_bytes': sum(self.shard_path(n).stat().st_size for n in shards if n not in missing),
            'download_metadata_revisions': sorted(revisions),
            'whole_checkpoint_checksum_verified': False,
        }


def vision_part(name):
    return (name.startswith(('vision.', 'aligner.')) or name in SENTINELS
            or name.endswith('.ffn.gate.bias_vl')
            or name in {f'layers.{i}.ffn.gate.bias' for i in range(3)})


def target_expert_part(name):
    """Return the target-expert component tuple, or ``None``.

    MTP experts are deliberately excluded: both NVIDIA and msuiche declare
    them outside the NVFP4 target-expert conversion.
    """
    match = TARGET_EXPERT_RE.fullmatch(name)
    if match is None:
        return None
    return (
        int(match['layer']), int(match['expert']),
        match['projection'], match['component'],
    )


def nvfp4_expert_schema(weights, *, layers, experts):
    """Validate the complete ModelOpt NVFP4 target-expert tensor inventory."""
    groups = {}
    unrecognized = []
    for name in weights:
        part = target_expert_part(name)
        if part is None:
            continue
        layer, expert, projection, component = part
        groups.setdefault((layer, expert, projection), set()).add(component)
        if component not in NVFP4_EXPERT_COMPONENTS:
            unrecognized.append(name)

    expected_groups = {
        (layer, expert, projection)
        for layer in range(layers)
        for expert in range(experts)
        for projection in ('w1', 'w2', 'w3')
    }
    actual_groups = set(groups)
    missing_groups = sorted(expected_groups-actual_groups)
    unexpected_groups = sorted(actual_groups-expected_groups)
    incomplete = []
    for group in sorted(expected_groups & actual_groups):
        missing = sorted(NVFP4_EXPERT_COMPONENTS-groups[group])
        extra = sorted(groups[group]-NVFP4_EXPERT_COMPONENTS)
        if missing or extra:
            incomplete.append({
                'layer': group[0], 'expert': group[1],
                'projection': group[2], 'missing': missing, 'extra': extra,
            })
    return {
        'expected_group_count': len(expected_groups),
        'actual_group_count': len(actual_groups),
        'expected_tensor_count': len(expected_groups)*len(NVFP4_EXPERT_COMPONENTS),
        'actual_tensor_count': sum(len(components) for components in groups.values()),
        'required_components': sorted(NVFP4_EXPERT_COMPONENTS),
        'missing_groups': [list(group) for group in missing_groups],
        'unexpected_groups': [list(group) for group in unexpected_groups],
        'incomplete_groups': incomplete,
        'unrecognized_target_expert_tensors': sorted(unrecognized),
        'complete': not (
            missing_groups or unexpected_groups or incomplete or unrecognized
        ),
    }


def full_passthrough_audit(vision, msuiche, names):
    source_hashes, source_bytes = vision.tensor_hashes(names)
    candidate_hashes, candidate_bytes = msuiche.tensor_hashes(names)
    mismatches = []
    manifest_hashes = {'vision_exp': hashlib.sha256(),
                       'msuiche_vision_nvfp4': hashlib.sha256()}
    matching = 0
    for name in sorted(names):
        source = source_hashes[name]
        candidate = candidate_hashes[name]
        for label, metadata in (('vision_exp', source),
                                ('msuiche_vision_nvfp4', candidate)):
            manifest_hashes[label].update(json.dumps(
                [name, metadata['dtype'], metadata['shape'],
                 metadata['tensor_bytes'], metadata['sha256']],
                separators=(',', ':'), ensure_ascii=True,
            ).encode())
            manifest_hashes[label].update(b'\n')
        if source == candidate:
            matching += 1
        else:
            mismatches.append({'tensor': name, 'vision_exp': source,
                               'msuiche_vision_nvfp4': candidate})
    return {
        'tensor_count': len(names),
        'matching_tensor_count': matching,
        'mismatches': mismatches,
        'payload_bytes': {'vision_exp': source_bytes,
                          'msuiche_vision_nvfp4': candidate_bytes},
        'manifest_sha256': {label: digest.hexdigest()
                            for label, digest in manifest_hashes.items()},
        'complete_match': not mismatches,
    }


def audit(text, nvidia, vision, msuiche=None, *, msuiche_passthrough='inventory'):
    models = {'text_0731': text, 'nvidia_nvfp4': nvidia, 'vision_exp': vision}
    if msuiche is not None:
        models['msuiche_vision_nvfp4'] = msuiche
    report = {
        'schema_version': 2, 'kind': 'cpu_metadata_and_bounded_payload_samples',
        'gpu_inference_run': False, 'image_quality_validated': False,
        'models': {name: model.identity() for name, model in models.items()},
        'structural_mismatches': {key: {name: model.config.get(key) for name, model in models.items()}
                                  for key in STRUCTURAL_FIELDS
                                  if any(model.config.get(key) != text.config.get(key) for model in models.values())},
        'config_choices_not_to_merge_silently': {
            key: {name: model.config.get(key) for name, model in models.items()}
            for key in ('rms_norm_eps', 'num_nextn_predict_layers', 'quantization_config')
        },
        'samples': [],
        'caveats': [
            'The NVIDIA model is text-only. A transplant is a new, unvalidated hybrid.',
            'Equal samples do not prove whole-checkpoint equality; unequal samples do not prove the hybrid cannot work.',
            'Matching shapes do not establish learned visual alignment or quality.',
            'No GPU allocation, model generation, whole-checkpoint hash, or all-expert value scan is performed.',
        ],
    }
    additions = sorted(set(vision.weights)-set(text.weights))
    report['vision_only_additions'] = additions
    report['unrecognized_additions'] = [name for name in additions if not vision_part(name)]
    report['additions_already_in_nvidia'] = [name for name in additions if name in nvidia.weights]
    if msuiche is not None:
        vision_passthrough_tensors = {
            name for name in vision.weights if target_expert_part(name) is None
        }
        msuiche_passthrough_tensors = {
            name for name in msuiche.weights if target_expert_part(name) is None
        }
        quant = msuiche.config.get('quantization_config') or {}
        advertised_architectures = msuiche.config.get('architectures')
        automatic_vision_route = (
            msuiche.config.get('vision_n_layers', 0) > 0
            and advertised_architectures == ['DeepseekV4ForCausalLM']
        )
        report['msuiche_contract'] = {
            'advertised_architectures': advertised_architectures,
            'vision_n_layers': msuiche.config.get('vision_n_layers'),
            'automatic_runtime_architecture_conversion_expected': automatic_vision_route,
            'expected_runtime_architectures': (
                ['DeepseekV4ForConditionalGeneration']
                if automatic_vision_route else advertised_architectures
            ),
            'requires_config_file_override': False if automatic_vision_route else None,
            'quant_method': quant.get('quant_method'),
            'moe_quant_algo': quant.get('moe_quant_algo'),
            'group_size': quant.get('group_size'),
            'scale_fmt': quant.get('scale_fmt'),
            'ignored_modules': quant.get('ignore'),
            'non_target_inventory': {
                'vision_count': len(vision_passthrough_tensors),
                'msuiche_count': len(msuiche_passthrough_tensors),
                'missing_from_msuiche': sorted(
                    vision_passthrough_tensors-msuiche_passthrough_tensors
                ),
                'unexpected_in_msuiche': sorted(
                    msuiche_passthrough_tensors-vision_passthrough_tensors
                ),
                'complete_match': (
                    vision_passthrough_tensors == msuiche_passthrough_tensors
                ),
                'payload_equality_checked': False,
            },
            'target_expert_schema': nvfp4_expert_schema(
                msuiche.weights,
                layers=vision.config['num_hidden_layers'],
                experts=vision.config['n_routed_experts'],
            ),
        }
        report['caveats'].extend([
            'The msuiche inventory match does not prove passthrough tensor payload equality.',
            'The msuiche expert schema check does not prove numerical conversion equality or kernel compatibility.',
        ])
        if msuiche_passthrough == 'full':
            report['msuiche_contract']['non_target_full_streaming_hashes'] = (
                full_passthrough_audit(
                    vision, msuiche, vision_passthrough_tensors
                )
            )
            report['msuiche_contract']['non_target_inventory'][
                'payload_equality_checked'
            ] = True
        elif msuiche_passthrough != 'inventory':
            raise ValueError(f'unknown msuiche passthrough mode {msuiche_passthrough!r}')
    for name in SAMPLES:
        samples = {label: model.sample(name) for label, model in models.items()}
        def equal(a, b):
            return all(samples[a][key] == samples[b][key] for key in ('shape','tensor_bytes','sample_sha256'))
        report['samples'].append({'tensor': name, 'models': samples,
                                  'text_nvidia_bytes_match': equal('text_0731','nvidia_nvfp4'),
                                  'text_vision_bytes_match': equal('text_0731','vision_exp'),
                                  'vision_msuiche_bytes_match': (
                                      equal('vision_exp','msuiche_vision_nvfp4')
                                      if msuiche is not None else None
                                  )})
    report['sample_payload_bytes'] = {name: model.payload_read for name, model in models.items()}
    report['nvidia_matches_text_samples'] = sum(r['text_nvidia_bytes_match'] for r in report['samples'])
    report['vision_matches_text_samples'] = sum(r['text_vision_bytes_match'] for r in report['samples'])
    if msuiche is not None:
        report['msuiche_matches_vision_samples'] = sum(
            r['vision_msuiche_bytes_match'] for r in report['samples']
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ('text', 'nvidia', 'vision'):
        parser.add_argument(f'--{model}-dir', type=Path, required=True)
    parser.add_argument('--msuiche-dir', type=Path)
    parser.add_argument('--msuiche-passthrough', choices=('inventory', 'full'),
                        default='inventory')
    parser.add_argument('--deadline-seconds', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    deadline = args.deadline_seconds or (
        3600 if args.msuiche_passthrough == 'full' else 90
    )
    if not 1 <= deadline <= 7200:
        parser.error('--deadline-seconds must be in [1, 7200]')
    if args.msuiche_passthrough == 'full' and args.msuiche_dir is None:
        parser.error('--msuiche-passthrough full requires --msuiche-dir')
    def expired(*_):
        raise TimeoutError(f'{deadline}-second metadata/sample audit deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(deadline)
    started = time.monotonic()
    # Never overwrite prior evidence, including failed attempts.
    with args.output.open('x') as out:
        try:
            report = audit(
                Checkpoint(args.text_dir), Checkpoint(args.nvidia_dir),
                Checkpoint(args.vision_dir),
                Checkpoint(args.msuiche_dir) if args.msuiche_dir else None,
                msuiche_passthrough=args.msuiche_passthrough,
            )
            report['status'] = 'audit_complete_not_inference_validated'
        except Exception as exc:
            report = {'status':'failed_audit', 'error':str(exc), 'gpu_inference_run':False}
            raise
        finally:
            signal.alarm(0)
            report['elapsed_s'] = time.monotonic()-started
            json.dump(report, out, indent=2, allow_nan=False)
            out.write('\n')
    summary_keys = ('status', 'elapsed_s', 'nvidia_matches_text_samples',
                    'vision_matches_text_samples',
                    'msuiche_matches_vision_samples', 'sample_payload_bytes')
    print(json.dumps({k: report[k] for k in summary_keys if k in report}))


if __name__ == '__main__':
    main()
