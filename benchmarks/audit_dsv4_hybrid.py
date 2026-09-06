#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only, bounded metadata/weight-sample audit; never a vision quality test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


def audit(text, nvidia, vision):
    models = {'text_0731': text, 'nvidia_nvfp4': nvidia, 'vision_exp': vision}
    report = {
        'schema_version': 1, 'kind': 'cpu_metadata_and_bounded_payload_samples',
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
            'No GPU allocation, model generation or complete checkpoint hashing is performed.',
        ],
    }
    additions = sorted(set(vision.weights)-set(text.weights))
    report['vision_only_additions'] = additions
    report['unrecognized_additions'] = [name for name in additions if not vision_part(name)]
    report['additions_already_in_nvidia'] = [name for name in additions if name in nvidia.weights]
    for name in SAMPLES:
        samples = {label: model.sample(name) for label, model in models.items()}
        def equal(a, b):
            return all(samples[a][key] == samples[b][key] for key in ('shape','tensor_bytes','sample_sha256'))
        report['samples'].append({'tensor': name, 'models': samples,
                                  'text_nvidia_bytes_match': equal('text_0731','nvidia_nvfp4'),
                                  'text_vision_bytes_match': equal('text_0731','vision_exp')})
    report['sample_payload_bytes'] = {name: model.payload_read for name, model in models.items()}
    report['nvidia_matches_text_samples'] = sum(r['text_nvidia_bytes_match'] for r in report['samples'])
    report['vision_matches_text_samples'] = sum(r['text_vision_bytes_match'] for r in report['samples'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ('text', 'nvidia', 'vision'):
        parser.add_argument(f'--{model}-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def expired(*_):
        raise TimeoutError('90-second metadata/sample audit deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(90)
    started = time.monotonic()
    # Never overwrite prior evidence, including failed attempts.
    with args.output.open('x') as out:
        try:
            report = audit(Checkpoint(args.text_dir), Checkpoint(args.nvidia_dir), Checkpoint(args.vision_dir))
            report['status'] = 'audit_complete_not_inference_validated'
        except Exception as exc:
            report = {'status':'failed_audit', 'error':str(exc), 'gpu_inference_run':False}
            raise
        finally:
            signal.alarm(0)
            report['elapsed_s'] = time.monotonic()-started
            json.dump(report, out, indent=2, allow_nan=False)
            out.write('\n')
    print(json.dumps({k: report[k] for k in ('status','elapsed_s','nvidia_matches_text_samples',
                                            'vision_matches_text_samples','sample_payload_bytes')}))


if __name__ == '__main__':
    main()
