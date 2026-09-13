#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare every routed-expert NVFP4 input scale between two checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import signal
import time

try:
    from benchmarks.audit_dsv4_hybrid import Checkpoint, target_expert_part
except ModuleNotFoundError:  # Direct ``python3 benchmarks/<script>.py`` use.
    from audit_dsv4_hybrid import Checkpoint, target_expert_part


def input_scale_names(checkpoint):
    return {
        name for name in checkpoint.weights
        if (part := target_expert_part(name)) is not None
        and part[3] == 'input_scale'
    }


def read_complete_scalars(checkpoint, names, *, workers):
    by_shard = {}
    for name in names:
        by_shard.setdefault(checkpoint.weights[name], []).append(name)

    def read_shard(item):
        shard, shard_names = item
        header, base, file_size = checkpoint.header(shard)
        results = {}
        with checkpoint.shard_path(shard).open('rb') as stream:
            for name in sorted(shard_names):
                metadata = header[name]
                start, end = metadata['data_offsets']
                if not 0 <= start <= end or base+end > file_size:
                    raise ValueError(f'{name}: tensor payload outside shard')
                stream.seek(base+start)
                payload = stream.read(end-start)
                if len(payload) != end-start:
                    raise ValueError(f'{name}: short tensor payload read')
                results[name] = {
                    'dtype': metadata['dtype'], 'shape': metadata['shape'],
                    'tensor_bytes': end-start,
                    'sha256': hashlib.sha256(payload).hexdigest(),
                }
        return results

    hashes = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shard_results in pool.map(read_shard, sorted(by_shard.items())):
            hashes.update(shard_results)
    return hashes, sum(value['tensor_bytes'] for value in hashes.values())


def compare_input_scales(reference, candidate, *, workers=8):
    reference_names = input_scale_names(reference)
    candidate_names = input_scale_names(candidate)
    common = reference_names & candidate_names
    reference_hashes, reference_bytes = read_complete_scalars(
        reference, common, workers=workers
    )
    candidate_hashes, candidate_bytes = read_complete_scalars(
        candidate, common, workers=workers
    )
    mismatches = []
    manifest = {'reference': hashlib.sha256(), 'candidate': hashlib.sha256()}
    for name in sorted(common):
        left = reference_hashes[name]
        right = candidate_hashes[name]
        for label, value in (('reference', left), ('candidate', right)):
            manifest[label].update(json.dumps(
                [name, value['dtype'], value['shape'], value['tensor_bytes'],
                 value['sha256']], separators=(',', ':'), ensure_ascii=True,
            ).encode())
            manifest[label].update(b'\n')
        if left != right:
            mismatches.append(name)
    return {
        'reference_tensor_count': len(reference_names),
        'candidate_tensor_count': len(candidate_names),
        'common_tensor_count': len(common),
        'missing_from_candidate': sorted(reference_names-candidate_names),
        'unexpected_in_candidate': sorted(candidate_names-reference_names),
        'mismatch_count': len(mismatches),
        'first_mismatches': mismatches[:20],
        'payload_bytes': {
            'reference': reference_bytes, 'candidate': candidate_bytes,
        },
        'manifest_sha256': {
            label: digest.hexdigest() for label, digest in manifest.items()
        },
        'complete_match': (
            reference_names == candidate_names and not mismatches
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=int, default=600)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 3600:
        parser.error('--deadline-seconds must be in [1, 3600]')
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be in [1, 32]')

    def expired(*_):
        raise TimeoutError(f'{args.deadline_seconds}-second input-scale deadline')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.deadline_seconds)
    started = time.monotonic()
    report = {
        'schema_version': 1,
        'kind': 'complete_nvfp4_routed_expert_input_scale_audit',
        'gpu_inference_run': False,
        'reference': str(args.reference_dir),
        'candidate': str(args.candidate_dir),
    }
    with args.output.open('x') as output:
        try:
            report['comparison'] = compare_input_scales(
                Checkpoint(args.reference_dir), Checkpoint(args.candidate_dir),
                workers=args.workers,
            )
            report['status'] = 'audit_complete_not_inference_validated'
        except BaseException as exc:
            report['status'] = (
                'interrupted_audit' if isinstance(exc, KeyboardInterrupt)
                else 'failed_audit'
            )
            report['error'] = str(exc)
            raise
        finally:
            signal.alarm(0)
            report['elapsed_s'] = time.monotonic()-started
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
    print(json.dumps({
        'status': report['status'],
        'complete_match': report['comparison']['complete_match'],
        'tensor_count': report['comparison']['common_tensor_count'],
        'mismatch_count': report['comparison']['mismatch_count'],
        'elapsed_s': report['elapsed_s'],
    }))


if __name__ == '__main__':
    main()
