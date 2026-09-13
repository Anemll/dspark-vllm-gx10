#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare one DeepSeek MXFP4 expert with its ModelOpt NVFP4 transcode.

This CPU-only gate reads exactly one expert's requested projections. It is a
logical-value audit, not an inference or full-checkpoint equality claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import struct
import time

import numpy as np

try:
    from benchmarks.audit_dsv4_hybrid import Checkpoint
except ModuleNotFoundError:  # Direct ``python3 benchmarks/<script>.py`` use.
    from audit_dsv4_hybrid import Checkpoint


E2M1_VALUES = np.asarray([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def decode_e8m0(raw):
    """Decode unsigned E8M0FN bytes using the DeepSeek/UE8M0 convention."""
    raw = np.asarray(raw, dtype=np.uint8)
    exponent = raw.astype(np.int16)-127
    decoded = np.exp2(exponent.astype(np.float64))
    return np.where(raw == 0, 0.0, decoded)


def decode_e4m3fn(raw):
    """Decode NVIDIA finite-only E4M3 bytes without a torch dependency."""
    raw = np.asarray(raw, dtype=np.uint8)
    sign = np.where(raw & 0x80, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0f
    mantissa = raw & 0x07
    normal = np.exp2(exponent.astype(np.int16)-7) * (1.0+mantissa/8.0)
    subnormal = mantissa.astype(np.float64) * 2.0**-9
    decoded = sign * np.where(exponent == 0, subnormal, normal)
    # E4M3FN reserves only the all-ones exponent/mantissa payload for NaN.
    return np.where((exponent == 15) & (mantissa == 7), np.nan, decoded)


def unpack_e2m1(raw):
    raw = np.asarray(raw, dtype=np.uint8)
    unpacked = np.empty(raw.shape[:-1]+(raw.shape[-1]*2,), dtype=np.float32)
    unpacked[..., 0::2] = E2M1_VALUES[raw & 0x0f]
    unpacked[..., 1::2] = E2M1_VALUES[raw >> 4]
    return unpacked


def read_tensor(checkpoint, name, *, byte_budget):
    shard = checkpoint.weights[name]
    header, base, file_size = checkpoint.header(shard)
    metadata = header[name]
    start, end = metadata['data_offsets']
    length = end-start
    if not 0 <= start <= end or base+end > file_size:
        raise ValueError(f'{name}: tensor payload outside shard')
    if length > byte_budget:
        raise ValueError(f'{name}: {length} bytes exceed {byte_budget}-byte tensor budget')
    with checkpoint.shard_path(shard).open('rb') as stream:
        stream.seek(base+start)
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError(f'{name}: short tensor payload read')
    return metadata, payload


def reshape_raw(metadata, payload):
    dtype = metadata['dtype']
    if dtype in {'I8', 'U8', 'F8_E8M0', 'F8_E4M3'}:
        array = np.frombuffer(payload, dtype=np.uint8)
    elif dtype == 'F32':
        array = np.frombuffer(payload, dtype='<f4')
    else:
        raise ValueError(f'unsupported audit dtype {dtype!r}')
    shape = tuple(metadata['shape'])
    if shape:
        array = array.reshape(shape)
    elif array.size == 1:
        array = array.reshape(())
    return array


def load_array(checkpoint, name, *, byte_budget=8*1024**2):
    metadata, payload = read_tensor(checkpoint, name, byte_budget=byte_budget)
    return metadata, reshape_raw(metadata, payload)


def metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f'metric shape mismatch {reference.shape} != {candidate.shape}')
    finite = np.isfinite(reference) & np.isfinite(candidate)
    delta = candidate-reference
    ref_norm = float(np.linalg.norm(reference[finite].reshape(-1)))
    delta_norm = float(np.linalg.norm(delta[finite].reshape(-1)))
    return {
        'element_count': int(reference.size),
        'finite_count': int(finite.sum()),
        'exact_equal_count': int(np.equal(reference, candidate).sum()),
        'exact_equal_fraction': float(np.equal(reference, candidate).mean()),
        'max_abs_error': float(np.max(np.abs(delta[finite]))) if finite.any() else None,
        'mean_abs_error': float(np.mean(np.abs(delta[finite]))) if finite.any() else None,
        'relative_l2': delta_norm/ref_norm if ref_norm else (0.0 if delta_norm == 0 else math.inf),
    }


def audit_projection(source, candidate, prefix):
    source_weight_meta, source_packed = load_array(source, prefix+'.weight')
    source_scale_meta, source_scale_raw = load_array(source, prefix+'.scale')
    candidate_weight_meta, candidate_packed = load_array(candidate, prefix+'.weight')
    candidate_scale_meta, candidate_scale_raw = load_array(
        candidate, prefix+'.weight_scale'
    )
    global_meta, global_array = load_array(candidate, prefix+'.weight_scale_2')
    input_meta, input_array = load_array(candidate, prefix+'.input_scale')

    if source_weight_meta['dtype'] != 'I8' or candidate_weight_meta['dtype'] != 'U8':
        raise ValueError(f'{prefix}: expected source I8 and candidate U8 packed weights')
    if source_scale_meta['dtype'] != 'F8_E8M0':
        raise ValueError(f'{prefix}: source scale is not F8_E8M0')
    if candidate_scale_meta['dtype'] != 'F8_E4M3':
        raise ValueError(f'{prefix}: candidate scale is not F8_E4M3')
    if global_meta['dtype'] != 'F32' or input_meta['dtype'] != 'F32':
        raise ValueError(f'{prefix}: candidate global/input scales are not F32')
    if source_packed.shape != candidate_packed.shape:
        raise ValueError(f'{prefix}: packed weight shapes differ')

    source_fp4 = unpack_e2m1(source_packed)
    candidate_fp4 = unpack_e2m1(candidate_packed)
    logical_cols = source_fp4.shape[-1]
    if source_scale_raw.shape != source_fp4.shape[:-1]+(logical_cols//32,):
        raise ValueError(f'{prefix}: source GS32 scale shape mismatch')
    if candidate_scale_raw.shape != candidate_fp4.shape[:-1]+(logical_cols//16,):
        raise ValueError(f'{prefix}: candidate GS16 scale shape mismatch')

    source_scales = decode_e8m0(source_scale_raw)
    candidate_global = float(global_array)
    candidate_scales = decode_e4m3fn(candidate_scale_raw)*candidate_global
    if not np.isfinite(candidate_global) or candidate_global <= 0:
        raise ValueError(f'{prefix}: invalid candidate weight_scale_2')
    if not np.isfinite(candidate_scales).all():
        raise ValueError(f'{prefix}: non-finite decoded NVFP4 scale')

    source_values = (
        source_fp4.reshape(*source_fp4.shape[:-1], logical_cols//32, 32)
        * source_scales[..., None]
    ).reshape(source_fp4.shape)
    candidate_values = (
        candidate_fp4.reshape(*candidate_fp4.shape[:-1], logical_cols//16, 16)
        * candidate_scales[..., None]
    ).reshape(candidate_fp4.shape)
    duplicated_source_scales = np.repeat(source_scales, 2, axis=-1)

    packed_equal = np.equal(source_packed, candidate_packed)
    return {
        'prefix': prefix,
        'source': {
            'weight_dtype': source_weight_meta['dtype'],
            'weight_shape': source_weight_meta['shape'],
            'scale_dtype': source_scale_meta['dtype'],
            'scale_shape': source_scale_meta['shape'],
            'group_size': 32,
        },
        'candidate': {
            'weight_dtype': candidate_weight_meta['dtype'],
            'weight_shape': candidate_weight_meta['shape'],
            'scale_dtype': candidate_scale_meta['dtype'],
            'scale_shape': candidate_scale_meta['shape'],
            'group_size': 16,
            'weight_scale_2': candidate_global,
            'input_scale': float(input_array),
        },
        'packed_nibbles': {
            'byte_count': int(source_packed.size),
            'equal_byte_count': int(packed_equal.sum()),
            'equal_byte_fraction': float(packed_equal.mean()),
        },
        'effective_scales': metrics(duplicated_source_scales, candidate_scales),
        'logical_values': metrics(source_values, candidate_values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--projections', default='w1,w2,w3')
    parser.add_argument('--deadline-seconds', type=int, default=120)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    projections = [value.strip() for value in args.projections.split(',') if value.strip()]
    if not projections or any(value not in {'w1', 'w2', 'w3'} for value in projections):
        parser.error('--projections must contain only w1,w2,w3')
    if args.deadline_seconds < 1 or args.deadline_seconds > 1800:
        parser.error('--deadline-seconds must be in [1, 1800]')

    def expired(*_):
        raise TimeoutError(f'{args.deadline_seconds}-second transcode audit deadline')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.deadline_seconds)
    started = time.monotonic()
    report = {
        'schema_version': 1,
        'kind': 'cpu_single_expert_mxfp4_to_nvfp4_value_audit',
        'gpu_inference_run': False,
        'full_checkpoint_equality_verified': False,
        'source': str(args.source_dir),
        'candidate': str(args.candidate_dir),
        'layer': args.layer,
        'expert': args.expert,
        'projections': [],
    }
    with args.output.open('x') as output:
        try:
            source = Checkpoint(args.source_dir)
            candidate = Checkpoint(args.candidate_dir)
            for projection in projections:
                prefix = f'layers.{args.layer}.ffn.experts.{args.expert}.{projection}'
                report['projections'].append(audit_projection(source, candidate, prefix))
            report['all_logical_values_exact'] = all(
                item['logical_values']['exact_equal_fraction'] == 1.0
                for item in report['projections']
            )
            report['all_effective_scales_exact'] = all(
                item['effective_scales']['exact_equal_fraction'] == 1.0
                for item in report['projections']
            )
            report['status'] = 'audit_complete_not_inference_validated'
        except Exception as exc:
            report['status'] = 'failed_audit'
            report['error'] = str(exc)
            raise
        finally:
            signal.alarm(0)
            report['elapsed_s'] = time.monotonic()-started
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
    print(json.dumps({
        'status': report['status'],
        'all_logical_values_exact': report['all_logical_values_exact'],
        'all_effective_scales_exact': report['all_effective_scales_exact'],
        'elapsed_s': report['elapsed_s'],
    }))


if __name__ == '__main__':
    main()
