#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Launch settings adapted from sudoingX (Copyright 2026 sudoingX) and
# HeNryous (Copyright 2026 HeNryous); see docs/dsv41-1017-deployment.md.
# Renderer implementation and local client-contract guards: Anemll contributors.
"""Render, but never execute, one rank of the 1017 vision candidate.

This profile requires the HeNryous v24 image. It deliberately does not consume
the repository's older DSv4 Compose, upstream.lock, or overlay.
"""

import argparse
import ipaddress
import json
from pathlib import Path
import re
import shlex
import sys

IMAGE_REPOSITORY = "ghcr.io/henryous/deepseek-v41-1017-dgx-spark"
IMAGE_DIGEST = "sha256:a796b0515cf6245f95cd97e2f5d12d000e7d99ad475d53d4dc21a0bc9a68cc81"
IMAGE_ID = "sha256:709f3991d0d3966608cfa1e59cd03bd8d9796467b1c307ecfcbc01adb9e7291e"
LAUNCHER_REVISION = "cc59c3609e8e671418f39eb92aa16b0e4f770b77"
RECIPE_REVISION = "382071728ce69700b28e7a5ea197ab0a48b2232c"
MARKOV_SHA256 = "0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a"
ALIASES = ("DeepSeek-v4.1-Flash-EXL3", "gemini-3-flash-preview")
STATUS = "NOT cluster validated"

DEFAULTS = {
    "PORT": "8888", "MASTER_PORT": "25000", "MAX_MODEL_LEN": "240000",
    "MAX_NUM_SEQS": "4", "MAX_NUM_BATCHED_TOKENS": "4096",
    "LONG_PREFILL_TOKEN_THRESHOLD": "512", "KV_CACHE_MEMORY_BYTES": "3758096384",
    "GPU_MEMORY_UTILIZATION": "0.86", "IMAGE_LIMIT": "4",
    "SPECULATIVE_TOKENS": "5", "EXL3_RDMA_CPU": "14",
    "EXL3_CPU_PIN": "5-9,15-19", "OFFLOAD_IO_CPUS": "0-4,10-13",
}
REQUIRED = {
    "IMAGE", "MODEL_DIR", "CACHE_DIR", "MARKOV_FILE", "NODE_RANK",
    "MASTER_ADDR", "SELF_ADDR", "FABRIC_INTERFACE", "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX", "EXL3_RDMA_DEVICES",
}

# sudoingX v1.1 baseline plus the overrides in HeNryous' v24 example.
# The confidence cost table is borrowed, and must be measured on this cluster.
KERNEL_ENV = dict(line.split("=", 1) for line in """
VLLM_EXL3_MOE_PARALLEL_GROUPS=1
EXL3_MOE_COOP_WIDE=1
EXL3_INT8_GEMV=0
VLLM_DSPARK_DEPTH_POLICY=conf
VLLM_DSPARK_ROUND_MS=33.26,36.33,38.91,42.23,46.03,48.15
VLLM_DSPARK_DRAFT_MS=0
VLLM_DSPARK_MARKOV_CACHE=/assets/markov_top_tokens.json
EXL3_L2PF=attn,moe,next
EXL3_RDMA_AR=1
VLLM_SHORT_CTX_BOUNDS=512,1025
EXL3_ROUTER=1
EXL3_L2PF_HC=1
VLLM_DSPARK_LOCAL_ARGMAX=1
EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1
VLLM_DSV41_MHC_OVERLAP=1
VLLM_ENGRAM_DISK_SPIN_MS=1000
EXL3_MROW_GROUPED=1
VLLM_EXL3_MOE_TF_PROMPT=1
TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
VLLM_PREFILL_EVERY_N_STEPS=100000
VLLM_PREFILL_THROTTLE_MIN_PRIO=5
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
NCCL_PROTO=^LL128
NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_INDEXER_TP_SHARD_MIN_ROWS=2048
EXL3_RDMA_AG=1
EXL3_RDMA_AG_MAX_KB=3072
VLLM_EXL3_HEAD_MROW=1
EXL3_HEAD_MROW_MAX_ROWS=8
VLLM_EXL3_MROW_MAX_ROWS=32
EXL3_MROW32_VARIANT=ring
EXL3_MROW32_MIN_ROWS=9
VLLM_MOE_GATE_FIRST=1
VLLM_EXL3_MOE_SMOE_SHARED=1
VLLM_DSV41_MHC_FUSED_POST=2
VLLM_DSV41_MHC_OVERLAP_MAX_TOKENS=24
VLLM_DSV41_ADAPTIVE_K=1
VLLM_DSV41_AK_MARGIN=1.0
VLLM_DSV41_AK_MIN=2
VLLM_DSV41_AK_HI=1.6
VLLM_DSV41_AK_MAX_REQS=4
VLLM_DSV41_KV_PAIR=3
VLLM_DSV41_KV_SWA_FILL=1
VLLM_DSV41_INDEXER_DEFER=1
VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63
VLLM_DSV41_MARKOV_FUSED=1
VLLM_DSV41_VOR_FUSED=1
VLLM_DSV41_BG_DEPTH=0
VLLM_DSV41_BG_PAUSE_PRIO=10
VLLM_DSV41_FG_WAIT_THROTTLE=1
VLLM_DSV41_BG_LPT_SOLO=1
VLLM_DSV41_REPLAY_SKIP=1
VLLM_DSV41_ENGRAM_LOOKAHEAD=4096
VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4
VLLM_DSV41_DROP_MTP_BUFFER=1
VLLM_EXL3_TF_PROMPT_TRANSIENT=1
VLLM_EXL3_TF_PROMPT_ALIAS=1
VLLM_DSV41_KV_CRC=1
VLLM_DSV41_OFFLOAD_EXPANDABLE_OK=1
VLLM_DSV41_OFFLOAD_MULTINODE=1
VLLM_DSV41_OFFLOAD_LOAD_FRACTION=0.45
""".strip().splitlines())


def read_env(path):
    values = {}
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"line {number}: expected KEY=value")
        key, raw_value = line.split("=", 1)
        if key in values:
            raise ValueError(f"line {number}: duplicate {key}")
        parts = shlex.split(raw_value, comments=True)
        if len(parts) != 1:
            raise ValueError(f"line {number}: quote values containing spaces")
        values[key] = parts[0]
    return values


def validate(values):
    unknown = set(values) - REQUIRED - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported settings: {', '.join(sorted(unknown))}")
    missing = REQUIRED - set(values)
    if missing:
        raise ValueError(f"required settings: {', '.join(sorted(missing))}")
    config = DEFAULTS | values
    for key, value in config.items():
        if not value or "CHANGEME" in value or any(c in value for c in "\n\r\x00"):
            raise ValueError(f"{key} must have a concrete, single-line value")
    if not re.fullmatch(re.escape(IMAGE_REPOSITORY) + r"@sha256:[0-9a-f]{64}", config["IMAGE"]):
        raise ValueError("IMAGE must be the selected HeNryous v24 repository pinned by sha256 digest")
    if config["IMAGE"] != f"{IMAGE_REPOSITORY}@{IMAGE_DIGEST}":
        raise ValueError("IMAGE digest does not match the reviewed v24 runtime")
    for key in ("MODEL_DIR", "CACHE_DIR", "MARKOV_FILE"):
        path = Path(config[key])
        if not path.is_absolute() or len(path.parts) < 3 or any(c in config[key] for c in ",$") or ".." in path.parts:
            raise ValueError(f"{key} must be an explicit absolute path without variables or commas")
    if config["MODEL_DIR"] == config["CACHE_DIR"]:
        raise ValueError("MODEL_DIR and CACHE_DIR must be separate directories")
    for key in ("MASTER_ADDR", "SELF_ADDR"):
        address = ipaddress.IPv4Address(config[key])
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError(f"{key} must be a routable fabric IPv4 address")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", config["FABRIC_INTERFACE"]):
        raise ValueError("FABRIC_INTERFACE must be a single network interface")
    for key in ("NCCL_IB_HCA", "EXL3_RDMA_DEVICES"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*", config[key]):
            raise ValueError(f"{key} must list exact local HCA names")
    for key in ("EXL3_CPU_PIN", "OFFLOAD_IO_CPUS"):
        if not re.fullmatch(r"[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*", config[key]):
            raise ValueError(f"{key} must be a CPU list")
    bounds = {
        "NODE_RANK": (0, 1), "PORT": (1, 65535), "MASTER_PORT": (1, 65535),
        "MAX_MODEL_LEN": (32769, 1048576), "MAX_NUM_SEQS": (1, 4),
        "MAX_NUM_BATCHED_TOKENS": (512, 16384),
        "LONG_PREFILL_TOKEN_THRESHOLD": (1, 16384),
        "KV_CACHE_MEMORY_BYTES": (1, 128 * 1024**3), "IMAGE_LIMIT": (1, 999),
        "SPECULATIVE_TOKENS": (0, 5), "NCCL_IB_GID_INDEX": (0, 255),
        "EXL3_RDMA_CPU": (0, 255),
    }
    for key, (minimum, maximum) in bounds.items():
        try:
            value = int(config[key])
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be in [{minimum}, {maximum}]")
        config[key] = str(value)
    if config["SPECULATIVE_TOKENS"] not in ("0", "5"):
        raise ValueError("SPECULATIVE_TOKENS must be 0 (target-only) or 5 (DSpark)")
    if config["NODE_RANK"] == "0" and config["SELF_ADDR"] != config["MASTER_ADDR"]:
        raise ValueError("head SELF_ADDR must equal MASTER_ADDR")
    if config["NODE_RANK"] == "1" and config["SELF_ADDR"] == config["MASTER_ADDR"]:
        raise ValueError("worker SELF_ADDR must differ from MASTER_ADDR")
    if int(config["MAX_NUM_BATCHED_TOKENS"]) > int(config["MAX_MODEL_LEN"]):
        raise ValueError("batch token budget must not exceed context")
    if int(config["LONG_PREFILL_TOKEN_THRESHOLD"]) * max(1, int(config["MAX_NUM_SEQS"]) - 1) >= int(config["MAX_NUM_BATCHED_TOKENS"]):
        raise ValueError("long-prefill chunks must leave batch space for a foreground request")
    if not 0 < float(config["GPU_MEMORY_UTILIZATION"]) < 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be between 0 and 1")
    return config


def compact(value):
    return json.dumps(value, separators=(",", ":"))


def bind(source, target, readonly=False):
    return ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]


def render(values, stage="serve"):
    config = validate(values)
    rank = config["NODE_RANK"]
    cache_bind = bind(config["CACHE_DIR"], "/root/.cache")
    if stage == "jit":
        env = {"TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
               "MAX_JOBS": "4", "EXL3_SMOE_FLAGS": "-DSMOE_RUNFLAGS=1"}
        argv = ["docker", "run", "--rm", "--pull=never", "--platform", "linux/arm64",
                "--gpus", "all", *cache_bind]
        for key, value in sorted(env.items()):
            argv += ["--env", f"{key}={value}"]
        argv += ["--entrypoint", "python3", IMAGE_ID, "-c",
                 "from vllm_exl3_moe import tf_experts as t, mrow, rdma, mhc, router, smoe, l2pf; "
                 "t._ext(); mrow.ext(); rdma.ext(); mhc.ext(); router.ext(); smoe.ext(); l2pf.ext()"]
        checks = [["test", "-d", config["CACHE_DIR"]]]
    elif stage == "serve":
        env = KERNEL_ENV | {
            "NCCL_IB_HCA": config["NCCL_IB_HCA"], "NCCL_IB_GID_INDEX": config["NCCL_IB_GID_INDEX"],
            "NCCL_IB_DISABLE": "0", "NCCL_NET": "IB",
            "NCCL_SOCKET_IFNAME": config["FABRIC_INTERFACE"],
            "GLOO_SOCKET_IFNAME": config["FABRIC_INTERFACE"],
            "TP_SOCKET_IFNAME": config["FABRIC_INTERFACE"], "VLLM_HOST_IP": config["SELF_ADDR"],
            "EXL3_RDMA_DEVICES": config["EXL3_RDMA_DEVICES"], "EXL3_RDMA_CPU": config["EXL3_RDMA_CPU"],
            "EXL3_CPU_PIN": config["EXL3_CPU_PIN"],
            "VLLM_DSV41_OFFLOAD_IO_CPUS": config["OFFLOAD_IO_CPUS"],
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_DISABLED_KERNELS": "FlashInferCutlassMxfp8LinearKernel", "HF_HUB_OFFLINE": "1",
        }
        args = ["serve", "/model", "--served-model-name", *ALIASES,
                "--host", "0.0.0.0", "--port", config["PORT"],
                "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", rank,
                "--master-addr", config["MASTER_ADDR"], "--master-port", config["MASTER_PORT"],
                "--gpu-memory-utilization", config["GPU_MEMORY_UTILIZATION"],
                "--max-model-len", config["MAX_MODEL_LEN"], "--max-num-seqs", config["MAX_NUM_SEQS"],
                "--max-num-batched-tokens", config["MAX_NUM_BATCHED_TOKENS"], "--load-format", "auto",
                "--kv-cache-memory-bytes", config["KV_CACHE_MEMORY_BYTES"],
                "--engram-config", compact({"cpu_offload": True, "dp_shared_memory": False,
                                             "disk_offload": True, "disk_threads": 256}),
                "--compilation-config", compact({"cudagraph_mode": "FULL_DECODE_ONLY"}),
                "--kernel-config", compact({"enable_flashinfer_autotune": False}),
                "--enable-auto-tool-choice", "--tool-call-parser", "deepseek_v41",
                "--reasoning-parser", "deepseek_v41",
                "--limit-mm-per-prompt", compact({"image": int(config["IMAGE_LIMIT"])}),
                "--mm-processor-cache-gb", "0", "--long-prefill-token-threshold", config["LONG_PREFILL_TOKEN_THRESHOLD"],
                "--scheduling-policy", "priority", "--enable-prompt-tokens-details",
                "--kv-transfer-config", compact({"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                    "kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 268435456,
                        "secondary_tiers": [{"type": "fs", "root_dir": "/root/.cache/kv_offload",
                                             "n_read_threads": 8, "n_write_threads": 4}]}})]
        if config["SPECULATIVE_TOKENS"] == "5":
            args += ["--speculative-config", compact({"method": "dspark", "num_speculative_tokens": 5,
                        "num_speculative_tokens_per_batch_size": [[1, 4, 5]]})]
        if rank == "1":
            args += ["--headless"]
        argv = ["docker", "run", "-d", "--pull=never", "--platform", "linux/arm64",
                "--name", f"dsv41-1017-candidate-rank{rank}",
                "--network", "host", "--ipc", "host", "--shm-size", "64g", "--gpus", "all",
                "--device", "/dev/infiniband", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                "--restart", "no", *bind(config["MODEL_DIR"], "/model", True), *cache_bind,
                *bind(config["MARKOV_FILE"], "/assets/markov_top_tokens.json", True)]
        for key, value in sorted(env.items()):
            argv += ["--env", f"{key}={value}"]
        argv += ["--entrypoint", "vllm", IMAGE_ID, *args]
        checks = [["test", "-d", config["CACHE_DIR"]]]
        checks += [["test", "-s", str(Path(config["MODEL_DIR"]) / name)] for name in
                   ("config.json", "model.safetensors.index.json", "tokenizer.json",
                    "model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")]
        checks += [["python3", "-c", "import hashlib,pathlib,sys; "
                    "actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); "
                    "assert actual==sys.argv[2], 'Pinned DSpark Markov asset hash mismatch'",
                    config["MARKOV_FILE"], MARKOV_SHA256]]
    else:
        raise ValueError("stage must be jit or serve")
    # docker save/load retains the config ID, but need not retain RepoDigests.
    # Use the independently pinned arm64 config ID on both disconnected ranks.
    checks += [["docker", "image", "inspect", "--format", "{{.Id}} {{.Os}}/{{.Architecture}}", IMAGE_ID]]
    return {"status": STATUS, "stage": stage, "node_rank": int(rank),
            "source": {"recipe_revision": RECIPE_REVISION, "launcher_revision": LAUNCHER_REVISION,
                       "image_family": "HeNryous v24", "image_digest": IMAGE_DIGEST,
                       "image_id": IMAGE_ID, "markov_sha256": MARKOV_SHA256},
            "client_contract": {"served_model_names": list(ALIASES), "context": int(config["MAX_MODEL_LEN"]),
                                "max_output_tokens": 32768, "native_vision": True},
            "preflight_argv": checks, "docker_argv": argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", type=Path)
    parser.add_argument("--stage", choices=("jit", "serve"), default="serve")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    try:
        result = render(read_env(args.env_file), args.stage)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("#!/usr/bin/env bash\nset -euo pipefail")
        print(f"# {STATUS}. Execute only after cluster preflight; worker before head.")
        for command in result["preflight_argv"]:
            print(shlex.join(command))
        print("exec " + shlex.join(result["docker_argv"]))


if __name__ == "__main__":
    main()
