#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded, model-free GB10 canary for the DeepSeek V4 mHC warmup overlay.

Run only on an idle GPU with an explicit coordination handoff. No checkpoint,
model implementation, distributed group, attention, or MoE is instantiated.
The controller starts a fresh process with unique cache paths and kills only
that process group at the deadline. Existing cache directories are untouched.

Example inside a compatible GPU image, with both source files mounted read-only:
  python benchmark_mhc_warmup.py --baseline-source /inputs/baseline.py \
      --warmup-source /inputs/candidate.py --output-dir /results/mhc-canary-001

Timing includes validation/instrumentation and is NOT a throughput benchmark.
Cache writes and compiler hooks are observations, not proof that every native
JIT path was intercepted. The report never asserts zero JIT from timing alone.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72]
MAX_TOKENS = 2048
CACHE_NAMES = {
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang-tmp",
    "DG_JIT_CACHE_DIR": "deep_gemm",
    "VLLM_CACHE_ROOT": "vllm",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "CUDA_CACHE_PATH": "cuda",
    "XDG_CACHE_HOME": "xdg",
    "TMPDIR": "tmp",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_info(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def cache_snapshot(root: Path) -> dict:
    """Hash bounded cache artifacts, never follow links out of the new root."""
    rows = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink in isolated cache: {path}")
        if not path.is_file():
            continue
        if len(rows) >= 10000:
            raise RuntimeError("canary cache exceeded 10,000-file budget")
        stat = path.stat()
        total_bytes += stat.st_size
        if total_bytes > 1024**3:
            raise RuntimeError("canary cache exceeded 1 GiB budget")
        rows[str(path.relative_to(root))] = {
            "bytes": stat.st_size, "sha256": sha256(path),
        }
    return rows


def changed_cache_entries(previous: dict, current: dict) -> list[str]:
    return sorted(key for key in previous.keys() | current.keys()
                  if previous.get(key) != current.get(key))


def compiler_process_observation(event: str, arguments) -> dict | None:
    """Separate actual compiler process invocations from compiler discovery.

    Python profile entries such as compile()/cached() are deliberately not
    inputs: they may only look up an existing compiled kernel.
    """
    if event == "subprocess.Popen":
        command = arguments[1]
        display = str(arguments[:3])
    elif event == "os.system":
        command = arguments[0]
        display = str(command)
    else:
        return None
    if isinstance(command, bytes):
        command = os.fsdecode(command)
    argv = list(command) if isinstance(command, (list, tuple)) else shlex.split(command)
    argv = [os.fsdecode(value) for value in argv]
    if argv and Path(argv[0]).name in ("sh", "bash", "dash") and "-c" in argv:
        position = argv.index("-c") + 1
        if position < len(argv):
            argv = shlex.split(argv[position])
    compiler = re.compile(r"(?:.*-)?(?:nvcc|ptxas|gcc|g\+\+|cc|c\+\+|clang|clang\+\+)(?:-\d+)?$")
    positions = [index for index, value in enumerate(argv)
                 if compiler.fullmatch(Path(value).name)]
    if not positions:
        return None
    first = positions[0]
    options = argv[first + 1:]
    query = bool(argv and Path(argv[0]).name in ("which", "whereis", "command"))
    compile_flags = {"-c", "--cubin", "-cubin", "-shared", "--fatbin", "-fatbin",
                     "--ptx", "-ptx", "-o", "--output-file"}
    if not any(option in compile_flags for option in options):
        query = query or any(option in ("--version", "-V", "-dumpmachine", "-dumpversion")
                             or option.startswith("-print-") for option in options)
    return {"event": event, "command": display[:4000],
            "is_compiler_invocation": not query,
            "observation_kind": "compiler_query" if query else "compiler_invocation"}


def verify_post_warmup_probe(probe: dict) -> None:
    """A bounded no-new-compilation-observed gate, not universal zero-JIT proof."""
    probe["no_new_compilation_observed"] = False
    if not probe.get("finite", False):
        raise RuntimeError("non-finite 294-token normalized pre output")
    if probe.get("cache_changed"):
        raise RuntimeError("post-warmup 294-token probe changed the isolated cache")
    invocations = [row for row in probe.get("compiler_process_observations", [])
                   if row.get("is_compiler_invocation", True)]
    if invocations:
        raise RuntimeError("post-warmup 294-token probe invoked an observed compiler")
    libraries = probe.get("tilelang_binaries", [])
    if not libraries or not all(
        row.get("library_inside_isolated_root", False)
        and row.get("library_provenance_verified", False)
        and row.get("library", {}).get("sha256")
        for row in libraries
    ):
        raise RuntimeError("post-warmup probe lost verified library provenance")
    probe["no_new_compilation_observed"] = True


def prepare_environment(output: Path) -> dict[str, str]:
    env = dict(os.environ)
    root = output / "cache"
    root.mkdir()
    for name, suffix in CACHE_NAMES.items():
        directory = root / suffix
        directory.mkdir()
        env[name] = str(directory)
    # Never alter HOME: it may contain credentials and runtime configuration.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # These knobs are verified in installed TileLang 0.1.9's env.py.
    env["TILELANG_PRINT_ON_COMPILATION"] = "1"
    env["TILELANG_DISABLE_CACHE"] = "0"
    env["TILELANG_CLEAR_CACHE"] = "0"
    env["TILELANG_CLEANUP_TEMP_FILES"] = "0"
    return env


def installed_cache_source_audit() -> dict:
    """Read installed source without importing GPU packages or guessing a path."""
    result = {}
    for package in ("tilelang", "deep-gemm", "vllm"):
        try:
            dist = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = {"installed_distribution": False}
            continue
        rows = []
        for entry in dist.files or []:
            name = str(entry)
            if not name.endswith(".py") or not any(
                part in name for part in ("cache", "env", "deep_gemm")
            ):
                continue
            path = Path(dist.locate_file(entry))
            if not path.is_file() or path.stat().st_size > 2 * 1024**2:
                continue
            lines = path.read_text(errors="replace").splitlines()
            matches = [
                {"line": number, "text": line.strip()}
                for number, line in enumerate(lines, 1)
                if any(key in line for key in (
                    "TILELANG_CACHE_DIR", "DG_JIT_CACHE_DIR", "VLLM_CACHE_ROOT",
                ))
            ]
            if matches:
                rows.append({**source_info(path), "matches": matches})
        result[package] = {"version": dist.version, "source_matches": rows}
    return result


def loaded_binary_evidence(cache_root: Path) -> list[dict]:
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return [{"available": False, "reason": "/proc/self/maps unavailable"}]
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            name = fields[5]
            if any(part in name for part in (
                "tilelang", "deep_gemm", str(cache_root),
            )):
                paths.add(Path(name))
    rows = []
    for path in sorted(paths):
        try:
            if path.is_file():
                rows.append(source_info(path))
            else:
                rows.append({"path": str(path), "available": False})
        except OSError as exc:
            rows.append({"path": str(path), "available": False, "error": str(exc)})
    return rows


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load explicit warmup source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_live_tilelang_caches(cache_module) -> dict:
    """Use the backend instances actually referenced by TileLang's dispatcher.

    TileLang 0.1.9 constructs one singleton per backend subclass. Constructing
    the base KernelCache makes a separate, unused cache that remains empty even
    while TVMFFIKernelCache compiles and retains kernels.
    """
    dispatch = getattr(cache_module, "_dispatch_map", None)
    cached = getattr(cache_module, "cached", None)
    if not isinstance(dispatch, dict) or not dispatch:
        raise RuntimeError("installed TileLang has no auditable cache dispatcher")
    if getattr(cached, "__globals__", {}).get("_dispatch_map") is not dispatch:
        raise RuntimeError("TileLang cached() does not reference the inspected map")
    for backend, cache in dispatch.items():
        if (not isinstance(backend, str)
                or not isinstance(getattr(cache, "_memory_cache", None), dict)
                or not callable(getattr(cache, "_get_cache_root", None))):
            raise RuntimeError("unsupported TileLang backend cache contract")
    return dict(dispatch)


def observe_tilelang_executable_exports(caches: dict, root: Path) -> tuple[dict, list]:
    """Observe the actual TVMFFI live-executable export, without compiling again.

    Fresh TVMFFIKernelAdapter objects have no libpath. The installed cache
    subclass serializes adapter.executable to executable.so in its staging
    directory, then atomically renames that directory under the final key.
    Record object identity and the emitted bytes at that exact export call.
    """
    root = root.resolve()
    observations, restore = {}, []
    for backend, cache in caches.items():
        if backend != "tvm_ffi":
            continue
        original = cache._save_so_cubin_to_disk
        restore.append((cache, original))

        def observed(kernel, cache_path, verbose=False, *, _original=original,
                     _cache=cache):
            directory = Path(cache_path).resolve()
            if not directory.is_relative_to(root):
                raise RuntimeError("TileLang executable export escaped isolated root")
            adapter = kernel.adapter
            executable = getattr(adapter, "executable", None)
            if executable is None:
                raise RuntimeError("TVMFFI export has no live executable")
            _original(kernel, cache_path, verbose)
            path = directory / _cache.kernel_lib_path
            observations[id(kernel)] = {
                "kernel": kernel, "adapter": adapter, "executable": executable,
                "exported_sha256": sha256(path), "staging_path": str(path),
            }

        cache._save_so_cubin_to_disk = observed
    return observations, restore


def tilelang_binary_evidence(kernel_cache, root: Path, *, backend=None,
                            exports=None) -> list[dict]:
    """Correlate live JITKernel adapter libraries with exact generated files."""
    root = root.resolve()
    rows = []
    for key, kernel in sorted(kernel_cache._memory_cache.items()):
        adapter = getattr(kernel, "adapter", None)
        libpath = getattr(adapter, "libpath", None)
        row = {"cache_key": key, "adapter_type": type(adapter).__name__}
        if backend is not None:
            row["execution_backend"] = backend
        if libpath:
            path = Path(libpath).resolve()
            row["library"] = source_info(path)
            row["library_inside_isolated_root"] = path.is_relative_to(root)
            row["library_provenance"] = "adapter_libpath"
            row["library_provenance_verified"] = True
        elif backend == "tvm_ffi":
            observation = (exports or {}).get(id(kernel))
            if (observation is not None and observation["kernel"] is kernel
                    and observation["adapter"] is adapter
                    and observation["executable"] is getattr(adapter, "executable", None)):
                path = (Path(kernel_cache._get_cache_path(key))
                        / kernel_cache.kernel_lib_path).resolve()
                row["library"] = source_info(path)
                row["library_inside_isolated_root"] = path.is_relative_to(root)
                row["library_provenance"] = "observed_live_executable_export"
                row["observed_export_sha256"] = observation["exported_sha256"]
                row["library_provenance_verified"] = (
                    row["library"]["sha256"] == observation["exported_sha256"]
                )
        source = getattr(kernel, "kernel_source", None)
        if isinstance(source, str):
            row["generated_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        rows.append(row)
    return rows


def live_tilelang_binary_evidence(caches: dict, root: Path, exports=None) -> list[dict]:
    return [
        row
        for backend, cache in sorted(caches.items())
        for row in tilelang_binary_evidence(cache, root, backend=backend, exports=exports)
    ]


def build_synthetic_models(torch, seed: int):
    """The target and DSpark draft use full residual width on each TP rank."""
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    def parameter(shape, *, dtype=torch.float32, scale=0.1, ones=False):
        value = (
            torch.ones(shape, dtype=dtype, device="cuda:0") if ones else
            torch.randn(shape, dtype=dtype, device="cuda:0", generator=generator)
            * scale
        )
        return torch.nn.Parameter(value, requires_grad=False)

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = parameter((4096,), dtype=torch.bfloat16, ones=True)
            self.variance_epsilon = 1e-6

    class DeepseekV4DecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size, self.hc_mult = 4096, 4
            self.rms_norm_eps = self.hc_eps = 1e-6
            self.hc_post_alpha, self.hc_sinkhorn_iters = 2.0, 20
            self.attn_norm, self.ffn_norm = Norm(), Norm()
            for prefix in ("hc_attn", "hc_ffn"):
                setattr(self, prefix + "_fn", parameter((24, 16384), scale=1e-4))
                setattr(self, prefix + "_scale", parameter((3,)))
                setattr(self, prefix + "_base", parameter((24,)))

    class DeepseekV4Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="deepseek_v4", hidden_size=4096)
            self.hc_mult, self.rms_norm_eps, self.hc_eps = 4, 1e-6, 1e-6
            self.layer = DeepseekV4DecoderLayer()
            self.hc_head_fn = parameter((4, 16384), scale=1e-4)
            self.hc_head_scale = parameter((1,))
            self.hc_head_base = parameter((4,))

    class DSparkDeepseekV4Model(torch.nn.Module):
        def __init__(self, target):
            super().__init__()
            # Share the shape contract, not another model/parameter allocation.
            for key in ("config", "hc_mult", "rms_norm_eps", "hc_eps", "layer",
                        "hc_head_fn", "hc_head_scale", "hc_head_base"):
                setattr(self, key, getattr(target, key))

    target = DeepseekV4Model()
    return target, DSparkDeepseekV4Model(target), generator


def worker(args) -> int:
    output = args.output_dir
    cache_root = output / "cache"
    started = time.monotonic()
    report = {
        "schema_version": 1, "status": "running", "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources": {"baseline": source_info(args.baseline_source),
                    "candidate": source_info(args.warmup_source),
                    "canary": source_info(Path(__file__))},
        "environment": {key: os.environ.get(key) for key in (
            *CACHE_NAMES, "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
            "TORCH_CUDA_ARCH_LIST", "VLLM_USE_DEEP_GEMM", "DG_JIT_DEBUG",
            "TILELANG_PRINT_ON_COMPILATION", "TILELANG_DISABLE_CACHE",
            "TILELANG_CLEAR_CACHE", "TILELANG_CLEANUP_TEMP_FILES",
        )},
        "phases": [], "compiler_process_observations": [],
        "zero_jit_proven": False, "no_new_compilation_observed": False,
        "limitations": [
            "Synthetic one-layer target and draft contract, not checkpoint validation.",
            "No TP collective, model, MoE, attention, or CUDA graph was executed.",
            "Timings include finite-output checks and Python instrumentation.",
            "Compiler entry calls can be cache lookups, not actual compilation.",
            "Native compiler calls can bypass Python audit/profile hooks.",
            "Mapped host binaries do not prove which device cubin was launched.",
            "Fresh requested paths alone do not prove every cache honored them.",
            "No-new-compilation-observed applies only to the two 294-token probes.",
        ],
    }
    phase = "imports"
    compile_entries = Counter()
    export_restore = []

    def save():
        report["elapsed_s"] = time.monotonic() - started
        write_json(output / "report.json", report)

    def event(name, **data):
        row = {"phase": name, "elapsed_s": time.monotonic() - started, **data}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        save()

    def audit(name, arguments):
        observation = compiler_process_observation(name, arguments)
        if observation is not None:
            report["compiler_process_observations"].append({"phase": phase, **observation})

    def profile(frame, event_name, arg):
        if event_name != "call":
            return
        code = frame.f_code
        if ("tilelang" in code.co_filename or "deep_gemm" in code.co_filename) and (
            "compil" in code.co_name or "build" in code.co_name
        ):
            compile_entries[(phase, code.co_filename, code.co_firstlineno,
                             code.co_name)] += 1

    def snapshot(name):
        rows = cache_snapshot(cache_root)
        write_json(output / f"cache-{name}.json", rows)
        return rows

    sys.addaudithook(audit)
    try:
        report["installed_cache_source_audit"] = installed_cache_source_audit()
        report["cache_before_imports"] = snapshot("before-imports")
        save()
        import torch
        import tilelang
        tilelang_cache = importlib.import_module("tilelang.cache")
        from vllm.model_executor.kernels.mhc import tilelang as mhc
        from vllm.model_executor.kernels.mhc import tilelang_kernels
        from vllm.utils import deep_gemm

        if torch.cuda.device_count() != 1:
            raise RuntimeError("canary requires exactly one visible GPU")
        if torch.cuda.get_device_capability(0) != (12, 1):
            raise RuntimeError("canary is scoped to GB10 SM121")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {"name": props.name, "sm": "12.1",
                         "num_sms": props.multi_processor_count,
                         "free_bytes_before": free, "total_bytes": total}
        if free < 1024**3:
            raise RuntimeError("canary needs at least 1 GiB free GPU memory")
        if not deep_gemm.is_deep_gemm_supported():
            raise RuntimeError("reported GB10 DeepGEMM path is not enabled")
        deep_gemm._lazy_init()
        dg = deep_gemm._import_deep_gemm()
        kernel_caches = select_live_tilelang_caches(tilelang_cache)
        export_observations, export_restore = observe_tilelang_executable_exports(
            kernel_caches, cache_root
        )
        tilelang_env = tilelang.env
        report["tilelang_isolation"] = {
            "cache_roots": {name: str(cache._get_cache_root())
                            for name, cache in kernel_caches.items()},
            "cache_types": {name: type(cache).__name__
                            for name, cache in kernel_caches.items()},
            "temporary_root": tilelang_env.TILELANG_TMP_DIR,
            "cache_enabled": tilelang_env.is_cache_enabled(),
            "initial_in_memory_entries": {
                name: len(cache._memory_cache) for name, cache in kernel_caches.items()
            },
        }
        if not (
            all(Path(cache._get_cache_root()).resolve().is_relative_to(cache_root)
                for cache in kernel_caches.values())
            and Path(tilelang_env.TILELANG_TMP_DIR).resolve().is_relative_to(cache_root)
            and tilelang_env.is_cache_enabled()
            and all(not cache._memory_cache for cache in kernel_caches.values())
        ):
            raise RuntimeError("TileLang did not honor fresh isolated cache contract")
        report["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "tilelang": getattr(tilelang, "__version__", None)}
        for name, module in (("tilelang", tilelang), ("mhc", mhc),
                             ("tilelang_cache_dispatch", tilelang_cache),
                             ("mhc_kernels", tilelang_kernels),
                             ("deep_gemm_wrapper", deep_gemm), ("deep_gemm", dg)):
            path = getattr(module, "__file__", None)
            if path:
                report["sources"][name] = source_info(Path(path))
        report["cache_runtime_values"] = {}
        for name, module in list(sys.modules.items()):
            if name.startswith(("tilelang", "deep_gemm", "vllm.third_party.deep_gemm")):
                values = {}
                for key in ("TILELANG_CACHE_DIR", "DG_JIT_CACHE_DIR", "CACHE_DIR"):
                    value = vars(module).get(key) if module is not None else None
                    if isinstance(value, (str, Path)):
                        values[key] = str(value)
                if values:
                    report["cache_runtime_values"][name] = values
        baseline = load_source(args.baseline_source, "baseline_mhc_canary")
        candidate = load_source(args.warmup_source, "candidate_mhc_canary")
        target, draft, generator = build_synthetic_models(torch, args.seed)
        original = {name: value.detach().cpu().clone()
                    for name, value in target.named_parameters()}
        report["synthetic_parameters"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in target.named_parameters()
        }
        report["detection"] = {
            "baseline_target": baseline._find_first_mhc_layer(target) is not None,
            "candidate_target": candidate._find_first_mhc_layer(target) is target.layer,
            "candidate_draft": candidate._find_first_mhc_layer(draft) is target.layer,
            "candidate_draft_head": candidate._find_deepseek_v4_model(draft) is draft,
        }
        if report["detection"]["baseline_target"] or not all(
            report["detection"][key] for key in (
                "candidate_target", "candidate_draft", "candidate_draft_head",
            )
        ):
            raise RuntimeError("warmup detection does not reproduce the intended A/B")
        allocated = torch.cuda.memory_allocated()
        baseline.deepseek_v4_mhc_warmup(
            target, max_tokens=MAX_TOKENS, cudagraph_capture_sizes=CAPTURE_SIZES
        )
        report["baseline_noop_allocation_delta"] = torch.cuda.memory_allocated() - allocated
        if report["baseline_noop_allocation_delta"] != 0:
            raise RuntimeError("baseline warmup unexpectedly allocated GPU memory")
        report["token_sizes"] = candidate._select_mhc_warmup_token_sizes(
            max_tokens=MAX_TOKENS, cudagraph_capture_sizes=CAPTURE_SIZES,
            hidden_size=4096, hc_mult=4, num_sms=props.multi_processor_count,
        )
        checks, calls, originals = [], [], {}
        for name in ("mhc_pre_tilelang", "mhc_fused_post_pre_tilelang",
                     "mhc_post_tilelang", "hc_head_fused_kernel_tilelang"):
            originals[name] = getattr(mhc, name)

            def checked(*positional, _name=name, **kwargs):
                result = originals[_name](*positional, **kwargs)
                outputs = result if isinstance(result, tuple) else (result,)
                calls.append({"name": _name, "input_shape": list(positional[0].shape),
                              "norm": kwargs.get("norm_weight") is not None})
                checks.extend(torch.isfinite(value).all() for value in outputs)
                return result

            setattr(mhc, name, checked)

        previous = snapshot("before-warmup")
        sys.setprofile(profile)
        phase = "candidate-warmup"
        event(phase, status="starting", token_sizes=report["token_sizes"])
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        candidate.deepseek_v4_mhc_warmup(
            target, max_tokens=MAX_TOKENS, cudagraph_capture_sizes=CAPTURE_SIZES
        )
        torch.cuda.synchronize()
        report["phases"].append({"name": phase, "wall_s": time.monotonic() - begin})
        report["warmup_finite"] = bool(torch.stack(checks).all().item())
        report["warmup_calls"] = calls
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        for name, original_function in originals.items():
            setattr(mhc, name, original_function)
        if not report["warmup_finite"]:
            raise RuntimeError("non-finite synthetic warmup output")
        current = snapshot("after-warmup")
        report["warmup_cache_changed"] = changed_cache_entries(previous, current)
        report["warmup_tilelang_binaries"] = live_tilelang_binary_evidence(
            kernel_caches, cache_root, export_observations
        )
        if not report["warmup_tilelang_binaries"] or not all(
            row.get("library_inside_isolated_root", False)
            and row.get("library_provenance_verified", False)
            for row in report["warmup_tilelang_binaries"]
        ):
            raise RuntimeError("cannot correlate warmup with isolated adapter libraries")
        previous = current
        residual = torch.randn((294, 4, 4096), dtype=torch.bfloat16,
                               device="cuda:0", generator=generator)
        layer = target.layer
        probe_results = []
        for trial in range(2):
            phase = f"post-warmup-294-{trial + 1}"
            event(phase, status="starting")
            begin = time.monotonic()
            result = mhc.mhc_pre_tilelang(
                residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                layer.rms_norm_eps, layer.hc_eps, layer.hc_eps,
                layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                norm_weight=layer.attn_norm.weight.data,
                norm_eps=layer.attn_norm.variance_epsilon,
            )
            torch.cuda.synchronize()
            duration = time.monotonic() - begin
            finite = all(bool(torch.isfinite(value).all().item()) for value in result)
            current = snapshot(f"after-probe-{trial + 1}")
            probe = {
                "name": phase, "wall_s": duration, "finite": finite,
                "cache_changed": changed_cache_entries(previous, current),
                "compiler_process_observations": [
                    row for row in report["compiler_process_observations"]
                    if row["phase"] == phase
                ],
                "tilelang_binaries": live_tilelang_binary_evidence(
                    kernel_caches, cache_root, export_observations
                ),
            }
            report["phases"].append(probe)
            verify_post_warmup_probe(probe)
            probe_results.append(probe)
            previous = current
        report["weights_unchanged"] = all(
            torch.equal(value.detach().cpu(), original[name])
            for name, value in target.named_parameters()
        )
        if not report["weights_unchanged"]:
            raise RuntimeError("synthetic parameters changed during canary")
        report["gpu"]["free_bytes_after"] = torch.cuda.mem_get_info()[0]
        report["no_new_compilation_observed"] = (
            len(probe_results) == 2
            and all(probe["no_new_compilation_observed"] for probe in probe_results)
        )
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        sys.setprofile(None)
        for cache, original_export in export_restore:
            cache._save_so_cubin_to_disk = original_export
        report["compile_entry_observations"] = [
            {"phase": item[0], "source": item[1], "line": item[2],
             "function": item[3], "calls": count}
            for item, count in sorted(compile_entries.items())
        ]
        report["loaded_host_binaries"] = loaded_binary_evidence(cache_root)
        save()
        event("complete", status=report["status"])
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--warmup-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be 1..600 seconds")
    for name in ("baseline_source", "warmup_source"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} must name an existing Python source file")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        return worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = prepare_environment(args.output_dir)
    command = [sys.executable, str(Path(__file__).resolve()),
               "--baseline-source", str(args.baseline_source),
               "--warmup-source", str(args.warmup_source),
               "--output-dir", str(args.output_dir), "--timeout", str(args.timeout),
               "--seed", str(args.seed), "--worker"]
    start = time.monotonic()
    with (args.output_dir / "worker.log").open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            while process.poll() is None:
                remaining = args.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    print(f"mHC canary active: {time.monotonic() - start:.0f}s; "
                          f"evidence: {args.output_dir}", flush=True)
        finally:
            # Also stop our detached compiler children if the caller interrupts.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
    write_json(args.output_dir / "controller.json", {
        "elapsed_s": time.monotonic() - start, "timeout_s": args.timeout,
        "timed_out": timed_out, "worker_exit_code": process.returncode,
        "process_group": process.pid,
    })
    return 124 if timed_out else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
