#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage a pinned HF snapshot and verify every file; never removes source data."""
import argparse
import concurrent.futures
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import time


def emit(event, **fields):
    sys.stdout.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")
    sys.stdout.flush()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate_manifest(manifest):
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("revision must be a full immutable commit")
    seen = set()
    if not manifest["files"]:
        raise ValueError("empty snapshot")
    for entry in manifest["files"]:
        path = PurePosixPath(entry["path"])
        if (not path.parts or entry["path"] != path.as_posix()
                or path.is_absolute() or ".." in path.parts or "\\" in entry["path"]
                or entry["path"] in seen or ".cache" in path.parts or ".staging-locks" in path.parts):
            raise ValueError("unsafe or duplicate snapshot path")
        seen.add(entry["path"])
        digest_length = 64 if entry["hash_type"] == "sha256" else 40
        if entry["hash_type"] not in ("sha256", "git-sha1"):
            raise ValueError("unknown hash type")
        if not re.fullmatch(r"[0-9a-f]{%d}" % digest_length, entry["hash"]):
            raise ValueError("invalid expected digest")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError("invalid file size")


def verify_file(path, entry):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked file: {path}")
    before = path.stat()
    if before.st_size != entry["size"]:
        raise ValueError(f"size mismatch: {path}")
    digest = hashlib.sha256() if entry["hash_type"] == "sha256" else hashlib.sha1()
    if entry["hash_type"] == "git-sha1":
        digest.update(f"blob {before.st_size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed during verification: {path}")
    if digest.hexdigest() != entry["hash"]:
        raise ValueError(f"content hash mismatch: {path}; preserved for inspection")
    return {**entry, "mtime_ns": after.st_mtime_ns, "verified_at": time.time()}


def plan(args):
    from huggingface_hub import HfApi
    info = HfApi(token=False).model_info(args.repo, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Hub returned a different revision than requested")
    entries = []
    for item in info.siblings:
        if args.file and item.rfilename not in args.file:
            continue
        entries.append({"path": item.rfilename, "size": item.size,
                        "hash_type": "sha256" if item.lfs else "git-sha1",
                        "hash": item.lfs.sha256 if item.lfs else item.blob_id})
    if args.file and set(args.file) != {entry["path"] for entry in entries}:
        raise ValueError("requested files absent from repository")
    manifest = {"schema_version": 1, "repo": args.repo, "revision": info.sha,
                "files": entries, "total_bytes": sum(entry["size"] for entry in entries)}
    validate_manifest(manifest)
    save_json(args.manifest, manifest)
    emit("planned", repo=args.repo, revision=info.sha, files=len(entries), bytes=manifest["total_bytes"])


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    dest = Path(args.destination).absolute()
    required_mount = Path(args.require_mount).absolute() if args.require_mount else None
    if required_mount and (not os.path.ismount(required_mount) or not dest.is_relative_to(required_mount)):
        raise ValueError("required destination volume is not mounted")
    if dest.is_symlink():
        raise ValueError("destination must not be a symlink")
    if required_mount and not dest.resolve().is_relative_to(required_mount.resolve()):
        raise ValueError("destination resolves outside required volume")
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / ".cache"
    for cache_entry in (cache, cache / "huggingface", cache / "huggingface" / "download"):
        if cache_entry.is_symlink():
            raise ValueError("download cache must not be a symlink")
    receipt_path = Path(args.receipt)
    receipt = {"repo": manifest["repo"], "revision": manifest["revision"],
               "destination": str(dest), "state": "running", "started_at": time.time(), "verified": {}}
    state_lock = threading.Lock()
    done = threading.Event()
    # Independent per-source locks permit the disjoint EXL3 + Engram downloads.
    lock_path = receipt_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with ExitStack() as locks:
        lock_file = locks.enter_context(lock_path.open("a+"))
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        # Lock each destination filename, even if another job uses a different
        # receipt. Disjoint EXL3 and Engram snapshots may share a directory.
        lock_dir = dest / ".staging-locks"
        if lock_dir.is_symlink():
            raise ValueError("lock directory must not be a symlink")
        lock_dir.mkdir(exist_ok=True)
        for entry in manifest["files"]:
            if not (dest / entry["path"]).resolve().is_relative_to(dest.resolve()):
                raise ValueError("snapshot path resolves outside destination")
            filename = hashlib.sha256(entry["path"].encode()).hexdigest() + ".lock"
            file_lock_path = lock_dir / filename
            if file_lock_path.is_symlink():
                raise ValueError("lock file must not be a symlink")
            file_lock = locks.enter_context(file_lock_path.open("a+"))
            fcntl.flock(file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(receipt_path, receipt)
        started = time.monotonic()

        def guard():
            while not done.wait(30):
                reason = None
                try:
                    if required_mount and not os.path.ismount(required_mount):
                        reason = "destination volume unmounted"
                    elif time.monotonic() - started > args.timeout_hours * 3600:
                        reason = "wall-clock deadline exceeded; resume from partial files"
                    elif shutil.disk_usage(dest).free < args.reserve_gib * 1024**3:
                        reason = "free storage below reserve"
                except OSError as exc:
                    reason = "destination storage check failed: " + str(exc)
                with state_lock:
                    if reason:
                        receipt.update(state="stopped", error=reason)
                        save_json(receipt_path, receipt)
                        emit("stopped", reason=reason)
                        os._exit(2)
                    emit("heartbeat", verified_files=len(receipt["verified"]),
                         total_files=len(manifest["files"]), elapsed_s=round(time.monotonic() - started))

        threading.Thread(target=guard, daemon=True).start()

        def stage(entry):
            path = dest / entry["path"]
            if path.is_symlink():
                raise ValueError("snapshot file must not be a symlink")
            if not path.resolve().is_relative_to(dest.resolve()):
                raise ValueError("snapshot path resolves outside destination")
            if not path.exists() and args.command == "download":
                from huggingface_hub import hf_hub_download
                emit("downloading", file=entry["path"], bytes=entry["size"])
                hf_hub_download(manifest["repo"], filename=entry["path"], revision=manifest["revision"],
                                local_dir=str(dest), token=False)
            emit("verifying", file=entry["path"])
            verified = verify_file(path, entry)
            with state_lock:
                receipt["verified"][entry["path"]] = verified
                save_json(receipt_path, receipt)
            emit("verified", file=entry["path"], bytes=entry["size"])

        try:
            if shutil.disk_usage(dest).free < args.reserve_gib * 1024**3:
                raise ValueError("insufficient storage reserve")
            # Keep downloads and hashing bounded on network storage.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
            entries = iter(manifest["files"])
            pending = {pool.submit(stage, entry) for entry in list(manifest["files"])[:args.workers]}
            for _ in range(min(args.workers, len(manifest["files"]))):
                next(entries)
            while pending:
                completed, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    future.result()
                for _ in completed:
                    entry = next(entries, None)
                    if entry is not None:
                        pending.add(pool.submit(stage, entry))
            pool.shutdown(wait=True)
            receipt.update(state="verified", finished_at=time.time())
            save_json(receipt_path, receipt)
            emit("complete", files=len(manifest["files"]), bytes=manifest["total_bytes"])
        except Exception as exc:
            with state_lock:
                receipt.update(state="failed", error=str(exc), finished_at=time.time())
                save_json(receipt_path, receipt)
            emit("failed", error=str(exc))
            # A sibling may be streaming a 100GB shard. End this job promptly;
            # HF partial files remain resumable. No further files are scheduled.
            os._exit(1)
        finally:
            done.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--repo", required=True)
    planner.add_argument("--revision", required=True)
    planner.add_argument("--file", action="append")
    planner.add_argument("--manifest", required=True)
    for name in ("download", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--destination", required=True)
        command.add_argument("--receipt", required=True)
        command.add_argument("--require-mount")
        command.add_argument("--workers", type=int, choices=range(1, 5), default=2)
        command.add_argument("--reserve-gib", type=int, default=100)
        command.add_argument("--timeout-hours", type=float, default=12)
    args = parser.parse_args()
    if args.command == "plan":
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            parser.error("pin a full revision before planning")
        plan(args)
    else:
        if args.timeout_hours <= 0 or args.reserve_gib < 0:
            parser.error("invalid deadline or reserve")
        run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("failed", error=str(error))
        sys.exit(1)
