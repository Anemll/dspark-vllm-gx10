#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage a pinned HF snapshot and verify every file; never removes source data."""
import argparse
import concurrent.futures
from contextlib import ExitStack
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


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


def load_verified_records(paths, manifest, destination):
    """Read local hash proofs before replacing a resumable download receipt."""
    records = {}
    for receipt_path in paths:
        previous = json.loads(Path(receipt_path).read_text())
        for key, expected in (("repo", manifest["repo"]), ("revision", manifest["revision"]),
                              ("destination", str(destination))):
            if previous.get(key) != expected:
                raise ValueError(f"verification receipt {key} mismatch: {receipt_path}")
        for name, record in previous.get("verified", {}).items():
            if (isinstance(record, dict) and isinstance(record.get("mtime_ns"), int)
                    and isinstance(record.get("verified_at"), (int, float))):
                records[name] = record
    return records


def reusable_verification(path, entry, record):
    if not record or any(record.get(key) != value for key, value in entry.items()):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    current = path.stat()
    return (current.st_size, current.st_mtime_ns) == (entry["size"], record["mtime_ns"])


def download_http(manifest, entry, path, deadline):
    """Dependency-free, bounded HTTP transport for ordinary checkpoint shards."""
    cache = path.parents[len(PurePosixPath(entry["path"]).parts) - 1] / ".cache" / "hf-direct"
    if cache.is_symlink():
        raise ValueError("HTTP cache must not be a symlink")
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / (hashlib.sha256(entry["path"].encode()).hexdigest() + ".part")
    if partial.is_symlink():
        raise ValueError("HTTP partial must not be a symlink")
    if entry["size"] == 0:
        partial.touch(exist_ok=True)
    url = ("https://huggingface.co/" + manifest["repo"] + "/resolve/" + manifest["revision"]
           + "/" + urllib.parse.quote(entry["path"], safe="/") + "?download=true")
    for attempt in range(1, 4):
        if time.time() >= deadline:
            raise TimeoutError("HTTP download deadline exceeded")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry["size"]:
            raise ValueError("HTTP partial exceeds expected file size")
        if offset == entry["size"]:
            break
        headers = {"Accept-Encoding": "identity", "Cache-Control": "no-cache"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=min(90, max(1, deadline-time.time()))) as response:
                status = response.status
                if status == 206:
                    expected = rf"bytes {offset}-([0-9]+)/{entry['size']}"
                    match = re.fullmatch(expected, response.headers.get("Content-Range", ""))
                    if not match or int(match.group(1)) != entry["size"] - 1:
                        raise ValueError("HTTP server returned a mismatched byte range")
                elif status != 200 or offset:
                    raise ValueError("HTTP server ignored resume range; partial preserved")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != entry["size"] - offset:
                    raise ValueError("HTTP response length differs from pinned manifest")
                last_report = 0
                with partial.open("ab") as target:
                    while True:
                        if time.time() >= deadline:
                            raise TimeoutError("HTTP download deadline exceeded")
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        if offset + len(chunk) > entry["size"]:
                            raise ValueError("HTTP payload exceeds pinned file size")
                        target.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report >= 20:
                            emit("download_progress", file=entry["path"], received=offset, total=entry["size"])
                            last_report = time.monotonic()
            if offset != entry["size"]:
                raise http.client.IncompleteRead(b"", entry["size"] - offset)
            break
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 3 or time.time() + 5 >= deadline:
                raise
            emit("http_retry", file=entry["path"], attempt=attempt,
                 error=type(exc).__name__, status=getattr(exc, "code", None))
            time.sleep(min(5 * attempt, max(0, deadline-time.time())))
    if path.exists() or path.is_symlink():
        raise ValueError("HTTP destination appeared during download; partial preserved")
    partial.replace(path)


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
    absolute_deadline = time.time() + args.timeout_hours * 3600
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
        prior_records = {}
        if args.command == "download":
            proof_paths = list(getattr(args, "verified_receipt", None) or [])
            if receipt_path.exists():
                proof_paths.append(receipt_path)
            prior_records = load_verified_records(proof_paths, manifest, dest)
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
                emit("downloading", file=entry["path"], bytes=entry["size"])
                if getattr(args, "transport", "hub") == "http":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    download_http(manifest, entry, path, absolute_deadline)
                else:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(manifest["repo"], filename=entry["path"], revision=manifest["revision"],
                                    local_dir=str(dest), token=False)
            previous = prior_records.get(entry["path"])
            if reusable_verification(path, entry, previous):
                verified = previous
                emit("reused_verification", file=entry["path"], bytes=entry["size"],
                     verified_at=previous["verified_at"])
            else:
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
        if name == "download":
            command.add_argument("--transport", choices=("hub", "http"), default="hub",
                                 help="http uses only Python's standard library; Hub transport is required for very large Xet-only files.")
            command.add_argument("--verified-receipt", action="append",
                                 help="Reuse local hash proofs for unchanged files; repeatable. "
                                      "The existing output receipt is reused automatically.")
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
