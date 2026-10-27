#!/usr/bin/env python3
"""Offline integrity and safety checks for checkpoint staging."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stage-hf-checkpoint.py"
SPEC = importlib.util.spec_from_file_location("stage_hf_checkpoint", SCRIPT)
stager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stager)
REVISION = "a" * 40


def entry(name="weights.safetensors", data=b"small test weights", algorithm="sha256"):
    if algorithm == "sha256":
        digest = hashlib.sha256(data).hexdigest()
    else:
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return {"path": name, "size": len(data), "hash_type": algorithm, "hash": digest}


def manifest(*entries):
    entries = entries or (entry(),)
    return {"schema_version": 1, "repo": "example/checkpoint", "revision": REVISION,
            "files": list(entries), "total_bytes": sum(item["size"] for item in entries)}


class CheckpointStagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "snapshot"
        self.manifest_path = self.root / "manifest.json"
        self.receipt_path = self.root / "receipt.json"
        self.args = types.SimpleNamespace(
            command="verify", manifest=str(self.manifest_path),
            destination=str(self.destination), receipt=str(self.receipt_path),
            require_mount=None, reserve_gib=0, timeout_hours=1, workers=1)

    def write_manifest(self, value=None):
        self.manifest_path.write_text(json.dumps(value or manifest()))

    def run_stager(self):
        # Production failures terminate streaming siblings immediately. Capture
        # that exit in unit tests so the failed receipt remains inspectable.
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(stager.os, "_exit", side_effect=SystemExit):
            try:
                stager.run(self.args)
            except SystemExit as exc:
                error = json.loads(self.receipt_path.read_text()).get("error", str(exc))
                raise ValueError(error) from exc

    def fake_hub(self, downloader):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = downloader
        return patch.dict("sys.modules", {"huggingface_hub": hub})

    def test_revision_requires_full_immutable_sha(self):
        for bad in ("main", "latest", "a" * 39, "a" * 41, "g" * 40):
            with self.subTest(revision=bad):
                value = manifest()
                value["revision"] = bad
                with self.assertRaises(ValueError):
                    stager.validate_manifest(value)
        stager.validate_manifest(manifest())

    def http_response(self, data, status=200, headers=None):
        response = io.BytesIO(data)
        response.status = status
        response.headers = headers if headers is not None else {"Content-Length": str(len(data))}
        return response

    def test_direct_http_uses_pinned_revision_and_verifies_content(self):
        data = b"model bytes"
        self.destination.mkdir()
        target = self.destination / "nested" / "weights.bin"
        target.parent.mkdir()
        expected = entry("nested/weights.bin", data)
        with patch.object(stager.urllib.request, "urlopen", return_value=self.http_response(data)) as fetch:
            with contextlib.redirect_stdout(io.StringIO()):
                stager.download_http(manifest(expected), expected, target, time.time()+60)
        self.assertIn('/resolve/'+REVISION+'/nested/weights.bin', fetch.call_args.args[0].full_url)
        self.assertEqual(stager.verify_file(target, expected)["hash"], expected["hash"])

    def test_direct_http_resumes_exact_range_after_short_body(self):
        data = b"abcdef"
        self.destination.mkdir()
        target = self.destination / "weights.bin"
        expected = entry("weights.bin", data)
        first = self.http_response(b"abc", headers={"Content-Length": "6"})
        second = self.http_response(b"def", status=206,
                                    headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"})
        with patch.object(stager.urllib.request, "urlopen", side_effect=[first, second]) as fetch, \
                patch.object(stager.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            stager.download_http(manifest(expected), expected, target, time.time()+60)
        self.assertEqual(fetch.call_args_list[1].args[0].get_header("Range"), "bytes=3-")
        self.assertEqual(target.read_bytes(), data)

    def test_direct_http_rejects_ignored_range_and_keeps_partial(self):
        self.destination.mkdir()
        expected = entry("weights.bin", b"abcdef")
        target = self.destination / "weights.bin"
        cache = self.destination / ".cache/hf-direct"
        cache.mkdir(parents=True)
        partial = cache/(hashlib.sha256(expected['path'].encode()).hexdigest()+'.part')
        partial.write_bytes(b"abc")
        with patch.object(stager.urllib.request, "urlopen", return_value=self.http_response(b"abcdef")):
            with self.assertRaisesRegex(ValueError, "ignored resume range"):
                stager.download_http(manifest(expected), expected, target, time.time()+60)
        self.assertEqual(partial.read_bytes(), b"abc")
        self.assertFalse(target.exists())

    def test_direct_http_rejects_wrong_content_range(self):
        self.destination.mkdir()
        expected = entry("weights.bin", b"abcdef")
        target = self.destination / "weights.bin"
        response = self.http_response(b"abcdef", 206, {"Content-Range": "bytes 1-5/6"})
        with patch.object(stager.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "mismatched byte range"):
                stager.download_http(manifest(expected), expected, target, time.time()+60)
        self.assertFalse(target.exists())

    def test_direct_http_rejects_extra_bytes(self):
        self.destination.mkdir()
        expected = entry("weights.bin", b"abc")
        target = self.destination / "weights.bin"
        with patch.object(stager.urllib.request, "urlopen", return_value=self.http_response(b"abcdef", headers={})):
            with self.assertRaisesRegex(ValueError, "exceeds pinned file size"):
                stager.download_http(manifest(expected), expected, target, time.time()+60)
        self.assertFalse(target.exists())

    def test_direct_http_empty_file_and_expired_deadline(self):
        self.destination.mkdir()
        expected = entry("empty", b"")
        with contextlib.redirect_stdout(io.StringIO()):
            stager.download_http(manifest(expected), expected, self.destination/'empty', time.time()+60)
        self.assertEqual((self.destination/'empty').read_bytes(), b"")
        with self.assertRaises(TimeoutError):
            stager.download_http(manifest(), entry(), self.destination/'weights.safetensors', time.time()-1)

    def test_http_transport_runs_without_huggingface_dependency(self):
        self.write_manifest(manifest(entry(data=b"weights")))
        self.args.command = "download"
        self.args.transport = "http"
        with patch.object(stager.urllib.request, "urlopen", return_value=self.http_response(b"weights")), \
                patch.dict('sys.modules', {'huggingface_hub': None}):
            self.run_stager()
        self.assertEqual(json.loads(self.receipt_path.read_text())['state'], 'verified')

    def test_rejects_unsafe_manifest_paths(self):
        for name in ("", ".", "/outside", "../outside", "a/../outside",
                     "a\\outside", ".cache/metadata", "a/.cache/metadata",
                     ".staging-locks/lock"):
            with self.subTest(path=name), self.assertRaises(ValueError):
                stager.validate_manifest(manifest(entry(name)))

    def test_rejects_duplicate_paths_including_normalized_aliases(self):
        for first, second in (("a.bin", "a.bin"), ("nested/a.bin", "nested//a.bin"),
                              ("a.bin", "./a.bin")):
            with self.subTest(paths=(first, second)), self.assertRaises(ValueError):
                stager.validate_manifest(manifest(entry(first), entry(second)))

    def test_rejects_bad_hashes_sizes_and_algorithms(self):
        for fields in ({"hash": "bad"}, {"hash_type": "md5"}, {"size": -1},
                       {"size": "10"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                value = manifest()
                value["files"][0].update(fields)
                stager.validate_manifest(value)

    def test_sha256_and_git_blob_sha1_match_known_bytes(self):
        data = b"hello\n"
        target = self.root / "payload"
        target.write_bytes(data)
        for algorithm in ("sha256", "git-sha1"):
            with self.subTest(algorithm=algorithm):
                result = stager.verify_file(target, entry("payload", data, algorithm))
                self.assertIn("verified_at", result)
                self.assertEqual(result["size"], len(data))
        self.assertEqual(entry("payload", data, "git-sha1")["hash"],
                         "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_mismatch_preserves_file_and_fails_receipt(self):
        self.write_manifest(manifest(entry(data=b"expected")))
        self.destination.mkdir()
        target = self.destination / "weights.safetensors"
        target.write_bytes(b"tampered")
        self.args.command = "download"
        calls = []
        with self.fake_hub(lambda *a, **kw: calls.append((a, kw))):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_stager()
        self.assertEqual(target.read_bytes(), b"tampered")
        self.assertEqual(calls, [])
        receipt = json.loads(self.receipt_path.read_text())
        self.assertEqual(receipt["state"], "failed")
        self.assertEqual(receipt["verified"], {})

    def test_success_receipt_requires_all_verified_files(self):
        data = b"tiny"
        self.write_manifest(manifest(entry("a", data), entry("b", data, "git-sha1")))
        self.destination.mkdir()
        (self.destination / "a").write_bytes(data)
        (self.destination / "b").write_bytes(data)
        self.run_stager()
        receipt = json.loads(self.receipt_path.read_text())
        self.assertEqual(receipt["state"], "verified")
        self.assertEqual(set(receipt["verified"]), {"a", "b"})

    def test_missing_file_is_failure_not_completion(self):
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.run_stager()
        receipt = json.loads(self.receipt_path.read_text())
        self.assertEqual(receipt["state"], "failed")
        self.assertEqual(receipt["verified"], {})

    def test_resumed_download_skips_verified_existing_file_and_pins_requests(self):
        data = b"tiny"
        self.write_manifest(manifest(entry("a", data), entry("b", data)))
        self.destination.mkdir()
        (self.destination / "a").write_bytes(data)
        self.args.command = "download"
        calls = []

        def download(repo, **kwargs):
            calls.append((repo, kwargs))
            (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(data)

        with self.fake_hub(download):
            self.run_stager()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "example/checkpoint")
        self.assertEqual(calls[0][1]["filename"], "b")
        self.assertEqual(calls[0][1]["revision"], REVISION)
        self.assertFalse(calls[0][1]["token"])
        self.assertEqual(json.loads(self.receipt_path.read_text())["state"], "verified")

    def test_required_mount_missing_rejects_before_destination_creation(self):
        self.write_manifest()
        self.args.require_mount = str(self.root)
        with patch.object(stager.os.path, "ismount", return_value=False):
            with self.assertRaisesRegex(ValueError, "mounted"):
                self.run_stager()
        self.assertFalse(self.destination.exists())

    def prepare_verified_download(self):
        self.write_manifest()
        self.destination.mkdir()
        target = self.destination / "weights.safetensors"
        target.write_bytes(b"small test weights")
        self.run_stager()
        proof = json.loads(self.receipt_path.read_text())
        self.args.command = "download"
        return target, proof

    def test_download_reuses_existing_receipt_without_rehash(self):
        _, proof = self.prepare_verified_download()
        with patch.object(stager, "verify_file", side_effect=AssertionError("unnecessary rehash")):
            self.run_stager()
        result = json.loads(self.receipt_path.read_text())
        self.assertEqual(result["state"], "verified")
        self.assertEqual(result["verified"], proof["verified"])

    def test_explicit_receipt_reuses_proof_when_output_receipt_absent(self):
        _, proof = self.prepare_verified_download()
        archived = self.root / "previous.json"
        self.receipt_path.rename(archived)
        self.args.verified_receipt = [str(archived)]
        with patch.object(stager, "verify_file", side_effect=AssertionError("unnecessary rehash")):
            self.run_stager()
        self.assertEqual(json.loads(self.receipt_path.read_text())["verified"], proof["verified"])

    def test_changed_mtime_requires_full_hash(self):
        target, _ = self.prepare_verified_download()
        stat = target.stat()
        stager.os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with patch.object(stager, "verify_file", wraps=stager.verify_file) as verify:
            self.run_stager()
        verify.assert_called_once()

    def test_changed_content_is_not_accepted_by_old_proof(self):
        target, _ = self.prepare_verified_download()
        target.write_bytes(b"wrong test weights")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_stager()

    def test_mismatched_receipt_identity_rejected_without_overwrite(self):
        _, proof = self.prepare_verified_download()
        for key in ("repo", "revision", "destination"):
            with self.subTest(field=key):
                wrong = dict(proof, **{key: "different"})
                self.receipt_path.write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, "receipt .* mismatch"):
                    self.run_stager()
                self.assertEqual(json.loads(self.receipt_path.read_text()), wrong)

    def test_wrong_expected_hash_in_receipt_does_not_skip_verification(self):
        _, proof = self.prepare_verified_download()
        proof["verified"]["weights.safetensors"]["hash"] = "b" * 64
        self.receipt_path.write_text(json.dumps(proof))
        with patch.object(stager, "verify_file", wraps=stager.verify_file) as verify:
            self.run_stager()
        verify.assert_called_once()

    def test_verify_command_always_rehashes(self):
        self.prepare_verified_download()
        self.args.command = "verify"
        self.args.verified_receipt = [str(self.receipt_path)]
        with patch.object(stager, "verify_file", wraps=stager.verify_file) as verify:
            self.run_stager()
        verify.assert_called_once()

    def test_destination_outside_required_mount_rejected(self):
        self.write_manifest()
        self.args.require_mount = str(self.root / "volume")
        with patch.object(stager.os.path, "ismount", return_value=True):
            with self.assertRaises(ValueError):
                self.run_stager()
        self.assertFalse(self.destination.exists())

    def test_symlinked_destination_ancestor_cannot_create_outside_mount(self):
        self.write_manifest()
        mount = self.root / "volume"
        outside = self.root / "outside"
        mount.mkdir()
        outside.mkdir()
        (mount / "escape").symlink_to(outside, target_is_directory=True)
        self.args.require_mount = str(mount)
        self.args.destination = str(mount / "escape" / "must-not-create")
        with patch.object(stager.os.path, "ismount", return_value=True):
            with self.assertRaises(ValueError):
                self.run_stager()
        self.assertFalse((outside / "must-not-create").exists())

    def test_manifest_child_symlink_escape_does_not_touch_outside_file(self):
        self.write_manifest(manifest(entry("nested/payload", b"tiny")))
        self.destination.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.destination / "nested").symlink_to(outside, target_is_directory=True)
        self.args.command = "download"
        calls = []
        with self.fake_hub(lambda *a, **kw: calls.append((a, kw))):
            with self.assertRaises(ValueError):
                self.run_stager()
        self.assertEqual(calls, [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_dangling_leaf_symlink_rejected_before_downloader(self):
        self.write_manifest()
        self.destination.mkdir()
        target = self.destination / "other-file"
        (self.destination / "weights.safetensors").symlink_to(target)
        self.args.command = "download"
        calls = []

        def download(repo, **kwargs):
            calls.append(kwargs)
            (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(b"small test weights")

        with self.fake_hub(download):
            with self.assertRaises(ValueError):
                self.run_stager()
        self.assertEqual(calls, [])
        self.assertFalse(target.exists())

    def test_hf_cache_symlink_escape_rejected_before_downloader(self):
        self.write_manifest()
        self.destination.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.destination / ".cache").symlink_to(outside, target_is_directory=True)
        self.args.command = "download"
        calls = []

        def download(repo, **kwargs):
            calls.append(kwargs)
            (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(b"small test weights")

        with self.fake_hub(download):
            with self.assertRaises(ValueError):
                self.run_stager()
        self.assertEqual(calls, [])

    def test_planner_rejects_api_revision_mismatch(self):
        class FakeApi:
            def __init__(self, **kwargs):
                pass

            def model_info(self, *args, **kwargs):
                return types.SimpleNamespace(sha="b" * 40, siblings=[
                    types.SimpleNamespace(rfilename="config.json", size=2,
                                          lfs=None, blob_id="c" * 40)])

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = FakeApi
        args = types.SimpleNamespace(repo="example/checkpoint", revision=REVISION,
                                     file=None, manifest=str(self.manifest_path))
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    stager.plan(args)
        self.assertFalse(self.manifest_path.exists())

    def test_filename_lock_prevents_overlap_with_different_receipt(self):
        data = b"tiny"
        self.write_manifest(manifest(entry("a", data)))
        self.destination.mkdir()
        (self.destination / "a").write_bytes(data)
        real_verify = stager.verify_file
        called = []

        def verify_with_second_job(path, expected):
            other_args = types.SimpleNamespace(**vars(self.args))
            other_args.receipt = str(self.root / "second-receipt.json")
            with self.assertRaises(BlockingIOError):
                stager.run(other_args)
            called.append(True)
            return real_verify(path, expected)

        with patch.object(stager, "verify_file", side_effect=verify_with_second_job):
            self.run_stager()
        self.assertEqual(called, [True])
        self.assertFalse((self.root / "second-receipt.json").exists())

    def test_failed_file_does_not_schedule_remaining_downloads(self):
        data = b"tiny"
        self.write_manifest(manifest(*(entry(name, data) for name in ("a", "b", "c"))))
        self.args.command = "download"
        calls = []

        def download(repo, **kwargs):
            calls.append(kwargs["filename"])
            raise OSError("download disconnected")

        with self.fake_hub(download), self.assertRaisesRegex(ValueError, "disconnected"):
            self.run_stager()
        self.assertEqual(calls, ["a"])
        self.assertEqual(json.loads(self.receipt_path.read_text())["state"], "failed")

    def test_disjoint_snapshot_jobs_can_share_destination(self):
        data = b"tiny"
        self.write_manifest(manifest(entry("weights", data)))
        self.destination.mkdir()
        (self.destination / "weights").write_bytes(data)
        (self.destination / "engrams").write_bytes(data)
        other_manifest = self.root / "engram-manifest.json"
        other_manifest.write_text(json.dumps(manifest(entry("engrams", data))))
        other_receipt = self.root / "engram-receipt.json"
        real_verify = stager.verify_file

        def verify_with_disjoint_job(path, expected):
            if path.name == "weights":
                other_args = types.SimpleNamespace(**vars(self.args))
                other_args.manifest = str(other_manifest)
                other_args.receipt = str(other_receipt)
                stager.run(other_args)
            return real_verify(path, expected)

        with patch.object(stager, "verify_file", side_effect=verify_with_disjoint_job):
            self.run_stager()
        self.assertEqual(json.loads(self.receipt_path.read_text())["state"], "verified")
        self.assertEqual(json.loads(other_receipt.read_text())["state"], "verified")

    def test_cli_failure_exits_and_preserves_failed_receipt(self):
        self.write_manifest(manifest(entry(data=b"expected")))
        self.destination.mkdir()
        target = self.destination / "weights.safetensors"
        target.write_bytes(b"tampered")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "verify", "--manifest", str(self.manifest_path),
             "--destination", str(self.destination), "--receipt", str(self.receipt_path),
             "--reserve-gib", "0", "--workers", "1"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(any(item["event"] == "failed" for item in events))
        self.assertFalse(any(item["event"] == "complete" for item in events))
        self.assertEqual(target.read_bytes(), b"tampered")
        self.assertEqual(json.loads(self.receipt_path.read_text())["state"], "failed")

    def test_cli_rejects_mutable_revision_before_hub_access(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "plan", "--repo", "example/checkpoint",
             "--revision", "main", "--manifest", str(self.manifest_path)],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pin a full revision", result.stderr)
        self.assertFalse(self.manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
