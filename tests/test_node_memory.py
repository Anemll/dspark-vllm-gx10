# SPDX-License-Identifier: MIT
"""Offline tests for units, counter lifetime and missing memory evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("node_memory", ROOT / "scripts/sample-node-memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


def sample():
    return {"monotonic_s": 10.0, "boot_id": "boot-a", "page_size_bytes": 4096,
            "vm_pages_or_events": dict.fromkeys(memory.VM_COUNTERS, 100),
            "mem_kib": {"SwapTotal": 16384, "SwapFree": 12000},
            "memory_psi": {"some": {"total_us": 1000}, "full": {"total_us": 1000}},
            "cgroup": None}


class MemoryTests(unittest.TestCase):
    def pair(self):
        before = sample()
        after = copy.deepcopy(before)
        after["monotonic_s"] = 12.0
        return before, after

    def test_proc_key_values_and_units(self):
        self.assertEqual(memory.key_values("MemAvailable: 123 kB\npswpout 15\nName: vllm\n"),
                         {"MemAvailable": 123, "pswpout": 15})

    def test_pressure_preserves_totals_and_averages(self):
        parsed = memory.pressure("some avg10=1.25 avg60=0.50 avg300=0.10 total=23456\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=100")
        self.assertEqual(parsed["some"]["total_us"], 23456)
        self.assertEqual(parsed["some"]["avg10"], 1.25)

    def test_no_since_boot_rate_on_first_sample(self):
        self.assertEqual(memory.interval(None, sample())["reason"], "first_sample")

    def test_swap_rates_use_page_size_and_actual_elapsed(self):
        before, after = self.pair()
        after["vm_pages_or_events"]["pswpout"] += 200
        after["vm_pages_or_events"]["pswpin"] += 10
        after["mem_kib"]["SwapFree"] -= 100
        result = memory.interval(before, after)
        self.assertTrue(result["valid"])
        self.assertEqual(result["swap_out_bytes_s"], 409600)
        self.assertEqual(result["swap_in_bytes_s"], 20480)
        self.assertEqual(result["swap_used_delta_kib"], 100)
        before["page_size_bytes"] = after["page_size_bytes"] = 65536
        self.assertEqual(memory.interval(before, after)["swap_out_bytes_s"], 6553600)

    def test_psi_uses_interval_total_not_boot_total_or_avg10(self):
        before, after = self.pair()
        after["memory_psi"]["full"] |= {"total_us": 201000, "avg10": 99.0}
        self.assertEqual(memory.interval(before, after)["memory_stall_percent"]["full"], 10.0)

    def test_counter_reset_not_negative_or_zero(self):
        before, after = self.pair()
        after["vm_pages_or_events"]["pswpout"] = 1
        result = memory.interval(before, after)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["swap_out_bytes_s"])
        self.assertIn("pswpout:reset", result["errors"])

    def test_missing_psi_not_a_zero_pressure_pass(self):
        before, after = self.pair()
        after["memory_psi"] = None
        result = memory.interval(before, after)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["memory_stall_percent"]["full"])

    def test_missing_vm_not_zero_swap(self):
        before, after = self.pair()
        after["vm_pages_or_events"] = None
        result = memory.interval(before, after)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["swap_out_bytes_s"])

    def test_changed_boot_or_unknown_boot_is_not_comparable(self):
        for boot in (None, "boot-b"):
            before, after = self.pair()
            after["boot_id"] = boot
            self.assertEqual(memory.interval(before, after)["reason"], "boot_unknown_or_changed")

    def test_clock_or_page_size_change_invalidates(self):
        for timestamp in (10.0, 9.0, float("nan")):
            before, after = self.pair()
            after["monotonic_s"] = timestamp
            self.assertFalse(memory.interval(before, after)["valid"])
        before, after = self.pair()
        after["page_size_bytes"] = 65536
        self.assertEqual(memory.interval(before, after)["reason"], "page_size_changed")

    def test_cgroup_oom_events(self):
        before, after = self.pair()
        before["cgroup"] = {"path": "group-a", "events": dict.fromkeys(("high", "max", "oom", "oom_kill"), 0)}
        after["cgroup"] = copy.deepcopy(before["cgroup"])
        after["cgroup"]["events"]["oom_kill"] = 1
        self.assertEqual(memory.interval(before, after)["cgroup_event_deltas"]["oom_kill"], 1)
        after["cgroup"]["path"] = "group-b"
        result = memory.interval(before, after)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["cgroup_event_deltas"])

    def test_falling_swap_usage_is_not_counter_reset(self):
        before, after = self.pair()
        after["mem_kib"]["SwapFree"] += 100
        result = memory.interval(before, after)
        self.assertTrue(result["valid"])
        self.assertEqual(result["swap_used_delta_kib"], -100)

    def test_missing_proc_files_record_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            result = memory.snapshot(proc_root=Path(directory), disk_path=Path(directory), pids=[123])
        self.assertIsNone(result["mem_kib"])
        self.assertIsNone(result["process_kib"]["123"])
        self.assertIsNone(result["boot_id"])
        self.assertGreaterEqual(len(result["errors"]), 5)
        self.assertGreater(result["disk_free_bytes"], 0)

    def test_cli_rejects_unbounded_and_invalid_sampling(self):
        for args in (("--samples", "0"), ("--samples", "3601"), ("--interval", "nan"),
                     ("--samples", "100", "--interval", "60"), ("--pid", "0")):
            with patch("sys.argv", ["sample-node-memory.py", *args]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as result:
                    memory.main()
                self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
