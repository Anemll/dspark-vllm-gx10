#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one command in a process group with a hard wall-clock deadline."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


TIMEOUT_EXIT_CODE = 124


def terminate_group(process: subprocess.Popen[bytes], grace_s: float) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_bounded(command: list[str], timeout_s: float, grace_s: float) -> int:
    if not command:
        raise ValueError("command is required")
    if timeout_s <= 0 or grace_s <= 0:
        raise ValueError("timeout and grace must be positive")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        terminate_group(process, grace_s)
        return TIMEOUT_EXIT_CODE
    except BaseException:
        terminate_group(process, grace_s)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--term-grace", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = run_bounded(command, args.timeout, args.term_grace)
    except ValueError as error:
        parser.error(str(error))
    raise SystemExit(result)


if __name__ == "__main__":
    main()
