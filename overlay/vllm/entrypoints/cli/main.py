# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import atexit
import importlib.metadata
import os
import shutil
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path


_prometheus_multiprocess_dir: str | None = None
_prometheus_multiprocess_owner_pid: int | None = None
_PROMETHEUS_ACTIVE_DIR_ENV = "VLLM_PROMETHEUS_MULTIPROC_ACTIVE_DIR"


def _cleanup_owned_prometheus_multiprocess_dir() -> None:
    """Remove only the directory created by this CLI process."""

    global _prometheus_multiprocess_dir
    if (
        _prometheus_multiprocess_dir is not None
        and _prometheus_multiprocess_owner_pid == os.getpid()
    ):
        shutil.rmtree(_prometheus_multiprocess_dir, ignore_errors=True)
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR") == (
            _prometheus_multiprocess_dir
        ):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        if os.environ.get(_PROMETHEUS_ACTIVE_DIR_ENV) == (
            _prometheus_multiprocess_dir
        ):
            os.environ.pop(_PROMETHEUS_ACTIVE_DIR_ENV, None)
        _prometheus_multiprocess_dir = None


def _setup_prometheus_multiprocess_for_serve(
    argv: list[str] | None = None,
) -> str | None:
    """Prepare an empty multiprocess directory before Prometheus is imported.

    The console entrypoint calls this at module import, before importing any
    other vLLM module.  Spawned/forked workers inherit the environment.  An
    explicitly supplied directory is accepted only when it already exists and
    is empty, matching prometheus-client's between-runs cleanup contract.
    """

    global _prometheus_multiprocess_dir
    global _prometheus_multiprocess_owner_pid

    argv = sys.argv if argv is None else argv
    if len(argv) < 2 or argv[1] != "serve":
        return None

    configured = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if configured:
        path = Path(configured)
        if not path.is_dir():
            raise RuntimeError(
                "PROMETHEUS_MULTIPROC_DIR must name an existing directory: "
                f"{configured}"
            )
        inherited_active_dir = os.environ.get(_PROMETHEUS_ACTIVE_DIR_ENV)
        if inherited_active_dir == configured:
            return configured
        if any(path.iterdir()):
            raise RuntimeError(
                "PROMETHEUS_MULTIPROC_DIR must be empty before vLLM starts: "
                f"{configured}"
            )
        os.environ[_PROMETHEUS_ACTIVE_DIR_ENV] = configured
        return configured

    path = tempfile.mkdtemp(prefix="vllm-prometheus-multiprocess-")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = path
    os.environ[_PROMETHEUS_ACTIVE_DIR_ENV] = path
    _prometheus_multiprocess_dir = path
    _prometheus_multiprocess_owner_pid = os.getpid()
    atexit.register(_cleanup_owned_prometheus_multiprocess_dir)
    return path


# This must remain above every vLLM/prometheus_client import.  The console
# script imports this module with the final argv already installed.
_setup_prometheus_multiprocess_for_serve()

from vllm.logger import init_logger  # noqa: E402

logger = init_logger(__name__)


def main():
    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.launch
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.serve.utils.api_utils import (
        VLLM_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,
        vllm.entrypoints.cli.launch,
        vllm.entrypoints.cli.benchmark.main,
        vllm.entrypoints.cli.collect_env,
        vllm.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" in sys.argv:
        # NOTE: Check the spec instead of importing directly here, since things could
        # fail with ImportError due to mismatched versions if things are moved around.
        spec = find_spec("vllm_omni")
        if spec is None:
            logger.error(
                "--omni flag requires a valid instance of vllm-omni to be installed."
            )
            sys.exit(1)

        from vllm_omni.entrypoints.cli.main import main as omni_main

        logger.info("Delegating entrypoint handling to vllm-omni")
        omni_main()
    else:
        # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
        if len(sys.argv) > 1 and sys.argv[1] == "bench":
            logger.debug(
                "Bench command detected, must ensure current platform is not "
                "UnspecifiedPlatform to avoid device type inference error"
            )
            from vllm import platforms

            if platforms.current_platform.is_unspecified():
                from vllm.platforms.cpu import CpuPlatform

                platforms.current_platform = CpuPlatform()
                logger.info(
                    "Unspecified platform detected, switching to CPU Platform instead."
                )

        parser = FlexibleArgumentParser(
            description="vLLM CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm"),
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
