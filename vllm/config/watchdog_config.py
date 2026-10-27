# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field

from vllm.config.utils import config

_DEFAULT_TIMEOUT = 300
_DEFAULT_CHECK_INTERVAL = 10


@config
class WatchdogConfig:
    """Configuration for the progress watchdog.

    The watchdog runs a background thread that periodically checks whether the
    main thread has fed it in time. If not, it dumps the stack traces of all
    threads for debugging and optionally terminates the process.
    """

    timeout: int = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """Timeout in seconds after which the watchdog considers the process to be
    stuck and dumps stack traces. Must be greater than 0."""

    check_interval: int = Field(default=_DEFAULT_CHECK_INTERVAL, gt=0)
    """Interval in seconds at which the watchdog checks for feed timeouts."""

    dump_dir: str = ""
    """Directory to save stack dump files. Without a directory, fatal timeout
    diagnostics are written to stderr."""

    abort_on_timeout: bool = False
    """Terminate a stalled process after dumping its stacks. The watchdog is
    disabled when this is false and dump_dir is empty."""
