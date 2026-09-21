# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped controls for MTP checkpoint completeness validation.

MTP was the first of these checks to meet a streamed reload, so it has its own
names. They now defer to `completeness`, which governs every such check.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from .completeness import completeness_checks_enabled, streaming_a_checkpoint


def is_mtp_completeness_check_enabled() -> bool:
    """Return whether MTP completeness validation is enabled in this scope."""
    return completeness_checks_enabled()


@contextmanager
def disable_mtp_completeness_check() -> Iterator[None]:
    """Temporarily disable MTP completeness validation for one weight load."""
    with streaming_a_checkpoint():
        yield
