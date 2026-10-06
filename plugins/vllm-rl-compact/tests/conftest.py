# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the tests from the source tree: make the package importable and its
entry point discoverable when the plugin is not installed."""

import importlib
import importlib.metadata
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not any(
    ep.name == "rl_compact"
    for ep in importlib.metadata.entry_points(group="vllm.endpoint_plugins")
):
    import dev_entry_point

    # Removed at interpreter exit (TemporaryDirectory finalizer).
    _ENTRY_POINT_DIR = tempfile.TemporaryDirectory(
        prefix="vllm-rl-compact-entry-point-"
    )
    dev_entry_point.write(Path(_ENTRY_POINT_DIR.name))
    sys.path.insert(0, _ENTRY_POINT_DIR.name)
    importlib.invalidate_caches()
