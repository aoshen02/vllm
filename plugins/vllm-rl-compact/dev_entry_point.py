# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Development only: make the ``rl_compact`` entry point discoverable without
installing the package (e.g. running from a source tree on ``PYTHONPATH``).

    .venv/bin/python dev_entry_point.py DIR   # then add DIR to PYTHONPATH

Writes ``DIR/vllm_rl_compact-0.1.0.dist-info`` (metadata and entry points as in
pyproject.toml). Installed packages do not need it; the tests call it on a
temporary directory (conftest.py)."""

import sys
from pathlib import Path

METADATA = """Metadata-Version: 2.1
Name: vllm-rl-compact
Version: 0.1.0
Summary: RL compact logprobs for /inference/v1/generate (vLLM endpoint plugin)
"""
ENTRY_POINTS = """[vllm.endpoint_plugins]
rl_compact = vllm_rl_compact:RLCompactPlugin
"""


def write(target: Path) -> Path:
    dist_info = target / "vllm_rl_compact-0.1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(METADATA)
    (dist_info / "entry_points.txt").write_text(ENTRY_POINTS)
    return dist_info


if __name__ == "__main__":
    print(write(Path(sys.argv[1])))
