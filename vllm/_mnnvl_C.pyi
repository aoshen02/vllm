# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

def make_capsule(
    pointer: int,
    rows: int,
    columns: int,
    row_stride: int,
    device_type: int,
    device_id: int,
) -> Any: ...
