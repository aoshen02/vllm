# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    ExpandedPlaceholderRangeInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import (
    normalize_expanded_placeholders,
)


def test_normalize_expanded_placeholders_uses_modality_ranges():
    placeholders = {
        "image": [
            ExpandedPlaceholderRangeInfo(
                offset=1,
                length=3,
                canonical_token_ids=[10],
            )
        ],
        "audio": [
            ExpandedPlaceholderRangeInfo(
                offset=5,
                length=2,
                canonical_token_ids=[20],
            )
        ],
    }

    result = normalize_expanded_placeholders(
        [1, 10, 10, 10, 2, 20, 20, 3],
        placeholders,
    )

    assert result == [1, 10, 2, 20, 3]
