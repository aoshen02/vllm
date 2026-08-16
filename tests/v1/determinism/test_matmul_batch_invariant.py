# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers.batch_invariant import matmul_batch_invariant
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_unsupported
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        # 2D x 2D
        ((32, 64), (64, 16)),
        # 2D x 3D
        ((64, 16), (4, 16, 32)),
        # 3D x 2D
        ((4, 32, 64), (64, 16)),
        # 4D x 2D
        ((1, 4, 32, 64), (64, 16)),
        # 3D x 3D
        ((4, 32, 64), (4, 64, 16)),
        # 3D x 4D
        ((2, 32, 64), (1, 2, 64, 16)),
        # 4D x 3D (Gemma4 pattern)
        ((1, 2, 32, 64), (2, 64, 16)),
        # 4D x 4D
        ((1, 2, 32, 64), (4, 2, 64, 16)),
        # 2D x 4D
        ((32, 64), (1, 2, 64, 16)),
        # 2D x 5D
        ((32, 64), (1, 2, 2, 64, 16)),
        # 5D x 2D
        ((1, 2, 2, 32, 64), (64, 16)),
        # 5D x 5D
        ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(a_shape, b_shape, dtype):
    """
    Compare matmul_batch_invariant against torch.matmul for various shapes.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a = torch.rand(a_shape, dtype=dtype, device=device)
    b = torch.rand(b_shape, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    standard_output = torch.matmul(a, b)

    # Batch-invariant implementation (Triton)
    triton_output = matmul_batch_invariant(a, b)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"matmul mismatch for a ndim={a.ndim}, b ndim={b.ndim},",
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_batch_invariance(dtype):
    """
    Verify that the result for one item is bitwise identical regardless
    of what other items are in the batch.
    """

    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a_single = torch.rand((1, 64, 32), dtype=dtype, device=device)
    b = torch.rand((32, 128), dtype=dtype, device=device)

    standard_output = matmul_batch_invariant(a_single, b)

    a_batch = torch.rand((8, 64, 32), dtype=dtype, device=device)
    a_batch[3] = a_single[0]

    batch_output = matmul_batch_invariant(a_batch, b)
    batch_output_a = batch_output[3]

    assert torch.equal(standard_output[0], batch_output_a)


@pytest.mark.parametrize(
    "M,N,K",
    [
        # The fp32 narrow-output config (BLOCK_M/N=64, BLOCK_K=128) is scoped to
        # (N, K) = (256, 4096) on SM100, the shape it was measured on. Its key
        # never includes M, so a row's reduction order must not move with M.
        # These shapes straddle the points where that could go wrong:
        (64, 256, 4096),  # exactly one M tile
        (65, 256, 4096),  # second M tile, partially filled
        (128, 256, 4096),  # second M tile, full
        (4096, 256, 4096),  # more tiles than SMs: persistent programs loop
    ],
)
@skip_unsupported
def test_matmul_fp32_narrow_output_rows_do_not_move_with_m(M, N, K):
    """A row's result must not depend on how many rows share the launch.

    The narrow-output tile config exists for the fp32 router gate; it changes
    BLOCK_M, so the M tiling and the persistent loop are where a row's reduction
    order could start tracking the batch. Compare the same row computed alone
    against the same row inside progressively larger launches.
    """
    device = torch.device(DEVICE_TYPE)
    gen = torch.Generator(device=device).manual_seed(M)
    b = torch.randn((K, N), generator=gen, device=device, dtype=torch.float32)
    a = torch.randn((M, K), generator=gen, device=device, dtype=torch.float32)

    full = matmul_batch_invariant(a, b)
    for row in (0, M // 2, M - 1):
        alone = matmul_batch_invariant(a[row : row + 1], b)
        torch.testing.assert_close(full[row : row + 1], alone, rtol=0, atol=0)


@skip_unsupported
def test_matmul_fp32_narrow_output_config_is_actually_selected():
    """The sweep above cannot tell which config it exercised.

    Every assertion there holds for the default 128x128x32 tile too, so a tree
    where the narrow config was never wired in -- or never reached because the
    module was not the one under test -- would pass it unchanged. Assert the
    selection itself.
    """
    from vllm.model_executor.layers.batch_invariant import (
        _persistent_matmul_config,
    )

    narrow = _persistent_matmul_config(torch.float32, 256, 4096)
    if not current_platform.is_device_capability_family(100):
        pytest.skip("the narrow-output config is scoped to SM100")

    assert (
        narrow["BLOCK_SIZE_M"],
        narrow["BLOCK_SIZE_N"],
        narrow["BLOCK_SIZE_K"],
        narrow["num_warps"],
    ) == (64, 64, 128, 4)

    default = _persistent_matmul_config(torch.float32, 512, 4096)
    assert default["BLOCK_SIZE_M"] == 128 and default["BLOCK_SIZE_K"] == 32, (
        "only (fp32, N=256, K=4096) is in scope; a wider N must keep the default"
    )
    assert _persistent_matmul_config(torch.bfloat16, 256, 4096) == (
        _persistent_matmul_config(torch.bfloat16, 512, 4096)
    ), "the narrow config must not leak into other dtypes"


@skip_unsupported
def test_matmul_config_key_cannot_include_m():
    """M is not a parameter, so no future edit can make the tile track the batch.

    Worth being precise about why, because the obvious reason is not the real
    one: on the fp32 path an M-keyed tile would still be bitwise invariant,
    since the k loop accumulates in increasing k whatever BLOCK_K is. The rule
    holds because that is a property of the current Triton lowering rather than
    of the contract -- it has no reason to survive tensor cores, and this
    helper serves every dtype. Enforced by signature rather than by comment.
    """
    import inspect

    from vllm.model_executor.layers.batch_invariant import (
        _persistent_matmul_config,
    )

    params = list(inspect.signature(_persistent_matmul_config).parameters)
    assert params == ["dtype", "N", "K"], (
        f"the tile config takes {params}; it must not be able to see M"
    )
