"""Opt-in process-local uint8 metadata-owner candidate, not a kernel change."""

import hashlib
from pathlib import Path

import torch


def install():
    import fi_w4a16_native_dlpack as native
    from flashinfer.comm import dlpack_utils, mnnvl

    source_sha = hashlib.sha256(Path(dlpack_utils.__file__).read_bytes()).hexdigest()
    assert source_sha == (
        "06baa8ea256f3360d6e223b4537f867fe1feac555828b0518a49f478617744af"
    )
    original = mnnvl.pack_strided_memory
    assert original is dlpack_utils.pack_strided_memory

    def pack(ptr, segment_size, segment_stride, num_segments, dtype, dev_id):
        if dtype != torch.uint8:
            return original(
                ptr, segment_size, segment_stride, num_segments, dtype, dev_id
            )
        capsule = native.make_capsule(
            ptr, num_segments, segment_size, segment_stride, 2, dev_id
        )
        return torch.utils.dlpack.from_dlpack(capsule)

    mnnvl.pack_strided_memory = pack
    fi_root = Path(dlpack_utils.__file__).parent.parent
    provenance = [
        Path(__file__),
        Path(__file__).with_name("fi_w4a16_native_dlpack.cpp"),
        Path(native.__file__),
        Path(dlpack_utils.__file__),
        Path(mnnvl.__file__),
        fi_root / "comm/trtllm_moe_alltoall.py",
        fi_root / "fused_moe/cute_dsl/blackwell/moe_w4a16.py",
        fi_root / "fused_moe/cute_dsl/tuner.py",
    ]
    return {
        "source_sha256": source_sha,
        "scope": "uint8 native metadata ownership only; no MnnvlMemory ownership",
        "counts_at_install": native.counts(),
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in provenance
        },
    }
