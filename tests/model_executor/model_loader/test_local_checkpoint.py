# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import zlib

import numpy as np
import pytest
import safetensors.numpy
import zstandard

from vllm.utils.local_checkpoint import pull_checkpoint


def _checksum(data: np.ndarray) -> str:
    return f"{zlib.adler32(data):08x}"


def _write_delta(
    source_dir,
    version: int,
    old: np.ndarray,
    new: np.ndarray,
    encoding: str,
) -> None:
    version_dir = source_dir / f"weight_v{version:06d}"
    version_dir.mkdir()
    old_bytes = old.view(np.uint8).reshape(-1)
    new_bytes = new.view(np.uint8).reshape(-1)
    if encoding == "xor":
        payload = old_bytes ^ new_bytes
    else:
        positions = np.flatnonzero(old_bytes != new_bytes).astype("<u4")
        payload = np.concatenate(
            (
                np.array([positions.size], dtype="<u4").view(np.uint8),
                positions.view(np.uint8),
                new_bytes[positions],
            )
        )
    compressed = np.frombuffer(
        zstandard.ZstdCompressor(level=1).compress(payload), dtype=np.uint8
    )
    tensor_file = "model-00000-of-00001.safetensors"
    safetensors.numpy.save_file(
        {"weight": compressed},
        version_dir / tensor_file,
        metadata={"weight": _checksum(new_bytes)},
    )
    index = {
        "metadata": {
            "version": f"{version:06d}",
            "base_version": f"{version - 1:06d}",
            "delta_encoding": encoding,
            "compression_format": "zstd",
            "checksum_format": "adler32",
        },
        "weight_map": {"weight": tensor_file},
    }
    (version_dir / "model.safetensors.index.json").write_text(json.dumps(index))


def _write_full(source_dir, version: int, weight: np.ndarray) -> None:
    version_dir = source_dir / f"weight_v{version:06d}"
    version_dir.mkdir()
    safetensors.numpy.save_file({"weight": weight}, version_dir / "model.safetensors")
    (version_dir / "config.json").write_text("{}")


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_pull_checkpoint_applies_vime_delta(tmp_path, encoding):
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "published"
    local_dir = tmp_path / "local"
    base_dir.mkdir()
    source_dir.mkdir()

    baseline = np.arange(12, dtype=np.float32).reshape(3, 4)
    updated = baseline.copy()
    updated[0, 1] = 100.0
    updated[2, 3] = -5.0
    safetensors.numpy.save_file({"weight": baseline}, base_dir / "model.safetensors")
    (base_dir / "config.json").write_text("{}")
    _write_delta(source_dir, 1, baseline, updated, encoding)

    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 0)
    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 1)
    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 1)

    actual = safetensors.numpy.load_file(local_dir / "model.safetensors")
    np.testing.assert_array_equal(actual["weight"], updated)
    state = json.loads((local_dir / ".weight_sync" / "state.json").read_text())
    assert state == {"version": "000001"}


def test_pull_checkpoint_resets_to_latest_full_version(tmp_path):
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "published"
    local_dir = tmp_path / "local"
    base_dir.mkdir()
    source_dir.mkdir()

    baseline = np.arange(12, dtype=np.float32).reshape(3, 4)
    first = baseline + 1
    reset = baseline + 10
    latest = reset.copy()
    latest[1, 2] = -7.0
    safetensors.numpy.save_file({"weight": baseline}, base_dir / "model.safetensors")
    (base_dir / "config.json").write_text("{}")
    _write_delta(source_dir, 1, baseline, first, "xor")
    _write_full(source_dir, 2, reset)
    _write_delta(source_dir, 3, reset, latest, "xor")

    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 1)
    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 3)

    actual = safetensors.numpy.load_file(local_dir / "model.safetensors")
    np.testing.assert_array_equal(actual["weight"], latest)
    state = json.loads((local_dir / ".weight_sync" / "state.json").read_text())
    assert state == {"version": "000003"}


def test_pull_checkpoint_does_not_advance_on_checksum_failure(tmp_path):
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "published"
    local_dir = tmp_path / "local"
    base_dir.mkdir()
    source_dir.mkdir()

    baseline = np.arange(12, dtype=np.float32).reshape(3, 4)
    updated = baseline + 1
    safetensors.numpy.save_file({"weight": baseline}, base_dir / "model.safetensors")
    (base_dir / "config.json").write_text("{}")
    _write_delta(source_dir, 1, baseline, updated, "xor")
    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 0)
    safetensors.numpy.save_file(
        {"weight": baseline + 2}, local_dir / "model.safetensors"
    )

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 1)

    state = json.loads((local_dir / ".weight_sync" / "state.json").read_text())
    assert state == {"version": "000000"}


def _publish_before_read(source_dir, target_version):
    from pathlib import Path

    source = Path(source_dir)
    (source / "hook-call.json").write_text(json.dumps(target_version))
    _write_full(source, target_version, np.full((2, 2), 7, dtype=np.float32))


def _reject_before_read(source_dir, target_version):
    raise RuntimeError("publication unavailable")


@pytest.mark.parametrize("target_version", [0, 1])
def test_pre_read_hook_publishes_before_version_lookup(tmp_path, target_version):
    base_dir, source_dir, local_dir = (
        tmp_path / "base",
        tmp_path / "published",
        tmp_path / "local",
    )
    base_dir.mkdir()
    source_dir.mkdir()
    baseline = np.ones((2, 2), dtype=np.float32)
    safetensors.numpy.save_file({"weight": baseline}, base_dir / "model.safetensors")
    pull_checkpoint(
        str(local_dir),
        str(base_dir),
        str(source_dir),
        target_version,
        pre_read_hook=f"{__name__}._publish_before_read",
    )
    actual = safetensors.numpy.load_file(local_dir / "model.safetensors")["weight"]
    expected = np.full((2, 2), 7, dtype=np.float32) if target_version else baseline
    np.testing.assert_array_equal(actual, expected)
    assert (source_dir / "hook-call.json").exists() == bool(target_version)


def test_pre_read_hook_failure_does_not_materialize_checkpoint(tmp_path):
    local_dir = tmp_path / "local"
    with pytest.raises(RuntimeError, match="publication unavailable"):
        pull_checkpoint(
            str(local_dir),
            str(tmp_path / "base"),
            str(tmp_path / "published"),
            1,
            pre_read_hook=f"{__name__}._reject_before_read",
        )
    assert not local_dir.exists()


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_delta_reads_only_indexed_shards(tmp_path, encoding):
    base_dir, source_dir, local_dir = (
        tmp_path / "base",
        tmp_path / "published",
        tmp_path / "local",
    )
    base_dir.mkdir()
    source_dir.mkdir()
    baseline = np.arange(4, dtype=np.float32).reshape(2, 2)
    updated = baseline + 3
    safetensors.numpy.save_file({"weight": baseline}, base_dir / "model.safetensors")
    stale_contents = b"\x02\x00\x00\x00\x00\x00\x00\x00!!"
    (base_dir / "stale.safetensors").write_bytes(stale_contents)
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}})
    )
    _write_delta(source_dir, 1, baseline, updated, encoding)
    pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 1)
    np.testing.assert_array_equal(
        safetensors.numpy.load_file(local_dir / "model.safetensors")["weight"], updated
    )
    assert (local_dir / "stale.safetensors").read_bytes() == stale_contents


def test_missing_indexed_shard_does_not_use_unreferenced_copy(tmp_path):
    base_dir, source_dir, local_dir = (
        tmp_path / "base",
        tmp_path / "published",
        tmp_path / "local",
    )
    base_dir.mkdir()
    source_dir.mkdir()
    baseline = np.arange(4, dtype=np.float32).reshape(2, 2)
    safetensors.numpy.save_file({"weight": baseline}, base_dir / "model.safetensors")
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    _write_delta(source_dir, 1, baseline, baseline + 1, "xor")
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        pull_checkpoint(str(local_dir), str(base_dir), str(source_dir), 1)
    state = json.loads((local_dir / ".weight_sync" / "state.json").read_text())
    assert state == {"version": "000000"}


def test_pre_read_hook_cli_is_owned_by_engine_args(tmp_path):
    from dataclasses import fields

    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--model",
                str(tmp_path),
                "--custom-pull-weights-pre-read-hook",
                "publisher.wait_for_version",
            ]
        )
    )
    assert args.custom_pull_weights_pre_read_hook == "publisher.wait_for_version"
    config_field = next(
        field
        for field in fields(ModelConfig)
        if field.name == "custom_pull_weights_pre_read_hook"
    )
    assert config_field.default is None
