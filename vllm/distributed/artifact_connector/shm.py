# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable execution artifacts stored in a shared-memory filesystem."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import mmap
import os
import stat
import struct
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import regex as re

from vllm.distributed.artifact_connector.store import ArtifactArray
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAGIC = b"VLLMART1"
_SUPERBLOCK_BYTES = 4096
_HEADER_LENGTH = struct.Struct("<Q")
_SAFE_ID = re.compile(r"^[a-f0-9]{32,64}$")


class ArtifactCapacityError(RuntimeError):
    """The artifact store cannot retain another object."""


class ArtifactCorruptionError(RuntimeError):
    """An artifact object failed structural or checksum validation."""


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class LocalSharedMemoryArtifactReader:
    """Read immutable artifact objects from a local SHM store."""

    def __init__(self, root: str, store_id: str) -> None:
        self._validate_id(store_id)
        self.root = Path(root) / store_id
        self.store_id = store_id
        self.blocks_dir = self.root / "blocks"
        self.tails_dir = self.root / "tails"
        self.manifests_dir = self.root / "manifests"

    @staticmethod
    def _validate_id(object_id: str) -> None:
        if not _SAFE_ID.fullmatch(object_id):
            raise ValueError(f"invalid artifact object id: {object_id!r}")

    def _path(self, kind: str, object_id: str) -> Path:
        self._validate_id(object_id)
        directory = {
            "block": self.blocks_dir,
            "tail": self.tails_dir,
            "manifest": self.manifests_dir,
        }[kind]
        suffix = ".json" if kind == "manifest" else ".bin"
        return directory / f"{object_id}{suffix}"

    @staticmethod
    def _open_regular_file(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid():
            os.close(fd)
            raise ArtifactCorruptionError(f"unsafe artifact file: {path}")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            os.close(fd)
            raise ArtifactCorruptionError(f"invalid artifact mode: {path}")
        return fd

    @staticmethod
    def _validate_signed_dict(value: dict[str, Any], checksum_key: str) -> None:
        expected = value.get(checksum_key)
        unsigned = dict(value)
        unsigned.pop(checksum_key, None)
        actual = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        if expected != actual:
            raise ArtifactCorruptionError(f"{checksum_key} mismatch")

    @contextmanager
    def open_array(self, kind: str, object_id: str) -> Iterator[np.ndarray]:
        """Map an immutable array without copying its payload."""
        if kind not in ("block", "tail"):
            raise ValueError(f"invalid artifact array kind: {kind}")
        path = self._path(kind, object_id)
        fd = self._open_regular_file(path)
        mapping: mmap.mmap | None = None
        array: np.ndarray | None = None
        payload_view: memoryview | None = None
        try:
            file_size = os.fstat(fd).st_size
            if file_size < _SUPERBLOCK_BYTES:
                raise ArtifactCorruptionError("artifact file is truncated")
            mapping = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
            if mapping[: len(_MAGIC)] != _MAGIC:
                raise ArtifactCorruptionError("invalid artifact magic")
            length_start = len(_MAGIC)
            length_end = length_start + _HEADER_LENGTH.size
            (header_length,) = _HEADER_LENGTH.unpack(mapping[length_start:length_end])
            max_header_length = _SUPERBLOCK_BYTES - length_end
            if header_length <= 0 or header_length > max_header_length:
                raise ArtifactCorruptionError("invalid artifact header length")
            header_end = length_end + header_length
            try:
                header = json.loads(mapping[length_end:header_end])
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArtifactCorruptionError("invalid artifact header") from error
            if not isinstance(header, dict):
                raise ArtifactCorruptionError("artifact header must be an object")
            self._validate_signed_dict(header, "header_sha256")
            if header.get("kind") != f"vllm.artifact_{kind}":
                raise ArtifactCorruptionError("artifact kind mismatch")
            payload_nbytes = header.get("payload_nbytes")
            if not isinstance(payload_nbytes, int) or payload_nbytes < 0:
                raise ArtifactCorruptionError("invalid artifact payload size")
            if file_size != _SUPERBLOCK_BYTES + payload_nbytes:
                raise ArtifactCorruptionError("artifact payload size mismatch")
            payload_view = memoryview(mapping)[_SUPERBLOCK_BYTES:]
            payload_sha256 = hashlib.sha256(payload_view).hexdigest()
            if payload_sha256 != header.get("payload_sha256"):
                raise ArtifactCorruptionError("artifact payload checksum mismatch")
            try:
                dtype = np.dtype(header["dtype"])
                shape = tuple(int(dimension) for dimension in header["shape"])
            except (KeyError, TypeError, ValueError) as error:
                raise ArtifactCorruptionError(
                    "invalid artifact array metadata"
                ) from error
            if any(dimension < 0 for dimension in shape):
                raise ArtifactCorruptionError("invalid artifact array shape")
            expected_nbytes = int(np.prod(shape)) * dtype.itemsize
            if expected_nbytes != payload_nbytes:
                raise ArtifactCorruptionError("artifact array size mismatch")
            array = np.ndarray(
                shape,
                dtype=dtype,
                buffer=mapping,
                offset=_SUPERBLOCK_BYTES,
            )
            array.setflags(write=False)
            yield array
        finally:
            array = None
            if payload_view is not None:
                payload_view.release()
            if mapping is not None:
                mapping.close()
            os.close(fd)

    def read_array(self, kind: str, object_id: str) -> np.ndarray:
        """Read an artifact array into owned memory."""
        with self.open_array(kind, object_id) as array:
            return array.copy()

    def read_manifest(self, sample_id: str) -> dict[str, Any]:
        path = self._path("manifest", sample_id)
        fd = self._open_regular_file(path)
        try:
            payload = b""
            while chunk := os.read(fd, 1 << 20):
                payload += chunk
        finally:
            os.close(fd)
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactCorruptionError("invalid artifact manifest") from error
        if not isinstance(manifest, dict):
            raise ArtifactCorruptionError("artifact manifest must be an object")
        self._validate_signed_dict(manifest, "manifest_sha256")
        return manifest

    def materialize(self, handle: dict[str, Any]) -> np.ndarray:
        """Materialize all manifest segments into one request-local array."""
        if handle.get("backend") != "shm" or handle.get("schema_version") != 1:
            raise ArtifactCorruptionError("unsupported artifact handle")
        if handle.get("store_id") != self.store_id:
            raise ArtifactCorruptionError("artifact store mismatch")
        sample_id = handle.get("artifact_sample_id")
        if not isinstance(sample_id, str):
            raise ArtifactCorruptionError("artifact sample id is missing")
        manifest = self.read_manifest(sample_id)
        for key in (
            "artifact_sample_id",
            "store_id",
            "profile_id",
            "manifest_sha256",
        ):
            if handle.get(key) != manifest.get(key):
                raise ArtifactCorruptionError(f"handle/manifest {key} mismatch")
        try:
            dtype = np.dtype(manifest["dtype"])
            shape = tuple(int(dimension) for dimension in manifest["shape"])
            segments = manifest["segments"]
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactCorruptionError("invalid artifact manifest schema") from error
        if not isinstance(segments, list):
            raise ArtifactCorruptionError("artifact segments must be a list")
        output = np.empty(shape, dtype=dtype)
        covered = 0
        for segment in segments:
            if not isinstance(segment, dict):
                raise ArtifactCorruptionError("invalid artifact segment")
            kind = segment.get("kind")
            object_id = segment.get("object_id")
            output_start = segment.get("output_start")
            valid_len = segment.get("valid_len")
            if (
                kind not in ("block", "tail")
                or not isinstance(object_id, str)
                or not isinstance(output_start, int)
                or not isinstance(valid_len, int)
                or output_start != covered
                or valid_len <= 0
            ):
                raise ArtifactCorruptionError("invalid artifact segment layout")
            with self.open_array(kind, object_id) as segment_array:
                if segment_array.dtype != dtype or segment_array.shape[0] != valid_len:
                    raise ArtifactCorruptionError("artifact segment shape mismatch")
                output[output_start : output_start + valid_len] = segment_array
            covered += valid_len
        if covered != shape[0]:
            raise ArtifactCorruptionError("artifact segments do not cover the output")
        return output


class LocalSharedMemoryArtifactStore(LocalSharedMemoryArtifactReader):
    """Single-writer immutable artifact store in a shared-memory filesystem."""

    def __init__(
        self,
        root: str,
        instance_id: str,
        dp_rank: int,
        *,
        max_bytes: int,
        ttl_seconds: int,
    ) -> None:
        store_id = hashlib.sha256(f"{instance_id}:{dp_rank}".encode()).hexdigest()[:32]
        super().__init__(root, store_id)
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._used_bytes = 0
        self._retained_blocks: Counter[str] = Counter()
        self._last_gc_time = 0.0
        self._gc_interval_seconds = min(max(ttl_seconds / 4, 1.0), 60.0)
        root_path = Path(root)
        self._prepare_directory(root_path)
        self._gc_stale_store_dirs(root_path)
        self._prepare_directory(self.root)
        for directory in (self.blocks_dir, self.tails_dir, self.manifests_dir):
            self._prepare_directory(directory)
        self._writer_lock_fd: int | None = self._acquire_writer_lock()
        self.gc()

    @staticmethod
    def _prepare_directory(path: Path) -> None:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError(f"artifact path is not a directory: {path}")
        if path_stat.st_uid != os.getuid():
            raise ValueError(f"artifact directory is not owned by this user: {path}")
        path.chmod(0o700)

    def _acquire_writer_lock(self) -> int:
        path = self.root / ".writer.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid():
                raise ValueError(f"unsafe artifact writer lock: {path}")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata = _canonical_json({"ttl_seconds": self.ttl_seconds})
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.write(fd, metadata) != len(metadata):
                raise OSError("short write while recording artifact store metadata")
            os.fsync(fd)
        except Exception:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def _stale_store_entries(
        store_root: Path,
    ) -> tuple[list[Path], list[Path], float] | None:
        """Return safe-to-remove entries and their newest modification time."""
        try:
            root_stat = store_root.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
            return None

        files: list[Path] = []
        directories: list[Path] = []
        newest_mtime = root_stat.st_mtime
        expected_directories = {"blocks", "tails", "manifests"}
        try:
            root_entries = list(os.scandir(store_root))
        except FileNotFoundError:
            return None
        for entry in root_entries:
            entry_stat = entry.stat(follow_symlinks=False)
            newest_mtime = max(newest_mtime, entry_stat.st_mtime)
            path = Path(entry.path)
            if entry.name == ".writer.lock":
                if not stat.S_ISREG(entry_stat.st_mode):
                    return None
                continue
            if entry.name not in expected_directories or not stat.S_ISDIR(
                entry_stat.st_mode
            ):
                return None
            if entry_stat.st_uid != os.getuid():
                return None
            directories.append(path)
            for child in os.scandir(path):
                child_stat = child.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(child_stat.st_mode)
                    or child_stat.st_uid != os.getuid()
                ):
                    return None
                newest_mtime = max(newest_mtime, child_stat.st_mtime)
                files.append(Path(child.path))
        return files, directories, newest_mtime

    def _gc_stale_store_dirs(self, root: Path) -> None:
        """Remove expired stores whose writer process is no longer alive."""
        now = time.time()
        for entry in os.scandir(root):
            if entry.name == self.store_id or not _SAFE_ID.fullmatch(entry.name):
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                store_root = Path(entry.path)
                scanned = self._stale_store_entries(store_root)
                cutoff = now - self._read_store_ttl(
                    store_root / ".writer.lock", self.ttl_seconds
                )
                if scanned is None or scanned[2] >= cutoff:
                    continue
                lock_path = store_root / ".writer.lock"
                lock_fd = self._open_regular_file(lock_path)
                try:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    # Re-scan after acquiring the lock to close the race with
                    # a writer that changed the store before exiting.
                    scanned = self._stale_store_entries(store_root)
                    if scanned is None or scanned[2] >= cutoff:
                        continue
                    files, directories, _ = scanned
                    for path in files:
                        path.unlink(missing_ok=True)
                    for path in directories:
                        path.rmdir()
                    lock_path.unlink(missing_ok=True)
                    store_root.rmdir()
                    logger.info("Removed expired artifact SHM store %s", store_root)
                finally:
                    os.close(lock_fd)
            except (ArtifactCorruptionError, FileNotFoundError, OSError, ValueError):
                logger.debug(
                    "Could not collect stale artifact SHM store %s",
                    entry.path,
                    exc_info=True,
                )

    @classmethod
    def _read_store_ttl(cls, lock_path: Path, fallback: int) -> int:
        """Read the writer's TTL, with compatibility for older stores."""
        try:
            fd = cls._open_regular_file(lock_path)
            try:
                value = json.loads(os.read(fd, 4096))
            finally:
                os.close(fd)
            if isinstance(value, dict):
                ttl_seconds = value.get("ttl_seconds")
                if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                    return ttl_seconds
        except (
            ArtifactCorruptionError,
            json.JSONDecodeError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ):
            pass
        return fallback

    def close(self) -> None:
        fd = getattr(self, "_writer_lock_fd", None)
        if fd is not None:
            self._writer_lock_fd = None
            os.close(fd)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _usage_bytes(self) -> int:
        total = 0
        for directory in (self.blocks_dir, self.tails_dir, self.manifests_dir):
            for entry in os.scandir(directory):
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
        return total

    def _reserve(self, additional_bytes: int) -> None:
        used = self._used_bytes
        if used + additional_bytes > self.max_bytes:
            self.gc(lock_held=True)
            used = self._used_bytes
            if used + additional_bytes > self.max_bytes:
                raise ArtifactCapacityError(
                    "artifact SHM capacity exceeded: "
                    f"used={used}, requested={additional_bytes}, "
                    f"limit={self.max_bytes}"
                )
        filesystem = os.statvfs(self.root)
        available = filesystem.f_bavail * filesystem.f_frsize
        if additional_bytes > available:
            raise ArtifactCapacityError(
                "artifact SHM filesystem is full: "
                f"available={available}, requested={additional_bytes}"
            )

    @staticmethod
    def _sign_dict(value: dict[str, Any], checksum_key: str) -> dict[str, Any]:
        signed = dict(value)
        signed[checksum_key] = hashlib.sha256(_canonical_json(signed)).hexdigest()
        return signed

    def _write_immutable_parts(
        self,
        path: Path,
        parts: tuple[bytes | memoryview, ...],
        total_size: int,
    ) -> bool:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        fd: int | None = None
        mapping: mmap.mmap | None = None
        try:
            fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.posix_fallocate(fd, 0, total_size)
            except OSError as error:
                if error.errno == errno.ENOSPC:
                    raise ArtifactCapacityError(
                        f"artifact SHM filesystem could not reserve {total_size} bytes"
                    ) from error
                if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
                os.ftruncate(fd, total_size)
            if total_size:
                mapping = mmap.mmap(
                    fd,
                    total_size,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
                offset = 0
                for part in parts:
                    end = offset + len(part)
                    mapping[offset:end] = part
                    offset = end
                if offset != total_size:
                    raise RuntimeError(
                        "artifact write size mismatch: "
                        f"written={offset}, expected={total_size}"
                    )
                mapping.flush()
                mapping.close()
                mapping = None
            os.fsync(fd)
            os.close(fd)
            fd = None
            try:
                os.link(temporary, path, follow_symlinks=False)
                return True
            except FileExistsError:
                return False
        finally:
            if mapping is not None:
                mapping.close()
            if fd is not None:
                os.close(fd)
            temporary.unlink(missing_ok=True)

    def _write_immutable(self, path: Path, payload: bytes) -> bool:
        return self._write_immutable_parts(path, (payload,), len(payload))

    @classmethod
    def _encode_array_superblock(
        cls, array: np.ndarray, metadata: dict[str, Any]
    ) -> bytes:
        payload_view = memoryview(array).cast("B")
        header = cls._sign_dict(
            {
                **metadata,
                "schema_version": 1,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "payload_nbytes": len(payload_view),
                "payload_sha256": hashlib.sha256(payload_view).hexdigest(),
            },
            "header_sha256",
        )
        payload_view.release()
        encoded_header = _canonical_json(header)
        header_prefix_bytes = len(_MAGIC) + _HEADER_LENGTH.size
        if len(encoded_header) > _SUPERBLOCK_BYTES - header_prefix_bytes:
            raise ValueError("artifact header does not fit in the superblock")
        return (
            _MAGIC
            + _HEADER_LENGTH.pack(len(encoded_header))
            + encoded_header
            + bytes(_SUPERBLOCK_BYTES - header_prefix_bytes - len(encoded_header))
        )

    def put_array(
        self,
        kind: str,
        object_id: str,
        array: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        if kind not in ("block", "tail"):
            raise ValueError(f"invalid artifact array kind: {kind}")
        self._validate_id(object_id)
        contiguous = np.ascontiguousarray(array)
        superblock = self._encode_array_superblock(
            contiguous,
            {**metadata, "kind": f"vllm.artifact_{kind}"},
        )
        payload_view = memoryview(contiguous).cast("B")
        object_size = len(superblock) + len(payload_view)
        path = self._path(kind, object_id)
        try:
            with self._lock:
                self._maybe_gc()
                if path.exists():
                    self._accept_existing(kind, object_id, contiguous)
                    return
                self._reserve(object_size)
                created = self._write_immutable_parts(
                    path,
                    (superblock, payload_view),
                    object_size,
                )
                if created:
                    self._used_bytes += object_size
                if not created:
                    self._accept_existing(kind, object_id, contiguous)
        finally:
            payload_view.release()

    def put_blocks(self, blocks: list[ArtifactArray]) -> None:
        """Publish one worker-step batch of completed full blocks."""
        retained: list[str] = []
        try:
            for block in blocks:
                self.put_array("block", block.object_id, block.array, block.metadata)
                with self._lock:
                    self._retained_blocks[block.object_id] += 1
                retained.append(block.object_id)
        except Exception:
            self.release_blocks(retained)
            raise

    def release_blocks(self, object_ids: list[str]) -> None:
        """Release in-progress references after manifest or discard."""
        with self._lock:
            for object_id in object_ids:
                count = self._retained_blocks.get(object_id, 0)
                if count <= 1:
                    self._retained_blocks.pop(object_id, None)
                else:
                    self._retained_blocks[object_id] = count - 1

    def _accept_existing(self, kind: str, object_id: str, incoming: np.ndarray) -> None:
        existing = self.read_array(kind, object_id)
        if existing.dtype != incoming.dtype or existing.shape != incoming.shape:
            raise ArtifactCorruptionError(f"artifact object id collision: {object_id}")
        # A full block is content-addressed by its KV-compatible key. As with
        # prefix-cached KV, the first successfully published value is the
        # canonical value even if a later execution shape produces different
        # floating-point routing decisions for the same logical content.
        if kind == "tail" and not np.array_equal(existing, incoming):
            raise ArtifactCorruptionError(f"artifact object id collision: {object_id}")

    def put_manifest(self, sample_id: str, manifest: dict[str, Any]) -> str:
        self._validate_id(sample_id)
        signed = self._sign_dict(manifest, "manifest_sha256")
        payload = _canonical_json(signed)
        path = self._path("manifest", sample_id)
        with self._lock:
            self._maybe_gc()
            if path.exists():
                if self.read_manifest(sample_id) != signed:
                    raise ArtifactCorruptionError(
                        f"artifact manifest id collision: {sample_id}"
                    )
                return signed["manifest_sha256"]
            self._reserve(len(payload))
            created = self._write_immutable(path, payload)
            if created:
                self._used_bytes += len(payload)
            if not created and self.read_manifest(sample_id) != signed:
                raise ArtifactCorruptionError(
                    f"artifact manifest id collision: {sample_id}"
                )
        return signed["manifest_sha256"]

    def _maybe_gc(self) -> None:
        if time.time() - self._last_gc_time >= self._gc_interval_seconds:
            self.gc(lock_held=True)

    def gc(self, *, lock_held: bool = False) -> None:
        """Remove expired manifests and their unreferenced payloads."""
        if not lock_held:
            with self._lock:
                self.gc(lock_held=True)
            return
        now = time.time()
        cutoff = now - self.ttl_seconds
        referenced: dict[str, set[str]] = {"block": set(), "tail": set()}
        for path in self.manifests_dir.glob("*.json"):
            if path.stat(follow_symlinks=False).st_mtime < cutoff:
                path.unlink(missing_ok=True)
                continue
            sample_id = path.stem
            try:
                manifest = self.read_manifest(sample_id)
            except (ArtifactCorruptionError, OSError, ValueError):
                logger.warning("Ignoring invalid artifact manifest %s", path)
                continue
            for segment in manifest.get("segments", ()):
                if not isinstance(segment, dict):
                    continue
                kind = segment.get("kind")
                object_id = segment.get("object_id")
                if kind in referenced and isinstance(object_id, str):
                    referenced[kind].add(object_id)
        for kind, directory in (("block", self.blocks_dir), ("tail", self.tails_dir)):
            for path in directory.glob("*.bin"):
                if (
                    path.stem not in referenced[kind]
                    and path.stat(follow_symlinks=False).st_mtime < cutoff
                    and not (kind == "block" and path.stem in self._retained_blocks)
                ):
                    path.unlink(missing_ok=True)
        for directory in (self.blocks_dir, self.tails_dir, self.manifests_dir):
            for path in directory.glob(".*.partial"):
                if path.stat(follow_symlinks=False).st_mtime < cutoff:
                    path.unlink(missing_ok=True)
        self._used_bytes = self._usage_bytes()
        self._last_gc_time = now
