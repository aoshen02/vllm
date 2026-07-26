# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable execution-artifact objects in a shared-memory filesystem."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import mmap
import os
import stat
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

import regex as re

from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
    ArtifactPutResult,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SAFE_STORE_ID = re.compile(r"^[a-f0-9]{32,64}$")


def make_shm_store_id(instance_id: str, dp_rank: int) -> str:
    """Return the process-group-stable SHM store identity."""
    return hashlib.sha256(f"{instance_id}:{dp_rank}".encode()).hexdigest()[:32]


class LocalSharedMemoryArtifactReader:
    """Read self-describing artifact objects from a local SHM store."""

    backend_name = "shm"

    def __init__(self, root: str, store_id: str) -> None:
        if not _SAFE_STORE_ID.fullmatch(store_id):
            raise ValueError(f"invalid artifact store id: {store_id!r}")
        self.root = Path(root) / store_id
        self.store_id = store_id
        self.objects_dir = self.root / "objects"

    @staticmethod
    def _object_id(key: str) -> str:
        if not key or "\x00" in key:
            raise ValueError("artifact object key must be a non-empty string")
        return hashlib.sha256(key.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.objects_dir / f"{self._object_id(key)}.bin"

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

    def exists(self, keys: list[str]) -> list[bool]:
        results: list[bool] = []
        for key in keys:
            path = self._path(key)
            try:
                fd = self._open_regular_file(path)
            except FileNotFoundError:
                results.append(False)
                continue
            else:
                os.close(fd)
                results.append(True)
        return results

    def get(self, keys: list[str]) -> list[bytes]:
        payloads: list[bytes] = []
        for key in keys:
            path = self._path(key)
            try:
                fd = self._open_regular_file(path)
            except FileNotFoundError as error:
                raise ArtifactNotFoundError(
                    f"artifact object does not exist: {key}"
                ) from error
            try:
                file_size = os.fstat(fd).st_size
                payload = bytearray(file_size)
                view = memoryview(payload)
                offset = 0
                while offset < file_size:
                    count = os.readv(fd, [view[offset:]])
                    if count <= 0:
                        raise ArtifactCorruptionError(
                            f"artifact object is truncated: {key}"
                        )
                    offset += count
                view.release()
                payloads.append(bytes(payload))
            finally:
                os.close(fd)
        return payloads

    def close(self) -> None:
        """Readers own no external resources."""


class LocalSharedMemoryArtifactStore(LocalSharedMemoryArtifactReader):
    """Single-writer immutable artifact store in `/dev/shm`."""

    def __init__(
        self,
        root: str,
        instance_id: str,
        dp_rank: int,
        *,
        max_bytes: int,
        ttl_seconds: int,
    ) -> None:
        super().__init__(root, make_shm_store_id(instance_id, dp_rank))
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._used_bytes = 0
        self._last_gc_time = 0.0
        self._gc_interval_seconds = min(max(ttl_seconds / 4, 1.0), 60.0)

        root_path = Path(root)
        self._prepare_directory(root_path)
        self._gc_stale_store_dirs(root_path)
        self._prepare_directory(self.root)
        self._prepare_directory(self.objects_dir)
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
            metadata = json.dumps(
                {"ttl_seconds": self.ttl_seconds},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.write(fd, metadata) != len(metadata):
                raise OSError("short write while recording artifact store metadata")
            os.fsync(fd)
        except Exception:
            os.close(fd)
            raise
        return fd

    @classmethod
    def _read_store_ttl(cls, lock_path: Path, fallback: int) -> int:
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

    @staticmethod
    def _stale_store_entries(
        store_root: Path,
    ) -> tuple[list[Path], list[Path], float] | None:
        try:
            root_stat = store_root.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
            return None

        files: list[Path] = []
        newest_mtime = root_stat.st_mtime
        try:
            entries = list(os.scandir(store_root))
        except FileNotFoundError:
            return None
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            newest_mtime = max(newest_mtime, entry_stat.st_mtime)
            if entry.name == ".writer.lock":
                if not stat.S_ISREG(entry_stat.st_mode):
                    return None
                continue
            if entry.name != "objects" or not stat.S_ISDIR(entry_stat.st_mode):
                return None
            if entry_stat.st_uid != os.getuid():
                return None
            for child in os.scandir(entry.path):
                child_stat = child.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(child_stat.st_mode)
                    or child_stat.st_uid != os.getuid()
                ):
                    return None
                newest_mtime = max(newest_mtime, child_stat.st_mtime)
                files.append(Path(child.path))
        return files, [store_root / "objects"], newest_mtime

    def _gc_stale_store_dirs(self, root: Path) -> None:
        now = time.time()
        for entry in os.scandir(root):
            if entry.name == self.store_id or not _SAFE_STORE_ID.fullmatch(entry.name):
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

    def _usage_bytes(self) -> int:
        return sum(
            entry.stat(follow_symlinks=False).st_size
            for entry in os.scandir(self.objects_dir)
            if entry.is_file(follow_symlinks=False)
        )

    def _reserve(self, additional_bytes: int) -> None:
        if self._used_bytes + additional_bytes > self.max_bytes:
            self.gc(lock_held=True)
            if self._used_bytes + additional_bytes > self.max_bytes:
                raise ArtifactCapacityError(
                    "artifact SHM capacity exceeded: "
                    f"used={self._used_bytes}, requested={additional_bytes}, "
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
    def _write_immutable(path: Path, payload: bytes) -> bool:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        fd: int | None = None
        mapping: mmap.mmap | None = None
        try:
            fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.posix_fallocate(fd, 0, len(payload))
            except OSError as error:
                if error.errno == errno.ENOSPC:
                    raise ArtifactCapacityError(
                        f"artifact SHM could not reserve {len(payload)} bytes"
                    ) from error
                if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
                os.ftruncate(fd, len(payload))
            if payload:
                mapping = mmap.mmap(
                    fd,
                    len(payload),
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
                mapping[:] = payload
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

    def _put_one(self, obj: ArtifactObject) -> None:
        path = self._path(obj.key)
        if path.exists():
            existing = self.get([obj.key])[0]
            if existing != obj.payload:
                raise ArtifactCorruptionError(
                    f"artifact object key collision: {obj.key}"
                )
            return
        self._reserve(len(obj.payload))
        created = self._write_immutable(path, obj.payload)
        if created:
            self._used_bytes += len(obj.payload)
            return
        existing = self.get([obj.key])[0]
        if existing != obj.payload:
            raise ArtifactCorruptionError(f"artifact object key collision: {obj.key}")

    def put(self, objects: list[ArtifactObject]) -> list[ArtifactPutResult]:
        results: list[ArtifactPutResult] = []
        with self._lock:
            self._maybe_gc()
            for obj in objects:
                try:
                    self._put_one(obj)
                except Exception as error:
                    results.append(
                        ArtifactPutResult(
                            key=obj.key,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                else:
                    results.append(ArtifactPutResult(key=obj.key))
        return results

    def _maybe_gc(self) -> None:
        if time.time() - self._last_gc_time >= self._gc_interval_seconds:
            self.gc(lock_held=True)

    def gc(self, *, lock_held: bool = False) -> None:
        if not lock_held:
            with self._lock:
                self.gc(lock_held=True)
            return
        cutoff = time.time() - self.ttl_seconds
        for path in self.objects_dir.glob("*.bin"):
            if path.stat(follow_symlinks=False).st_mtime < cutoff:
                path.unlink(missing_ok=True)
        for path in self.objects_dir.glob(".*.partial"):
            if path.stat(follow_symlinks=False).st_mtime < cutoff:
                path.unlink(missing_ok=True)
        self._used_bytes = self._usage_bytes()
        self._last_gc_time = time.time()

    def close(self) -> None:
        fd = getattr(self, "_writer_lock_fd", None)
        if fd is not None:
            self._writer_lock_fd = None
            os.close(fd)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
