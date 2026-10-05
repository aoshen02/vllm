# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental ``numpy2base64`` of routed-experts (R3) chunks."""

import io

import numpy as np
import pybase64

from vllm.logprobs import _Base64Stream


def npy_header(dtype: np.dtype, shape: tuple[int, ...]) -> bytes:
    """The ``.npy`` header ``np.save`` writes for a C-ordered array (format
    version 1.0, which ``np.save`` picks for any header below 64 KiB)."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.getvalue()


class RoutedExpertsNpyBase64:
    """Builds ``numpy2base64(np.concatenate(chunks, axis=0))`` incrementally.

    The ``.npy`` data is the C-order bytes of the chunks, preceded by a
    header that depends on the final row count. Base64 works in 3-byte
    groups, so the data stream is encoded from the byte offset that a
    header of ``max_rows_hint`` rows would leave: the first
    ``(3 - len(header) % 3) % 3`` data bytes are held back and encoded with
    the header at the end. If the final header length has a different
    residue mod 3 (a much longer shape string), :meth:`parts` returns None
    and :meth:`array` gives the data for a regular ``numpy2base64``.

    Holds about 1.33x the data (base64) plus at most one 768 KiB segment of
    raw bytes; the chunks themselves are not kept.
    """

    def __init__(self, max_rows_hint: int) -> None:
        self.max_rows_hint = max(int(max_rows_hint), 0)
        self.dtype: np.dtype | None = None
        self.row_shape: tuple[int, ...] = ()
        self.rows = 0
        self._header_mod = 0
        self._hold_bytes = 0
        self._held = b""
        self._stream = _Base64Stream()

    @property
    def has_data(self) -> bool:
        """Whether any chunk (possibly with zero rows) was appended."""
        return self.dtype is not None

    def try_append(self, chunk: np.ndarray) -> bool:
        """Encode ``chunk``; False if it cannot be encoded byte-identically:
        dtype or row shape differs from the first chunk, non-native byte
        order (concatenation normalizes it), or not C-contiguous (the
        concatenation could be Fortran-ordered). The caller then keeps
        plain chunks, as without the encoder."""
        if chunk.dtype.byteorder not in ("=", "|") or (
            chunk.ndim > 1 and not chunk.flags.c_contiguous
        ):
            return False
        if self.dtype is None:
            self.dtype = chunk.dtype
            self.row_shape = tuple(chunk.shape[1:])
            hint = max(self.max_rows_hint, len(chunk))
            header_len = len(npy_header(self.dtype, (hint, *self.row_shape)))
            self._header_mod = header_len % 3
            self._hold_bytes = (3 - self._header_mod) % 3
        elif chunk.dtype != self.dtype or tuple(chunk.shape[1:]) != self.row_shape:
            return False
        data = np.ascontiguousarray(chunk).reshape(-1).view(np.uint8)
        missing = self._hold_bytes - len(self._held)
        if missing > 0 and len(data):
            self._held += data[:missing].tobytes()
            data = data[missing:]
        if len(data):
            self._stream.write(data)
        self.rows += len(chunk)
        return True

    def _shape(self) -> tuple[int, ...]:
        return (self.rows, *self.row_shape)

    def parts(self) -> list[bytes] | None:
        """Base64 text parts of ``numpy2base64`` of all appended rows."""
        assert self.dtype is not None
        header = npy_header(self.dtype, self._shape())
        if len(header) % 3 != self._header_mod:
            return None
        head = pybase64.b64encode(header + self._held)
        return [head, *self._stream.parts()]

    def array(self) -> np.ndarray:
        """The concatenated rows (decoded; for fallbacks)."""
        assert self.dtype is not None
        data = self._held + self._stream.decode()
        return np.frombuffer(data, dtype=self.dtype).reshape(self._shape()).copy()
