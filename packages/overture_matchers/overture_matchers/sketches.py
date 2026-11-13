# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Reader for the Spark count-min sketches stored in the model jar, usable from Python workers.

A count-min sketch is a compact table of counts that never undercounts. Spark's
JVM code wrote the sketches into the jar as binary files, and Python workers
read them directly and reproduce Spark's hashing so they get the same answers
the JVM does.
"""

from __future__ import annotations

import struct

import numpy as np

_MASK = 0xFFFFFFFF


def _rotate_left_32(x: int, bits: int) -> int:
    return ((x << bits) | (x >> (32 - bits))) & _MASK


def _mix_k1(k1: int) -> int:
    return (_rotate_left_32((k1 * 0xCC9E2D51) & _MASK, 15) * 0x1B873593) & _MASK


def _mix_h1(h1: int, k1: int) -> int:
    return (_rotate_left_32(h1 ^ k1, 13) * 5 + 0xE6546B64) & _MASK


def _to_signed_int32(x: int) -> int:
    return x - (1 << 32) if x & 0x80000000 else x


def murmur3_32(data: bytes, seed: int) -> int:
    """Spark's ``Murmur3_x86_32.hashUnsafeBytes``, as a signed 32-bit int.

    Spark hashes each tail byte as its own sign-extended block, which differs
    from standard Murmur3, so ``mmh3`` gives other values.
    """
    seed &= _MASK
    aligned = len(data) - len(data) % 4
    h1 = seed
    for (word,) in struct.iter_unpack("<I", data[:aligned]):
        h1 = _mix_h1(h1, _mix_k1(word))
    for byte in data[aligned:]:
        h1 = _mix_h1(h1, _mix_k1(byte - 256 if byte > 127 else byte & _MASK))
    h1 ^= len(data)
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & _MASK
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & _MASK
    h1 ^= h1 >> 16
    return _to_signed_int32(h1)


class CountMinSketch:
    """Read-only Spark ``CountMinSketch`` (version 1 binary format)."""

    def __init__(self, data: bytes):
        """Parse bytes written by ``CountMinSketch.writeTo``.

        Args:
            data: The serialized sketch.

        Raises:
            ValueError: The version is not 1.
        """
        version, _, self._depth, self._width = struct.unpack_from(">iqii", data)
        if version != 1:
            raise ValueError(f"unsupported CountMinSketch version {version}")
        # The table stays big-endian and unconverted, so it costs no more memory than the bytes.
        table_start = struct.calcsize(">iqii") + 8 * self._depth
        self._table = np.frombuffer(data, dtype=">i8", offset=table_start).reshape(
            self._depth, self._width
        )

    def estimate_count(self, value: str) -> int:
        """Upper bound of how many times ``value`` was added."""
        raw = value.encode("utf-8")
        h1 = murmur3_32(raw, 0)
        h2 = murmur3_32(raw, h1)
        return min(
            int(
                self._table[
                    i, abs(_to_signed_int32((h1 + i * h2) & _MASK)) % self._width
                ]
            )
            for i in range(self._depth)
        )
