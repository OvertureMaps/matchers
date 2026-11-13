# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Readers for the Bloom filter and count-min sketch stored in the model jar, usable from Python workers.

A Bloom filter is a compact set that answers "is this item in it?" with
occasionally wrong "yes" answers and never wrong "no" answers. A count-min
sketch is a compact table of counts that never undercounts. Spark's JVM code
wrote both into the jar as binary files. The setup stage asks the Bloom filter
which tokens are common in a given H3 cell, and the blocking step asks the
sketch how common a name token is. Python reads the files directly and
reproduces Spark's hashing, so it gets the same answers the JVM does.
"""

from __future__ import annotations

import struct

_MASK = 0xFFFFFFFF


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _MASK


def _mix_k1(k1: int) -> int:
    return (_rotl((k1 * 0xCC9E2D51) & _MASK, 15) * 0x1B873593) & _MASK


def _mix_h1(h1: int, k1: int) -> int:
    return (_rotl(h1 ^ k1, 13) * 5 + 0xE6546B64) & _MASK


def _signed(x: int) -> int:
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
    return _signed(h1)


_HEADER = 12


class BloomFilter:
    """Read-only Spark ``BloomFilter`` (version 1 binary format)."""

    def __init__(self, data: bytes):
        """Parse bytes written by ``BloomFilter.writeTo``.

        Args:
            data: The serialized filter.

        Raises:
            ValueError: The version is not 1.
        """
        version, self._num_hashes, num_words = struct.unpack_from(">iii", data)
        if version != 1:
            raise ValueError(f"unsupported BloomFilter version {version}")
        self._data = data
        # The bitset stays packed. Unpacking it into Python ints costs about 5x the memory.
        self._bit_size = num_words * 64

    def might_contain_string(self, value: str) -> bool:
        """Whether ``value`` may be in the filter. False means definitely absent."""
        raw = value.encode("utf-8")
        h1 = murmur3_32(raw, 0)
        h2 = murmur3_32(raw, h1)
        for i in range(1, self._num_hashes + 1):
            combined = _signed((h1 + i * h2) & _MASK)
            if combined < 0:
                combined = ~combined
            index = combined % self._bit_size
            (word,) = struct.unpack_from(">Q", self._data, _HEADER + 8 * (index >> 6))
            if not (word >> (index & 63)) & 1:
                return False
        return True


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
        self._data = data
        # The table stays packed, like the Bloom filter's bitset.
        self._table_start = struct.calcsize(">iqii") + 8 * self._depth

    def estimate_count(self, value: str) -> int:
        """Upper bound of how many times ``value`` was added."""
        raw = value.encode("utf-8")
        h1 = murmur3_32(raw, 0)
        h2 = murmur3_32(raw, h1)
        return min(
            struct.unpack_from(">q", self._data, self._cell(i, h1, h2))[0]
            for i in range(self._depth)
        )

    def _cell(self, row: int, h1: int, h2: int) -> int:
        column = abs(_signed((h1 + row * h2) & _MASK)) % self._width
        return self._table_start + 8 * (row * self._width + column)
