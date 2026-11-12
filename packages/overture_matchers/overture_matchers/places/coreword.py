# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Core-word background stripping: drops locally common name tokens before name scoring.

A token is background when the model jar's Bloom filter holds
``<res>:<cell>:<token>`` for the record's H3 cell at resolution 9, 7, 5 or 3.
Only functions here run on Spark workers, which read the filter out of the
model jar that ``sc.addFile`` shipped to them.
"""

from __future__ import annotations

import zipfile
from functools import cache

from pyspark import SparkFiles
from pyspark.sql import Column
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from overture_matchers.sketches import BloomFilter

FILTER_RESOURCE = "config/places/coreword_bg_filter.bin"
# Finest to coarsest. The coarser cells are the H3 parents of the res-9 cell.
RESOLUTIONS = (9, 7, 5, 3)
_H3_RES_SHIFT = 52
_H3_DIGIT_BITS = 3
_H3_MAX_RES = 15
_UINT64 = (1 << 64) - 1


def h3_parent(cell: int, resolution: int) -> int:
    """H3 parent of ``cell`` at a coarser ``resolution``.

    Bit arithmetic keeps workers free of an H3 library.
    """
    unused_digits = (1 << (_H3_MAX_RES - resolution) * _H3_DIGIT_BITS) - 1
    parent = cell & ~(0xF << _H3_RES_SHIFT) & ~unused_digits
    return parent | (resolution << _H3_RES_SHIFT) | unused_digits


@cache
def _load_filter(jar_name: str) -> BloomFilter:
    with zipfile.ZipFile(SparkFiles.get(jar_name)) as jar:
        return BloomFilter(jar.read(FILTER_RESOURCE))


def strip_background(
    tokens: list[str] | None, h3_res9: int | None, bloom: BloomFilter
) -> list[str]:
    """Drop background tokens, keeping all tokens if every one is background.

    Args:
        tokens: Name tokens.
        h3_res9: The record's H3 resolution 9 cell, or None without geometry.
            Without a cell nothing counts as background.
        bloom: The background filter.

    Returns:
        The kept tokens, in order.
    """
    if not tokens:
        return []
    keys = (
        [f"{res}:{h3_parent(h3_res9, res) & _UINT64}:" for res in RESOLUTIONS]
        if h3_res9 is not None
        else []
    )
    kept = [
        t
        for t in tokens
        if t and not any(bloom.might_contain_string(k + t) for k in keys)
    ]
    return kept or tokens


def strip_background_udf(jar_name: str, tokens: Column, h3_res9: Column) -> Column:
    """Column of ``strip_background`` over a token array and an H3 cell column.

    Args:
        jar_name: File name of the model jar that ``sc.addFile`` shipped.
        tokens: Array-of-string column.
        h3_res9: Long column.
    """

    def strip(tokens, cell):
        return strip_background(tokens, cell, _load_filter(jar_name))

    return F.udf(strip, ArrayType(StringType()))(tokens, h3_res9)
