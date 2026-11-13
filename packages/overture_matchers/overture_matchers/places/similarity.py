# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""String and token similarity for place-pair scoring.

Lengths, edit distances and prefixes count UTF-16 code units, as the model
was served. Code points and UTF-16 units differ only on characters outside
the Basic Multilingual Plane, such as emoji.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

TOKEN_THRESHOLD = 0.9
_INT_MASK = 0xFFFFFFFF


def _units(text: str) -> str | list[int]:
    """``text`` itself if it is all BMP, else its UTF-16 code units."""
    if text.isascii() or max(text) <= "\uffff":
        return text
    return memoryview(text.encode("utf-16-le")).cast("H").tolist()


def _comparable(a: str, b: str) -> tuple[Sequence, Sequence]:
    """Both strings as sequences of UTF-16 units of the same element type."""
    ua, ub = _units(a), _units(b)
    if isinstance(ua, str) and isinstance(ub, str):
        return ua, ub
    as_list = lambda u: u if isinstance(u, list) else [ord(c) for c in u]
    return as_list(ua), as_list(ub)


def normalized_similarity(left: str | None, right: str | None) -> float:
    """One minus the edit distance over the longer length. Null is empty; two empties give 1.0."""
    a, b = _comparable(left or "", right or "")
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / longest


def _shared_prefix_ratio(a: Sequence, b: Sequence) -> float:
    shortest = min(len(a), len(b))
    shared = 0
    while shared < shortest and a[shared] == b[shared]:
        shared += 1
    return shared / shortest


def prefix_score(left: str | None, right: str | None) -> float:
    """Shared prefix length over the shorter length, or 0.0 if either is null or empty."""
    if not left or not right:
        return 0.0
    return _shared_prefix_ratio(*_comparable(left, right))


def suffix_score(left: str | None, right: str | None) -> float:
    """``prefix_score`` of the strings reversed by UTF-16 unit."""
    if not left or not right:
        return 0.0
    a, b = _comparable(left, right)
    return _shared_prefix_ratio(a[::-1], b[::-1])


# simplification: a per-process memo of token comparisons, bounded by its size.
@lru_cache(maxsize=1 << 16)
def _similar(a: str | None, b: str | None) -> bool:
    return normalized_similarity(a, b) >= TOKEN_THRESHOLD


def fuzzy_jaccard(left: list | None, right: list | None) -> float:
    """Jaccard over tokens, where each left token claims the first unclaimed similar right token.

    Two empty lists give 1.0 and one empty list gives 0.0. Null is empty.
    """
    a, b = left or [], right or []
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    claimed: set[int] = set()
    for token in a:
        for j, other in enumerate(b):
            if j not in claimed and _similar(token, other):
                claimed.add(j)
                break
    shared = len(claimed)
    return shared / (len(a) + len(b) - shared)


def shared_token_ratio(left: list | None, right: list | None) -> float:
    """Distinct right tokens claimed by similar left tokens, over the shorter list's length.

    A right token, once claimed, is skipped wherever it repeats. Null or empty gives 0.0.
    """
    if not left or not right:
        return 0.0
    claimed: set[str | None] = set()
    for token in left:
        for other in right:
            if other not in claimed and _similar(token, other):
                claimed.add(other)
                break
    return len(claimed) / min(len(left), len(right))


def is_token_subset(left: list | None, right: list | None) -> bool:
    """Whether every token of one side has a similar token on the other. Null or empty is False."""
    if not left or not right:
        return False

    def covered(source: list, target: list) -> bool:
        return all(any(_similar(t, o) for o in target) for t in source)

    return covered(left, right) or covered(right, left)


def _java_string_hash(text: str) -> int:
    h = 0
    for unit in memoryview(text.encode("utf-16-le")).cast("H"):
        h = (31 * h + unit) & _INT_MASK
    return h


def _hash_trie_order(token: str) -> tuple[int, ...]:
    """Sort key for a hash trie keyed on 5-bit chunks of an improved string hash.

    The trie branches on the lowest chunk first and visits each level in chunk
    order.
    """
    h = _java_string_hash(token)
    h = (h + (~(h << 9) & _INT_MASK)) & _INT_MASK
    h ^= h >> 14
    h = (h + (h << 4)) & _INT_MASK
    h ^= h >> 10
    return tuple((h >> shift) & 0x1F for shift in range(0, 32, 5))


def _set_order(tokens: Iterable[str]) -> list[str]:
    """Distinct tokens in the iteration order the served model's sums use.

    Up to four tokens keep insertion order. Larger sets follow hash trie order.
    """
    distinct = list(dict.fromkeys(tokens))
    return distinct if len(distinct) <= 4 else sorted(distinct, key=_hash_trie_order)


def tfidf_overlap(
    left: list | None, right: list | None, idf: Callable[[str], float]
) -> float:
    """Summed IDF of the shared distinct tokens over the smaller distinct count.

    The sum runs in ``_set_order``, so float rounding matches the served
    model's. Null is empty, and an empty side gives 0.0.
    """
    left_set = _set_order(t for t in left or () if t is not None)
    right_set = {t for t in right or () if t is not None}
    smaller = min(len(left_set), len(right_set))
    if smaller == 0:
        return 0.0
    shared = [t for t in left_set if t in right_set]
    return sum(map(idf, shared)) / smaller


def residual_idf_max(
    left: list | None, right: list | None, idf: Callable[[str], float]
) -> float:
    """Largest IDF among tokens with no similar token on the other side, or -1.0 if none."""
    a, b = left or [], right or []
    residual = {t for t in a if not any(_similar(t, o) for o in b)} | {
        t for t in b if not any(_similar(t, o) for o in a)
    }
    residual.discard(None)
    return max(map(idf, residual), default=-1.0)


def website_path_similarity(left: list | None, right: list | None) -> float:
    """Best path-segment Jaccard over same-host URL pairs, or -1.0 if no host is shared.

    A URL's host is the text before its first ``/``, and its path is the set
    of non-empty segments after it. Two empty paths give 1.0, one gives 0.0.
    """
    if not left or not right:
        return -1.0

    def parse(urls: list) -> list[tuple[str, set[str]]]:
        parsed = []
        for url in (u for u in urls if u is not None):
            host, _, path = url.partition("/")
            parsed.append((host, {s for s in path.split("/") if s}))
        return parsed

    best = -1.0
    for host_a, path_a in parse(left):
        for host_b, path_b in parse(right):
            if host_a != host_b:
                continue
            if not path_a and not path_b:
                score = 1.0
            elif not path_a or not path_b:
                score = 0.0
            else:
                score = len(path_a & path_b) / len(path_a | path_b)
            best = max(best, score)
    return best
