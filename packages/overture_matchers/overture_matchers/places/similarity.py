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


def _to_utf16_code_units(text: str) -> str | list[int]:
    """``text`` itself if it is all BMP, else its UTF-16 code units."""
    if text.isascii() or max(text) <= "\uffff":
        return text
    return memoryview(text.encode("utf-16-le")).cast("H").tolist()


def _to_comparable_utf16_sequences(left: str, right: str) -> tuple[Sequence, Sequence]:
    """Both strings as sequences of UTF-16 units of the same element type."""
    left_units, right_units = _to_utf16_code_units(left), _to_utf16_code_units(right)
    if isinstance(left_units, str) and isinstance(right_units, str):
        return left_units, right_units
    as_unit_list = lambda units: (
        units if isinstance(units, list) else [ord(character) for character in units]
    )
    return as_unit_list(left_units), as_unit_list(right_units)


def compute_normalized_similarity(left: str | None, right: str | None) -> float:
    """One minus the edit distance over the longer length. Null is empty; two empties give 1.0."""
    left_units, right_units = _to_comparable_utf16_sequences(left or "", right or "")
    longest = max(len(left_units), len(right_units))
    if longest == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(left_units, right_units) / longest


def _compute_shared_prefix_ratio(left_units: Sequence, right_units: Sequence) -> float:
    shortest = min(len(left_units), len(right_units))
    shared = 0
    while shared < shortest and left_units[shared] == right_units[shared]:
        shared += 1
    return shared / shortest


def get_string_prefix_score(left: str | None, right: str | None) -> float:
    """Shared prefix length over the shorter length, or 0.0 if either is null or empty."""
    if not left or not right:
        return 0.0
    return _compute_shared_prefix_ratio(*_to_comparable_utf16_sequences(left, right))


def get_string_suffix_score(left: str | None, right: str | None) -> float:
    """``get_string_prefix_score`` of the strings reversed by UTF-16 unit."""
    if not left or not right:
        return 0.0
    left_units, right_units = _to_comparable_utf16_sequences(left, right)
    return _compute_shared_prefix_ratio(left_units[::-1], right_units[::-1])


# simplification: a per-process memo of token comparisons, bounded by its size.
@lru_cache(maxsize=1 << 16)
def _are_tokens_similar(left_token: str | None, right_token: str | None) -> bool:
    return compute_normalized_similarity(left_token, right_token) >= TOKEN_THRESHOLD


def compute_fuzzy_jaccard_similarity(
    left_tokens: list | None, right_tokens: list | None
) -> float:
    """Jaccard over tokens, where each left token claims the first unclaimed similar right token.

    Two empty lists give 1.0 and one empty list gives 0.0. Null is empty.
    """
    left, right = left_tokens or [], right_tokens or []
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    claimed_indices: set[int] = set()
    for token in left:
        for right_index, other_token in enumerate(right):
            if right_index not in claimed_indices and _are_tokens_similar(
                token, other_token
            ):
                claimed_indices.add(right_index)
                break
    shared = len(claimed_indices)
    return shared / (len(left) + len(right) - shared)


def compute_shared_token_ratio(
    left_tokens: list | None, right_tokens: list | None
) -> float:
    """Distinct right tokens claimed by similar left tokens, over the shorter list's length.

    A right token, once claimed, is skipped wherever it repeats. Null or empty gives 0.0.
    """
    if not left_tokens or not right_tokens:
        return 0.0
    claimed_tokens: set[str | None] = set()
    for token in left_tokens:
        for other_token in right_tokens:
            if other_token not in claimed_tokens and _are_tokens_similar(
                token, other_token
            ):
                claimed_tokens.add(other_token)
                break
    return len(claimed_tokens) / min(len(left_tokens), len(right_tokens))


def is_token_subset(left_tokens: list | None, right_tokens: list | None) -> bool:
    """Whether every token of one side has a similar token on the other. Null or empty is False."""
    if not left_tokens or not right_tokens:
        return False

    def are_all_tokens_matched(source_tokens: list, target_tokens: list) -> bool:
        return all(
            any(
                _are_tokens_similar(source_token, target_token)
                for target_token in target_tokens
            )
            for source_token in source_tokens
        )

    return are_all_tokens_matched(left_tokens, right_tokens) or are_all_tokens_matched(
        right_tokens, left_tokens
    )


def _compute_utf16_string_hash(text: str) -> int:
    hash_value = 0
    for code_unit in memoryview(text.encode("utf-16-le")).cast("H"):
        hash_value = (31 * hash_value + code_unit) & _INT_MASK
    return hash_value


def _get_hash_trie_sort_key(token: str) -> tuple[int, ...]:
    """Sort key for a hash trie keyed on 5-bit chunks of an improved string hash.

    The trie branches on the lowest chunk first and visits each level in chunk
    order.
    """
    hash_value = _compute_utf16_string_hash(token)
    hash_value = (hash_value + (~(hash_value << 9) & _INT_MASK)) & _INT_MASK
    hash_value ^= hash_value >> 14
    hash_value = (hash_value + (hash_value << 4)) & _INT_MASK
    hash_value ^= hash_value >> 10
    return tuple((hash_value >> shift) & 0x1F for shift in range(0, 32, 5))


def _order_distinct_tokens(tokens: Iterable[str]) -> list[str]:
    """Distinct tokens in the iteration order the served model's sums use.

    Up to four tokens keep insertion order. Larger sets follow hash trie order.
    """
    distinct = list(dict.fromkeys(tokens))
    return (
        distinct
        if len(distinct) <= 4
        else sorted(distinct, key=_get_hash_trie_sort_key)
    )


def get_tfidf_overlap(
    left_tokens: list | None,
    right_tokens: list | None,
    compute_idf: Callable[[str], float],
) -> float:
    """Summed IDF of the shared distinct tokens over the smaller distinct count.

    The sum runs in ``_order_distinct_tokens``, so float rounding matches the
    served model's. Null is empty, and an empty side gives 0.0.
    """
    left_set = _order_distinct_tokens(
        token for token in left_tokens or () if token is not None
    )
    right_set = {token for token in right_tokens or () if token is not None}
    smaller = min(len(left_set), len(right_set))
    if smaller == 0:
        return 0.0
    shared = [token for token in left_set if token in right_set]
    return sum(map(compute_idf, shared)) / smaller


def compute_residual_idf_max(
    left_tokens: list | None,
    right_tokens: list | None,
    compute_idf: Callable[[str], float],
) -> float:
    """Largest IDF among tokens with no similar token on the other side, or -1.0 if none."""
    left, right = left_tokens or [], right_tokens or []
    residual = {
        token
        for token in left
        if not any(_are_tokens_similar(token, other) for other in right)
    } | {
        token
        for token in right
        if not any(_are_tokens_similar(token, other) for other in left)
    }
    residual.discard(None)
    return max(map(compute_idf, residual), default=-1.0)


def compute_website_path_similarity(
    left_urls: list | None, right_urls: list | None
) -> float:
    """Best path-segment Jaccard over same-host URL pairs, or -1.0 if no host is shared.

    A URL's host is the text before its first ``/``, and its path is the set
    of non-empty segments after it. Two empty paths give 1.0, one gives 0.0.
    """
    if not left_urls or not right_urls:
        return -1.0

    def parse_website_urls(urls: list) -> list[tuple[str, set[str]]]:
        parsed = []
        for url in (entry for entry in urls if entry is not None):
            host, _, path = url.partition("/")
            parsed.append((host, {segment for segment in path.split("/") if segment}))
        return parsed

    best = -1.0
    for left_host, left_segments in parse_website_urls(left_urls):
        for right_host, right_segments in parse_website_urls(right_urls):
            if left_host != right_host:
                continue
            if not left_segments and not right_segments:
                score = 1.0
            elif not left_segments or not right_segments:
                score = 0.0
            else:
                score = len(left_segments & right_segments) / len(
                    left_segments | right_segments
                )
            best = max(best, score)
    return best
