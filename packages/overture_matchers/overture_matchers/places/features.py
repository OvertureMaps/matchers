# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Place-pair features for blocking and scoring.

Each feature reads the two sides of a pair as columns: the left side under
a column's own name and the right side under ``base_<name>``. Blocking runs
on every spatial pair, so its features stay built-in expressions, never
Python UDFs. Scoring runs on far fewer pairs. Its cheap features are
expressions too, and the string and token features come from one Python UDF,
so each pair crosses to Python once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from pyspark.sql import Column
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from overture_matchers.places import similarity

if TYPE_CHECKING:
    from pyspark.sql import Row

# Supplied by the spatial join rather than computed from the two sides.
GEO_DISTANCE = "geo_distance"
NAME_IDF = "name_idf"
_UNAVAILABLE = -1.0
# Phone numbers of unknown country match on this many trailing digits.
_PHONE_SUFFIX_LENGTH = 6


def name_similarity() -> Column:
    """One minus the edit distance over the longer length, with 1.0 for two empty names."""
    left = F.coalesce(F.col("name_clean"), F.lit(""))
    right = F.coalesce(F.col("base_name_clean"), F.lit(""))
    longest = F.greatest(F.length(left), F.length(right))
    return F.when(longest == 0, F.lit(1.0)).otherwise(
        1.0 - F.levenshtein(left, right) / longest
    )


def _equal_or_unavailable(left: str, right: str, equal: Column | None = None) -> Column:
    """1.0 or 0.0 for present values, -1.0 when either side is null or empty."""
    left_col, right_col = F.col(left), F.col(right)
    unavailable = (
        left_col.isNull() | right_col.isNull() | (left_col == "") | (right_col == "")
    )
    same = equal if equal is not None else left_col == right_col
    return (
        F.when(unavailable, F.lit(_UNAVAILABLE))
        .when(same, F.lit(1.0))
        .otherwise(F.lit(0.0))
    )


def phone_number_similarity_availability() -> Column:
    """Phone match, from the whole number or, without both countries, its last digits."""
    left, right = F.col("normalized_phone"), F.col("base_normalized_phone")
    country_unknown = (
        F.col("country").isNull()
        | F.col("base_country").isNull()
        | (F.col("country") == "")
        | (F.col("base_country") == "")
    )
    suffixes_match = left.substr(-_PHONE_SUFFIX_LENGTH, _PHONE_SUFFIX_LENGTH) == (
        right.substr(-_PHONE_SUFFIX_LENGTH, _PHONE_SUFFIX_LENGTH)
    )
    return _equal_or_unavailable(
        "normalized_phone",
        "base_normalized_phone",
        F.when(country_unknown, suffixes_match).otherwise(left == right),
    )


def house_number_similarity_availability() -> Column:
    """House number match."""
    return _equal_or_unavailable("house_number", "base_house_number")


def name_tfidf() -> Column:
    """IDF weight of the shared name tokens over the smaller token count.

    Reads the ``name_idf`` maps of both sides.
    """
    left, right = F.col(NAME_IDF), F.col(f"base_{NAME_IDF}")
    shared = F.array_intersect(F.map_keys(left), F.map_keys(right))
    weight = F.aggregate(shared, F.lit(0.0), lambda total, token: total + left[token])
    smaller = F.least(F.size(left), F.size(right))
    return F.when(smaller > 0, weight / smaller).otherwise(F.lit(0.0))


BLOCKING_FEATURES = {
    GEO_DISTANCE: lambda: F.col(GEO_DISTANCE),
    "name_similarity": name_similarity,
    "phone_number_similarity_availability": phone_number_similarity_availability,
    "house_number_similarity_availability": house_number_similarity_availability,
    "name_tfidf": name_tfidf,
}
# The columns each side of a pair carries for a feature. Every pair also has
# GEO_DISTANCE.
FEATURE_INPUTS = {
    GEO_DISTANCE: (),
    "name_similarity": ("name_clean",),
    "phone_number_similarity_availability": ("normalized_phone", "country"),
    "house_number_similarity_availability": ("house_number",),
    "name_tfidf": (NAME_IDF,),
}


def phone_last_digits_match() -> Column:
    """1.0 if the last four phone digits match, else 0.0. Null when either phone is missing."""
    left, right = F.col("normalized_phone"), F.col("base_normalized_phone")
    missing = left.isNull() | right.isNull() | (left == "") | (right == "")
    return F.when(missing, F.lit(None).cast("double")).otherwise(
        (left.substr(-4, 4) == right.substr(-4, 4)).cast("double")
    )


def website_domain_match() -> Column:
    """1 if the two sides share a website domain, 0 if not, -1 if either has none."""
    left, right = F.col("website_domains"), F.col("base_website_domains")
    missing = (
        left.isNull() | right.isNull() | (F.size(left) <= 0) | (F.size(right) <= 0)
    )
    return (
        F.when(missing, F.lit(-1))
        .when(F.arrays_overlap(left, right), F.lit(1))
        .otherwise(F.lit(0))
    )


def geo_distance() -> Column:
    """Great-circle distance between the two places in meters."""
    return F.expr(
        "ST_DistanceSphere(ST_Point(longitude, latitude), "
        "ST_Point(base_longitude, base_latitude))"
    )


SCORING_EXPRESSIONS = {
    "phone_number_similarity_availability": phone_number_similarity_availability,
    "house_number_similarity_availability": house_number_similarity_availability,
    "phone_last_digits_match": phone_last_digits_match,
    "website_domain_match": website_domain_match,
    GEO_DISTANCE: geo_distance,
}

# Columns scoring adds to each side before the pairwise UDF runs.
NAME_H3_CELL = "name_h3_cell"
NAME_H3_RESOLUTION = 5
NAME_VECTOR = "name_vector"
TAXONOMY_VECTOR = "taxonomy_vector"
PAIRWISE_INPUTS = (
    "name",
    "name_clean",
    "name_clean_tokens",
    "name_clean_stripped",
    "name_clean_stripped_tokens",
    "address_clean",
    "address_clean_tokens",
    "website_urls",
    NAME_H3_CELL,
    NAME_VECTOR,
    TAXONOMY_VECTOR,
)
NAME_COSINE = "name_cosine_similarity"
TAXONOMY_COSINE = "taxonomy_cosine_similarity"
PAIRWISE_FEATURES = StructType(
    [
        StructField(
            name, IntegerType() if name == "address_token_subset" else DoubleType()
        )
        for name in (
            "name_similarity",
            "name_jaccard_similarity",
            "name_strict_prefix_score",
            "name_strict_suffix_score",
            "name_strict_shared_token_ratio",
            "name_strict_token_subset",
            "address_similarity_availability",
            "address_prefix_score",
            "address_suffix_score",
            "address_jaccard_similarity",
            "address_shared_token_ratio",
            "address_token_subset",
            "name_similarity_stripped",
            "name_jaccard_similarity_stripped",
            "name_prefix_score_stripped",
            "name_suffix_score_stripped",
            "website_path_similarity",
            "name_tfidf",
            "residual_idf_max",
            "address_tfidf",
            "name_tfidf_stripped",
            "residual_idf_max_stripped",
            "unique_name_h3_cell_count_bin",
            NAME_COSINE,
            TAXONOMY_COSINE,
        )
    ]
)
SCORING_FEATURES = {*SCORING_EXPRESSIONS, *PAIRWISE_FEATURES.fieldNames()}


class PairwiseLookups(NamedTuple):
    """Token weights and name counts the pairwise features read.

    Attributes:
        name_idf: IDF of a name token.
        address_idf: IDF of an address token.
        name_repeated: Whether an H3 name key occurs at least twice.
        name_common: Whether an H3 name key occurs at least ten times.
    """

    name_idf: Callable[[str], float]
    address_idf: Callable[[str], float]
    name_repeated: Callable[[str], bool]
    name_common: Callable[[str], bool]


# Java's String.trim strips every character up to and including the space.
_JAVA_WHITESPACE = "".join(map(chr, range(33)))
_INT8_SCALE = 1.0 / (127.0 * 127.0)


def _name_h3_bin(name: str | None, cell: int | None, lookups: PairwiseLookups) -> int:
    """2 if the name is common in its H3 cell, 1 if it repeats there, else 0."""
    if name is None or cell is None:
        return 0
    name_key = name.strip(_JAVA_WHITESPACE).lower()
    if not name_key:
        return 0
    key = f"{NAME_H3_RESOLUTION}:{cell}:{name_key}"
    if lookups.name_common(key):
        return 2
    return 1 if lookups.name_repeated(key) else 0


def _cosine(left: bytes | None, right: bytes | None) -> float | None:
    """Cosine of two int8 embeddings scaled to unit length.

    None if either is missing or the lengths differ.
    """
    if not left or not right or len(left) != len(right):
        return None
    dot = int(
        np.dot(
            np.frombuffer(left, np.int8).astype(np.int32),
            np.frombuffer(right, np.int8).astype(np.int32),
        )
    )
    return min(1.0, max(-1.0, dot * _INT8_SCALE))


def _pairwise(p: Row, lookups: PairwiseLookups) -> dict:
    """Every ``PAIRWISE_FEATURES`` value of the pair ``p``."""
    name_idf, address_idf = lookups.name_idf, lookups.address_idf
    s = similarity
    name_similarity = s.normalized_similarity(p.name_clean, p.base_name_clean)
    # Name features marked strict count only where the full names differ.
    strict = 1.0 - name_similarity
    names, base_names = p.name_clean_tokens, p.base_name_clean_tokens
    stripped, base_stripped = (
        p.name_clean_stripped_tokens,
        p.base_name_clean_stripped_tokens,
    )
    addresses, base_addresses = p.address_clean_tokens, p.base_address_clean_tokens
    has_addresses = bool(p.address_clean) and bool(p.base_address_clean)
    return {
        "name_similarity": name_similarity,
        "name_jaccard_similarity": s.fuzzy_jaccard(names, base_names),
        "name_strict_prefix_score": s.prefix_score(p.name_clean, p.base_name_clean)
        * strict,
        "name_strict_suffix_score": s.suffix_score(p.name_clean, p.base_name_clean)
        * strict,
        "name_strict_shared_token_ratio": s.shared_token_ratio(names, base_names)
        * strict,
        "name_strict_token_subset": s.is_token_subset(names, base_names) * strict,
        "address_similarity_availability": s.normalized_similarity(
            p.address_clean, p.base_address_clean
        )
        if has_addresses
        else -1.0,
        "address_prefix_score": s.prefix_score(p.address_clean, p.base_address_clean),
        "address_suffix_score": s.suffix_score(p.address_clean, p.base_address_clean),
        "address_jaccard_similarity": s.fuzzy_jaccard(addresses, base_addresses),
        "address_shared_token_ratio": s.shared_token_ratio(addresses, base_addresses),
        "address_token_subset": int(s.is_token_subset(addresses, base_addresses)),
        "name_similarity_stripped": s.normalized_similarity(
            p.name_clean_stripped, p.base_name_clean_stripped
        ),
        "name_jaccard_similarity_stripped": s.fuzzy_jaccard(stripped, base_stripped),
        "name_prefix_score_stripped": s.prefix_score(
            p.name_clean_stripped, p.base_name_clean_stripped
        ),
        "name_suffix_score_stripped": s.suffix_score(
            p.name_clean_stripped, p.base_name_clean_stripped
        ),
        "website_path_similarity": s.website_path_similarity(
            p.website_urls, p.base_website_urls
        ),
        "name_tfidf": s.tfidf_overlap(names, base_names, name_idf),
        "residual_idf_max": s.residual_idf_max(names, base_names, name_idf),
        "address_tfidf": s.tfidf_overlap(addresses, base_addresses, address_idf),
        "name_tfidf_stripped": s.tfidf_overlap(stripped, base_stripped, name_idf),
        "residual_idf_max_stripped": s.residual_idf_max(
            stripped, base_stripped, name_idf
        ),
        "unique_name_h3_cell_count_bin": float(
            max(
                _name_h3_bin(p.name, p.name_h3_cell, lookups),
                _name_h3_bin(p.base_name, p.base_name_h3_cell, lookups),
            )
        ),
        NAME_COSINE: _cosine(p.name_vector, p.base_name_vector),
        TAXONOMY_COSINE: _cosine(p.taxonomy_vector, p.base_taxonomy_vector),
    }


def pairwise_features(lookups: Callable[[], PairwiseLookups]) -> Column:
    """Struct of every ``PAIRWISE_FEATURES`` field, from one Python UDF call per pair.

    Reads ``PAIRWISE_INPUTS`` on both sides.

    Args:
        lookups: Builds the lookups. It runs on the workers, once per pair,
            so it should cache what it builds.
    """
    inputs = F.struct(*PAIRWISE_INPUTS, *[f"base_{c}" for c in PAIRWISE_INPUTS])
    udf = F.udf(lambda p: _pairwise(p, lookups()), PAIRWISE_FEATURES)
    return udf(inputs)
