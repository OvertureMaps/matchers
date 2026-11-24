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


def calculate_name_similarity() -> Column:
    """One minus the edit distance over the longer length, with 1.0 for two empty names."""
    left = F.coalesce(F.col("name_clean"), F.lit(""))
    right = F.coalesce(F.col("base_name_clean"), F.lit(""))
    longest = F.greatest(F.length(left), F.length(right))
    return F.when(longest == 0, F.lit(1.0)).otherwise(
        1.0 - F.levenshtein(left, right) / longest
    )


def _calculate_equality_or_unavailable(
    left_column: str, right_column: str, equality_condition: Column | None = None
) -> Column:
    """1.0 or 0.0 for present values, -1.0 when either side is null or empty."""
    left, right = F.col(left_column), F.col(right_column)
    unavailable = left.isNull() | right.isNull() | (left == "") | (right == "")
    same = equality_condition if equality_condition is not None else left == right
    return (
        F.when(unavailable, F.lit(_UNAVAILABLE))
        .when(same, F.lit(1.0))
        .otherwise(F.lit(0.0))
    )


def calculate_phone_number_similarity_availability() -> Column:
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
    return _calculate_equality_or_unavailable(
        "normalized_phone",
        "base_normalized_phone",
        F.when(country_unknown, suffixes_match).otherwise(left == right),
    )


def calculate_house_number_similarity_availability() -> Column:
    """House number match."""
    return _calculate_equality_or_unavailable("house_number", "base_house_number")


def calculate_name_tfidf_overlap() -> Column:
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
    "name_similarity": calculate_name_similarity,
    "phone_number_similarity_availability": calculate_phone_number_similarity_availability,
    "house_number_similarity_availability": calculate_house_number_similarity_availability,
    "name_tfidf": calculate_name_tfidf_overlap,
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


def calculate_phone_last_digits_match() -> Column:
    """1.0 if the last four phone digits match, else 0.0. Null when either phone is missing."""
    left, right = F.col("normalized_phone"), F.col("base_normalized_phone")
    missing = left.isNull() | right.isNull() | (left == "") | (right == "")
    return F.when(missing, F.lit(None).cast("double")).otherwise(
        (left.substr(-4, 4) == right.substr(-4, 4)).cast("double")
    )


def calculate_website_domain_match() -> Column:
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


def calculate_geo_distance() -> Column:
    """Great-circle distance between the two places in meters."""
    return F.expr(
        "ST_DistanceSphere(ST_Point(longitude, latitude), "
        "ST_Point(base_longitude, base_latitude))"
    )


def calculate_unique_name_h3_cell_count_bin(
    compute_unique_name_bin: Callable[[Column, Column, Column], Column],
) -> Column:
    """Larger H3 resolution-5 name-repetition bin (0, 1 or 2) across the two places."""
    left_bin = compute_unique_name_bin(
        F.col("name"), F.col("latitude"), F.col("longitude")
    )
    right_bin = compute_unique_name_bin(
        F.col("base_name"), F.col("base_latitude"), F.col("base_longitude")
    )
    return F.greatest(left_bin, right_bin).cast("double")


SCORING_EXPRESSIONS = {
    "phone_number_similarity_availability": calculate_phone_number_similarity_availability,
    "house_number_similarity_availability": calculate_house_number_similarity_availability,
    "phone_last_digits_match": calculate_phone_last_digits_match,
    "website_domain_match": calculate_website_domain_match,
    GEO_DISTANCE: calculate_geo_distance,
}

UNIQUE_NAME_H3_CELL_COUNT_BIN = "unique_name_h3_cell_count_bin"
# Columns scoring adds to each side before the pairwise UDF runs.
NAME_VECTOR = "name_vector"
TAXONOMY_VECTOR = "taxonomy_vector"
PAIRWISE_INPUTS = (
    "name_clean",
    "name_clean_tokens",
    "name_clean_stripped",
    "name_clean_stripped_tokens",
    "address_clean",
    "address_clean_tokens",
    "website_urls",
    NAME_VECTOR,
    TAXONOMY_VECTOR,
)
NAME_COSINE_SIMILARITY = "name_cosine_similarity"
TAXONOMY_COSINE_SIMILARITY = "taxonomy_cosine_similarity"
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
            NAME_COSINE_SIMILARITY,
            TAXONOMY_COSINE_SIMILARITY,
        )
    ]
)
SCORING_FEATURES = {
    *SCORING_EXPRESSIONS,
    UNIQUE_NAME_H3_CELL_COUNT_BIN,
    *PAIRWISE_FEATURES.fieldNames(),
}


class PairwiseLookups(NamedTuple):
    """Token weights the pairwise features read.

    Attributes:
        compute_name_idf: IDF of a name token.
        compute_address_idf: IDF of an address token.
    """

    compute_name_idf: Callable[[str], float]
    compute_address_idf: Callable[[str], float]


_INT8_SCALE = 1.0 / (127.0 * 127.0)


def _compute_cosine_similarity(
    left_vector: bytes | None, right_vector: bytes | None
) -> float | None:
    """Cosine of two int8 embeddings scaled to unit length.

    None if either is missing or the lengths differ.
    """
    if not left_vector or not right_vector or len(left_vector) != len(right_vector):
        return None
    dot = int(
        np.dot(
            np.frombuffer(left_vector, np.int8).astype(np.int32),
            np.frombuffer(right_vector, np.int8).astype(np.int32),
        )
    )
    return min(1.0, max(-1.0, dot * _INT8_SCALE))


def _compute_pairwise_features_for_row(pair_row: Row, lookups: PairwiseLookups) -> dict:
    """Every ``PAIRWISE_FEATURES`` value of ``pair_row``."""
    compute_name_idf = lookups.compute_name_idf
    compute_address_idf = lookups.compute_address_idf
    name_similarity = similarity.compute_normalized_similarity(
        pair_row.name_clean, pair_row.base_name_clean
    )
    # Name features marked strict count only where the full names differ.
    strict_weight = 1.0 - name_similarity
    names, base_names = pair_row.name_clean_tokens, pair_row.base_name_clean_tokens
    stripped, base_stripped = (
        pair_row.name_clean_stripped_tokens,
        pair_row.base_name_clean_stripped_tokens,
    )
    addresses, base_addresses = (
        pair_row.address_clean_tokens,
        pair_row.base_address_clean_tokens,
    )
    has_addresses = bool(pair_row.address_clean) and bool(pair_row.base_address_clean)
    return {
        "name_similarity": name_similarity,
        "name_jaccard_similarity": similarity.compute_fuzzy_jaccard_similarity(
            names, base_names
        ),
        "name_strict_prefix_score": similarity.get_string_prefix_score(
            pair_row.name_clean, pair_row.base_name_clean
        )
        * strict_weight,
        "name_strict_suffix_score": similarity.get_string_suffix_score(
            pair_row.name_clean, pair_row.base_name_clean
        )
        * strict_weight,
        "name_strict_shared_token_ratio": similarity.compute_shared_token_ratio(
            names, base_names
        )
        * strict_weight,
        "name_strict_token_subset": similarity.is_token_subset(names, base_names)
        * strict_weight,
        "address_similarity_availability": similarity.compute_normalized_similarity(
            pair_row.address_clean, pair_row.base_address_clean
        )
        if has_addresses
        else -1.0,
        "address_prefix_score": similarity.get_string_prefix_score(
            pair_row.address_clean, pair_row.base_address_clean
        ),
        "address_suffix_score": similarity.get_string_suffix_score(
            pair_row.address_clean, pair_row.base_address_clean
        ),
        "address_jaccard_similarity": similarity.compute_fuzzy_jaccard_similarity(
            addresses, base_addresses
        ),
        "address_shared_token_ratio": similarity.compute_shared_token_ratio(
            addresses, base_addresses
        ),
        "address_token_subset": int(
            similarity.is_token_subset(addresses, base_addresses)
        ),
        "name_similarity_stripped": similarity.compute_normalized_similarity(
            pair_row.name_clean_stripped, pair_row.base_name_clean_stripped
        ),
        "name_jaccard_similarity_stripped": similarity.compute_fuzzy_jaccard_similarity(
            stripped, base_stripped
        ),
        "name_prefix_score_stripped": similarity.get_string_prefix_score(
            pair_row.name_clean_stripped, pair_row.base_name_clean_stripped
        ),
        "name_suffix_score_stripped": similarity.get_string_suffix_score(
            pair_row.name_clean_stripped, pair_row.base_name_clean_stripped
        ),
        "website_path_similarity": similarity.compute_website_path_similarity(
            pair_row.website_urls, pair_row.base_website_urls
        ),
        "name_tfidf": similarity.get_tfidf_overlap(names, base_names, compute_name_idf),
        "residual_idf_max": similarity.compute_residual_idf_max(
            names, base_names, compute_name_idf
        ),
        "address_tfidf": similarity.get_tfidf_overlap(
            addresses, base_addresses, compute_address_idf
        ),
        "name_tfidf_stripped": similarity.get_tfidf_overlap(
            stripped, base_stripped, compute_name_idf
        ),
        "residual_idf_max_stripped": similarity.compute_residual_idf_max(
            stripped, base_stripped, compute_name_idf
        ),
        NAME_COSINE_SIMILARITY: _compute_cosine_similarity(
            pair_row.name_vector, pair_row.base_name_vector
        ),
        TAXONOMY_COSINE_SIMILARITY: _compute_cosine_similarity(
            pair_row.taxonomy_vector, pair_row.base_taxonomy_vector
        ),
    }


def compute_pairwise_features(get_lookups: Callable[[], PairwiseLookups]) -> Column:
    """Struct of every ``PAIRWISE_FEATURES`` field, from one Python UDF call per pair.

    Reads ``PAIRWISE_INPUTS`` on both sides.

    Args:
        get_lookups: Builds the lookups. It runs on the workers, once per pair,
            so it should cache what it builds.
    """
    inputs = F.struct(
        *PAIRWISE_INPUTS, *[f"base_{column_name}" for column_name in PAIRWISE_INPUTS]
    )
    udf = F.udf(
        lambda pair_row: _compute_pairwise_features_for_row(pair_row, get_lookups()),
        PAIRWISE_FEATURES,
    )
    return udf(inputs)
