# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Place-pair features built from Spark expressions.

Each feature reads the two sides of a pair as columns: the left side under
a column's own name and the right side under ``base_<name>``. Blocking runs
on every spatial pair, so these stay built-in expressions, never Python UDFs.
"""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

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
    "name_similarity": name_similarity,
    "phone_number_similarity_availability": phone_number_similarity_availability,
    "house_number_similarity_availability": house_number_similarity_availability,
    "name_tfidf": name_tfidf,
}
# The columns each side of a pair carries for a feature.
FEATURE_INPUTS = {
    "name_similarity": ("name_clean",),
    "phone_number_similarity_availability": ("normalized_phone", "country"),
    "house_number_similarity_availability": ("house_number",),
    "name_tfidf": (NAME_IDF,),
}
