# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Score blocking pairs with the places matcher model.

Adds every scoring feature the model asks for, then the model's outputs as
Spark ML vectors: ``features`` (the feature values, null as NaN),
``rawPrediction`` (negated and raw margin) and ``probability`` (negative and
positive class). ``prediction`` is 1 when the positive probability, which is
also ``composite_score``, reaches the model threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyspark.ml.functions import array_to_vector
from pyspark.sql import functions as F

from overture_matchers.places import features
from overture_matchers.places.model import MARGIN, PROBABILITY

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


@dataclass(frozen=True)
class PlacesPairScorer:
    """The scoring model.

    Attributes:
        feature_order: Scoring features in model order.
        prediction_threshold: Smallest probability that makes a pair a match.
        get_lookups: Builds the pairwise features' lookups on a worker.
        compute_unique_name_bin: Resolution-5 H3 name-repetition bin (0, 1 or 2)
            for one side's name, latitude and longitude.
        predict_scores: Struct of ``MARGIN`` and ``PROBABILITY`` from the
            feature columns in model order.
    """

    feature_order: tuple[str, ...]
    prediction_threshold: float
    get_lookups: Callable[[], features.PairwiseLookups]
    compute_unique_name_bin: Callable[[Column, Column, Column], Column]
    predict_scores: Callable[[list[Column]], Column]

    @classmethod
    def from_config(
        cls,
        config: dict,
        get_lookups: Callable[[], features.PairwiseLookups],
        compute_unique_name_bin: Callable[[Column, Column, Column], Column],
        predict_scores: Callable[[list[Column]], Column],
    ) -> PlacesPairScorer:
        """The scorer of the jar's parsed scoring config and the given engines.

        Raises:
            ValueError: The feature order names a feature that ``features.py``
                does not produce.
        """
        feature_order = tuple(config["feature_order"])
        unknown_features = [
            feature_name
            for feature_name in feature_order
            if feature_name not in features.SCORING_FEATURES
        ]
        if unknown_features:
            raise ValueError(
                f"scoring feature_order needs {unknown_features}, but features.py "
                f"produces only {sorted(features.SCORING_FEATURES)}"
            )
        return cls(
            feature_order=feature_order,
            prediction_threshold=float(config["prediction_threshold"]),
            get_lookups=get_lookups,
            compute_unique_name_bin=compute_unique_name_bin,
            predict_scores=predict_scores,
        )

    @property
    def needs_embeddings(self) -> bool:
        """Whether a cosine feature reads the embeddings cache."""
        return not {
            features.NAME_COSINE_SIMILARITY,
            features.TAXONOMY_COSINE_SIMILARITY,
        }.isdisjoint(self.feature_order)


MISSING_EMBEDDINGS_CACHE_MESSAGE = (
    "the model's cosine features need the embeddingsCacheTable param"
)


# The cache row of an embedding: a SHA-256 of the normalized text, keyed by field type.
CACHE_FIELD_TYPE = "field_type"
CACHE_HASH = "input_hash"
CACHE_VECTOR = "embedding"


def _normalize_name_for_hash(name: Column) -> Column:
    return F.trim(F.lower(name))


def _normalize_taxonomy_for_hash(taxonomy: Column) -> Column:
    return F.regexp_replace(taxonomy, "_", " ")


# Feature column, text column, cache field type and hash normalization.
_EMBEDDINGS = (
    (features.NAME_VECTOR, "name", "name", _normalize_name_for_hash),
    (
        features.TAXONOMY_VECTOR,
        "taxonomy_primary",
        "taxonomy",
        _normalize_taxonomy_for_hash,
    ),
)


def _compute_text_hash(text: Column) -> Column:
    """SHA-256 of ``text``, or null when it is null or blank."""
    return F.when(
        text.isNull() | (F.length(F.trim(text)) == 0), F.lit(None).cast("string")
    ).otherwise(F.sha2(text, 256))


def _attach_cached_embeddings(
    pairs: DataFrame, embeddings_cache: DataFrame | None, needs_embeddings: bool
) -> DataFrame:
    """Add each side's name and taxonomy embedding from the cache.

    A text with no cache row gets a null embedding. Without ``needs_embeddings``
    every embedding is null and the cache is never read.
    """
    if not needs_embeddings:
        for vector_column, *_ in _EMBEDDINGS:
            for side_prefix in ("", "base_"):
                pairs = pairs.withColumn(
                    side_prefix + vector_column, F.lit(None).cast("binary")
                )
        return pairs
    if embeddings_cache is None:
        raise ValueError(MISSING_EMBEDDINGS_CACHE_MESSAGE)
    for vector_column, text_column, field_type, normalize_text in _EMBEDDINGS:
        hash_columns = {
            side_prefix: f"_{side_prefix}{vector_column}_hash"
            for side_prefix in ("", "base_")
        }
        for side_prefix, hash_column in hash_columns.items():
            pairs = pairs.withColumn(
                hash_column,
                _compute_text_hash(normalize_text(F.col(side_prefix + text_column))),
            )
        needed_hashes = (
            pairs.select(F.col(hash_columns[""]).alias("hash"))
            .unionByName(pairs.select(F.col(hash_columns["base_"]).alias("hash")))
            .filter(F.col("hash").isNotNull())
            .distinct()
        )
        # One vector per hash, so a cache holding a hash twice cannot duplicate
        # pairs. The smallest vector wins, so reruns agree.
        embeddings = (
            embeddings_cache.filter(F.col(CACHE_FIELD_TYPE) == field_type)
            .select(
                F.col(CACHE_HASH).cast("string").alias("hash"),
                F.col(CACHE_VECTOR).cast("binary").alias("vector"),
            )
            .join(needed_hashes, "hash")
            .groupBy("hash")
            .agg(F.min("vector").alias("vector"))
        )
        if field_type == "taxonomy":
            embeddings = F.broadcast(embeddings)
        for side_prefix, hash_column in hash_columns.items():
            pairs = pairs.join(
                embeddings.select(
                    F.col("hash").alias(hash_column),
                    F.col("vector").alias(side_prefix + vector_column),
                ),
                hash_column,
                "left",
            ).drop(hash_column)
    return pairs


def _build_scoring_feature_column(
    feature_name: str,
    pairwise_column: str,
    pairwise_names: set[str],
    scorer: PlacesPairScorer,
) -> Column:
    if feature_name in pairwise_names:
        return F.col(f"{pairwise_column}.{feature_name}").alias(feature_name)
    if feature_name == features.UNIQUE_NAME_H3_CELL_COUNT_BIN:
        return features.calculate_unique_name_h3_cell_count_bin(
            scorer.compute_unique_name_bin
        ).alias(feature_name)
    return features.SCORING_EXPRESSIONS[feature_name]().alias(feature_name)


def score_pairs(
    pairs: DataFrame,
    scorer: PlacesPairScorer,
    embeddings_cache: DataFrame | None,
) -> DataFrame:
    """Add the scoring features and the model's outputs to each pair.

    Args:
        pairs: Blocking pairs.
        scorer: The scoring model.
        embeddings_cache: Name and taxonomy embedding cache. Read only when the
            model uses a cosine feature.

    Returns:
        ``pairs`` with one column per scoring feature, then ``features``,
        ``rawPrediction``, ``probability``, ``prediction`` (an int) and
        ``composite_score`` (the match probability).
    """
    feature_order = scorer.feature_order
    pairwise_column = "_pairwise"
    scored = _attach_cached_embeddings(
        pairs, embeddings_cache, scorer.needs_embeddings
    ).withColumn(
        pairwise_column, features.compute_pairwise_features(scorer.get_lookups)
    )
    pairwise_names = set(features.PAIRWISE_FEATURES.fieldNames())
    feature_columns = [
        _build_scoring_feature_column(
            feature_name, pairwise_column, pairwise_names, scorer
        )
        for feature_name in feature_order
    ]
    # Selecting only the input columns drops the helper columns added above.
    scored = scored.select(
        *[
            column_name
            for column_name in pairs.columns
            if column_name not in feature_order
        ],
        *feature_columns,
    )
    output_column = "_model_output"
    margin = F.col(f"{output_column}.{MARGIN}")
    probability = F.col(f"{output_column}.{PROBABILITY}")
    nan = F.lit(float("nan"))
    feature_values = [
        F.coalesce(F.col(feature_name).cast("double"), nan)
        for feature_name in feature_order
    ]
    return (
        scored.withColumn(
            output_column,
            scorer.predict_scores(
                [F.col(feature_name) for feature_name in feature_order]
            ),
        )
        .withColumn("features", array_to_vector(F.array(*feature_values)))
        .withColumn("rawPrediction", array_to_vector(F.array(-margin, margin)))
        .withColumn(
            "probability",
            array_to_vector(F.array(F.lit(1.0) - probability, probability)),
        )
        .withColumn(
            "prediction", (probability >= scorer.prediction_threshold).cast("int")
        )
        .withColumn("composite_score", probability)
        .drop(output_column)
    )
