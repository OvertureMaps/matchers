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
class Scorer:
    """The scoring model.

    Attributes:
        feature_order: Scoring features in model order.
        threshold: Smallest probability that makes a pair a match.
        lookups: Builds the pairwise features' lookups on a worker.
        predict: Struct of ``MARGIN`` and ``PROBABILITY`` from the feature
            columns in model order.
    """

    feature_order: tuple[str, ...]
    threshold: float
    lookups: Callable[[], features.PairwiseLookups]
    predict: Callable[[list[Column]], Column]

    @classmethod
    def from_config(
        cls,
        config: dict,
        lookups: Callable[[], features.PairwiseLookups],
        predict: Callable[[list[Column]], Column],
    ) -> Scorer:
        """The scorer of the jar's parsed scoring config and the given engines.

        Raises:
            ValueError: The feature order names a feature that ``features.py``
                does not produce.
        """
        order = tuple(config["feature_order"])
        unknown = [f for f in order if f not in features.SCORING_FEATURES]
        if unknown:
            raise ValueError(
                f"scoring feature_order needs {unknown}, but features.py produces "
                f"only {sorted(features.SCORING_FEATURES)}"
            )
        return cls(order, float(config["prediction_threshold"]), lookups, predict)


# The cache row of an embedding: a SHA-256 of the normalized text, keyed by field type.
CACHE_FIELD_TYPE = "field_type"
CACHE_HASH = "input_hash"
CACHE_VECTOR = "embedding"


def _name_hash_text(name: Column) -> Column:
    return F.trim(F.lower(name))


def _taxonomy_hash_text(taxonomy: Column) -> Column:
    return F.regexp_replace(taxonomy, "_", " ")


# Feature column, text column, cache field type and hash normalization.
_EMBEDDINGS = (
    (features.NAME_VECTOR, "name", "name", _name_hash_text),
    (features.TAXONOMY_VECTOR, "taxonomy_primary", "taxonomy", _taxonomy_hash_text),
)


def _text_hash(text: Column) -> Column:
    """SHA-256 of ``text``, or null when it is null or blank."""
    return F.when(
        text.isNull() | (F.length(F.trim(text)) == 0), F.lit(None).cast("string")
    ).otherwise(F.sha2(text, 256))


def _h3_cell(prefix: str) -> Column:
    lon, lat = F.col(f"{prefix}longitude"), F.col(f"{prefix}latitude")
    cell = F.expr(
        f"ST_H3CellIDs(ST_Point({prefix}longitude, {prefix}latitude), "
        f"{features.NAME_H3_RESOLUTION}, false)[0]"
    )
    return F.when(lon.isNull() | lat.isNull(), F.lit(None)).otherwise(cell)


def _with_embeddings(
    pairs: DataFrame, cache: DataFrame | None, needed: bool
) -> DataFrame:
    """Add each side's name and taxonomy embedding from the cache.

    A text with no cache row gets a null embedding. Without ``needed`` every
    embedding is null and the cache is never read.
    """
    if not needed:
        for vector, *_ in _EMBEDDINGS:
            for prefix in ("", "base_"):
                pairs = pairs.withColumn(prefix + vector, F.lit(None).cast("binary"))
        return pairs
    if cache is None:
        raise ValueError(
            "the model's cosine features need the embeddingsCacheTable param"
        )
    for vector, text, field_type, normalize in _EMBEDDINGS:
        hashes = {p: f"_{p}{vector}_hash" for p in ("", "base_")}
        for prefix, hash_col in hashes.items():
            pairs = pairs.withColumn(
                hash_col, _text_hash(normalize(F.col(prefix + text)))
            )
        needed_hashes = (
            pairs.select(F.col(hashes[""]).alias("hash"))
            .unionByName(pairs.select(F.col(hashes["base_"]).alias("hash")))
            .filter(F.col("hash").isNotNull())
            .distinct()
        )
        # One vector per hash, so a cache holding a hash twice cannot duplicate pairs.
        embeddings = (
            cache.filter(F.col(CACHE_FIELD_TYPE) == field_type)
            .select(
                F.col(CACHE_HASH).cast("string").alias("hash"),
                F.col(CACHE_VECTOR).cast("binary").alias("vector"),
            )
            .join(needed_hashes, "hash")
            .dropDuplicates(["hash"])
        )
        if field_type == "taxonomy":
            embeddings = F.broadcast(embeddings)
        for prefix, hash_col in hashes.items():
            pairs = pairs.join(
                embeddings.select(
                    F.col("hash").alias(hash_col),
                    F.col("vector").alias(prefix + vector),
                ),
                hash_col,
                "left",
            ).drop(hash_col)
    return pairs


def score_pairs(
    pairs: DataFrame, scorer: Scorer, embeddings: DataFrame | None
) -> DataFrame:
    """Add the scoring features and the model's outputs to each pair.

    Args:
        pairs: Blocking pairs.
        scorer: The scoring model.
        embeddings: Name and taxonomy embedding cache. Read only when the
            model uses a cosine feature.

    Returns:
        ``pairs`` with one column per scoring feature, then ``features``,
        ``rawPrediction``, ``probability``, ``prediction`` (an int) and
        ``composite_score`` (the match probability).
    """
    order = scorer.feature_order
    cosine = {features.NAME_COSINE, features.TAXONOMY_COSINE} & set(order)
    pairwise = "_pairwise"
    scored = (
        _with_embeddings(pairs, embeddings, bool(cosine))
        .withColumn(features.NAME_H3_CELL, _h3_cell(""))
        .withColumn(f"base_{features.NAME_H3_CELL}", _h3_cell("base_"))
        .withColumn(pairwise, features.pairwise_features(scorer.lookups))
    )
    pairwise_names = set(features.PAIRWISE_FEATURES.fieldNames())
    feature_columns = [
        F.col(f"{pairwise}.{name}").alias(name)
        if name in pairwise_names
        else features.SCORING_EXPRESSIONS[name]().alias(name)
        for name in order
    ]
    # Selecting only the input columns drops the helper columns added above.
    scored = scored.select(
        *[c for c in pairs.columns if c not in order], *feature_columns
    )
    output = scorer.predict([F.col(name) for name in order])
    margin = F.col(f"{pairwise}.{MARGIN}")
    probability = F.col(f"{pairwise}.{PROBABILITY}")
    nan = F.lit(float("nan"))
    feature_values = [F.coalesce(F.col(name).cast("double"), nan) for name in order]
    return (
        scored.withColumn(pairwise, output)
        .withColumn("features", array_to_vector(F.array(*feature_values)))
        .withColumn("rawPrediction", array_to_vector(F.array(-margin, margin)))
        .withColumn(
            "probability",
            array_to_vector(F.array(F.lit(1.0) - probability, probability)),
        )
        .withColumn("prediction", (probability >= scorer.threshold).cast("int"))
        .withColumn("composite_score", probability)
        .drop(pairwise)
    )
