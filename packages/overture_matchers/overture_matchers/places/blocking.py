# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Place blocking: spatial candidate pairs, filtered by the blocking model.

Candidates without history are unioned with the whole baseline and joined with
themselves within the model's radius. The blocking booster scores each pair
on a few cheap features, and only pairs above its threshold go on to scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from overture_matchers.places import features
from overture_matchers.prematch import BASELINE, CANDIDATE

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyspark.sql import Column, DataFrame

    from overture_matchers.places.model import PlacesModel

# Columns each side carries into scoring.
PAIR_COLUMNS = (
    "longitude",
    "latitude",
    "house_number",
    "address",
    "address_clean",
    "address_clean_tokens",
    "country",
    "normalized_phone",
    "name",
    "name_clean",
    "name_clean_tokens",
    "name_clean_stripped",
    "name_clean_stripped_tokens",
    "taxonomy_primary",
    "website_domains",
    "website_urls",
)
# A record is unique by id and provider together.
KEY = ("id", "provider")
SHUFFLE_PARTITIONS_CONF = "spark.sql.shuffle.partitions"
DEFAULT_SHUFFLE_PARTITIONS = "1920"
SPATIAL_PARTITIONS_PER_CORE = 20
# Session settings ``spatial_pairs`` needs. Sedona partitions and indexes the
# right side, and adaptive execution would re-plan the join's partitions.
SPATIAL_JOIN_CONF = {
    "sedona.join.spatitionside": "right",
    "sedona.join.indexbuildside": "right",
    "spark.sql.adaptive.enabled": "false",
}


def _by_pair_key(df: DataFrame, partitions: int) -> DataFrame:
    return df.repartition(
        partitions, F.expr("xxhash64(id, provider, base_id, base_provider)")
    )


def spatial_pairs(
    records: DataFrame, carried: list[str], radius_m: float, partitions: int
) -> DataFrame:
    """Pairs of records closer than ``radius_m``, one per pair, in id order.

    Keep the distance test in the form ``ST_DistanceSphere(a, b) < radius``:
    only then does Sedona plan a distance join instead of a cartesian product.
    The session needs ``SPATIAL_JOIN_CONF``.

    Args:
        records: Records with ``id``, ``provider``, ``longitude``, ``latitude`` and ``carried``.
        carried: Columns to carry to both sides of each pair.
        radius_m: Radius in meters.
        partitions: Partitions of the records going into the join.

    Returns:
        ``id``, ``provider``, ``base_id`` and ``base_provider`` for the sides,
        ``geo_distance``, and each carried column for the left side and as
        ``base_<name>`` for the right.
    """
    points = records.select(
        *KEY, F.expr("ST_Point(longitude, latitude)").alias("geom"), *carried
    ).repartition(partitions)
    distance = F.expr("ST_DistanceSphere(s.geom, t.geom)")
    return (
        points.alias("s")
        .join(
            points.alias("t"),
            (distance < radius_m) & (F.col("s.id") < F.col("t.id")),
            "inner",
        )
        .select(
            F.col("s.id").alias("id"),
            F.col("s.provider").alias("provider"),
            F.col("t.id").alias("base_id"),
            F.col("t.provider").alias("base_provider"),
            distance.alias(features.GEO_DISTANCE),
            *[
                col
                for c in carried
                for col in (
                    F.col(f"s.{c}").alias(c),
                    F.col(f"t.{c}").alias(f"base_{c}"),
                )
            ],
        )
    )


def prune(
    pairs: DataFrame,
    model: PlacesModel,
    score: Callable[[list[Column]], Column],
    partitions: int,
) -> DataFrame:
    """Keys of the pairs ``score`` puts at or above the model's threshold.

    Args:
        pairs: Output of ``spatial_pairs``.
        model: The model, whose feature order and threshold apply.
        score: Blocking score from the feature columns in model order.
        partitions: Partitions of the result.

    Returns:
        ``id``, ``provider``, ``base_id`` and ``base_provider``.
    """
    columns: list[Column] = [
        features.BLOCKING_FEATURES[f]() for f in model.blocking_features
    ]
    keys = pairs.select(
        "id", "provider", "base_id", "base_provider", score(columns).alias("score")
    )
    survivors = keys.filter(F.col("score") >= model.blocking_threshold).drop("score")
    return _by_pair_key(survivors, partitions)


def blocking_pairs(
    candidates: DataFrame,
    baseline: DataFrame,
    model: PlacesModel,
    name_idf: Callable[[Column], Column],
    score: Callable[[list[Column]], Column],
) -> DataFrame:
    """Candidate pairs that pass blocking, with both sides' columns.

    Args:
        candidates: Candidates without history matches, from setup.
        baseline: The whole baseline, inactive records included.
        model: The model.
        name_idf: Map of token to weight, from an array of name tokens.
        score: Blocking score from the feature columns in model order.

    Returns:
        The right side's key, ``record_type`` and ``PAIR_COLUMNS``, each with
        ``base_`` in front, then the same columns of the left side. The left
        side is the smaller id. Each read reruns blocking, so callers
        checkpoint the result.
    """
    spark = candidates.sparkSession
    shuffle_partitions = int(
        spark.conf.get(SHUFFLE_PARTITIONS_CONF, DEFAULT_SHUFFLE_PARTITIONS)
    )
    columns = [*KEY, *PAIR_COLUMNS]
    unioned = (
        candidates.select(*columns)
        .withColumn("record_type", F.lit(CANDIDATE))
        .unionByName(
            baseline.select(*columns).withColumn("record_type", F.lit(BASELINE))
        )
    )

    inputs = [c for f in model.blocking_features for c in features.FEATURE_INPUTS[f]]
    needs_idf = features.NAME_IDF in inputs
    carried = list(dict.fromkeys(inputs))
    spatial_input = (
        unioned.withColumn(features.NAME_IDF, name_idf(F.col("name_clean_tokens")))
        if needs_idf
        else unioned
    )
    pairs = spatial_pairs(
        spatial_input,
        carried,
        model.blocking_radius_m,
        spark.sparkContext.defaultParallelism * SPATIAL_PARTITIONS_PER_CORE,
    )
    survivors = prune(pairs, model, score, shuffle_partitions)

    output = [*KEY, "record_type", *PAIR_COLUMNS]
    left, right = unioned.alias("left"), unioned.alias("right")
    joined = (
        survivors.alias("pairs")
        .join(
            left,
            (F.col("left.id") == F.col("pairs.id"))
            & (F.col("left.provider") == F.col("pairs.provider")),
        )
        .join(
            right,
            (F.col("right.id") == F.col("pairs.base_id"))
            & (F.col("right.provider") == F.col("pairs.base_provider")),
        )
        .select(
            *[F.col(f"right.{c}").alias(f"base_{c}") for c in output],
            *[F.col(f"left.{c}") for c in output],
        )
    )
    return _by_pair_key(joined, shuffle_partitions)
