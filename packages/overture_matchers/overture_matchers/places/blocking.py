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

    from pyspark.sql import Column, DataFrame, SparkSession

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
RECORD_KEY = ("id", "provider")
SHUFFLE_PARTITIONS_CONF = "spark.sql.shuffle.partitions"
DEFAULT_SHUFFLE_PARTITIONS = "1920"
SPATIAL_PARTITIONS_PER_CORE = 20
# Session settings ``get_spatial_blocking_pairs`` needs. Sedona partitions and
# indexes the right side, and adaptive execution would re-plan the join's partitions.
SPATIAL_JOIN_CONF = {
    "sedona.join.spatitionside": "right",
    "sedona.join.indexbuildside": "right",
    "spark.sql.adaptive.enabled": "false",
}


def get_shuffle_partition_count(spark: SparkSession) -> int:
    """Partitions for the shuffles of blocking and scoring."""
    return int(spark.conf.get(SHUFFLE_PARTITIONS_CONF, DEFAULT_SHUFFLE_PARTITIONS))


def repartition_by_pair_key(df: DataFrame, num_partitions: int) -> DataFrame:
    """``df`` hash partitioned on both sides' ``id`` and ``provider``."""
    return df.repartition(
        num_partitions, F.expr("xxhash64(id, provider, base_id, base_provider)")
    )


def get_spatial_blocking_pairs(
    records: DataFrame,
    carried_columns: list[str],
    radius_meters: float,
    num_partitions: int,
) -> DataFrame:
    """Pairs of records closer than ``radius_meters``, one per pair, in id order.

    Keep the distance test in the form ``ST_DistanceSphere(a, b) < radius``:
    only then does Sedona plan a distance join instead of a cartesian product.
    The session needs ``SPATIAL_JOIN_CONF``.

    Args:
        records: Records with ``id``, ``provider``, ``longitude``, ``latitude``
            and ``carried_columns``.
        carried_columns: Columns to carry to both sides of each pair.
        radius_meters: Radius in meters.
        num_partitions: Partitions of the records going into the join.

    Returns:
        ``id``, ``provider``, ``base_id`` and ``base_provider`` for the sides,
        ``geo_distance``, and each carried column for the left side and as
        ``base_<name>`` for the right.
    """
    points = records.select(
        *RECORD_KEY,
        F.expr("ST_Point(longitude, latitude)").alias("geometry"),
        *carried_columns,
    ).repartition(num_partitions)
    distance = F.expr("ST_DistanceSphere(s.geometry, t.geometry)")
    return (
        points.alias("s")
        .join(
            points.alias("t"),
            (distance < radius_meters) & (F.col("s.id") < F.col("t.id")),
            "inner",
        )
        .select(
            F.col("s.id").alias("id"),
            F.col("s.provider").alias("provider"),
            F.col("t.id").alias("base_id"),
            F.col("t.provider").alias("base_provider"),
            distance.alias(features.GEO_DISTANCE),
            *[
                column
                for column_name in carried_columns
                for column in (
                    F.col(f"s.{column_name}").alias(column_name),
                    F.col(f"t.{column_name}").alias(f"base_{column_name}"),
                )
            ],
        )
    )


def prune_blocking_pairs(
    spatial_pairs: DataFrame,
    model: PlacesModel,
    score_pairs: Callable[[list[Column]], Column],
    num_partitions: int,
) -> DataFrame:
    """Keys of the pairs ``score_pairs`` puts at or above the model's threshold.

    Args:
        spatial_pairs: Output of ``get_spatial_blocking_pairs``.
        model: The model, whose feature order and threshold apply.
        score_pairs: Blocking score from the feature columns in model order.
        num_partitions: Partitions of the result.

    Returns:
        ``id``, ``provider``, ``base_id`` and ``base_provider``.
    """
    feature_columns: list[Column] = [
        features.BLOCKING_FEATURES[feature_name]()
        for feature_name in model.blocking_features
    ]
    scored_keys = spatial_pairs.select(
        "id",
        "provider",
        "base_id",
        "base_provider",
        score_pairs(feature_columns).alias("score"),
    )
    surviving_keys = scored_keys.filter(
        F.col("score") >= model.blocking_threshold
    ).drop("score")
    return repartition_by_pair_key(surviving_keys, num_partitions)


def get_blocking_pairs(
    candidates: DataFrame,
    baseline: DataFrame,
    model: PlacesModel,
    compute_name_idf: Callable[[Column], Column],
    score_pairs: Callable[[list[Column]], Column],
) -> DataFrame:
    """Candidate pairs that pass blocking, with both sides' columns.

    Args:
        candidates: Candidates without history matches, from setup.
        baseline: The whole baseline, inactive records included.
        model: The model.
        compute_name_idf: Map of token to weight, from an array of name tokens.
        score_pairs: Blocking score from the feature columns in model order.

    Returns:
        The right side's key, ``record_type`` and ``PAIR_COLUMNS``, each with
        ``base_`` in front, then the same columns of the left side. The left
        side is the smaller id. Each read reruns blocking, so callers
        checkpoint the result.
    """
    spark = candidates.sparkSession
    shuffle_partitions = get_shuffle_partition_count(spark)
    record_columns = [*RECORD_KEY, *PAIR_COLUMNS]
    unioned = (
        candidates.select(*record_columns)
        .withColumn("record_type", F.lit(CANDIDATE))
        .unionByName(
            baseline.select(*record_columns).withColumn("record_type", F.lit(BASELINE))
        )
    )

    feature_inputs = [
        column_name
        for feature_name in model.blocking_features
        for column_name in features.FEATURE_INPUTS[feature_name]
    ]
    needs_idf = features.NAME_IDF in feature_inputs
    carried_columns = list(dict.fromkeys(feature_inputs))
    spatial_input = (
        unioned.withColumn(
            features.NAME_IDF, compute_name_idf(F.col("name_clean_tokens"))
        )
        if needs_idf
        else unioned
    )
    spatial_pairs = get_spatial_blocking_pairs(
        spatial_input,
        carried_columns,
        model.blocking_radius_meters,
        spark.sparkContext.defaultParallelism * SPATIAL_PARTITIONS_PER_CORE,
    )
    surviving_keys = prune_blocking_pairs(
        spatial_pairs, model, score_pairs, shuffle_partitions
    )

    output_columns = [*RECORD_KEY, "record_type", *PAIR_COLUMNS]
    left, right = unioned.alias("left"), unioned.alias("right")
    joined = (
        surviving_keys.alias("pairs")
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
            *[
                F.col(f"right.{column_name}").alias(f"base_{column_name}")
                for column_name in output_columns
            ],
            *[F.col(f"left.{column_name}") for column_name in output_columns],
        )
    )
    return repartition_by_pair_key(joined, shuffle_partitions)
