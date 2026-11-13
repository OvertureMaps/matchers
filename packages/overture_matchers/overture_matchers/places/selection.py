# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Cluster winner selection: one surviving candidate per cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F

from overture_matchers.clustering import (
    BASELINE_BASELINE,
    CANDIDATE_BASELINE,
    CANDIDATE_CANDIDATE,
    NO_MATCH,
)
from overture_matchers.prematch import CANDIDATE

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


STRING_COLUMNS = (
    "id",
    "record_type",
    "base_id",
    "base_record_type",
    "match_type",
    "match_sub_type",
    "cluster_id",
    "provider",
    "base_provider",
)


def coalesce_match_schema(matches: DataFrame) -> DataFrame:
    """Add the match columns that ``matches`` lacks, with null (or default) values."""
    for name in STRING_COLUMNS:
        if name not in matches.columns:
            matches = matches.withColumn(name, F.lit(None).cast("string"))
    if "composite_score" not in matches.columns:
        matches = matches.withColumn("composite_score", F.lit(-1.0))
    if "base_active" not in matches.columns:
        matches = matches.withColumn("base_active", F.lit(1))
    return matches


def add_base_active(matches: DataFrame, baseline: DataFrame) -> DataFrame:
    """Set ``base_active`` to the max ``active`` of the baseline ID, or 1 if unknown."""
    active_by_id = baseline.groupBy(F.col("id").alias("_active_id")).agg(
        F.max("active").alias("_active")
    )
    return (
        matches.drop("base_active")
        .join(active_by_id, F.col("base_id") == F.col("_active_id"), "left")
        .withColumn("base_active", F.coalesce("_active", F.lit(1)).cast("int"))
        .drop("_active_id", "_active")
    )


def _column_equals(column_name: str, value: str) -> Column:
    # A null column never equals the value.
    return F.coalesce(F.col(column_name) == value, F.lit(False))


def _classify_clusters(matches: DataFrame) -> DataFrame:
    """Add ``_cluster_kind``: nomatch, baseline_only, candidate_only or best_match."""
    cluster_window = Window.partitionBy("cluster_id")
    cluster_size = F.count(F.lit(1)).over(cluster_window)
    all_baseline_baseline = (
        F.sum(_column_equals("match_sub_type", BASELINE_BASELINE).cast("int")).over(
            cluster_window
        )
        == cluster_size
    )
    all_candidate_candidate = (
        F.sum(_column_equals("match_sub_type", CANDIDATE_CANDIDATE).cast("int")).over(
            cluster_window
        )
        == cluster_size
    )
    cluster_kind = (
        F.when(
            (cluster_size == 1) & _column_equals("match_type", NO_MATCH),
            "nomatch",
        )
        .when(all_baseline_baseline, "baseline_only")
        .when(all_candidate_candidate, "candidate_only")
        .otherwise("best_match")
    )
    return matches.withColumn("_cluster_kind", cluster_kind)


def _select_no_match_winners(classified_matches: DataFrame) -> DataFrame:
    return classified_matches.where(F.col("_cluster_kind") == "nomatch").select(
        "cluster_id", "id", F.lit(None).cast("string").alias("base_id")
    )


def _select_best_inner_candidate(classified_matches: DataFrame) -> DataFrame:
    """Select the ID seen most often on either side, then by max score, then by ID."""
    candidate_pairs = classified_matches.where(
        F.col("_cluster_kind") == "candidate_only"
    )
    sides = candidate_pairs.select("cluster_id", "id", "composite_score").unionByName(
        candidate_pairs.select(
            "cluster_id", F.col("base_id").alias("id"), "composite_score"
        )
    )
    per_id = sides.groupBy("cluster_id", "id").agg(
        F.count(F.lit(1)).alias("occurrences"),
        F.max("composite_score").alias("score"),
    )
    return (
        per_id.groupBy("cluster_id")
        .agg(F.max(F.struct("occurrences", "score", "id")).alias("winner"))
        .select(
            "cluster_id",
            F.col("winner.id").alias("id"),
            F.lit(None).cast("string").alias("base_id"),
        )
    )


def _select_best_baseline_match(classified_matches: DataFrame) -> DataFrame:
    """Select the baseline ID with best (active, occurrences, score) and candidate.

    A baseline ID without a candidate-baseline pair cannot win.
    """
    best_match_pairs = classified_matches.where(F.col("_cluster_kind") == "best_match")
    is_candidate_baseline_pair = _column_equals("match_sub_type", CANDIDATE_BASELINE)
    per_baseline_id = (
        best_match_pairs.groupBy("cluster_id", "base_id")
        .agg(
            F.count(F.lit(1)).alias("occurrences"),
            F.max(
                F.when(
                    is_candidate_baseline_pair,
                    F.struct("composite_score", "id", "base_active"),
                )
            ).alias("best"),
        )
        .where(F.col("best").isNotNull())
    )
    ranked = per_baseline_id.select(
        "cluster_id",
        "base_id",
        F.struct(
            F.col("best.base_active").alias("active"),
            "occurrences",
            F.col("best.composite_score").alias("score"),
            "base_id",
            F.col("best.id").alias("id"),
        ).alias("rank"),
    )
    return (
        ranked.groupBy("cluster_id")
        .agg(F.max("rank").alias("winner"))
        .select(
            "cluster_id",
            F.col("winner.id").alias("id"),
            F.col("winner.base_id").alias("base_id"),
        )
    )


def _extract_pair_side_ids(
    matches: DataFrame, candidates_only: bool = True
) -> DataFrame:
    """Extract distinct ``(cluster_id, id)`` from either side of each pair."""
    left, right = matches, matches
    if candidates_only:
        left = matches.where(F.col("record_type") == CANDIDATE)
        right = matches.where(F.col("base_record_type") == CANDIDATE)
    return (
        left.select("cluster_id", "id")
        .unionByName(right.select("cluster_id", F.col("base_id").alias("id")))
        .distinct()
    )


def _join_on_cluster_id(
    left: DataFrame, right: DataFrame, how: str = "inner"
) -> DataFrame:
    """Join on ``cluster_id``, treating a null ``cluster_id`` as one cluster."""
    joined = left.alias("left").join(
        right.alias("right"),
        F.col("left.cluster_id").eqNullSafe(F.col("right.cluster_id")),
        how,
    )
    return joined.drop(F.col("right.cluster_id")) if how != "left_anti" else joined


def select_cluster_winners_and_fallout(
    matches: DataFrame, baseline: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Select one winning candidate per cluster and map losing candidates to it.

    Args:
        matches: Match pairs with a ``cluster_id``.
        baseline: The prepared baseline, which gives each ID its ``active`` state.

    Returns:
        ``(winners, fallout)``. ``winners`` has ``id`` and ``base_id``: the candidate
        that survives and the baseline ID it reuses, or null if it needs a new ID.
        ``fallout`` has ``id`` and ``duplicate_of``: each candidate that lost its
        cluster, and the winner it folds into. Clusters with only baseline records
        have no winner.

    Raises:
        ValueError: A cluster with candidates has no winner. This happens when it
            mixes baseline and candidate records without a candidate-baseline pair,
            and its candidates would otherwise be in neither output.
    """
    classified = _classify_clusters(
        add_base_active(coalesce_match_schema(matches), baseline)
    )
    winners_by_cluster = (
        _select_no_match_winners(classified)
        .unionByName(_select_best_inner_candidate(classified))
        .unionByName(_select_best_baseline_match(classified))
        .cache()
    )
    orphan_clusters = [
        row["cluster_id"]
        for row in _join_on_cluster_id(
            _extract_pair_side_ids(
                classified.where(F.col("_cluster_kind") == "best_match")
            )
            .select("cluster_id")
            .distinct(),
            winners_by_cluster,
            "left_anti",
        )
        .limit(5)
        .collect()
    ]
    if orphan_clusters:
        raise ValueError(f"clusters without a winner, for example {orphan_clusters}")
    # Candidate-only clusters fold in every ID, whatever its record type.
    candidate_ids_by_cluster = _extract_pair_side_ids(
        classified.where(F.col("_cluster_kind").isin("nomatch", "best_match"))
    ).unionByName(
        _extract_pair_side_ids(
            classified.where(F.col("_cluster_kind") == "candidate_only"),
            candidates_only=False,
        )
    )
    fallout = (
        _join_on_cluster_id(
            candidate_ids_by_cluster,
            winners_by_cluster.select("cluster_id", F.col("id").alias("duplicate_of")),
        )
        .where(~F.col("id").eqNullSafe(F.col("duplicate_of")))
        .select("id", "duplicate_of")
        .distinct()
    )
    return winners_by_cluster.select("id", "base_id"), fallout
