# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Cluster winner selection: one surviving candidate per cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

CANDIDATE = "candidate"
NOMATCH = "nomatch"
BASELINE_BASELINE = "baseline_baseline"
CANDIDATE_CANDIDATE = "candidate_candidate"
CANDIDATE_BASELINE = "candidate_baseline"

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


def coalesce_schema(matches: DataFrame) -> DataFrame:
    """Add the match columns that ``matches`` lacks, with null (or default) values."""
    for name in STRING_COLUMNS:
        if name not in matches.columns:
            matches = matches.withColumn(name, F.lit(None).cast("string"))
    if "composite_score" not in matches.columns:
        matches = matches.withColumn("composite_score", F.lit(-1.0))
    if "base_active" not in matches.columns:
        matches = matches.withColumn("base_active", F.lit(1))
    return matches


def with_base_active(matches: DataFrame, baseline: DataFrame) -> DataFrame:
    """Set ``base_active`` to the max ``active`` of the baseline ID, or 1 if unknown."""
    active = baseline.groupBy(F.col("id").alias("_active_id")).agg(
        F.max("active").alias("_active")
    )
    return (
        matches.drop("base_active")
        .join(active, F.col("base_id") == F.col("_active_id"), "left")
        .withColumn("base_active", F.coalesce("_active", F.lit(1)).cast("int"))
        .drop("_active_id", "_active")
    )


def _is(column: str, value: str):
    # A null column never equals the value.
    return F.coalesce(F.col(column) == value, F.lit(False))


def _flag_clusters(matches: DataFrame) -> DataFrame:
    """Add ``_kind``: nomatch, baseline_only, candidate_only or best_match."""
    w = Window.partitionBy("cluster_id")
    size = F.count(F.lit(1)).over(w)
    all_bb = F.sum(_is("match_sub_type", BASELINE_BASELINE).cast("int")).over(w) == size
    all_cc = (
        F.sum(_is("match_sub_type", CANDIDATE_CANDIDATE).cast("int")).over(w) == size
    )
    kind = (
        F.when((size == 1) & _is("match_type", NOMATCH), "nomatch")
        .when(all_bb, "baseline_only")
        .when(all_cc, "candidate_only")
        .otherwise("best_match")
    )
    return matches.withColumn("_kind", kind)


def _nomatch_winners(rows: DataFrame) -> DataFrame:
    return rows.where(F.col("_kind") == "nomatch").select(
        "cluster_id", "id", F.lit(None).cast("string").alias("base_id")
    )


def _candidate_only_winners(rows: DataFrame) -> DataFrame:
    """The ID seen most often on either side, then by max score, then by ID."""
    rows = rows.where(F.col("_kind") == "candidate_only")
    sides = rows.select("cluster_id", "id", "composite_score").unionByName(
        rows.select("cluster_id", F.col("base_id").alias("id"), "composite_score")
    )
    per_id = sides.groupBy("cluster_id", "id").agg(
        F.count(F.lit(1)).alias("n"), F.max("composite_score").alias("score")
    )
    return (
        per_id.groupBy("cluster_id")
        .agg(F.max(F.struct("n", "score", "id")).alias("w"))
        .select(
            "cluster_id",
            F.col("w.id").alias("id"),
            F.lit(None).cast("string").alias("base_id"),
        )
    )


def _best_match_winners(rows: DataFrame) -> DataFrame:
    """The baseline ID with the best (active, occurrences, score) and its best candidate.

    A baseline ID without a candidate-baseline pair cannot win.
    """
    rows = rows.where(F.col("_kind") == "best_match")
    is_pair = _is("match_sub_type", CANDIDATE_BASELINE)
    per_base = (
        rows.groupBy("cluster_id", "base_id")
        .agg(
            F.count(F.lit(1)).alias("occurrences"),
            F.max(
                F.when(is_pair, F.struct("composite_score", "id", "base_active"))
            ).alias("best"),
        )
        .where(F.col("best").isNotNull())
    )
    ranked = per_base.select(
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
        .agg(F.max("rank").alias("w"))
        .select(
            "cluster_id", F.col("w.id").alias("id"), F.col("w.base_id").alias("base_id")
        )
    )


def _candidate_ids(rows: DataFrame) -> DataFrame:
    """Distinct (cluster_id, id) for candidate records on either side of a pair."""
    left = rows.where(F.col("record_type") == CANDIDATE).select("cluster_id", "id")
    right = rows.where(F.col("base_record_type") == CANDIDATE).select(
        "cluster_id", F.col("base_id").alias("id")
    )
    return left.unionByName(right).distinct()


def select_winners(
    matches: DataFrame, baseline: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Pick one winner per cluster.

    Args:
        matches: Match pairs with a ``cluster_id``.
        baseline: The prepared baseline, which gives each ID its ``active`` state.

    Returns:
        ``(winners, fallout)``. ``winners`` has ``id`` and ``base_id``: the candidate
        that survives and the baseline ID it reuses, or null if it needs a new ID.
        ``fallout`` has ``id`` and ``duplicate_of``: each candidate that lost its
        cluster, and the winner it folds into. Clusters with only baseline records
        have no winner.
    """
    rows = _flag_clusters(with_base_active(coalesce_schema(matches), baseline))
    by_cluster = (
        _nomatch_winners(rows)
        .unionByName(_candidate_only_winners(rows))
        .unionByName(_best_match_winners(rows))
        .cache()
    )
    fallout = (
        _candidate_ids(rows.where(F.col("_kind") != "baseline_only"))
        .join(
            by_cluster.select("cluster_id", F.col("id").alias("duplicate_of")),
            "cluster_id",
        )
        .where(F.col("id") != F.col("duplicate_of"))
        .select("id", "duplicate_of")
    )
    return by_cluster.select("id", "base_id"), fallout
