# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Best match selection: one selected match per cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F

from overture_matchers.clustering import (
    BASELINE_BASELINE,
    CANDIDATE_BASELINE,
    CANDIDATE_CANDIDATE,
    MATCH_RESULT_COLUMNS,
    NO_MATCH,
)
from overture_matchers.prematch import CANDIDATE

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# Defaults of the numeric match columns. The rest are strings.
NUMERIC_COLUMN_DEFAULTS = {"composite_score": -1.0, "base_active": 1}


def coalesce_match_schema(matches: DataFrame) -> DataFrame:
    """Add the match columns that ``matches`` lacks, with null (or default) values."""
    for column_name in MATCH_RESULT_COLUMNS:
        if column_name not in matches.columns:
            default = NUMERIC_COLUMN_DEFAULTS.get(column_name)
            value = F.lit(None).cast("string") if default is None else F.lit(default)
            matches = matches.withColumn(column_name, value)
    return matches


def add_base_active(matches: DataFrame, baseline: DataFrame) -> DataFrame:
    """Check if any of the providers for a given GERS ID are active.

    If they are, or the GERS ID is missing from the baseline, set base_active=1 on the
    matches to that GERS ID.
    """
    active_by_id = baseline.groupBy(F.col("id").alias("_active_id")).agg(
        F.max("active").alias("_active")
    )
    return (
        matches.drop("base_active")
        .join(active_by_id, F.col("base_id") == F.col("_active_id"), "left")
        .withColumn("base_active", F.coalesce("_active", F.lit(1)).cast("int"))
        .drop("_active_id", "_active")
    )


def _classify_clusters(matches: DataFrame) -> DataFrame:
    """Add ``_cluster_type``: nomatch, baseline_only, candidate_only or best_match."""
    cluster_window = Window.partitionBy("cluster_id")
    cluster_size = F.count(F.lit(1)).over(cluster_window)
    all_baseline_baseline = F.bool_and(
        F.col("match_sub_type").eqNullSafe(BASELINE_BASELINE)
    ).over(cluster_window)
    all_candidate_candidate = F.bool_and(
        F.col("match_sub_type").eqNullSafe(CANDIDATE_CANDIDATE)
    ).over(cluster_window)
    cluster_type = (
        F.when(
            (cluster_size == 1) & F.col("match_type").eqNullSafe(NO_MATCH),
            "nomatch",
        )
        .when(all_baseline_baseline, "baseline_only")
        .when(all_candidate_candidate, "candidate_only")
        .otherwise("best_match")
    )
    return matches.withColumn("_cluster_type", cluster_type)


def _select_no_matches(classified_matches: DataFrame) -> DataFrame:
    return classified_matches.where(F.col("_cluster_type") == "nomatch").select(
        "cluster_id", "id", F.lit(None).cast("string").alias("base_id")
    )


def _select_best_inner_candidate(classified_matches: DataFrame) -> DataFrame:
    """Select the ID seen most often on either side, then by max score, then by ID."""
    candidate_pairs = classified_matches.where(
        F.col("_cluster_type") == "candidate_only"
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
        .agg(F.max(F.struct("occurrences", "score", "id")).alias("survivor"))
        .select(
            "cluster_id",
            F.col("survivor.id").alias("id"),
            F.lit(None).cast("string").alias("base_id"),
        )
    )


def _select_best_baseline_match(classified_matches: DataFrame) -> DataFrame:
    """Select the best baseline ID in each cluster, and its best candidate.

    Each baseline ID's best candidate comes from its candidate-baseline pairs. Higher
    scores rank higher, then larger candidate IDs. Baseline IDs without
    candidate-baseline pairs are never selected.

    Baseline IDs are then ranked by these keys, in order of priority:

    1. Active baseline IDs rank above inactive ones.
    2. Baseline IDs that appear in more of the cluster's pairs rank higher.
    3. Higher best-candidate scores rank higher.
    4. Larger baseline IDs rank higher, then larger candidate IDs.
    """
    best_match_pairs = classified_matches.where(F.col("_cluster_type") == "best_match")
    is_candidate_baseline_pair = F.col("match_sub_type").eqNullSafe(CANDIDATE_BASELINE)
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
        .agg(F.max("rank").alias("best_match"))
        .select(
            "cluster_id",
            F.col("best_match.id").alias("id"),
            F.col("best_match.base_id").alias("base_id"),
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


def select_best_matches_and_fallout(
    matches: DataFrame, baseline: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Select one match per cluster, and map the other candidates to it as fallout.

    Args:
        matches: Match pairs with a ``cluster_id``.
        baseline: The prepared baseline, which gives each ID its ``active`` state.

    Returns:
        ``(selected_matches, fallout)``. ``selected_matches`` has ``id`` and
        ``base_id``: the candidate that survives and the baseline ID it reuses, or
        null if it needs a new ID. ``fallout`` has ``id`` and ``duplicate_of``: each
        other candidate in the cluster, and the selected candidate it duplicates.
        Clusters with only baseline records have no selected match.

    Raises:
        ValueError: A cluster with candidates has no selected match. This happens when it
            mixes baseline and candidate records without a candidate-baseline pair,
            and its candidates would otherwise be in neither output.
    """
    classified = _classify_clusters(
        add_base_active(coalesce_match_schema(matches), baseline)
    )
    selected_matches = (
        _select_no_matches(classified)
        .unionByName(_select_best_inner_candidate(classified))
        .unionByName(_select_best_baseline_match(classified))
        .cache()
    )
    orphan_clusters = [
        row["cluster_id"]
        for row in _join_on_cluster_id(
            _extract_pair_side_ids(
                classified.where(F.col("_cluster_type") == "best_match")
            )
            .select("cluster_id")
            .distinct(),
            selected_matches,
            "left_anti",
        )
        .limit(5)
        .collect()
    ]
    if orphan_clusters:
        raise ValueError(
            f"clusters without a selected match, for example {orphan_clusters}"
        )
    # In candidate-only clusters, every other ID is fallout, whatever its record type.
    candidate_ids_by_cluster = _extract_pair_side_ids(
        classified.where(F.col("_cluster_type").isin("nomatch", "best_match"))
    ).unionByName(
        _extract_pair_side_ids(
            classified.where(F.col("_cluster_type") == "candidate_only"),
            candidates_only=False,
        )
    )
    fallout = (
        _join_on_cluster_id(
            candidate_ids_by_cluster,
            selected_matches.select("cluster_id", F.col("id").alias("duplicate_of")),
        )
        .where(~F.col("id").eqNullSafe(F.col("duplicate_of")))
        .select("id", "duplicate_of")
        .distinct()
    )
    return selected_matches.select("id", "base_id"), fallout
