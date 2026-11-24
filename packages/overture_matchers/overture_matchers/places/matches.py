# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Accepted place pairs, their clusters and the NO_MATCH rows of ``matches``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from overture_matchers.clustering import (
    BASELINE_BASELINE,
    CANDIDATE_BASELINE,
    CANDIDATE_CANDIDATE,
    MATCH,
    MATCH_RESULT_COLUMNS,
    NO_MATCH,
    compute_connected_components,
)
from overture_matchers.prematch import BASELINE, CANDIDATE

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

SCORED_PAIR_COLUMN_TYPES = {
    "id": "string",
    "record_type": "string",
    "base_id": "string",
    "base_record_type": "string",
    "composite_score": "double",
    "prediction": "int",
    "provider": "string",
    "base_provider": "string",
}
# Paired sides that swap so the candidate lands on the ``id`` side.
SWAPPED_SIDE_COLUMNS = (
    ("id", "base_id"),
    ("record_type", "base_record_type"),
    ("provider", "base_provider"),
)


def select_accepted_pairs(scores: DataFrame, history_pairs: DataFrame) -> DataFrame:
    """Select scored pairs and history pairs with ``prediction`` of at least 1.

    Args:
        scores: Scored pairs, with at least the scored-pair columns.
        history_pairs: History pairs, with the scored-pair columns.

    Returns:
        The scored-pair columns, cast to ``SCORED_PAIR_COLUMN_TYPES``.
    """
    columns = [
        F.col(column_name).cast(column_type).alias(column_name)
        for column_name, column_type in SCORED_PAIR_COLUMN_TYPES.items()
    ]
    return (
        scores.select(*columns)
        .unionByName(history_pairs.select(*columns))
        .where(F.col("prediction") >= 1)
    )


def _classify_match_sub_type(record_type: Column, base_record_type: Column) -> Column:
    """Classify the pair's sub-type, raising on an unknown record type."""
    return (
        F.when(
            ((record_type == CANDIDATE) & (base_record_type == BASELINE))
            | ((record_type == BASELINE) & (base_record_type == CANDIDATE)),
            CANDIDATE_BASELINE,
        )
        .when(
            (record_type == BASELINE) & (base_record_type == BASELINE),
            BASELINE_BASELINE,
        )
        .when(
            (record_type == CANDIDATE) & (base_record_type == CANDIDATE),
            CANDIDATE_CANDIDATE,
        )
        .otherwise(
            F.raise_error(
                F.format_string(
                    "unknown record types: %s, %s", record_type, base_record_type
                )
            )
        )
    )


def _require_non_null_score(composite_score: Column) -> Column:
    """Return ``composite_score`` as non-nullable, raising on a null."""
    # coalesce is non-nullable when its last argument is. raise_error fires first.
    checked = F.when(
        composite_score.isNull(), F.raise_error(F.lit("null composite_score"))
    )
    return F.coalesce(checked.otherwise(composite_score), F.lit(0.0))


def build_clustered_match_rows(
    accepted_pairs: DataFrame, components: DataFrame
) -> DataFrame:
    """Build one MATCH row per accepted pair, in the cluster of its ``id``.

    Args:
        accepted_pairs: Accepted pairs, in the scored-pair columns.
        components: ``id`` and ``component`` for every ID in ``accepted_pairs``.

    Returns:
        ``matches`` rows. A baseline-candidate pair swaps sides, so the
        candidate is on the ``id`` side.
    """
    aliased_pairs = accepted_pairs.alias("pairs")
    should_swap = (F.col("pairs.record_type") == BASELINE) & (
        F.col("pairs.base_record_type") == CANDIDATE
    )
    sides = {}
    for left_column, right_column in SWAPPED_SIDE_COLUMNS:
        sides[left_column] = F.when(
            should_swap, F.col(f"pairs.{right_column}")
        ).otherwise(F.col(f"pairs.{left_column}"))
        sides[right_column] = F.when(
            should_swap, F.col(f"pairs.{left_column}")
        ).otherwise(F.col(f"pairs.{right_column}"))
    sub_type = _classify_match_sub_type(
        F.col("pairs.record_type"), F.col("pairs.base_record_type")
    )
    columns = {
        **sides,
        "composite_score": _require_non_null_score(F.col("pairs.composite_score")),
        "cluster_id": F.col("components.component").cast("string"),
        # Always MATCH. Deriving it from the data keeps it nullable in the
        # written schema, where a literal would be folded to non-nullable.
        "match_type": F.when(sub_type.isNotNull(), F.lit(MATCH)),
        "match_sub_type": sub_type,
        "base_active": F.lit(1),
    }
    return aliased_pairs.join(
        components.alias("components"), F.col("pairs.id") == F.col("components.id")
    ).select(
        *(
            columns[column_name].alias(column_name)
            for column_name in MATCH_RESULT_COLUMNS
        )
    )


def build_no_match_rows(candidates: DataFrame, accepted_pairs: DataFrame) -> DataFrame:
    """Build one NO_MATCH row, in a new random cluster, per unpaired candidate.

    Args:
        candidates: Candidates with ``id`` and ``provider``.
        accepted_pairs: Accepted pairs. An ID on either side counts as paired.

    Returns:
        ``matches`` rows with no baseline side and a score of 0.
    """
    paired_ids = accepted_pairs.select("id").union(
        accepted_pairs.select(F.col("base_id").alias("id"))
    )
    columns = {
        "id": F.col("id"),
        "provider": F.col("provider"),
        "record_type": F.lit(CANDIDATE),
        "base_id": F.lit(None).cast("string"),
        "base_provider": F.lit(None).cast("string"),
        "base_record_type": F.lit(None).cast("string"),
        "composite_score": F.lit(0.0),
        "cluster_id": F.expr("uuid()"),
        "match_type": F.lit(NO_MATCH),
        "match_sub_type": F.lit(NO_MATCH),
        "base_active": F.lit(1),
    }
    return (
        candidates.select("id", "provider")
        .join(paired_ids, "id", "left_anti")
        .select(
            *(
                columns[column_name].alias(column_name)
                for column_name in MATCH_RESULT_COLUMNS
            )
        )
    )


def build_match_results(accepted_pairs: DataFrame, candidates: DataFrame) -> DataFrame:
    """Build the ``matches`` output from clustered pairs and unpaired candidates.

    Args:
        accepted_pairs: Accepted pairs, in the scored-pair columns.
        candidates: Every candidate, with ``id`` and ``provider``.

    Returns:
        NO_MATCH rows, then MATCH rows. Unpaired baseline records get no row.
    """
    edges = accepted_pairs.select(
        F.col("id").alias("src"), F.col("base_id").alias("dst")
    )
    return build_no_match_rows(candidates, accepted_pairs).unionByName(
        build_clustered_match_rows(accepted_pairs, compute_connected_components(edges))
    )
