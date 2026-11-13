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
    MATCH_COLUMNS,
    NO_MATCH,
    connected_components,
)
from overture_matchers.prematch import BASELINE, CANDIDATE

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

SCORED_PAIR_TYPES = {
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
SWAPPED_SIDES = (
    ("id", "base_id"),
    ("record_type", "base_record_type"),
    ("provider", "base_provider"),
)


def accepted_pairs(scores: DataFrame, history: DataFrame) -> DataFrame:
    """Scored pairs and history pairs with ``prediction`` of at least 1.

    Args:
        scores: Scored pairs, with at least the scored-pair columns.
        history: History pairs, with the scored-pair columns.

    Returns:
        The scored-pair columns, cast to ``SCORED_PAIR_TYPES``.
    """
    columns = [F.col(c).cast(t).alias(c) for c, t in SCORED_PAIR_TYPES.items()]
    return (
        scores.select(*columns)
        .unionByName(history.select(*columns))
        .where(F.col("prediction") >= 1)
    )


def _match_sub_type(record_type: Column, base_record_type: Column) -> Column:
    """The pair's sub-type. A record type other than the two known ones raises."""
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


def _required_score(score: Column) -> Column:
    """``score``, raising on a null. The schema marks it non-nullable."""
    # coalesce is non-nullable when its last argument is. raise_error fires first.
    checked = F.when(score.isNull(), F.raise_error(F.lit("null composite_score")))
    return F.coalesce(checked.otherwise(score), F.lit(0.0))


def match_rows(pairs: DataFrame, components: DataFrame) -> DataFrame:
    """One MATCH row per accepted pair, in the cluster of its ``id``.

    Args:
        pairs: Accepted pairs, in the scored-pair columns.
        components: ``id`` and ``component`` for every ID in ``pairs``.

    Returns:
        ``matches`` rows. A baseline-candidate pair swaps sides, so the
        candidate is on the ``id`` side.
    """
    p = pairs.alias("p")
    swap = (F.col("p.record_type") == BASELINE) & (
        F.col("p.base_record_type") == CANDIDATE
    )
    sides = {}
    for left, right in SWAPPED_SIDES:
        sides[left] = F.when(swap, F.col(f"p.{right}")).otherwise(F.col(f"p.{left}"))
        sides[right] = F.when(swap, F.col(f"p.{left}")).otherwise(F.col(f"p.{right}"))
    sub_type = _match_sub_type(F.col("p.record_type"), F.col("p.base_record_type"))
    columns = {
        **sides,
        "composite_score": _required_score(F.col("p.composite_score")),
        "cluster_id": F.col("c.component").cast("string"),
        # Always MATCH. Deriving it from the data keeps it nullable in the
        # written schema, where a literal would be folded to non-nullable.
        "match_type": F.when(sub_type.isNotNull(), F.lit(MATCH)),
        "match_sub_type": sub_type,
        "base_active": F.lit(1),
    }
    return p.join(components.alias("c"), F.col("p.id") == F.col("c.id")).select(
        *(columns[c].alias(c) for c in MATCH_COLUMNS)
    )


def no_match_rows(candidates: DataFrame, pairs: DataFrame) -> DataFrame:
    """One NO_MATCH row, in a new random cluster, per candidate in no pair.

    Args:
        candidates: Candidates with ``id`` and ``provider``.
        pairs: Accepted pairs. An ID on either side counts as paired.

    Returns:
        ``matches`` rows with no baseline side and a score of 0.
    """
    paired = pairs.select("id").union(pairs.select(F.col("base_id").alias("id")))
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
        .join(paired, "id", "left_anti")
        .select(*(columns[c].alias(c) for c in MATCH_COLUMNS))
    )


def match_results(pairs: DataFrame, candidates: DataFrame) -> DataFrame:
    """The ``matches`` output: clustered pairs plus unpaired candidates.

    Args:
        pairs: Accepted pairs, in the scored-pair columns.
        candidates: Every candidate, with ``id`` and ``provider``.

    Returns:
        NO_MATCH rows, then MATCH rows. Unpaired baseline records get no row.
    """
    edges = pairs.select(F.col("id").alias("src"), F.col("base_id").alias("dst"))
    return no_match_rows(candidates, pairs).unionByName(
        match_rows(pairs, connected_components(edges))
    )
