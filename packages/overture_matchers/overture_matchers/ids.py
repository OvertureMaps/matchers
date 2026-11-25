# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""GERS ID assignment for selected matches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from overture_matchers.core import io

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

ID_MAP = "idMap"


def assign_ids(
    candidates: DataFrame,
    selected_matches: DataFrame,
    provider: str,
    output_path: str | None = None,
    feature_type: str = "place",
) -> DataFrame:
    """Give each selected candidate its baseline ID, or a new UUID if it has none.

    If a candidate is not in ``selected_matches``, we drop it. We draw a new UUID
    for each row, so duplicate raw rows get different IDs. We checkpoint the rows
    before writing ``idMap``, so ``idMap`` and the returned rows have the same
    UUIDs.

    Args:
        candidates: Raw candidates.
        selected_matches: ``id`` and ``base_id`` of each selected match.
        provider: The provider written to the result.
        output_path: Run folder for ``idMap``, or None to stay in memory. With a
            run folder, the SparkContext needs a checkpoint directory.
        feature_type: Feature type of the ``idMap`` folder.

    Returns:
        The selected candidates with their assigned ``id`` and ``provider``.
    """
    assigned = (
        candidates.alias("candidates")
        .join(
            selected_matches.alias("selected_matches"),
            F.col("candidates.id") == F.col("selected_matches.id"),
        )
        .select(
            "candidates.*",
            F.coalesce("selected_matches.base_id", F.expr("uuid()")).alias(
                "assigned_id"
            ),
        )
    )
    if output_path is None:
        assigned = assigned.cache()
    else:
        assigned = assigned.checkpoint()
        io.write_output(
            output_path, ID_MAP, feature_type, assigned.select("id", "assigned_id")
        )
    return (
        assigned.withColumn("id", F.col("assigned_id"))
        .drop("assigned_id")
        .withColumn("provider", F.lit(provider))
    )
