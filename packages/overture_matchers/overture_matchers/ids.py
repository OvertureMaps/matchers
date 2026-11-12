# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""GERS ID minting for candidates that win their cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from overture_matchers.core import io

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

ID_MAP = "idMap"


def assign_ids(
    candidates: DataFrame,
    winners: DataFrame,
    provider: str,
    output_path: str | None = None,
    feature_type: str = "place",
) -> DataFrame:
    """Give each winning candidate its baseline ID, or a new UUID if it has none.

    Candidates that are not winners are dropped. The ``id`` to ``assigned_id`` mapping
    is checkpointed as ``idMap``, and the result is built from that checkpoint so the
    mapping and the result agree on the random UUIDs.

    Args:
        candidates: Raw candidates.
        winners: ``id`` and ``base_id`` of each winner.
        provider: The provider written to the result.
        output_path: Run folder for the ``idMap`` checkpoint, or None to stay in memory.
        feature_type: Feature type of the ``idMap`` folder.

    Returns:
        The winning candidates with their assigned ``id`` and ``provider``.
    """
    mapping = winners.select(
        "id", F.coalesce("base_id", F.expr("uuid()")).alias("assigned_id")
    ).cache()
    mapping = io.checkpoint(mapping, output_path, ID_MAP, feature_type)
    return (
        candidates.alias("c")
        .join(mapping.alias("m"), F.col("c.id") == F.col("m.id"))
        .select("c.*", "m.assigned_id")
        .withColumn("id", F.col("assigned_id"))
        .drop("assigned_id")
        .withColumn("provider", F.lit(provider))
    )
