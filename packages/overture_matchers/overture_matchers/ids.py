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
    """Give each winning candidate row its baseline ID, or a new UUID if it has none.

    Candidates that are not winners are dropped. Each row gets its own UUID, so
    duplicate raw rows get distinct IDs. The rows are pinned before the ``id`` to
    ``assigned_id`` mapping is written as ``idMap``, so the mapping and the result
    agree on the random UUIDs.

    Args:
        candidates: Raw candidates.
        winners: ``id`` and ``base_id`` of each winner.
        provider: The provider written to the result.
        output_path: Run folder for ``idMap``, or None to stay in memory. With a
            run folder, the SparkContext needs a checkpoint directory.
        feature_type: Feature type of the ``idMap`` folder.

    Returns:
        The winning candidates with their assigned ``id`` and ``provider``.
    """
    assigned = (
        candidates.alias("c")
        .join(winners.alias("w"), F.col("c.id") == F.col("w.id"))
        .select("c.*", F.coalesce("w.base_id", F.expr("uuid()")).alias("assigned_id"))
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
