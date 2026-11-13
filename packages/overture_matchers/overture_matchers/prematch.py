# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Pairs accepted before blocking and scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

CANDIDATE = "candidate"
BASELINE = "baseline"


def history_pairs(candidates: DataFrame) -> DataFrame:
    """One accepted pair per baseline ID in a candidate's ``base_ids``.

    Setup fills ``base_ids`` (or leaves it null when history matching is off),
    so this never looks at the ``matchHistory`` setting.

    Args:
        candidates: Candidates with ``id``, ``provider`` and ``base_ids``.

    Returns:
        Scored-pair columns, with ``composite_score`` 1.0 and ``prediction`` 1.
        The baseline side takes the candidate's provider.
    """
    return candidates.filter(F.col("base_ids").isNotNull()).select(
        "id",
        F.lit(CANDIDATE).alias("record_type"),
        F.explode("base_ids").alias("base_id"),
        F.lit(BASELINE).alias("base_record_type"),
        F.lit(1.0).alias("composite_score"),
        F.lit(1).alias("prediction"),
        "provider",
        F.col("provider").alias("base_provider"),
    )
