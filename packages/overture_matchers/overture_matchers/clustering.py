# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Match-result vocabulary and connected components over accepted pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from overture_matchers.prematch import BASELINE, CANDIDATE

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

MATCH = "match"
NO_MATCH = "nomatch"
BASELINE_BASELINE = f"{BASELINE}_{BASELINE}"
CANDIDATE_CANDIDATE = f"{CANDIDATE}_{CANDIDATE}"
CANDIDATE_BASELINE = f"{CANDIDATE}_{BASELINE}"


def connected_components(edges: DataFrame) -> DataFrame:
    """The connected component of every vertex in ``edges``.

    Runs the GraphFrames JVM class through py4j, so only the GraphFrames jar
    has to be on the classpath. The SparkContext needs a checkpoint directory.
    Turns adaptive execution off for the session, because GraphFrames 0.8.3
    on Spark 3.5 drops edges under it (graphframes/graphframes#453). The
    returned frame is lazy, so the setting stays off.

    Args:
        edges: ``src`` and ``dst`` vertex IDs.

    Returns:
        ``id`` and ``component``, a long shared by the vertices of one
        component. Its value depends on the vertices' partition order, so
        compare components by membership.
    """
    spark: SparkSession = edges.sparkSession
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    vertices = (
        edges.select(F.col("src").alias("id"))
        .union(edges.select(F.col("dst").alias("id")))
        .distinct()
    )
    graph = spark._jvm.org.graphframes.GraphFrame(vertices._jdf, edges._jdf)
    return DataFrame(graph.connectedComponents().run(), spark)
