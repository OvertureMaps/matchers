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
MATCH_RESULT_COLUMNS = (
    "id",
    "provider",
    "record_type",
    "base_id",
    "base_provider",
    "base_record_type",
    "composite_score",
    "cluster_id",
    "match_type",
    "match_sub_type",
    "base_active",
)
# Session settings ``compute_connected_components`` needs. GraphFrames 0.8.3 on
# Spark 3.5 drops edges under adaptive execution (graphframes/graphframes#453),
# and Sedona 1.8.1's shaded GraphFrames 0.10.0 falls back to slow RDD
# checkpoints unless local checkpoints are enabled.
CONNECTED_COMPONENTS_CONF = {
    "spark.sql.adaptive.enabled": "false",
    "spark.graphframes.useLocalCheckpoints": "true",
}


def compute_connected_components(edges: DataFrame) -> DataFrame:
    """Compute the connected component of every vertex in ``edges``.

    Runs the GraphFrames JVM class through py4j, so only the GraphFrames jar
    has to be on the classpath, and Spark Connect is not supported. The
    SparkContext needs a checkpoint directory when GraphFrames 0.8.3 is first
    on the classpath, and the session needs ``CONNECTED_COMPONENTS_CONF`` until
    the result is written.

    Args:
        edges: ``src`` and ``dst`` vertex IDs.

    Returns:
        ``id`` and ``component``, a long shared by the vertices of one
        component. Its value depends on the vertices' partition order, so
        compare components by membership.
    """
    spark: SparkSession = edges.sparkSession
    # GraphFrames 0.10.0 reads conf from SparkSession.getActiveSession, which
    # is thread-local in the JVM and can be unset on a fresh Py4J thread.
    spark._jvm.org.apache.spark.sql.SparkSession.setActiveSession(spark._jsparkSession)
    vertices = (
        edges.select(F.col("src").alias("id"))
        .union(edges.select(F.col("dst").alias("id")))
        .distinct()
    )
    graph = spark._jvm.org.graphframes.GraphFrame(vertices._jdf, edges._jdf)
    return DataFrame(graph.connectedComponents().run(), spark)
