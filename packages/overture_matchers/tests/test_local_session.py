# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Smoke tests that the local session loads Sedona and GraphFrames."""

import pytest


def test_sedona_functions_blocking_uses(spark):
    row = spark.sql(
        "SELECT ST_DistanceSphere(ST_Point(0, 0), ST_Point(0, 1)) AS meters,"
        " ST_H3CellIDs(ST_Point(0, 0), 9, false) AS cells"
    ).first()
    assert row.meters == pytest.approx(111_195, rel=1e-3)
    assert len(row.cells) == 1


def test_graphframes_connected_components(spark, tmp_path):
    from graphframes import GraphFrame

    spark.sparkContext.setCheckpointDir(str(tmp_path))
    vertices = spark.createDataFrame([(i,) for i in range(4)], ["id"])
    edges = spark.createDataFrame([(0, 1), (2, 3)], ["src", "dst"])
    rows = GraphFrame(vertices, edges).connectedComponents().collect()
    component = {r.id: r.component for r in rows}
    assert component[0] == component[1] != component[2] == component[3]
