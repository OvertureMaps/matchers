# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Smoke tests that the local session loads Sedona and GraphFrames."""

import pytest

from overture_matchers.clustering import connected_components


def test_sedona_functions_blocking_uses(spark):
    row = spark.sql(
        "SELECT ST_DistanceSphere(ST_Point(0, 0), ST_Point(0, 1)) AS meters,"
        " ST_H3CellIDs(ST_Point(0, 0), 9, false) AS cells"
    ).first()
    assert row.meters == pytest.approx(111_195, rel=1e-3)
    assert len(row.cells) == 1


def test_graphframes_connected_components(graph_session):
    edges = graph_session.createDataFrame(
        [("a", "b"), ("b", "c"), ("d", "e")], ["src", "dst"]
    )
    component = {r.id: r.component for r in connected_components(edges).collect()}
    assert component["a"] == component["b"] == component["c"] != component["d"]
    assert component["d"] == component["e"]
