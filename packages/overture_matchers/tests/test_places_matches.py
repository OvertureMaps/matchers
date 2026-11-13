# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for accepted place pairs, their clusters and the NO_MATCH rows."""

import pytest
from pyspark.sql import functions as F

from overture_matchers.places.matches import (
    MATCH_COLUMNS,
    SCORED_PAIR_TYPES,
    accepted_pairs,
    match_results,
    match_rows,
    no_match_rows,
)

C, B = "candidate", "baseline"
PAIR_SCHEMA = ", ".join(f"{c} {t}" for c, t in SCORED_PAIR_TYPES.items())


@pytest.fixture
def checkpoint_dir(spark, tmp_path):
    spark.sparkContext.setCheckpointDir(str(tmp_path / "checkpoints"))


def pairs(spark, *rows):
    """Accepted pairs from ``(id, record_type, base_id, base_record_type)``."""
    return spark.createDataFrame(
        [(a, ra, b, rb, 0.5, 1, f"p{a}", f"p{b}") for a, ra, b, rb in rows],
        PAIR_SCHEMA,
    )


def clusters(rows):
    """The set of ID sets that share a ``cluster_id``."""
    members = {}
    for r in rows:
        members.setdefault(r.cluster_id, set()).update({r.id, r.base_id} - {None})
    return {frozenset(m) for m in members.values()}


def test_accepted_pairs_keep_predicted_matches_and_history(spark):
    scores = spark.createDataFrame(
        [
            ("a", C, "b", B, 0.9, 1, "p", "q", "x"),
            ("c", C, "d", B, 0.2, 0, "p", "q", "x"),
        ],
        "id string, record_type string, base_id string, base_record_type string,"
        "composite_score float, prediction long, provider string,"
        "base_provider string, other string",
    )
    history = spark.createDataFrame([("e", C, "f", B, 1.0, 1, "p", "p")], PAIR_SCHEMA)
    out = accepted_pairs(scores, history)
    assert dict(out.dtypes) == SCORED_PAIR_TYPES
    assert sorted(r.id for r in out.collect()) == ["a", "e"]


def test_match_rows_swap_baseline_candidate_pairs_and_set_sub_types(spark):
    accepted = pairs(
        spark,
        ("b1", B, "c1", C),
        ("c2", C, "b2", B),
        ("b3", B, "b4", B),
        ("c3", C, "c4", C),
    )
    components = spark.createDataFrame(
        [(i, n) for n, i in enumerate(["b1", "c2", "b3", "c3"])],
        "id string, component long",
    )
    out = match_rows(accepted, components)
    assert out.columns == list(MATCH_COLUMNS)
    rows = {r.id: r for r in out.collect()}
    assert rows["c1"].asDict() == {
        "id": "c1",
        "provider": "pc1",
        "record_type": C,
        "base_id": "b1",
        "base_provider": "pb1",
        "base_record_type": B,
        "composite_score": 0.5,
        "cluster_id": "0",
        "match_type": "match",
        "match_sub_type": "candidate_baseline",
        "base_active": 1,
    }
    assert (rows["c2"].base_id, rows["c2"].match_sub_type) == (
        "b2",
        "candidate_baseline",
    )
    assert rows["b3"].match_sub_type == "baseline_baseline"
    assert rows["c3"].match_sub_type == "candidate_candidate"


def components_of(spark, *ids):
    return spark.createDataFrame([(i, 0) for i in ids], "id string, component long")


def test_match_rows_raise_on_an_unknown_record_type(spark):
    out = match_rows(pairs(spark, ("x", "other", "y", B)), components_of(spark, "x"))
    with pytest.raises(Exception, match="unknown record types: other, baseline"):
        out.collect()


def test_match_rows_raise_on_a_null_score(spark):
    accepted = pairs(spark, ("c1", C, "b1", B)).withColumn(
        "composite_score", F.lit(None).cast("double")
    )
    out = match_rows(accepted, components_of(spark, "c1"))
    with pytest.raises(Exception, match="null composite_score"):
        out.collect()


def test_no_match_rows_cover_candidates_in_no_pair(spark):
    candidates = spark.createDataFrame(
        [("c1", "p"), ("c2", "p"), ("c3", "p"), ("c4", "p")],
        "id string, provider string",
    )
    accepted = pairs(spark, ("c1", C, "b1", B), ("b2", B, "c2", C))
    out = no_match_rows(candidates, accepted)
    assert out.columns == list(MATCH_COLUMNS)
    rows = out.collect()
    assert sorted(r.id for r in rows) == ["c3", "c4"]
    assert {
        (
            r.record_type,
            r.base_id,
            r.composite_score,
            r.match_type,
            r.match_sub_type,
            r.base_active,
        )
        for r in rows
    } == {(C, None, 0.0, "nomatch", "nomatch", 1)}
    assert len({r.cluster_id for r in rows}) == 2


def test_match_results_cluster_connected_pairs(spark, checkpoint_dir):
    candidates = spark.createDataFrame(
        [("c1", "p"), ("c2", "p"), ("c3", "p")], "id string, provider string"
    )
    accepted = pairs(spark, ("c1", C, "b1", B), ("b1", B, "b2", B), ("b3", B, "c2", C))
    out = match_results(accepted, candidates)
    required = {f.name for f in out.schema if not f.nullable}
    assert {"composite_score", "base_active"} <= required
    rows = out.collect()
    assert clusters(r for r in rows if r.match_type == "match") == {
        frozenset({"c1", "b1", "b2"}),
        frozenset({"c2", "b3"}),
    }
    assert [r.id for r in rows if r.match_type == "nomatch"] == ["c3"]
