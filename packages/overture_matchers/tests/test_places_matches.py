# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for accepted place pairs, their clusters and the NO_MATCH rows."""

import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F

from overture_matchers.clustering import MATCH_RESULT_COLUMNS
from overture_matchers.places.matches import (
    SCORED_PAIR_COLUMN_TYPES,
    build_clustered_match_rows,
    build_match_results,
    build_no_match_rows,
    select_accepted_pairs,
)
from overture_matchers.prematch import BASELINE, CANDIDATE

PAIR_SCHEMA = ", ".join(
    f"{column_name} {column_type}"
    for column_name, column_type in SCORED_PAIR_COLUMN_TYPES.items()
)


def build_accepted_pairs(spark, *rows):
    """Build accepted pairs from ``(id, record_type, base_id, base_record_type)``."""
    return spark.createDataFrame(
        [
            (
                left_id,
                left_type,
                right_id,
                right_type,
                0.5,
                1,
                f"p{left_id}",
                f"p{right_id}",
            )
            for left_id, left_type, right_id, right_type in rows
        ],
        PAIR_SCHEMA,
    )


def collect_cluster_sets(rows):
    """Collect the set of ID sets that share a ``cluster_id``."""
    members_by_cluster = {}
    for row in rows:
        members_by_cluster.setdefault(row.cluster_id, set()).update(
            {row.id, row.base_id} - {None}
        )
    return {frozenset(members) for members in members_by_cluster.values()}


def test_select_accepted_pairs_keeps_predicted_matches_and_history(spark):
    scores = spark.createDataFrame(
        [
            ("a", CANDIDATE, "b", BASELINE, 0.9, 1, "p", "q", "x"),
            ("c", CANDIDATE, "d", BASELINE, 0.2, 0, "p", "q", "x"),
        ],
        "id string, record_type string, base_id string, base_record_type string,"
        "composite_score float, prediction long, provider string,"
        "base_provider string, other string",
    )
    history_pairs = spark.createDataFrame(
        [("e", CANDIDATE, "f", BASELINE, 1.0, 1, "p", "p")], PAIR_SCHEMA
    )
    out = select_accepted_pairs(scores, history_pairs)
    assert dict(out.dtypes) == SCORED_PAIR_COLUMN_TYPES
    assert sorted(row.id for row in out.collect()) == ["a", "e"]


def test_build_clustered_match_rows_swaps_baseline_candidate_pairs_and_sets_sub_types(
    spark,
):
    accepted_pairs = build_accepted_pairs(
        spark,
        ("b1", BASELINE, "c1", CANDIDATE),
        ("c2", CANDIDATE, "b2", BASELINE),
        ("b3", BASELINE, "b4", BASELINE),
        ("c3", CANDIDATE, "c4", CANDIDATE),
    )
    components = spark.createDataFrame(
        [
            (entity_id, index)
            for index, entity_id in enumerate(["b1", "c2", "b3", "c3"])
        ],
        "id string, component long",
    )
    out = build_clustered_match_rows(accepted_pairs, components)
    assert out.columns == list(MATCH_RESULT_COLUMNS)
    rows = {row.id: row for row in out.collect()}
    assert rows["c1"].asDict() == {
        "id": "c1",
        "provider": "pc1",
        "record_type": CANDIDATE,
        "base_id": "b1",
        "base_provider": "pb1",
        "base_record_type": BASELINE,
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


def build_components(spark, *entity_ids):
    return spark.createDataFrame(
        [(entity_id, 0) for entity_id in entity_ids], "id string, component long"
    )


def test_build_clustered_match_rows_raises_on_an_unknown_record_type(spark):
    out = build_clustered_match_rows(
        build_accepted_pairs(spark, ("x", "other", "y", BASELINE)),
        build_components(spark, "x"),
    )
    with pytest.raises(Exception, match="unknown record types: other, baseline"):
        out.collect()


def test_build_clustered_match_rows_raises_on_a_null_score(spark):
    accepted_pairs = build_accepted_pairs(
        spark, ("c1", CANDIDATE, "b1", BASELINE)
    ).withColumn("composite_score", F.lit(None).cast("double"))
    out = build_clustered_match_rows(accepted_pairs, build_components(spark, "c1"))
    with pytest.raises(Exception, match="null composite_score"):
        out.collect()


def test_build_no_match_rows_covers_candidates_in_no_pair(spark):
    candidates = spark.createDataFrame(
        [("c1", "p"), ("c2", "p"), ("c3", "p"), ("c4", "p")],
        "id string, provider string",
    )
    accepted_pairs = build_accepted_pairs(
        spark, ("c1", CANDIDATE, "b1", BASELINE), ("b2", BASELINE, "c2", CANDIDATE)
    )
    out = build_no_match_rows(candidates, accepted_pairs)
    assert out.columns == list(MATCH_RESULT_COLUMNS)
    rows = out.collect()
    assert sorted(row.id for row in rows) == ["c3", "c4"]
    assert {
        (
            row.record_type,
            row.base_id,
            row.composite_score,
            row.match_type,
            row.match_sub_type,
            row.base_active,
        )
        for row in rows
    } == {(CANDIDATE, None, 0.0, "nomatch", "nomatch", 1)}
    assert len({row.cluster_id for row in rows}) == 2


def test_build_match_results_clusters_connected_pairs(spark, graph_session, tmp_path):
    candidates = spark.createDataFrame(
        [("c1", "p"), ("c2", "p"), ("c3", "p")], "id string, provider string"
    )
    accepted_pairs = build_accepted_pairs(
        spark,
        ("c1", CANDIDATE, "b1", BASELINE),
        ("b1", BASELINE, "b2", BASELINE),
        ("b3", BASELINE, "c2", CANDIDATE),
    )
    out = build_match_results(accepted_pairs, candidates)
    out.write.parquet(str(tmp_path / "matches"))
    written = pq.read_schema(next((tmp_path / "matches").glob("*.parquet")))
    required = {field.name for field in written if not field.nullable}
    assert required == {"composite_score", "base_active"}
    rows = out.collect()
    assert collect_cluster_sets(row for row in rows if row.match_type == "match") == {
        frozenset({"c1", "b1", "b2"}),
        frozenset({"c2", "b3"}),
    }
    assert [row.id for row in rows if row.match_type == "nomatch"] == ["c3"]


def test_build_match_results_without_pairs_returns_all_no_match(spark, graph_session):
    candidates = spark.createDataFrame([("c1", "p")], "id string, provider string")
    out = build_match_results(build_accepted_pairs(spark), candidates).collect()
    assert [(row.id, row.match_type) for row in out] == [("c1", "nomatch")]
