"""Tests winner selection and ID assignment on a few clusters."""

import pytest
from pyspark.sql import functions as F

from overture_matchers.ids import assign_ids
from overture_matchers.places.selection import select_winners

PAIR_COLUMNS = [
    "cluster_id",
    "id",
    "record_type",
    "base_id",
    "base_record_type",
    "match_type",
    "match_sub_type",
    "composite_score",
]
NOMATCH = ("nomatch", "nomatch")
BB = ("match", "baseline_baseline")
CC = ("match", "candidate_candidate")
CB = ("match", "candidate_baseline")


def pair(cluster, id_, base_id, kind, score=0.5):
    left = "baseline" if kind == BB else "candidate"
    right = "candidate" if kind == CC else "baseline"
    if kind == NOMATCH:
        right = None
    return (cluster, id_, left, base_id, right, *kind, score)


MATCHES = [
    pair("nomatch", "n", None, NOMATCH, -1.0),
    pair("baseline_only", "b1", "b2", BB),
    pair("cc", "c1", "c2", CC, 0.8),
    pair("cc", "c2", "c3", CC, 0.9),
    # B-active beats B-deleted, though the deleted one has the higher score
    pair("active", "x", "B-active", CB, 0.6),
    pair("active", "y", "B-active", CB, 0.7),
    pair("active", "x", "B-deleted", CB, 0.95),
    # occurrences beat score
    pair("occurrences", "z", "B-one", CB, 0.9),
    pair("occurrences", "z", "B-two", CB, 0.4),
    pair("occurrences", "w", "B-two", CB, 0.3),
    pair("occurrences", "z", "B-two", BB, 0.1),
    # a baseline ID with no candidate pair cannot win
    pair("no_candidate_pair", "q", "B-q", CB, 0.5),
    pair("no_candidate_pair", "B-r", "B-s", BB, 0.5),
    # a null candidate ID still folds into the winner
    pair("null_id", None, "d1", CC, 0.5),
    pair("null_id", "d1", "d2", CC, 0.5),
    # candidate-only clusters fold in IDs without a record type
    ("untyped", "e1", None, "e2", None, *CC, 0.4),
    # a null cluster_id is one cluster
    pair(None, "m", "B-m", CB, 0.5),
]
BASELINE = [("B-active", 1), ("B-deleted", 0), ("B-q", 1)]


@pytest.fixture
def selected(spark):
    matches = spark.createDataFrame(MATCHES, PAIR_COLUMNS)
    baseline = spark.createDataFrame(BASELINE, "id string, active int")
    winners, fallout = select_winners(matches, baseline)
    return (
        {tuple(r) for r in winners.collect()},
        {tuple(r) for r in fallout.collect()},
    )


def test_winners(selected):
    winners, _ = selected
    assert winners == {
        ("n", None),
        ("c2", None),
        ("y", "B-active"),
        ("z", "B-two"),
        ("q", "B-q"),
        ("d1", None),
        ("e2", None),
        ("m", "B-m"),
    }


def test_a_cluster_without_a_winner_raises(spark):
    # "s" is both a candidate and a baseline ID, so the cluster has no CB pair.
    matches = spark.createDataFrame(
        [pair("orphan", "c1", "s", CC), pair("orphan", "s", "b1", BB)], PAIR_COLUMNS
    )
    baseline = spark.createDataFrame([("b1", 1)], "id string, active int")
    with pytest.raises(ValueError, match="orphan"):
        select_winners(matches, baseline)


def test_fallout_folds_losing_candidates_into_the_winner(selected):
    _, fallout = selected
    assert fallout == {
        ("c1", "c2"),
        ("c3", "c2"),
        ("x", "y"),
        ("w", "z"),
        (None, "d1"),
        ("d2", "d1"),
        ("e1", "e2"),
    }


def test_select_winners_adds_missing_match_columns(spark):
    matches = spark.createDataFrame([("n",)], "id string").withColumn(
        "cluster_id", F.lit("c")
    )
    matches = matches.withColumn("match_type", F.lit("nomatch"))
    baseline = spark.createDataFrame([], "id string, active int")
    winners, fallout = select_winners(matches, baseline)
    assert [tuple(r) for r in winners.collect()] == [("n", None)]
    assert fallout.count() == 0


@pytest.mark.parametrize("output_path", [None, "run"])
def test_assign_ids_reuses_baseline_ids_and_mints_uuids(spark, tmp_path, output_path):
    # "b" is in the raw candidates twice; "ghost" wins but is not a raw candidate.
    candidates = spark.createDataFrame(
        [("a", "A"), ("b", "B"), ("b", "B2"), ("c", "C")], "id string, name string"
    )
    winners = spark.createDataFrame(
        [("a", "B-1"), ("b", None), ("ghost", None)], "id string, base_id string"
    )
    run = str(tmp_path / output_path) if output_path else None
    if run:
        spark.sparkContext.setCheckpointDir(str(tmp_path / "checkpoints"))
    final = assign_ids(candidates, winners, "acme", run)
    rows = {r["name"]: r for r in final.collect()}
    assert final.columns == ["id", "name", "provider"]
    assert set(rows) == {"A", "B", "B2"}
    assert rows["A"]["id"] == "B-1"
    assert len(rows["B"]["id"]) == 36
    assert rows["B"]["id"] != rows["B2"]["id"]
    assert {r["provider"] for r in rows.values()} == {"acme"}
    if run:
        id_map = sorted(
            tuple(r) for r in spark.read.parquet(f"{run}/idMap/type=place").collect()
        )
        assert id_map == sorted(
            [("a", "B-1"), ("b", rows["B"]["id"]), ("b", rows["B2"]["id"])]
        )
