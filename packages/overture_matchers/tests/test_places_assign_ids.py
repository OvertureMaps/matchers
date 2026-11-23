"""Tests winner selection and ID assignment on a few clusters."""

import pytest
from pyspark.sql import functions as F

from overture_matchers.ids import assign_ids
from overture_matchers.places.selection import select_cluster_winners_and_fallout

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
NO_MATCH_KIND = ("nomatch", "nomatch")
BASELINE_BASELINE_KIND = ("match", "baseline_baseline")
CANDIDATE_CANDIDATE_KIND = ("match", "candidate_candidate")
CANDIDATE_BASELINE_KIND = ("match", "candidate_baseline")


def build_match_pair(cluster_id, entity_id, base_id, kind, score=0.5):
    left_type = "baseline" if kind == BASELINE_BASELINE_KIND else "candidate"
    right_type = "candidate" if kind == CANDIDATE_CANDIDATE_KIND else "baseline"
    if kind == NO_MATCH_KIND:
        right_type = None
    return (cluster_id, entity_id, left_type, base_id, right_type, *kind, score)


MATCHES = [
    build_match_pair("nomatch", "n", None, NO_MATCH_KIND, -1.0),
    build_match_pair("baseline_only", "b1", "b2", BASELINE_BASELINE_KIND),
    build_match_pair("cc", "c1", "c2", CANDIDATE_CANDIDATE_KIND, 0.8),
    build_match_pair("cc", "c2", "c3", CANDIDATE_CANDIDATE_KIND, 0.9),
    # B-active beats B-deleted, though the deleted one has the higher score
    build_match_pair("active", "x", "B-active", CANDIDATE_BASELINE_KIND, 0.6),
    build_match_pair("active", "y", "B-active", CANDIDATE_BASELINE_KIND, 0.7),
    build_match_pair("active", "x", "B-deleted", CANDIDATE_BASELINE_KIND, 0.95),
    # occurrences beat score
    build_match_pair("occurrences", "z", "B-one", CANDIDATE_BASELINE_KIND, 0.9),
    build_match_pair("occurrences", "z", "B-two", CANDIDATE_BASELINE_KIND, 0.4),
    build_match_pair("occurrences", "w", "B-two", CANDIDATE_BASELINE_KIND, 0.3),
    build_match_pair("occurrences", "z", "B-two", BASELINE_BASELINE_KIND, 0.1),
    # a baseline ID with no candidate pair cannot win
    build_match_pair("no_candidate_pair", "q", "B-q", CANDIDATE_BASELINE_KIND, 0.5),
    build_match_pair("no_candidate_pair", "B-r", "B-s", BASELINE_BASELINE_KIND, 0.5),
    # a null candidate ID still folds into the winner
    build_match_pair("null_id", None, "d1", CANDIDATE_CANDIDATE_KIND, 0.5),
    build_match_pair("null_id", "d1", "d2", CANDIDATE_CANDIDATE_KIND, 0.5),
    # candidate-only clusters fold in IDs without a record type
    ("untyped", "e1", None, "e2", None, *CANDIDATE_CANDIDATE_KIND, 0.4),
    # a null cluster_id is one cluster
    build_match_pair(None, "m", "B-m", CANDIDATE_BASELINE_KIND, 0.5),
]
BASELINE = [("B-active", 1), ("B-deleted", 0), ("B-q", 1)]


@pytest.fixture
def selected(spark):
    matches = spark.createDataFrame(MATCHES, PAIR_COLUMNS)
    baseline = spark.createDataFrame(BASELINE, "id string, active int")
    winners, fallout = select_cluster_winners_and_fallout(matches, baseline)
    return (
        {tuple(row) for row in winners.collect()},
        {tuple(row) for row in fallout.collect()},
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
        [
            build_match_pair("orphan", "c1", "s", CANDIDATE_CANDIDATE_KIND),
            build_match_pair("orphan", "s", "b1", BASELINE_BASELINE_KIND),
        ],
        PAIR_COLUMNS,
    )
    baseline = spark.createDataFrame([("b1", 1)], "id string, active int")
    with pytest.raises(ValueError, match="orphan"):
        select_cluster_winners_and_fallout(matches, baseline)


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


def test_select_cluster_winners_and_fallout_adds_missing_match_columns(spark):
    matches = spark.createDataFrame([("n",)], "id string").withColumn(
        "cluster_id", F.lit("c")
    )
    matches = matches.withColumn("match_type", F.lit("nomatch"))
    baseline = spark.createDataFrame([], "id string, active int")
    winners, fallout = select_cluster_winners_and_fallout(matches, baseline)
    assert [tuple(row) for row in winners.collect()] == [("n", None)]
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
