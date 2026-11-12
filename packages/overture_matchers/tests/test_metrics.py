"""Tests the shared and places metrics on a few rows."""

from overture_matchers import metrics as shared
from overture_matchers.places import metrics as places

PAIRS = [
    "cluster_id",
    "id",
    "base_id",
    "match_type",
    "match_sub_type",
    "record_type",
    "base_record_type",
]


def inputs(spark, **optional):
    candidates = spark.createDataFrame([("a",), ("b",), ("c",)], "id string")
    baseline = spark.createDataFrame([("B1",), ("B2",)], "id string")
    prepared_candidates = spark.createDataFrame(
        [("a", ["B1"]), ("b", None), ("c", None)], "id string, base_ids array<string>"
    )
    prepared_baseline = spark.createDataFrame(
        [("B1", "acme", 1), ("B2", "other", 1), ("B3", "acme", 0)],
        "id string, provider string, active int",
    )
    matches = spark.createDataFrame(
        [
            ("1", "a", "B1", "match", "candidate_baseline", "candidate", "baseline"),
            ("2", "b", "c", "match", "candidate_candidate", "candidate", "candidate"),
            ("3", "z", None, "nomatch", "nomatch", "candidate", None),
        ],
        PAIRS,
    )
    results = spark.createDataFrame(
        [("B1", "acme"), ("new", "acme")], "id string, provider string"
    )
    return shared.MetricInputs(
        candidates,
        baseline,
        prepared_candidates,
        prepared_baseline,
        matches,
        results,
        "acme",
        **optional,
    )


def by_id(rows):
    return {(m, s): v for m, v, s in rows}


def test_total_counts(spark):
    counts = by_id(shared.total_counts(inputs(spark)))
    assert counts[("MatchCandidateInputCount", "")] == 3
    assert counts[("MatchBaselineInputCount", "")] == 2
    assert counts[("MatchProviderBaselineInputCount", "")] == 2
    assert counts[("MatchActiveBaselineEntityCount", "")] == 2
    assert counts[("MatchResultsCount", "")] == 2


def test_history_match(spark):
    assert by_id(shared.history_match(inputs(spark))) == {
        ("MatchHistoryMatchCount", ""): 1,
        ("MatchHistoryMatchRate", ""): 33.33,
    }


def test_match_type_distribution(spark):
    rows = by_id(places.match_type_distribution(inputs(spark)))
    assert rows[("MatchTypeDistribution", "candidate_baseline")] == 1
    assert rows[("MatchCandidateDuplicates", "")] == 2


def test_blocking_metrics_need_blocking(spark):
    assert places.blocking_metrics(inputs(spark)) == []


def test_round_half_up():
    assert shared.round_half_up(0.125, 2) == 0.13


def test_compute_skips_a_failing_metric(spark):
    def boom(_):
        raise ValueError

    rows = shared.compute(
        inputs(spark), [("boom", boom), ("History", shared.history_match)]
    )
    assert len(rows) == 2


def test_to_frame_keeps_first_row_per_key(spark):
    frame = shared.to_frame(
        spark,
        [("M", 1.0, ""), ("M", 2.0, ""), ("M", 3.0, "x")],
        "places",
        "place",
        "acme",
    )
    rows = {(r["metric_id"], r["sub_partition"]): r for r in frame.collect()}
    assert {k: r["value"] for k, r in rows.items()} == {("M", ""): 1.0, ("M", "x"): 3.0}
    assert {r["primary_partition"] for r in rows.values()} == {"acme-places"}
