# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests the shared and places metrics on a few rows."""

import json
import zipfile

import pytest

from overture_matchers import metrics as shared
from overture_matchers.places import metrics as places
from overture_matchers.places import model_jar

PAIRS = [
    "cluster_id",
    "id",
    "base_id",
    "match_type",
    "match_sub_type",
    "record_type",
    "base_record_type",
]


def build_metric_inputs(spark, **optional):
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


def index_metrics_by_key(rows):
    return {
        (metric_id, sub_partition): value for metric_id, value, sub_partition in rows
    }


def test_total_counts(spark):
    counts = index_metrics_by_key(
        shared.compute_total_counts(build_metric_inputs(spark))
    )
    assert counts[("MatchCandidateInputCount", "")] == 3
    assert counts[("MatchBaselineInputCount", "")] == 2
    assert counts[("MatchProviderBaselineInputCount", "")] == 2
    assert counts[("MatchActiveBaselineEntityCount", "")] == 2
    assert counts[("MatchResultsCount", "")] == 2


def test_history_match(spark):
    assert index_metrics_by_key(
        shared.compute_history_match_metrics(build_metric_inputs(spark))
    ) == {
        ("MatchHistoryMatchCount", ""): 1,
        ("MatchHistoryMatchRate", ""): 33.33,
    }


def test_match_type_distribution(spark):
    rows = index_metrics_by_key(
        places.compute_match_type_distribution(build_metric_inputs(spark))
    )
    assert rows[("MatchTypeDistribution", "candidate_baseline")] == 1
    assert rows[("MatchCandidateDuplicates", "")] == 2


def test_match_result_breakdown(spark):
    rows = index_metrics_by_key(
        shared.compute_match_result_breakdown(build_metric_inputs(spark))
    )
    assert rows[("MatchNewEntityCount", "")] == 1
    assert rows[("MatchRetainedEntityCount", "")] == 1
    assert rows[("MatchResurrectedEntityCount", "")] == 0
    assert rows[("MatchNewProviderRecordCount", "")] == 1
    assert rows[("MatchReductionCount", "")] == 1


def test_cluster_stats(spark):
    rows = index_metrics_by_key(
        places.compute_cluster_stats(build_metric_inputs(spark))
    )
    for kind in ("match", "candidates-only", "no-match"):
        assert rows[("MatchClusterCountByType", kind)] == 1
    assert rows[("MatchClusterSizeStatsAvg", "")] == 2
    assert rows[("MatchClusterCandidateSizeStatsMax", "")] == 2


def test_cluster_size_by_country(spark):
    inputs = build_metric_inputs(spark)
    inputs = inputs._replace(
        prepared_candidates=spark.createDataFrame(
            [("a", "US"), ("b", "CA"), ("c", "CA")], "id string, country string"
        ),
        prepared_baseline=spark.createDataFrame(
            [("B1", "US")], "id string, country string"
        ),
    )
    rows = index_metrics_by_key(places.compute_cluster_size_by_country(inputs))
    assert rows[("MatchClusterStatsByCountryTotal", "US")] == 1
    assert rows[("MatchClusterStatsByCountryMax", "CA")] == 2


def test_gers_uniqueness(spark):
    inputs = build_metric_inputs(spark)
    assert places.compute_gers_id_uniqueness(inputs) == [
        ("MatchGERSIdUniquenessCheck", 1.0, "")
    ]
    duplicated_results = inputs.results.unionByName(inputs.results)
    assert (
        places.compute_gers_id_uniqueness(inputs._replace(results=duplicated_results))[
            0
        ][1]
        == 0.0
    )


def test_match_rate_without_blocking(spark):
    rows = index_metrics_by_key(places.compute_match_rates(build_metric_inputs(spark)))
    assert rows == {("MatchRateIngest", ""): pytest.approx(100 / 3)}


def test_blocking_metrics_need_blocking(spark):
    assert places.compute_blocking_metrics(build_metric_inputs(spark)) == []


def test_round_half_up():
    assert shared.round_half_up(0.125, 2) == 0.13


def test_compute_metrics_skips_a_failing_metric(spark):
    def raise_error(_):
        raise ValueError

    rows = shared.compute_metrics(
        build_metric_inputs(spark),
        [("boom", raise_error), ("History", shared.compute_history_match_metrics)],
    )
    assert len(rows) == 2


def test_build_metrics_dataframe_keeps_first_row_per_key(spark):
    frame = shared.build_metrics_dataframe(
        spark,
        [("M", 1.0, ""), ("M", 2.0, ""), ("M", 3.0, "x")],
        "places",
        "place",
        "acme",
    )
    rows = {(row["metric_id"], row["sub_partition"]): row for row in frame.collect()}
    assert {key: row["value"] for key, row in rows.items()} == {
        ("M", ""): 1.0,
        ("M", "x"): 3.0,
    }
    assert {row["primary_partition"] for row in rows.values()} == {"acme-places"}


def test_feature_stats_with_a_model_jar(spark, tmp_path):
    reference = tmp_path / "reference"
    spark.createDataFrame([(0.0,), (2.0,)], "f double").coalesce(1).write.parquet(
        str(reference)
    )
    jar_path = tmp_path / "model.jar"
    with zipfile.ZipFile(jar_path, "w") as jar_file:
        jar_file.writestr(
            model_jar.CONFIG_RESOURCE, json.dumps({"feature_order": ["f"]})
        )
        jar_file.write(next(reference.glob("*.parquet")), model_jar.REFERENCE_RESOURCE)
    sample = spark.createDataFrame([(3.0,), (5.0,)], "f double")
    rows = index_metrics_by_key(
        places.compute_feature_stats(
            build_metric_inputs(
                spark,
                score_sample=sample,
                feature_order=model_jar.read_feature_order(spark, str(jar_path)),
                reference_dataset=model_jar.read_reference_dataset(
                    spark, str(jar_path), str(tmp_path / "run")
                ),
            )
        )
    )
    assert rows[("MatchFeature.f", "mean")] == 4.0
    # (4 - 1) / sqrt(2)
    assert rows[("MatchFeatureZScore", "f")] == 2.1213


def test_summarize_match_clusters_keeps_a_null_candidate_id(spark):
    matches = spark.createDataFrame(
        [
            ("k", None, "candidate", "b", "baseline", "match", "candidate_baseline"),
            ("k", "b", "baseline", "c", "candidate", "match", "candidate_baseline"),
        ],
        "cluster_id string, id string, record_type string, base_id string, "
        "base_record_type string, match_type string, match_sub_type string",
    )
    [row] = places.summarize_match_clusters(matches).collect()
    assert sorted(row["candidate_ids"], key=str) == [None, "c"]
    assert row["candidate_count"] == 2


def test_blocking_metrics_and_relative_match_rate(spark):
    blocking = spark.createDataFrame(
        [("b", "candidate", "B1", "baseline"), ("b", "candidate", "c", "candidate")],
        "id string, record_type string, base_id string, base_record_type string",
    )
    inputs = build_metric_inputs(spark, blocking=blocking)
    rows = index_metrics_by_key(places.compute_blocking_metrics(inputs))
    assert rows[("MatchBlockingTotalCandidates", "")] == 2
    assert rows[("MatchBlockingPassedRate", "")] == 1.0
    assert rows[("MatchBlockingVolume", "")] == 2
    assert rows[("MatchBlockingClusterDistribution", "mean")] == 1.5
    assert rows[("MatchBlockingClusterDistribution", "max")] == 2
    # The one match is a history match, so none is left for the relative rate.
    assert (
        index_metrics_by_key(places.compute_match_rates(inputs))[
            ("MatchRateRelative", "")
        ]
        == 0.0
    )


def test_score_distribution(spark):
    scores = spark.createDataFrame([(0.05,), (0.5,), (1.0,)], "composite_score double")
    rows = index_metrics_by_key(
        places.compute_score_distribution(build_metric_inputs(spark, scores=scores))
    )
    assert rows[("MatchScoreDistribution", "clear_no_match")] == 1
    assert rows[("MatchScoreDistribution", "uncertain_no_match")] == 0
    assert rows[("MatchScoreDistributionShare", "uncertain")] == pytest.approx(100 / 3)
    assert (
        sum(
            count
            for (metric_id, _), count in rows.items()
            if metric_id == "MatchScoreDistribution"
        )
        == 3
    )
    empty = scores.limit(0)
    assert (
        places.compute_score_distribution(build_metric_inputs(spark, scores=empty))
        == []
    )
    assert places.compute_score_distribution(build_metric_inputs(spark)) == []


def test_feature_stats_against_the_previous_run(spark):
    sample = spark.createDataFrame([(3.0, 1.0), (5.0, 1.0)], "f double, g double")
    previous_metrics = spark.createDataFrame(
        [("MatchFeature.f", "mean", 2.0), ("MatchFeature.f", "std", 1.0)],
        "metric_id string, sub_partition string, value double",
    )
    reference_dataset = spark.createDataFrame([(1.0,)], "other double")
    rows = index_metrics_by_key(
        places.compute_feature_stats(
            build_metric_inputs(
                spark,
                score_sample=sample,
                feature_order=("f", "g"),
                previous_metrics=previous_metrics,
                reference_dataset=reference_dataset,
            )
        )
    )
    assert rows[("MatchFeatureZScorePrevRelease", "f")] == 2.0
    assert ("MatchFeatureZScorePrevRelease", "g") not in rows
    assert not any(metric_id == "MatchFeatureZScore" for metric_id, _ in rows)
    assert (
        places.compute_feature_stats(build_metric_inputs(spark, score_sample=sample))
        == []
    )
