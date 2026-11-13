# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places metrics: cluster shape, match rates, blocking, and score profiles."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from overture_matchers.clustering import (
    BASELINE_BASELINE,
    CANDIDATE_CANDIDATE,
    NO_MATCH,
)
from overture_matchers.metrics import (
    SHARED_METRICS,
    Metric,
    MetricInputs,
    MetricRow,
    compute_percentage,
    round_half_up,
)
from overture_matchers.prematch import CANDIDATE

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, Row

SCORE_BUCKETS = (
    ("clear_no_match", 0.0, 0.1),
    ("uncertain_no_match", 0.1, 0.3),
    ("uncertain", 0.3, 0.7),
    ("uncertain_match", 0.7, 0.9),
    # The upper bound is above 1 so that a score of 1.0 lands in the last bucket.
    ("clear_match", 0.9, 1.01),
)


def _coerce_to_float(value: object) -> float:
    return 0.0 if value is None else float(value)


def _compute_exact_percentile(column_name: str, quantile: float) -> Column:
    return F.expr(f"percentile({column_name}, {quantile})")


def _build_stat_metric_rows(
    metric_id_prefix: str,
    row: Row,
    suffix_by_field: dict[str, str],
    sub_partition: str = "",
) -> list[MetricRow]:
    """Build one metric row per entry of ``suffix_by_field``."""
    return [
        (metric_id_prefix + suffix, _coerce_to_float(row[field]), sub_partition)
        for field, suffix in suffix_by_field.items()
    ]


def _count_distinct_pair_ids(pairs: DataFrame) -> int:
    return (
        pairs.select("id")
        .union(pairs.select(F.col("base_id").alias("id")))
        .distinct()
        .count()
    )


def compute_match_type_distribution(inputs: MetricInputs) -> list[MetricRow]:
    """Compute pairs by sub-type, and the share of IDs duplicated on their own side."""
    by_type = (
        inputs.matches.groupBy("match_sub_type")
        .agg(F.count("id").alias("count"))
        .collect()
    )
    candidate_duplicate_count = _count_distinct_pair_ids(
        inputs.matches.where(F.col("match_sub_type") == CANDIDATE_CANDIDATE)
    )
    baseline_duplicate_count = _count_distinct_pair_ids(
        inputs.matches.where(F.col("match_sub_type") == BASELINE_BASELINE)
    )
    return [
        ("MatchTypeDistribution", float(row["count"]), row["match_sub_type"] or "")
        for row in by_type
    ] + [
        ("MatchCandidateDuplicates", float(candidate_duplicate_count), ""),
        ("MatchBaselineDuplicates", float(baseline_duplicate_count), ""),
        (
            "MatchCandidateDuplicateShare",
            compute_percentage(
                candidate_duplicate_count, inputs.prepared_candidates.count()
            ),
            "",
        ),
        (
            "MatchBaselineDuplicateShare",
            compute_percentage(baseline_duplicate_count, inputs.baseline.count()),
            "",
        ),
    ]


def summarize_match_clusters(matches: DataFrame) -> DataFrame:
    """Summarize ``matches`` to one row per cluster.

    Columns: ``cluster_id``, ``type`` (baseline-only, candidates-only, no-match or
    match), ``entity_count`` (distinct IDs on either side, a null ``base_id``
    included), ``candidate_ids``, ``candidate_count`` and ``first_id``.
    """

    def all_rows_have_sub_type(sub_type: str) -> Column:
        return F.sum((F.col("match_sub_type") == sub_type).cast("int")) == F.count(
            F.lit(1)
        )

    # Structs keep a null candidate ID apart from a non-candidate side.
    candidate_side = F.array(
        F.struct(
            (F.col("record_type") == CANDIDATE).alias("is_candidate"),
            F.col("id").alias("entity_id"),
        ),
        F.struct(
            (F.col("base_record_type") == CANDIDATE).alias("is_candidate"),
            F.col("base_id").alias("entity_id"),
        ),
    )
    return (
        matches.groupBy("cluster_id")
        .agg(
            F.when(all_rows_have_sub_type(BASELINE_BASELINE), "baseline-only")
            .when(all_rows_have_sub_type(CANDIDATE_CANDIDATE), "candidates-only")
            .when(all_rows_have_sub_type(NO_MATCH), "no-match")
            .otherwise("match")
            .alias("type"),
            F.array_distinct(F.flatten(F.collect_list(F.array("id", "base_id")))).alias(
                "entities"
            ),
            F.flatten(F.collect_list(candidate_side)).alias("candidate_entries"),
            F.min("id").alias("first_id"),
        )
        .select(
            "cluster_id",
            "type",
            F.size("entities").alias("entity_count"),
            F.array_distinct(
                F.transform(
                    F.filter("candidate_entries", lambda entry: entry["is_candidate"]),
                    lambda entry: entry["entity_id"],
                )
            ).alias("candidate_ids"),
            "first_id",
        )
        .withColumn("candidate_count", F.size("candidate_ids"))
    )


# Keeps the last matches table's clusters cached until the next call.
@lru_cache(maxsize=1)
def _get_cached_match_clusters(matches: DataFrame) -> DataFrame:
    return summarize_match_clusters(matches).cache()


def _build_size_stat_columns(column_name: str) -> list[Column]:
    column = F.col(column_name)
    return [
        _compute_exact_percentile(column_name, 0.99).alias("p99"),
        _compute_exact_percentile(column_name, 0.75).alias("p75"),
        _compute_exact_percentile(column_name, 0.5).alias("p50"),
        _compute_exact_percentile(column_name, 0.25).alias("p25"),
        F.avg(column).alias("avg"),
        F.stddev(column).alias("std"),
        F.min(column).alias("min"),
        F.max(column).alias("max"),
    ]


CLUSTER_SIZE_SUFFIXES = {
    "p99": "P99",
    "p75": "P75",
    "p50": "P50",
    "p25": "P25",
    "avg": "Avg",
    "std": "Std",
    "min": "Min",
    "max": "Max",
}


def compute_cluster_stats(inputs: MetricInputs) -> list[MetricRow]:
    """Compute clusters by type, singleton rate, duplicated IDs and size profiles."""
    clusters_df = _get_cached_match_clusters(inputs.matches)
    by_type = {
        row["type"]: row["count"]
        for row in clusters_df.groupBy("type")
        .agg(F.count(F.lit(1)).alias("count"))
        .collect()
    }
    relevant = clusters_df.where(F.col("type").isin("match", "candidates-only"))
    sizes = relevant.agg(*_build_size_stat_columns("entity_count")).first()
    candidate_sizes = relevant.agg(*_build_size_stat_columns("candidate_count")).first()
    duplicated = clusters_df.where(
        F.col("entity_count") - F.col("candidate_count") > 1
    ).count()
    rows: list[MetricRow] = [
        ("MatchClusterCountByType", float(count), cluster_type)
        for cluster_type, count in by_type.items()
    ]
    rows.append(
        (
            "MatchSingletonRate",
            compute_percentage(
                by_type.get("no-match", 0), inputs.prepared_candidates.count()
            ),
            "",
        )
    )
    rows.append(("MatchDuplicatedIdCount", float(duplicated), ""))
    rows += _build_stat_metric_rows(
        "MatchClusterSizeStats", sizes, CLUSTER_SIZE_SUFFIXES
    )
    rows += _build_stat_metric_rows(
        "MatchClusterCandidateSizeStats", candidate_sizes, CLUSTER_SIZE_SUFFIXES
    )
    return rows


def compute_cluster_size_by_country(inputs: MetricInputs) -> list[MetricRow]:
    """Profile candidate count per cluster by the country of its smallest ID.

    An ID in both candidates and baseline has two country rows, so its cluster is
    counted once per row.
    """
    countries = inputs.prepared_candidates.select("id", "country").unionByName(
        inputs.prepared_baseline.select("id", "country")
    )
    relevant = _get_cached_match_clusters(inputs.matches).where(
        F.col("type").isin("match", "candidates-only")
    )
    stats = (
        relevant.join(countries, relevant["first_id"] == countries["id"])
        .groupBy("country")
        .agg(
            _compute_exact_percentile("candidate_count", 0.99).alias("p99"),
            _compute_exact_percentile("candidate_count", 0.95).alias("p95"),
            _compute_exact_percentile("candidate_count", 0.75).alias("p75"),
            _compute_exact_percentile("candidate_count", 0.5).alias("p50"),
            _compute_exact_percentile("candidate_count", 0.25).alias("p25"),
            F.max("candidate_count").alias("max"),
            F.min("candidate_count").alias("min"),
            F.avg("candidate_count").alias("avg"),
            F.count("cluster_id").alias("total"),
        )
        .collect()
    )
    suffixes = {
        "p99": "P99",
        "p95": "P95",
        "p75": "P75",
        "p50": "P50",
        "p25": "P25",
        "max": "Max",
        "min": "Min",
        "avg": "Avg",
        "total": "Total",
    }
    rows: list[MetricRow] = []
    for row in stats:
        rows += _build_stat_metric_rows(
            "MatchClusterStatsByCountry", row, suffixes, row["country"] or ""
        )
    return rows


def compute_gers_id_uniqueness(inputs: MetricInputs) -> list[MetricRow]:
    """Return 1 if every result row has a distinct GERS ID, else 0."""
    unique = inputs.results.count() == inputs.results.select("id").distinct().count()
    return [("MatchGERSIdUniquenessCheck", 1.0 if unique else 0.0, "")]


def _filter_non_history_candidates(inputs: MetricInputs) -> DataFrame:
    """Filter to candidates that setup did not already match through history."""
    return inputs.prepared_candidates.where(F.col("base_ids").isNull())


def _filter_candidates_passing_blocking(
    candidate_feed: DataFrame, blocking_pairs: DataFrame
) -> DataFrame:
    """Filter ``candidate_feed`` to IDs appearing on either side of ``blocking_pairs``."""
    passing_ids = (
        blocking_pairs.where(F.col("record_type") == CANDIDATE)
        .select("id")
        .union(
            blocking_pairs.where(F.col("base_record_type") == CANDIDATE).select(
                F.col("base_id").alias("id")
            )
        )
        .distinct()
    )
    return candidate_feed.join(passing_ids, "id", "left_semi")


def compute_blocking_metrics(inputs: MetricInputs) -> list[MetricRow]:
    """Compute pass and block rates of the candidate feed, and partners per candidate."""
    if inputs.blocking is None:
        return []
    candidate_feed = _filter_non_history_candidates(inputs)
    total = candidate_feed.count()
    passed = _filter_candidates_passing_blocking(
        candidate_feed, inputs.blocking
    ).count()
    blocked_rate = (total - passed) / total if total > 0 else 0.0

    blocking_pairs = inputs.blocking
    per_candidate = (
        blocking_pairs.where(F.col("record_type") == CANDIDATE)
        .groupBy("id")
        .agg(F.count("base_id").alias("partner_count"))
        .union(
            blocking_pairs.where(F.col("base_record_type") == CANDIDATE)
            .groupBy(F.col("base_id").alias("id"))
            .agg(F.count("id").alias("partner_count"))
        )
        .groupBy("id")
        .agg(F.sum("partner_count").alias("partner_count"))
        .join(candidate_feed.select("id"), "id")
    )
    partners = per_candidate.agg(
        F.avg("partner_count").alias("mean"),
        F.stddev("partner_count").alias("std"),
        _compute_exact_percentile("partner_count", 0.5).alias("median"),
        _compute_exact_percentile("partner_count", 0.99).alias("p99"),
        F.max("partner_count").alias("max"),
    ).first()
    rows: list[MetricRow] = [
        ("MatchBlockingTotalCandidates", float(total), ""),
        ("MatchBlockingPassedCandidates", float(passed), ""),
        ("MatchBlockingBlockedCandidates", float(total - passed), ""),
        ("MatchBlockingBlockedRate", blocked_rate, ""),
        ("MatchBlockingPassedRate", 1.0 - blocked_rate, ""),
        ("MatchBlockingVolume", float(blocking_pairs.count()), ""),
    ]
    for stat in ("mean", "std", "median", "p99", "max"):
        rows.append(
            ("MatchBlockingClusterDistribution", _coerce_to_float(partners[stat]), stat)
        )
    return rows


def compute_match_rates(inputs: MetricInputs) -> list[MetricRow]:
    """Compute the share of candidates that matched a baseline record.

    The ingest rate is over every candidate. The relative rate leaves out history
    matches and candidates that did not pass blocking, so it needs ``blocking``.
    """
    candidates = inputs.prepared_candidates.count()
    history = inputs.prepared_candidates.where(F.col("base_ids").isNotNull()).count()
    matched = (
        _get_cached_match_clusters(inputs.matches)
        .where(F.col("type") == "match")
        .select(F.explode("candidate_ids").alias("id"))
        .distinct()
        .count()
    )
    rows: list[MetricRow] = [
        ("MatchRateIngest", compute_percentage(matched, candidates), "")
    ]
    if inputs.blocking is not None:
        passed = _filter_candidates_passing_blocking(
            _filter_non_history_candidates(inputs), inputs.blocking
        ).count()
        rows.append(
            ("MatchRateRelative", compute_percentage(matched - history, passed), "")
        )
    return rows


def _compute_z_score(
    current_mean: float,
    baseline_mean: float,
    baseline_standard_deviation: float,
) -> float:
    return round_half_up(
        (current_mean - baseline_mean) / baseline_standard_deviation, 4
    )


def compute_feature_stats(inputs: MetricInputs) -> list[MetricRow]:
    """Compute mean, std, median, p10 and p90 per scoring feature, and drift z-scores.

    Drift is measured against the model's reference dataset and against the previous
    run's metrics, when each is available.
    """
    if inputs.score_sample is None or not inputs.feature_order:
        return []
    feature_names = inputs.feature_order
    sample = inputs.score_sample.select(
        *(
            F.col(feature_name).cast("double").alias(feature_name)
            for feature_name in feature_names
        )
    )
    aggregations = []
    for index, feature_name in enumerate(feature_names):
        aggregations += [
            F.mean(feature_name).alias(f"mean{index}"),
            F.stddev(feature_name).alias(f"std{index}"),
            _compute_exact_percentile(feature_name, 0.5).alias(f"median{index}"),
            _compute_exact_percentile(feature_name, 0.1).alias(f"p10{index}"),
            _compute_exact_percentile(feature_name, 0.9).alias(f"p90{index}"),
        ]
    row = sample.agg(*aggregations).first()
    rows: list[MetricRow] = []
    current_means = {}
    for index, feature_name in enumerate(feature_names):
        current_means[feature_name] = _coerce_to_float(row[f"mean{index}"])
        for stat in ("mean", "std", "median", "p10", "p90"):
            value = row[f"{stat}{index}"]
            if value is not None:
                rows.append((f"MatchFeature.{feature_name}", float(value), stat))
    rows += _compute_reference_z_scores(
        inputs.reference_dataset, feature_names, current_means
    )
    rows += _compute_previous_release_z_scores(
        inputs.previous_metrics, feature_names, current_means
    )
    return rows


def _compute_reference_z_scores(
    reference_dataset: DataFrame | None,
    feature_names: tuple[str, ...],
    current_means: dict[str, float],
) -> list[MetricRow]:
    if reference_dataset is None:
        return []
    available_features = [
        feature_name
        for feature_name in feature_names
        if feature_name in reference_dataset.columns
    ]
    if not available_features:
        return []
    aggregations = []
    for index, feature_name in enumerate(available_features):
        feature_column = F.col(feature_name).cast("double")
        aggregations += [
            F.avg(feature_column).alias(f"mean{index}"),
            F.stddev(feature_column).alias(f"std{index}"),
        ]
    row = reference_dataset.agg(*aggregations).first()
    rows = []
    for index, feature_name in enumerate(available_features):
        standard_deviation = _coerce_to_float(row[f"std{index}"])
        if standard_deviation > 0:
            rows.append(
                (
                    "MatchFeatureZScore",
                    _compute_z_score(
                        current_means[feature_name],
                        _coerce_to_float(row[f"mean{index}"]),
                        standard_deviation,
                    ),
                    feature_name,
                )
            )
    return rows


def _compute_previous_release_z_scores(
    previous_metrics: DataFrame | None,
    feature_names: tuple[str, ...],
    current_means: dict[str, float],
) -> list[MetricRow]:
    if previous_metrics is None:
        return []
    prefix = "MatchFeature."
    stats_by_feature: dict[str, dict[str, float]] = {}
    for row in previous_metrics.where(F.col("metric_id").startswith(prefix)).collect():
        stats_by_feature.setdefault(row["metric_id"][len(prefix) :], {})[
            row["sub_partition"]
        ] = float(row["value"])
    rows = []
    for feature_name in feature_names:
        previous_stats = stats_by_feature.get(feature_name)
        if previous_stats is not None and previous_stats.get("std", 0.0) > 0:
            rows.append(
                (
                    "MatchFeatureZScorePrevRelease",
                    _compute_z_score(
                        current_means[feature_name],
                        previous_stats.get("mean", 0.0),
                        previous_stats["std"],
                    ),
                    feature_name,
                )
            )
    return rows


def compute_score_distribution(inputs: MetricInputs) -> list[MetricRow]:
    """Compute pair counts and shares in each ``composite_score`` bucket."""
    if inputs.scores is None:
        return []
    score = F.col("composite_score")
    row = inputs.scores.agg(
        F.count(F.lit(1)).alias("total"),
        *(
            F.count(F.when((score >= low) & (score < high), 1)).alias(name)
            for name, low, high in SCORE_BUCKETS
        ),
    ).first()
    total = row["total"]
    if total == 0:
        return []
    counts = [(name, float(row[name])) for name, _, _ in SCORE_BUCKETS]
    return [("MatchScoreDistribution", count, name) for name, count in counts] + [
        ("MatchScoreDistributionShare", count / total * 100, name)
        for name, count in counts
    ]


PLACES_METRICS: tuple[tuple[str, Metric], ...] = (
    ("Places Match Type Distribution", compute_match_type_distribution),
    ("Places Cluster Stats", compute_cluster_stats),
    ("Cluster Sizes By Country", compute_cluster_size_by_country),
    ("GERS ID Uniqueness", compute_gers_id_uniqueness),
    ("Places Blocking Metrics", compute_blocking_metrics),
    ("Places Match Rates", compute_match_rates),
    ("Places Feature Stats", compute_feature_stats),
    ("Places Score Distribution", compute_score_distribution),
)

METRICS = SHARED_METRICS + PLACES_METRICS
