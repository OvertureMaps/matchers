# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places metrics: cluster shape, match rates, blocking, and score profiles."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from overture_matchers.metrics import (
    SHARED_METRICS,
    Metric,
    MetricInputs,
    MetricRow,
    percentage,
    round_half_up,
)
from overture_matchers.places.selection import (
    BASELINE_BASELINE,
    CANDIDATE,
    CANDIDATE_CANDIDATE,
    NOMATCH,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# Sampled score files stop once they add up to this many bytes.
SAMPLE_BYTES = 200 * 1024 * 1024
SAMPLE_SEED = 42

SCORE_BUCKETS = (
    ("clear_no_match", 0.0, 0.1),
    ("uncertain_no_match", 0.1, 0.3),
    ("uncertain", 0.3, 0.7),
    ("uncertain_match", 0.7, 0.9),
    # The upper bound is above 1 so that a score of 1.0 lands in the last bucket.
    ("clear_match", 0.9, 1.01),
)


def _num(value) -> float:
    return 0.0 if value is None else float(value)


def _pct(column: str, q: float):
    return F.expr(f"percentile({column}, {q})")


def _stat_rows(
    metric_id: str, row, names: dict[str, str], sub: str = ""
) -> list[MetricRow]:
    """One row per entry of ``names``, which maps a Row field to a metric suffix."""
    return [
        (metric_id + suffix, _num(row[field]), sub) for field, suffix in names.items()
    ]


def _distinct_ids(pairs: DataFrame) -> int:
    return (
        pairs.select("id")
        .union(pairs.select(F.col("base_id").alias("id")))
        .distinct()
        .count()
    )


def match_type_distribution(inp: MetricInputs) -> list[MetricRow]:
    """Pairs by sub-type, and the share of IDs that have duplicates on their own side."""
    by_type = (
        inp.matches.groupBy("match_sub_type").agg(F.count("id").alias("n")).collect()
    )
    candidate_dups = _distinct_ids(
        inp.matches.where(F.col("match_sub_type") == CANDIDATE_CANDIDATE)
    )
    baseline_dups = _distinct_ids(
        inp.matches.where(F.col("match_sub_type") == BASELINE_BASELINE)
    )
    return [
        ("MatchTypeDistribution", float(r["n"]), r["match_sub_type"] or "")
        for r in by_type
    ] + [
        ("MatchCandidateDuplicates", float(candidate_dups), ""),
        ("MatchBaselineDuplicates", float(baseline_dups), ""),
        (
            "MatchCandidateDuplicateShare",
            percentage(candidate_dups, inp.prepared_candidates.count()),
            "",
        ),
        (
            "MatchBaselineDuplicateShare",
            percentage(baseline_dups, inp.baseline.count()),
            "",
        ),
    ]


def clusters(matches: DataFrame) -> DataFrame:
    """One row per cluster.

    Columns: ``cluster_id``, ``type`` (baseline-only, candidates-only, no-match or
    match), ``entity_count`` (distinct IDs on either side, a null ``base_id``
    included), ``candidate_ids``, ``candidate_count`` and ``first_id``.
    """

    def all_of(sub_type: str):
        return F.sum((F.col("match_sub_type") == sub_type).cast("int")) == F.count(
            F.lit(1)
        )

    # Structs keep a null candidate ID apart from a non-candidate side.
    candidate_side = F.array(
        F.struct(
            (F.col("record_type") == CANDIDATE).alias("c"), F.col("id").alias("v")
        ),
        F.struct(
            (F.col("base_record_type") == CANDIDATE).alias("c"),
            F.col("base_id").alias("v"),
        ),
    )
    return (
        matches.groupBy("cluster_id")
        .agg(
            F.when(all_of(BASELINE_BASELINE), "baseline-only")
            .when(all_of(CANDIDATE_CANDIDATE), "candidates-only")
            .when(all_of(NOMATCH), "no-match")
            .otherwise("match")
            .alias("type"),
            F.array_distinct(F.flatten(F.collect_list(F.array("id", "base_id")))).alias(
                "entities"
            ),
            F.flatten(F.collect_list(candidate_side)).alias("cands"),
            F.min("id").alias("first_id"),
        )
        .select(
            "cluster_id",
            "type",
            F.size("entities").alias("entity_count"),
            F.array_distinct(
                F.transform(F.filter("cands", lambda x: x["c"]), lambda x: x["v"])
            ).alias("candidate_ids"),
            "first_id",
        )
        .withColumn("candidate_count", F.size("candidate_ids"))
    )


# simplification: keeps the last matches table's clusters cached until the next call.
@lru_cache(maxsize=1)
def _cached_clusters(matches: DataFrame) -> DataFrame:
    return clusters(matches).cache()


def _size_stats(column: str) -> list[F.Column]:
    c = F.col(column)
    return [
        _pct(column, 0.99).alias("p99"),
        _pct(column, 0.75).alias("p75"),
        _pct(column, 0.5).alias("p50"),
        _pct(column, 0.25).alias("p25"),
        F.avg(c).alias("avg"),
        F.stddev(c).alias("std"),
        F.min(c).alias("min"),
        F.max(c).alias("max"),
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


def cluster_stats(inp: MetricInputs) -> list[MetricRow]:
    """Clusters by type, singleton rate, duplicated IDs and cluster size profiles."""
    cl = _cached_clusters(inp.matches)
    by_type = {
        r["type"]: r["n"]
        for r in cl.groupBy("type").agg(F.count(F.lit(1)).alias("n")).collect()
    }
    relevant = cl.where(F.col("type").isin("match", "candidates-only"))
    sizes = relevant.agg(*_size_stats("entity_count")).first()
    candidate_sizes = relevant.agg(*_size_stats("candidate_count")).first()
    duplicated = cl.where(F.col("entity_count") - F.col("candidate_count") > 1).count()
    rows: list[MetricRow] = [
        ("MatchClusterCountByType", float(n), t) for t, n in by_type.items()
    ]
    rows.append(
        (
            "MatchSingletonRate",
            percentage(by_type.get("no-match", 0), inp.prepared_candidates.count()),
            "",
        )
    )
    rows.append(("MatchDuplicatedIdCount", float(duplicated), ""))
    rows += _stat_rows("MatchClusterSizeStats", sizes, CLUSTER_SIZE_SUFFIXES)
    rows += _stat_rows(
        "MatchClusterCandidateSizeStats", candidate_sizes, CLUSTER_SIZE_SUFFIXES
    )
    return rows


def cluster_size_by_country(inp: MetricInputs) -> list[MetricRow]:
    """Candidate count per cluster, profiled by the country of the cluster's smallest ID.

    An ID in both candidates and baseline has two country rows, so its cluster is
    counted once per row.
    """
    countries = inp.prepared_candidates.select("id", "country").unionByName(
        inp.prepared_baseline.select("id", "country")
    )
    relevant = _cached_clusters(inp.matches).where(
        F.col("type").isin("match", "candidates-only")
    )
    stats = (
        relevant.join(countries, relevant["first_id"] == countries["id"])
        .groupBy("country")
        .agg(
            _pct("candidate_count", 0.99).alias("p99"),
            _pct("candidate_count", 0.95).alias("p95"),
            _pct("candidate_count", 0.75).alias("p75"),
            _pct("candidate_count", 0.5).alias("p50"),
            _pct("candidate_count", 0.25).alias("p25"),
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
        rows += _stat_rows(
            "MatchClusterStatsByCountry", row, suffixes, row["country"] or ""
        )
    return rows


def gers_uniqueness(inp: MetricInputs) -> list[MetricRow]:
    """1 if every result has its own ID, else 0."""
    unique = inp.results.count() == inp.results.select("id").distinct().count()
    return [("MatchGERSIdUniquenessCheck", 1.0 if unique else 0.0, "")]


def _candidate_feed(inp: MetricInputs) -> DataFrame:
    """Candidates that setup did not already match through history."""
    return inp.prepared_candidates.where(F.col("base_ids").isNull())


def _passed_blocking(feed: DataFrame, blocking: DataFrame) -> DataFrame:
    """Feed records whose ID appears on either side of a blocking pair."""
    ids = (
        blocking.where(F.col("record_type") == CANDIDATE)
        .select("id")
        .union(
            blocking.where(F.col("base_record_type") == CANDIDATE).select(
                F.col("base_id").alias("id")
            )
        )
        .distinct()
    )
    return feed.join(ids, "id", "left_semi")


def blocking_metrics(inp: MetricInputs) -> list[MetricRow]:
    """Pass and block rates of the candidate feed, and partners per candidate."""
    if inp.blocking is None:
        return []
    feed = _candidate_feed(inp)
    total = feed.count()
    passed = _passed_blocking(feed, inp.blocking).count()
    blocked_rate = (total - passed) / total if total > 0 else 0.0

    blocking = inp.blocking
    per_candidate = (
        blocking.where(F.col("record_type") == CANDIDATE)
        .groupBy("id")
        .agg(F.count("base_id").alias("n"))
        .union(
            blocking.where(F.col("base_record_type") == CANDIDATE)
            .groupBy(F.col("base_id").alias("id"))
            .agg(F.count("id").alias("n"))
        )
        .groupBy("id")
        .agg(F.sum("n").alias("n"))
        .join(feed.select("id"), "id")
    )
    partners = per_candidate.agg(
        F.avg("n").alias("mean"),
        F.stddev("n").alias("std"),
        _pct("n", 0.5).alias("median"),
        _pct("n", 0.99).alias("p99"),
        F.max("n").alias("max"),
    ).first()
    rows: list[MetricRow] = [
        ("MatchBlockingTotalCandidates", float(total), ""),
        ("MatchBlockingPassedCandidates", float(passed), ""),
        ("MatchBlockingBlockedCandidates", float(total - passed), ""),
        ("MatchBlockingBlockedRate", blocked_rate, ""),
        ("MatchBlockingPassedRate", 1.0 - blocked_rate, ""),
        ("MatchBlockingVolume", float(blocking.count()), ""),
    ]
    for stat in ("mean", "std", "median", "p99", "max"):
        rows.append(("MatchBlockingClusterDistribution", _num(partners[stat]), stat))
    return rows


def match_rate(inp: MetricInputs) -> list[MetricRow]:
    """Share of the candidate feed that matched a baseline record.

    The ingest rate is over every candidate. The relative rate leaves out history
    matches and candidates that did not pass blocking, so it needs ``blocking``.
    """
    candidates = inp.prepared_candidates.count()
    history = inp.prepared_candidates.where(F.col("base_ids").isNotNull()).count()
    matched = (
        _cached_clusters(inp.matches)
        .where(F.col("type") == "match")
        .select(F.explode("candidate_ids").alias("id"))
        .distinct()
        .count()
    )
    rows: list[MetricRow] = [("MatchRateIngest", percentage(matched, candidates), "")]
    if inp.blocking is not None:
        passed = _passed_blocking(_candidate_feed(inp), inp.blocking).count()
        rows.append(("MatchRateRelative", percentage(matched - history, passed), ""))
    return rows


def sample_score_files(spark: SparkSession, scores: DataFrame) -> DataFrame:
    """Read a random subset of the score files, about ``SAMPLE_BYTES`` in total.

    Scores are hash-partitioned by pair, so any subset of files is representative.
    Reads every file if the whole table is smaller than the target.
    """
    files = [f for f in scores.inputFiles() if f.endswith(".parquet")]
    random.Random(SAMPLE_SEED).shuffle(files)
    jvm, conf = spark._jvm, spark._jsc.hadoopConfiguration()
    chosen, total = [], 0
    for f in files:
        if total >= SAMPLE_BYTES:
            break
        path = jvm.org.apache.hadoop.fs.Path(f)
        total += path.getFileSystem(conf).getFileStatus(path).getLen()
        chosen.append(f)
    return spark.read.parquet(*(chosen or files))


def _z_score(current: float, mean: float, std: float) -> float:
    return round_half_up((current - mean) / std, 4)


def feature_stats(inp: MetricInputs) -> list[MetricRow]:
    """Mean, std, median, p10 and p90 per scoring feature, and drift z-scores.

    Drift is measured against the model's reference dataset and against the previous
    run's metrics, when each is available.
    """
    if inp.score_sample is None or not inp.feature_order:
        return []
    features = inp.feature_order
    sample = inp.score_sample.select(
        *(F.col(f).cast("double").alias(f) for f in features)
    )
    aggs = []
    for i, f in enumerate(features):
        aggs += [
            F.mean(f).alias(f"mean{i}"),
            F.stddev(f).alias(f"std{i}"),
            _pct(f, 0.5).alias(f"median{i}"),
            _pct(f, 0.1).alias(f"p10{i}"),
            _pct(f, 0.9).alias(f"p90{i}"),
        ]
    row = sample.agg(*aggs).first()
    rows: list[MetricRow] = []
    means = {}
    for i, f in enumerate(features):
        means[f] = _num(row[f"mean{i}"])
        for stat in ("mean", "std", "median", "p10", "p90"):
            value = row[f"{stat}{i}"]
            if value is not None:
                rows.append((f"MatchFeature.{f}", float(value), stat))
    rows += _reference_z_scores(inp.reference, features, means)
    rows += _previous_z_scores(inp.previous, features, means)
    return rows


def _reference_z_scores(reference, features, means) -> list[MetricRow]:
    if reference is None:
        return []
    available = [f for f in features if f in reference.columns]
    if not available:
        return []
    aggs = []
    for i, f in enumerate(available):
        c = F.col(f).cast("double")
        aggs += [F.avg(c).alias(f"mean{i}"), F.stddev(c).alias(f"std{i}")]
    row = reference.agg(*aggs).first()
    rows = []
    for i, f in enumerate(available):
        std = _num(row[f"std{i}"])
        if std > 0:
            rows.append(
                (
                    "MatchFeatureZScore",
                    _z_score(means[f], _num(row[f"mean{i}"]), std),
                    f,
                )
            )
    return rows


def _previous_z_scores(previous, features, means) -> list[MetricRow]:
    if previous is None:
        return []
    prefix = "MatchFeature."
    stats: dict[str, dict[str, float]] = {}
    for r in previous.where(F.col("metric_id").startswith(prefix)).collect():
        stats.setdefault(r["metric_id"][len(prefix) :], {})[r["sub_partition"]] = float(
            r["value"]
        )
    rows = []
    for f in features:
        prev = stats.get(f)
        if prev is not None and prev.get("std", 0.0) > 0:
            rows.append(
                (
                    "MatchFeatureZScorePrevRelease",
                    _z_score(means[f], prev.get("mean", 0.0), prev["std"]),
                    f,
                )
            )
    return rows


def score_distribution(inp: MetricInputs) -> list[MetricRow]:
    """Pairs, and their share, in each ``composite_score`` bucket."""
    if inp.scores is None:
        return []
    score = F.col("composite_score")
    row = inp.scores.agg(
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
    return [("MatchScoreDistribution", n, name) for name, n in counts] + [
        ("MatchScoreDistributionShare", n / total * 100, name) for name, n in counts
    ]


PLACES_METRICS: tuple[tuple[str, Metric], ...] = (
    ("Places Match Type Distribution", match_type_distribution),
    ("Places Cluster Stats", cluster_stats),
    ("Cluster Sizes By Country", cluster_size_by_country),
    ("GERS ID Uniqueness", gers_uniqueness),
    ("Places Blocking Metrics", blocking_metrics),
    ("Places Match Rates", match_rate),
    ("Places Feature Stats", feature_stats),
    ("Places Score Distribution", score_distribution),
)

METRICS = SHARED_METRICS + PLACES_METRICS
